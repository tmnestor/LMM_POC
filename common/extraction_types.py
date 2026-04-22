"""Foundation dataclasses for the agentic extraction engine.

Zero dependencies on existing modules -- safe to import anywhere.
All graph execution state, results, and session types live here.

Named ``NodeGenParams`` (not ``GenerationParams``) to avoid collision
with ``models/backend.py:GenerationParams`` which is a frozen dataclass
used by the orchestrator pipeline.
"""

from dataclasses import dataclass, field
from typing import Any

_MISSING = object()  # sentinel for WorkflowState.get() default


@dataclass
class NodeGenParams:
    """Per-node generation parameters built from YAML node definition.

    The generate_fn wrapper dispatches via ``match`` on these fields
    to build backend-specific sampling parameters (vLLM SamplingParams,
    HF GenerationConfig, etc.).
    """

    max_tokens: int = 4096
    temperature: float = 0.0
    stop: list[str] | None = None
    output_schema: dict | None = None  # JSON schema -> vLLM StructuredOutputsParams
    logprobs: int | None = None  # top-k logprobs per token


@dataclass
class GenerateResult:
    """Response from a single model call."""

    text: str
    logprobs: list[dict[str, float]] | None = None


@dataclass
class NodeResult:
    """Result of executing a single graph node."""

    key: str  # matches YAML node key
    image_ref: str  # which image this node ran against
    prompt_sent: str  # after template injection (empty for routers/validators)
    raw_response: str  # raw model output (empty for routers/validators)
    parsed: dict[str, Any]  # structured parse of this node's output
    elapsed: float  # seconds
    attempt: int  # 1-indexed retry attempt number
    edge_taken: str  # which edge was followed


@dataclass
class WorkflowState:
    """Typed state accumulated across graph execution.

    Implements the MetaGPT pattern: structured outputs between nodes
    prevent garbage propagation through the graph.
    """

    node_results: dict[str, NodeResult] = field(default_factory=dict)

    def get(self, dot_path: str, default: Any = _MISSING) -> Any:
        """Resolve a dot-path reference.

        Example: ``'detect_headers.column_mapping.balance'``

        If *default* is provided, return it when the path cannot be
        resolved (missing node or missing key).  Without a default,
        raise ``KeyError`` with a diagnostic message.
        """
        parts = dot_path.split(".")
        node_key = parts[0]
        if node_key not in self.node_results:
            if default is not _MISSING:
                return default
            msg = f"Node '{node_key}' not found in state (have: {list(self.node_results)})"
            raise KeyError(msg)
        result: Any = self.node_results[node_key].parsed
        for part in parts[1:]:
            try:
                if isinstance(result, dict):
                    result = result[part]
                else:
                    result = getattr(result, part)
            except (KeyError, AttributeError):
                if default is not _MISSING:
                    return default
                error_ctx = self.node_results[node_key].parsed.get("error", "")
                hint = f" (node parse failed: {error_ctx})" if error_ctx else ""
                msg = f"Cannot resolve '{dot_path}': key '{part}' missing{hint}"
                raise KeyError(msg) from None
        return result

    def has(self, node_key: str) -> bool:
        """Check whether a node has produced a result."""
        return node_key in self.node_results


@dataclass
class WorkflowTrace:
    """Observability record for full graph execution."""

    nodes_visited: list[str] = field(default_factory=list)
    edges_taken: list[tuple[str, str, str]] = field(
        default_factory=list
    )  # (from_node, edge, to_node)
    retries: dict[str, int] = field(default_factory=dict)
    total_model_calls: int = 0
    total_elapsed: float = 0.0


@dataclass
class ExtractionSession:
    """Complete graph execution for one image (or image pair)."""

    image_path: str
    image_name: str
    document_type: str
    node_results: list[NodeResult]
    final_fields: dict[str, str]
    strategy: str  # last extraction node key
    trace: WorkflowTrace
    input_images: dict[str, str] | None = None  # cross-image only

    @property
    def raw_response(self) -> str:
        """Flat FIELD: value string -- same contract as existing pipeline."""
        return "\n".join(f"{k}: {v}" for k, v in self.final_fields.items())

    def to_record(self) -> dict[str, Any]:
        """Serialize for raw_extractions.jsonl."""
        record: dict[str, Any] = {
            "image_name": self.image_name,
            "image_path": self.image_path,
            "document_type": self.document_type,
            "raw_response": self.raw_response,
            "nodes": [
                {
                    "key": n.key,
                    "image_ref": n.image_ref,
                    "raw_response": n.raw_response,
                    "elapsed": n.elapsed,
                    "attempt": n.attempt,
                    "edge_taken": n.edge_taken,
                }
                for n in self.node_results
            ],
            "trace": {
                "nodes_visited": self.trace.nodes_visited,
                "edges_taken": self.trace.edges_taken,
                "retries": self.trace.retries,
                "total_model_calls": self.trace.total_model_calls,
            },
            "strategy": self.strategy,
            "processing_time": sum(n.elapsed for n in self.node_results),
            "error": None,
        }
        if self.input_images:
            record["input_images"] = self.input_images
        return record
