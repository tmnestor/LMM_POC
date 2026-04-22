"""Graph walker engine for multi-turn extraction workflows.

Walks a YAML-defined node graph against a model via a ``generate_fn``
callback.  Does not modify any existing code -- it is a new consumer
of the existing model infrastructure.

Patterns implemented:
- **Self-Refine**: retry with reflection on parse failure
- **MetaGPT**: typed state between nodes (WorkflowState)
- **Observability**: full WorkflowTrace

~200 lines.  No framework dependency.
"""

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from common.extraction_types import (
    ExtractionSession,
    GenerateResult,
    NodeGenParams,
    NodeResult,
    WorkflowState,
    WorkflowTrace,
)
from common.turn_parsers import (
    ParseError,
    TurnParser,
    dedup_by_field,
    enforce_amount_gate,
)

logger = logging.getLogger(__name__)

# Map match-response field names -> transaction_link schema field names
_TX_LINK_FIELD_MAP: dict[str, str] = {
    "TRANSACTION_DATE": "BANK_TRANSACTION_DATE",
    "TRANSACTION_AMOUNT": "BANK_TRANSACTION_DEBIT",
    "TRANSACTION_DESCRIPTION": "BANK_TRANSACTION_DESCRIPTION",
    "RECEIPT_STORE": "RECEIPT_DESCRIPTION",
}


class GraphExecutor:
    """Walk a YAML-defined node graph against a model.

    Args:
        generate_fn: ``(image, prompt, params) -> GenerateResult``.
        parsers: name -> TurnParser mapping.
        default_max_tokens: fallback when YAML node omits ``max_tokens``.
        max_graph_steps: circuit breaker for infinite loops.
    """

    def __init__(
        self,
        generate_fn: Callable[[Image.Image, str, NodeGenParams], GenerateResult],
        parsers: dict[str, TurnParser],
        *,
        default_max_tokens: int = 4096,
        max_graph_steps: int = 20,
    ) -> None:
        self._generate_fn = generate_fn
        self._parsers = parsers
        self._default_max_tokens = default_max_tokens
        self._max_graph_steps = max_graph_steps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        document_type: str,
        definition: dict[str, Any],
        *,
        images: dict[str, str] | None = None,
        image_path: str | None = None,
        image_name: str | None = None,
        extra_fields: dict[str, str] | None = None,
    ) -> ExtractionSession:
        """Entry point -- dispatch via match on definition shape."""
        match definition:
            case {"inputs": [*_], "nodes": dict() as nodes, **rest}:
                if images is None:
                    msg = "Cross-image workflow requires 'images' kwarg"
                    raise ValueError(msg)
                img_paths = images
            case {"nodes": dict() as nodes, **rest}:
                if image_path is None:
                    msg = "Single-image workflow requires 'image_path' kwarg"
                    raise ValueError(msg)
                img_paths = {"primary": image_path}
            case _:
                msg = f"Unrecognized workflow definition: {list(definition.keys())}"
                raise ValueError(msg)

        resolved_name = image_name or Path(next(iter(img_paths.values()))).stem
        return self._walk(
            nodes=nodes,
            images=img_paths,
            document_type=document_type,
            workflow_meta=rest,
            image_name=resolved_name,
            extra_fields=extra_fields or {},
        )

    # ------------------------------------------------------------------
    # Graph walk loop
    # ------------------------------------------------------------------

    def _walk(
        self,
        nodes: dict[str, dict[str, Any]],
        images: dict[str, str],
        document_type: str,
        workflow_meta: dict[str, Any],
        image_name: str,
        extra_fields: dict[str, str],
    ) -> ExtractionSession:
        state = WorkflowState()
        trace = WorkflowTrace()
        all_results: list[NodeResult] = []
        retries: dict[str, int] = {}
        model_calls = 0

        loaded_images: dict[str, Image.Image] = {
            name: Image.open(path).convert("RGB") for name, path in images.items()
        }

        current = self._find_start_node(nodes)
        steps = 0

        while current != "done":
            if steps >= self._max_graph_steps:
                msg = (
                    f"Graph exceeded {self._max_graph_steps} steps at node '{current}'"
                )
                raise RuntimeError(msg)
            steps += 1

            node_def = nodes[current]
            trace.nodes_visited.append(current)

            result, edge = self._execute_node(
                current, node_def, state, loaded_images, retries
            )
            all_results.append(result)
            state.node_results[current] = result

            if result.raw_response:
                model_calls += 1

            next_node = node_def["edges"][edge]
            trace.edges_taken.append((current, edge, next_node))
            logger.debug("%s -[%s]-> %s", current, edge, next_node)
            current = next_node

        # Post-processing
        post_processing = workflow_meta.get("post_processing", [])
        self._run_post_processing(post_processing, state)

        # Build final output
        trace.total_model_calls = model_calls
        trace.total_elapsed = sum(r.elapsed for r in all_results)
        trace.retries = dict(retries)

        final_fields = self._build_final_fields(document_type, state, extra_fields)

        strategy = ""
        for r in reversed(all_results):
            if r.raw_response:
                strategy = r.key
                break

        return ExtractionSession(
            image_path=next(iter(images.values())),
            image_name=image_name,
            document_type=document_type,
            node_results=all_results,
            final_fields=final_fields,
            strategy=strategy,
            trace=trace,
            input_images=images if len(images) > 1 else None,
        )

    # ------------------------------------------------------------------
    # Node execution via match
    # ------------------------------------------------------------------

    def _execute_node(
        self,
        key: str,
        node_def: dict[str, Any],
        state: WorkflowState,
        images: dict[str, Image.Image],
        retries: dict[str, int],
    ) -> tuple[NodeResult, str]:
        """Execute a single node.  Returns (result, edge_name)."""
        match node_def:
            # --- Validator: no model call, check accumulated state ---
            case {"type": "validator", "check": str(check_name), "edges": dict()}:
                start = time.time()
                ok, parsed = self._run_validator(check_name, state)
                elapsed = time.time() - start
                edge = "ok" if ok else "failed"
                return NodeResult(
                    key=key,
                    image_ref="",
                    prompt_sent="",
                    raw_response="",
                    parsed=parsed,
                    elapsed=elapsed,
                    attempt=1,
                    edge_taken=edge,
                ), edge

            # --- Router: no model call, evaluate conditions ---
            case {"type": "router", "edges": dict(edges)}:
                edge = self._evaluate_router(edges, state)
                return NodeResult(
                    key=key,
                    image_ref="",
                    prompt_sent="",
                    raw_response="",
                    parsed={},
                    elapsed=0.0,
                    attempt=1,
                    edge_taken=edge,
                ), edge

            # --- Model call: template + parser ---
            case {"template": str(template), "edges": dict(), **rest}:
                return self._execute_model_call(
                    key, template, rest, state, images, retries
                )

            case _:
                msg = f"Unrecognized node definition for '{key}': {node_def!r}"
                raise ValueError(msg)

    def _execute_model_call(
        self,
        key: str,
        template: str,
        rest: dict[str, Any],
        state: WorkflowState,
        images: dict[str, Image.Image],
        retries: dict[str, int],
    ) -> tuple[NodeResult, str]:
        """Execute a model-call node with Self-Refine retry."""
        image_ref = rest.get("image", "primary")
        image = images[image_ref]
        max_retries = rest.get("max_retries", 0)
        reflection_template = rest.get("reflection")

        # Resolve inject placeholders from accumulated state
        injections = rest.get("inject", {})
        prompt = self._resolve_inject(template, injections, state)

        # If retrying, append reflection with previous error
        attempt = retries.get(key, 0) + 1
        if attempt > 1 and reflection_template and state.has(key):
            prev_error = state.node_results[key].parsed.get("error", "")
            reflection = reflection_template.replace("{error}", prev_error)
            prompt = prompt + "\n\n" + reflection

        gen_params = NodeGenParams(
            max_tokens=rest.get("max_tokens", self._default_max_tokens),
            temperature=rest.get("temperature", 0.0),
            stop=rest.get("stop"),
            output_schema=rest.get("output_schema"),
            logprobs=rest.get("logprobs"),
        )

        start = time.time()
        result = self._generate_fn(image, prompt, gen_params)
        elapsed = time.time() - start

        # Dispatch on generation mode via match
        match gen_params:
            case NodeGenParams(output_schema=dict()):
                parsed = json.loads(result.text)
                edge = "ok"
            case _:
                parser_name = rest.get("parser", "field_value")
                try:
                    parsed = self._parsers[parser_name].parse(result.text, state)
                    edge = "ok"
                except ParseError as exc:
                    if attempt <= max_retries and reflection_template:
                        retries[key] = attempt
                        edge = "parse_failed"
                    else:
                        edge = "ok"  # exhausted retries -- proceed with error
                    parsed = {"error": str(exc), "raw": result.text}

        if result.logprobs is not None:
            parsed["_logprobs"] = result.logprobs

        return NodeResult(
            key=key,
            image_ref=image_ref,
            prompt_sent=prompt,
            raw_response=result.text,
            parsed=parsed,
            elapsed=elapsed,
            attempt=attempt,
            edge_taken=edge,
        ), edge

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_start_node(nodes: dict[str, Any]) -> str:
        """Return the first node key (YAML preserves insertion order)."""
        return next(iter(nodes))

    @staticmethod
    def _resolve_inject(
        template: str,
        injections: dict[str, str],
        state: WorkflowState,
    ) -> str:
        """Replace ``{placeholder}`` references with values from state.

        Raises RuntimeError with diagnostics when an upstream node's
        parse failed and the required key is missing.
        """
        prompt = template
        for placeholder, dot_path in injections.items():
            try:
                value = state.get(dot_path)
            except KeyError as exc:
                node_key = dot_path.split(".")[0]
                raw = ""
                if state.has(node_key):
                    raw = state.node_results[node_key].raw_response[:500]
                msg = (
                    f"Inject failed for '{placeholder}' <- '{dot_path}': {exc}. "
                    f"Upstream raw_response: {raw!r}"
                )
                raise RuntimeError(msg) from None
            prompt = prompt.replace(
                f"{{{placeholder}}}",
                str(value) if value is not None else placeholder,
            )
        return prompt

    @staticmethod
    def _evaluate_router(
        edges: dict[str, str],
        state: WorkflowState,
    ) -> str:
        """Select edge based on accumulated state."""
        if state.has("detect_headers"):
            mapping = state.get("detect_headers.column_mapping")
            if isinstance(mapping, dict):
                has_balance = mapping.get("balance") is not None
                has_debit = mapping.get("debit") is not None
                has_amount = mapping.get("amount") is not None

                if has_balance and has_debit and "has_balance_debit" in edges:
                    return "has_balance_debit"
                if has_balance and has_amount and "has_amount" in edges:
                    return "has_amount"
                if (
                    has_debit or mapping.get("credit")
                ) and "has_debit_or_credit" in edges:
                    return "has_debit_or_credit"

        return "default"

    @staticmethod
    def _run_validator(
        check_name: str,
        state: WorkflowState,
    ) -> tuple[bool, dict[str, Any]]:
        """Run a named validator check.  Returns ``(ok, parsed_dict)``."""
        match check_name:
            case "amount_gate":
                receipts = (
                    state.get("extract_receipts.receipts")
                    if state.has("extract_receipts")
                    else []
                )
                matches = (
                    state.get("match_to_statement.matches")
                    if state.has("match_to_statement")
                    else []
                )
                validated = enforce_amount_gate(receipts, matches)
                overridden = sum(
                    1
                    for orig, val in zip(matches, validated, strict=False)
                    if orig.get("MATCHED_TRANSACTION") != val.get("MATCHED_TRANSACTION")
                )
                return overridden == 0, {
                    "validated_matches": validated,
                    "overridden_count": overridden,
                }
            case _:
                return True, {}

    @staticmethod
    def _run_post_processing(
        post_processing: list[dict[str, Any]],
        state: WorkflowState,
    ) -> None:
        """Run declarative post-processing steps on accumulated state."""
        for step in post_processing:
            match step:
                case {"type": "dedup", "field": str(field_name)}:
                    # Apply to validated matches first, then raw matches
                    if state.has("validate_amounts"):
                        parsed = state.node_results["validate_amounts"].parsed
                        if "validated_matches" in parsed:
                            parsed["validated_matches"] = dedup_by_field(
                                parsed["validated_matches"], field_name
                            )
                    elif state.has("match_to_statement"):
                        parsed = state.node_results["match_to_statement"].parsed
                        if "matches" in parsed:
                            parsed["matches"] = dedup_by_field(
                                parsed["matches"], field_name
                            )

    @staticmethod
    def _build_final_fields(
        document_type: str,
        state: WorkflowState,
        extra_fields: dict[str, str],
    ) -> dict[str, str]:
        """Build flat ``FIELD: value`` dict from accumulated state."""
        fields: dict[str, str] = {"DOCUMENT_TYPE": document_type}
        fields.update(extra_fields)

        # Transaction linking: combine receipts + matches
        if state.has("match_to_statement") or state.has("validate_amounts"):
            # Prefer validated matches
            if state.has("validate_amounts"):
                validated = state.node_results["validate_amounts"].parsed
                matches = validated.get("validated_matches", [])
            else:
                matches = state.get("match_to_statement.matches")

            receipts = (
                state.get("extract_receipts.receipts")
                if state.has("extract_receipts")
                else []
            )

            fields["RECEIPT_DATE"] = " | ".join(
                r.get("DATE", "NOT_FOUND") for r in receipts
            )
            fields["RECEIPT_DESCRIPTION"] = " | ".join(
                m.get("RECEIPT_STORE", "NOT_FOUND") for m in matches
            )
            fields["RECEIPT_TOTAL"] = " | ".join(
                m.get("RECEIPT_TOTAL", "NOT_FOUND") for m in matches
            )
            fields["BANK_TRANSACTION_DATE"] = " | ".join(
                m.get("TRANSACTION_DATE", "NOT_FOUND") for m in matches
            )
            fields["BANK_TRANSACTION_DESCRIPTION"] = " | ".join(
                m.get("TRANSACTION_DESCRIPTION", "NOT_FOUND") for m in matches
            )
            fields["BANK_TRANSACTION_DEBIT"] = " | ".join(
                m.get("TRANSACTION_AMOUNT", "NOT_FOUND") for m in matches
            )
            return fields

        # Standard docs: merge flat parsed dicts from model-call nodes
        for node_key in state.node_results:
            parsed = state.node_results[node_key].parsed
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    if k.startswith("_"):
                        continue
                    if isinstance(v, str) and k.isupper():
                        fields[k] = v

        return fields
