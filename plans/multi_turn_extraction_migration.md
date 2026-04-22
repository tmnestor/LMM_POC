# Multi-Turn Agentic Extraction Migration Plan

## References

- [Agentic Workflows: Emerging Architectures (Vellum AI, 2026)](https://www.vellum.ai/blog/agentic-workflows-emerging-architectures-and-design-patterns) — Level 1/2/3 taxonomy, ReAct, Self-Refine, Reflexion, PlaG patterns
- [Agentic Document Workflows (LlamaIndex, 2026)](https://www.llamaindex.ai/blog/introducing-agentic-document-workflows) — cross-document coordination, state preservation, validation patterns
- [2026 Guide to Agentic Workflow Architectures (StackAI)](https://www.stackai.com/blog/the-2026-guide-to-agentic-workflow-architectures) — observability, retry/escalation, production checklist
- [PEP 634 — Structural Pattern Matching](https://peps.python.org/pep-0634/) — Python 3.10+ match/case syntax
- [vLLM Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — `StructuredOutputsParams(json=schema)`, xgrammar/guidance/outlines backends
- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/automatic_prefix_caching.html) — hash-based KV block caching for shared prefixes across requests
- [vLLM SamplingParams](https://docs.vllm.ai/en/latest/api/params/sampling_params.html) — `logprobs`, `stop`, `temperature`, `structured_output` parameters

## Status Quo

- **Bank statements**: Already multi-turn via `UnifiedBankExtractor` (turn 0: header detection, turn 1: strategy-specific extraction). Hardwired 2-turn flow with strategy enum.
- **Standard docs** (invoice, receipt): Single-turn via `processor.process_document_aware()`. Prompt sent, raw response returned.
- **Transaction linking** (`transaction-linking` branch): 3-stage **cross-image** pipeline hardwired in a notebook. Stage 1 extracts receipts, Stage 2a detects bank columns, Stage 2b matches receipts to bank debits using text from Stage 1 + column names from Stage 2a.
- **Pipeline contract**: Extract stage produces `raw_response` string → clean stage parses it via `ResponseHandler.handle()`.

## Classification

Per the [Vellum taxonomy](https://www.vellum.ai/blog/agentic-workflows-emerging-architectures-and-design-patterns), this pipeline is a **Level 2 Router Workflow** — the model's outputs drive routing decisions (column detection → strategy selection), but the execution environment is predefined. This is the right level for auditable financial document extraction: predictable enough to trace, flexible enough to handle layout variation.

## Goal

Generalize multi-turn extraction as a **directed state graph** where:
- **Nodes** are actions (model calls, routers, validators).
- **Edges** are transitions, including retry cycles.
- Any document type declares its graph in YAML — no Python changes for new workflows.
- Bank statements, standard docs, and transaction linking are all instances of the same engine.

---

## Data Structures

### A. Workflow Graph Definition (YAML)

Each document type declares a **directed graph of nodes** with named edges.
This replaces the previous linear `turns:` list design.

```yaml
# prompts/internvl3_prompts.yaml
prompts:
  invoice:
    nodes:
      extract:
        template: |
          Extract ALL data from this invoice image.
          For each field, write FIELD_NAME: value ...
        edges:
          ok: done

  bank_statement:
    nodes:
      detect_headers:
        template: |
          Look at the transaction table in this bank statement.
          List the exact column header names, one per line.
        max_tokens: 500
        parser: header_list
        max_retries: 2
        reflection: |
          Your previous response could not be parsed as column headers:
          {error}
          Please list only the column header names, one per line.
        edges:
          ok: select_strategy
          parse_failed: detect_headers       # retry with reflection

      select_strategy:
        type: router
        edges:
          has_balance_debit: extract_balance
          has_amount: extract_amount
          has_debit_or_credit: extract_debit_credit
          default: table_fallback

      extract_balance:
        template: |
          List all balances in the {balance_col} column ...
        inject:
          balance_col: "detect_headers.column_mapping.balance"
          desc_col: "detect_headers.column_mapping.description"
          debit_col: "detect_headers.column_mapping.debit"
          credit_col: "detect_headers.column_mapping.credit"
        max_tokens: 4096
        parser: balance_description
        max_retries: 1
        edges:
          ok: validate_balances
          parse_failed: extract_balance

      extract_amount:
        template: |
          Extract all transactions using the {amount_col} column ...
        inject:
          amount_col: "detect_headers.column_mapping.amount"
        max_tokens: 4096
        parser: amount_description
        edges:
          ok: done

      extract_debit_credit:
        template: |
          Extract debit and credit transactions ...
        inject:
          debit_col: "detect_headers.column_mapping.debit"
          credit_col: "detect_headers.column_mapping.credit"
        max_tokens: 4096
        parser: debit_credit_description
        edges:
          ok: done

      table_fallback:
        template: |
          Extract all transactions from the table ...
        max_tokens: 4096
        parser: field_value
        edges:
          ok: done

      validate_balances:
        type: validator
        check: balance_sequence
        edges:
          ok: done
          failed: extract_balance            # re-extract with error context
```

**Key differences from previous linear design:**

- **Edges replace implicit ordering** — each node declares where to go on success (`ok`),
  parse failure (`parse_failed`), or validation failure (`failed`). This makes retry
  cycles and conditional branching first-class graph concepts, not special-case turn types.

- **`reflection` key** — template appended to the prompt on retry, with `{error}` placeholder
  for the parse error message. This is the **Self-Refine pattern** from
  [Vellum](https://www.vellum.ai/blog/agentic-workflows-emerging-architectures-and-design-patterns):
  instead of silently falling back, the model gets a second chance with error context.

- **`max_retries`** — caps retry cycles to prevent infinite loops. After max retries,
  `parse_failed` escalates to an error record instead of looping.

- **`type: validator` nodes** — no model call, just check accumulated state (e.g.,
  `balance_sequence` verifies monotonic balance progression). On failure, routes back to
  the extraction node with the validation error as context. This replaces `BalanceCorrector`
  as a post-hoc fixup with a proper feedback loop.

- **`done` is a reserved terminal** — reaching `done` exits the graph and produces
  `final_fields` from the last extraction node's parsed output.

- **Single-turn docs** (invoice) degenerate to one node with `edges: {ok: done}` —
  fully backward compatible.

### A2. Cross-Image Workflows: Transaction Linking

Transaction linking operates across **two images** (receipt + bank statement) with data
flowing between them. The existing notebook (`transaction-linking` branch,
`staged_transaction_linking.ipynb`) hardwires a 3-stage pipeline:

```
Stage 1:  Receipt image     → extract store/date/total per receipt
Stage 2a: Bank stmt image   → extract column headers (reuses bank header detection)
Stage 2b: Bank stmt image   → match receipts to debit transactions
           + Stage 1 text      (purchases_text injected as plain text)
           + Stage 2a cols     (date_column, description_column, debit_column injected)
```

Workflows live under a `workflows:` key with named `inputs` and a node graph:

```yaml
# prompts/staged_transaction_linking.yaml
workflows:
  transaction_link:
    inputs:
      - name: receipt
        type: image
      - name: bank_statement
        type: image

    nodes:
      extract_receipts:
        image: receipt
        template: |
          You are given a single image of a scanned page ...
          --- RECEIPT 1 ---
          STORE: ...
          DATE: ...
          TOTAL: ...
        max_tokens: 500
        parser: receipt_list
        max_retries: 1
        reflection: |
          Could not parse receipt blocks from your response:
          {error}
          Use exactly the --- RECEIPT N --- format.
        edges:
          ok: detect_headers
          parse_failed: extract_receipts

      detect_headers:
        image: bank_statement
        template: |
          Find the transaction table and list column headers ...
        max_tokens: 200
        parser: header_list
        edges:
          ok: match_to_statement
          parse_failed: detect_headers

      match_to_statement:
        image: bank_statement
        template: |
          The following purchases were extracted from receipts:

          {purchases_text}

          Match each to a debit in the "{debit_column}" column ...
        inject:
          purchases_text: "extract_receipts.formatted_text"
          date_column: "detect_headers.column_mapping.date"
          description_column: "detect_headers.column_mapping.description"
          debit_column: "detect_headers.column_mapping.debit"
        max_tokens: 2000
        parser: transaction_match
        edges:
          ok: validate_amounts

      validate_amounts:
        type: validator
        check: amount_gate
        edges:
          ok: done
          failed: done                       # override, don't re-extract

    post_processing:
      - type: dedup
        field: RECEIPT_DESCRIPTION
```

**Key design decisions:**

- **`image` key per node** — names which input image the model sees. Defaults to the
  primary image for single-image workflows. Cross-image workflows require it on every
  model-call node.

- **`inject` still uses dot-path** — `extract_receipts.formatted_text` resolves from
  accumulated state the same way as single-image injection. State accumulates across
  all nodes regardless of which image they ran against.

- **`validate_amounts` routes to `done` on failure** — the amount gate *overrides*
  match status rather than requesting re-extraction (the model already saw the best
  data). This is an intentional design choice: some validators correct, others reject.

- **`post_processing`** — runs after the graph reaches `done`. Declarative hooks for
  operations that don't fit the graph model (dedup, array alignment).

### B. Runtime State (Dataclasses)

```python
# common/extraction_types.py  (new file, zero project deps like bank_types.py)

@dataclass
class NodeResult:
    """Result of executing a single graph node."""
    key: str                        # matches YAML node key
    image_ref: str                  # which image this node ran against
    prompt_sent: str                # after template injection (empty for routers/validators)
    raw_response: str               # raw model output (empty for routers/validators)
    parsed: dict[str, Any]          # structured parse of this node's output
    elapsed: float                  # seconds
    attempt: int                    # 1-indexed retry attempt number
    edge_taken: str                 # which edge was followed ("ok", "parse_failed", etc.)

@dataclass
class WorkflowState:
    """Typed state accumulated across graph execution.

    Implements the MetaGPT pattern (Vellum, 2026): enforce structured outputs
    between agents/nodes to prevent garbage propagation through the graph.
    """
    node_results: dict[str, NodeResult]     # key -> most recent result

    def get(self, dot_path: str) -> Any:
        """Resolve a dot-path reference (e.g., 'detect_headers.column_mapping.balance').

        Used by inject to pull values from earlier nodes' parsed output.
        """
        parts = dot_path.split(".")
        node_key = parts[0]
        result = self.node_results[node_key].parsed
        for part in parts[1:]:
            if isinstance(result, dict):
                result = result[part]
            else:
                result = getattr(result, part)
        return result

    def has(self, node_key: str) -> bool:
        """Check if a node has produced results."""
        return node_key in self.node_results

@dataclass
class GenerationParams:
    """Per-node generation parameters, built from YAML node def + defaults.

    Carries vLLM-specific options (structured output, logprobs, stop sequences)
    alongside standard generation config.  The backend dispatches via match:

        match params:
            case GenerationParams(output_schema=dict() as schema):
                # vLLM StructuredOutputsParams path — guaranteed valid JSON
                sampling = SamplingParams(
                    max_tokens=params.max_tokens,
                    temperature=params.temperature,
                    structured_output=StructuredOutputsParams(json=schema),
                )
            case GenerationParams(logprobs=int() as n):
                # Logprobs path — for confidence scoring
                sampling = SamplingParams(
                    max_tokens=params.max_tokens,
                    temperature=params.temperature,
                    logprobs=n,
                )
            case GenerationParams(stop=list() as stops) if stops:
                sampling = SamplingParams(
                    max_tokens=params.max_tokens,
                    temperature=params.temperature,
                    stop=stops,
                )
            case _:
                sampling = SamplingParams(
                    max_tokens=params.max_tokens,
                    temperature=params.temperature,
                )
    """
    max_tokens: int = 4096
    temperature: float = 0.0
    stop: list[str] | None = None
    output_schema: dict | None = None   # JSON schema → vLLM StructuredOutputsParams
    logprobs: int | None = None         # top-k logprobs per token

@dataclass
class GenerateResult:
    """Response from a single model call."""
    text: str
    logprobs: list[dict[str, float]] | None = None  # per-token top-k logprobs

@dataclass
class WorkflowTrace:
    """Observability record for the full graph execution.

    Per StackAI (2026) production checklist: track 'prompts, tool calls,
    intermediate outputs, decisions, and costs' end-to-end.
    """
    nodes_visited: list[str]                    # ordered keys of nodes executed
    edges_taken: list[tuple[str, str, str]]     # (from_node, edge_name, to_node)
    retries: dict[str, int]                     # node_key -> retry count
    total_model_calls: int
    total_elapsed: float

@dataclass
class ExtractionSession:
    """Complete graph execution for one image (or image pair)."""
    image_path: str                 # primary image (or first image for workflows)
    image_name: str
    document_type: str
    node_results: list[NodeResult]
    final_fields: dict[str, str]    # schema-format output for clean stage
    strategy: str                   # last extraction node key
    trace: WorkflowTrace
    input_images: dict[str, str] | None = None  # workflow only: {name: path}

    @property
    def raw_response(self) -> str:
        """Flat FIELD: value string for the clean stage contract."""
        return "\n".join(
            f"{k}: {v}" for k, v in self.final_fields.items()
        )

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
```

### C. Inter-Stage Serialization (JSONL)

```jsonc
// raw_extractions.jsonl — bank statement example with one retry
{
  "image_name": "bank_001.png",
  "document_type": "BANK_STATEMENT",
  "raw_response": "DOCUMENT_TYPE: BANK_STATEMENT\nTRANSACTION_DATES: 03/05 | 04/05",
  "nodes": [
    {"key": "detect_headers", "image_ref": "primary", "elapsed": 1.2, "attempt": 1, "edge_taken": "parse_failed"},
    {"key": "detect_headers", "image_ref": "primary", "elapsed": 1.4, "attempt": 2, "edge_taken": "ok"},
    {"key": "select_strategy", "image_ref": "primary", "elapsed": 0.0, "attempt": 1, "edge_taken": "has_balance_debit"},
    {"key": "extract_balance", "image_ref": "primary", "elapsed": 3.4, "attempt": 1, "edge_taken": "ok"},
    {"key": "validate_balances", "image_ref": "primary", "elapsed": 0.01, "attempt": 1, "edge_taken": "ok"}
  ],
  "trace": {
    "nodes_visited": ["detect_headers", "detect_headers", "select_strategy", "extract_balance", "validate_balances"],
    "edges_taken": [["detect_headers", "parse_failed", "detect_headers"], ["detect_headers", "ok", "select_strategy"], ["select_strategy", "has_balance_debit", "extract_balance"], ["extract_balance", "ok", "validate_balances"], ["validate_balances", "ok", "done"]],
    "retries": {"detect_headers": 1},
    "total_model_calls": 3
  },
  "strategy": "extract_balance",
  "processing_time": 6.01,
  "error": null
}
```

- `raw_response` is the parseable flat text (clean stage reads this — unchanged).
- `nodes` + `trace` are the debug/observability artifacts (clean stage ignores them).
- Retries are visible: `detect_headers` appears twice with attempts 1 and 2.

---

## Execution Engine

### `common/graph_executor.py` (new file)

The engine walks the YAML-defined graph using **structural pattern matching** (PEP 634)
for exhaustive, auditable dispatch. ~200 lines — no framework dependency.

```python
class GraphExecutor:
    """Walk a YAML-defined node graph against a model.

    Handles single-image graphs (from `prompts:`) and cross-image workflows
    (from `workflows:` with multiple named input images).

    Implements:
    - Self-Refine pattern (Vellum, 2026): retry with reflection on parse failure
    - MetaGPT pattern (Vellum, 2026): typed state between nodes
    - Observability (StackAI, 2026): full WorkflowTrace of nodes visited + edges taken
    """

    def __init__(
        self,
        generate_fn: Callable[[Image, str, GenerationParams], GenerateResult],
        parsers: dict[str, TurnParser],
        *,
        default_max_tokens: int = 4096,
        max_graph_steps: int = 20,          # circuit breaker
    ) -> None: ...

    def run(
        self,
        document_type: str,
        definition: dict,
        **kwargs,
    ) -> ExtractionSession:
        """Unified entry point — dispatch via match."""
        match definition:
            case {"inputs": list(), "nodes": dict(nodes), **rest}:
                return self._walk(nodes, kwargs["images"], document_type, rest)
            case {"nodes": dict(nodes), **rest}:
                images = {"primary": kwargs["image_path"]}
                return self._walk(nodes, images, document_type, rest)
            case {"template": str(), **rest}:
                # Legacy single-turn — wrap as one-node graph
                nodes = {"extract": {**definition, "edges": {"ok": "done"}}}
                images = {"primary": kwargs["image_path"]}
                return self._walk(nodes, images, document_type, {})
            case unknown:
                raise ValueError(f"Unrecognized definition: {unknown!r}")
```

### Graph Walk Loop

```python
def _walk(
    self,
    nodes: dict[str, dict],
    images: dict[str, str],
    document_type: str,
    workflow_meta: dict,
) -> ExtractionSession:
    state = WorkflowState(node_results={})
    trace_nodes: list[str] = []
    trace_edges: list[tuple[str, str, str]] = []
    all_results: list[NodeResult] = []
    retries: dict[str, int] = {}
    loaded_images: dict[str, Image] = {
        name: Image.open(path) for name, path in images.items()
    }

    current = self._find_start_node(nodes)
    steps = 0

    while current != "done":
        if steps >= self.max_graph_steps:
            raise RuntimeError(f"Graph exceeded {self.max_graph_steps} steps — cycle?")
        steps += 1

        node_def = nodes[current]
        trace_nodes.append(current)

        result, edge = self._execute_node(
            current, node_def, state, loaded_images, retries
        )
        all_results.append(result)
        state.node_results[current] = result

        next_node = node_def["edges"][edge]
        trace_edges.append((current, edge, next_node))
        current = next_node

    # ... build ExtractionSession from state + trace
```

### Node Execution via `match`

```python
def _execute_node(
    self,
    key: str,
    node_def: dict,
    state: WorkflowState,
    images: dict[str, Image],
    retries: dict[str, int],
) -> tuple[NodeResult, str]:
    """Execute a single node. Returns (result, edge_name)."""

    match node_def:
        # --- Router: no model call, evaluate conditions ---
        case {"type": "router", "edges": dict(edges)}:
            edge = self._evaluate_router(edges, state)
            return NodeResult(
                key=key, image_ref="", prompt_sent="", raw_response="",
                parsed={}, elapsed=0.0, attempt=1, edge_taken=edge,
            ), edge

        # --- Validator: no model call, check accumulated state ---
        case {"type": "validator", "check": str(check_name), "edges": dict(edges)}:
            ok, error_msg = self._run_validator(check_name, state)
            edge = "ok" if ok else "failed"
            if not ok:
                # Inject error into state so next node can use {error} in reflection
                state.node_results[key] = NodeResult(
                    key=key, image_ref="", prompt_sent="", raw_response="",
                    parsed={"error": error_msg}, elapsed=0.0,
                    attempt=1, edge_taken=edge,
                )
            return state.node_results.get(key, NodeResult(
                key=key, image_ref="", prompt_sent="", raw_response="",
                parsed={}, elapsed=0.0, attempt=1, edge_taken=edge,
            )), edge

        # --- Model call: template + optional inject + parser ---
        case {"template": str(template), "edges": dict(edges), **rest}:
            image_ref = rest.get("image", "primary")
            image = images[image_ref]
            max_retries = rest.get("max_retries", 0)
            reflection_template = rest.get("reflection")

            # Resolve inject placeholders
            injections = rest.get("inject", {})
            prompt = self._resolve_inject(template, injections, state)

            # Build GenerationParams from YAML node def
            gen_params = GenerationParams(
                max_tokens=rest.get("max_tokens", self.default_max_tokens),
                temperature=rest.get("temperature", 0.0),
                stop=rest.get("stop"),
                output_schema=rest.get("output_schema"),
                logprobs=rest.get("logprobs"),
            )

            # Retry loop (Self-Refine pattern)
            attempt = retries.get(key, 0) + 1
            start = time.time()
            result = self.generate_fn(image, prompt, gen_params)
            elapsed = time.time() - start

            # Dispatch on generation mode via match
            match gen_params:
                case GenerationParams(output_schema=dict()):
                    # Structured output — model response guaranteed valid JSON
                    parsed = json.loads(result.text)
                    edge = "ok"
                case _:
                    # Free-text parsing with retry
                    parser_name = rest.get("parser", "field_value")
                    try:
                        parsed = self.parsers[parser_name].parse(result.text, state)
                        edge = "ok"
                    except ParseError as e:
                        if attempt <= max_retries and reflection_template:
                            retries[key] = attempt
                            edge = "parse_failed"
                            parsed = {"error": str(e), "raw": result.text}
                        else:
                            edge = "parse_failed"
                            parsed = {"error": str(e), "raw": result.text}

            # Attach logprobs to parsed output for downstream confidence scoring
            if result.logprobs is not None:
                parsed["_logprobs"] = result.logprobs

            return NodeResult(
                key=key, image_ref=image_ref, prompt_sent=prompt,
                raw_response=result.text, parsed=parsed, elapsed=elapsed,
                attempt=attempt, edge_taken=edge,
            ), edge

        case unknown:
            raise ValueError(f"Unrecognized node definition for '{key}': {unknown!r}")
```

### Strategy Router via Dataclass Matching

```python
def _evaluate_router(
    self, edges: dict[str, str], state: WorkflowState,
) -> str:
    """Select edge based on accumulated state."""
    # Bank strategy selection — match on ColumnMapping dataclass
    if state.has("detect_headers"):
        mapping = state.get("detect_headers.column_mapping")
        match mapping:
            case ColumnMapping(balance=str(), debit=str()):
                return "has_balance_debit"
            case ColumnMapping(balance=str(), amount=str()):
                return "has_amount"
            case ColumnMapping(debit=str()) | ColumnMapping(credit=str()):
                return "has_debit_or_credit"
            case ColumnMapping():
                return "default"

    return "default"
```

### Response Structure Detection

```python
def _detect_response_format(
    self, text: str, expected_fields: list[str],
) -> dict[str, str]:
    """Detect and parse response format via match."""
    try:
        data = json.loads(self._strip_markdown_fences(text))
    except json.JSONDecodeError:
        return self._parse_plain_text(text, expected_fields)

    match data:
        case {field: str()} if field in expected_fields:
            return {f: str(data.get(f, "NOT_FOUND")) for f in expected_fields}
        case [{"DOCUMENT_TYPE": str(), **fields}, *_]:
            return {f: str(fields.get(f, "NOT_FOUND")) for f in expected_fields}
        case {"turn0": str(), **_}:
            return {f: "NOT_FOUND" for f in expected_fields}
        case _:
            return self._parse_plain_text(text, expected_fields)
```

### Turn Parsers

Register named parsers that match YAML `parser:` keys:

```python
# common/turn_parsers.py
class TurnParser(Protocol):
    def parse(self, raw_response: str, context: WorkflowState) -> dict[str, Any]: ...

class ParseError(Exception):
    """Raised when a parser cannot extract structured data."""
    pass

# Concrete implementations (refactored from existing code):
class HeaderListParser(TurnParser):          # from UnifiedBankExtractor._parse_headers
class BalanceDescriptionParser(TurnParser):  # from ResponseParser.parse_balance_description
class AmountDescriptionParser(TurnParser):   # from ResponseParser.parse_amount_description
class DebitCreditParser(TurnParser):         # from ResponseParser.parse_debit_credit_description
class FieldValueParser(TurnParser):          # from hybrid_parse_response (standard docs)
class ReceiptListParser(TurnParser):         # from notebook parse_stage1_response()
class TransactionMatchParser(TurnParser):    # from notebook match parsing
```

Existing parsing logic moves into these classes — no new algorithms, just reorganization.
Each parser raises `ParseError` on failure, which the graph walker catches to trigger
retry-with-reflection.

### vLLM Backend Integration

The pipeline runs on vLLM (`models/model_loader.py` `VllmSpec`, `models/backends/vllm_backend.py`).
Current config uses only basic features: `gpu_memory_utilization=0.90`, `max_model_len=8192`,
`SamplingParams(max_tokens=..., temperature=0)`. Several vLLM capabilities map directly
to graph executor features with zero external dependencies.

#### 1. Structured Outputs (Constrained Decoding)

vLLM's `StructuredOutputsParams` guarantees valid JSON matching a schema.
This eliminates parsing and retries for nodes with known output structure.

**API** (vLLM 0.8+):
```python
from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams

sampling = SamplingParams(
    max_tokens=4096,
    temperature=0.0,
    structured_output=StructuredOutputsParams(
        json={"type": "object", "properties": {...}, "required": [...]}
    ),
)
```

Backends: xgrammar (default, fastest), guidance, outlines. All enforce the
schema token-by-token during decoding — the model physically cannot produce
invalid output.

**YAML per-node opt-in:**
```yaml
nodes:
  extract:
    template: |
      Extract fields from this invoice ...
    output_schema:
      type: object
      properties:
        INVOICE_NUMBER: {type: string}
        INVOICE_DATE: {type: string}
        TOTAL_AMOUNT: {type: string}
      required: [INVOICE_NUMBER, INVOICE_DATE, TOTAL_AMOUNT]
    edges:
      ok: done
```

**Where it helps:**
- Invoice extraction (fixed field set) — eliminates `FieldValueParser` + retries
- Receipt extraction (fixed field set) — same benefit
- Transaction match response (fixed structure per receipt block)

**Where it does NOT help** (genuinely free-form output):
- Header detection (unknown column names, variable count)
- Balance/amount extraction (variable-length transaction lists)

The `_execute_node` dispatch uses `match` on `GenerationParams` to select
the structured vs free-text path (see Node Execution code above).

#### 2. Automatic Prefix Caching (APC)

Multi-turn workflows on the **same image** share a long common prefix (system
prompt + image tokens). APC caches the KV states of this shared prefix so
subsequent turns skip redundant prefill.

**Activation** (engine-level, one line):
```python
# models/model_loader.py — VllmSpec or LLM() kwargs
llm = LLM(
    model=spec.model_path,
    enable_prefix_caching=True,   # <-- only change
    gpu_memory_utilization=spec.gpu_memory_utilization,
    ...
)
```

No code changes in `GraphExecutor` or `VllmBackend` — vLLM detects shared
prefixes automatically via hash-based block matching. Turns 2+ on the same
image are ~30-50% faster (image tokens dominate prefill cost).

**Biggest wins:**
- Bank statement graphs (2-5 turns on same image)
- Transaction linking Stage 2a → 2b (both hit the bank statement image)
- Retry-with-reflection (same image + prompt prefix, only reflection suffix differs)

#### 3. Logprobs for Confidence Scoring

Request per-token logprobs on nodes where confidence matters. The graph
executor passes `logprobs=N` through `GenerationParams` to `SamplingParams`:

```python
SamplingParams(
    max_tokens=...,
    temperature=0.0,
    logprobs=5,           # top-5 logprobs per generated token
)
```

**Use cases:**
- **Transaction match confidence** — average logprob of the `MATCHED_TRANSACTION: FOUND`
  tokens is a better confidence signal than the model's self-reported `CONFIDENCE: HIGH`.
- **Validator input** — `validate_amounts` can weight logprob-derived confidence when
  the amount gate is borderline (receipt $83.48 vs bank $83.49).
- **Retry trigger** — nodes with `confidence_threshold` in YAML can route to
  `parse_failed` when mean logprob falls below threshold, even if parsing succeeds.

**YAML:**
```yaml
nodes:
  match_to_statement:
    template: |
      ...
    logprobs: 5
    confidence_threshold: -0.5    # mean logprob below this → retry
    edges:
      ok: validate_amounts
      low_confidence: match_to_statement   # retry with reflection
      parse_failed: match_to_statement
```

Logprobs attach to `NodeResult.parsed["_logprobs"]` for downstream access
without polluting the parsed field dict.

#### 4. Stop Sequences

Stop sequences terminate generation early, reducing wasted tokens and
preventing the model from hallucinating beyond the expected response boundary.

```python
SamplingParams(
    max_tokens=4096,
    stop=["---END---", "\n\n\n"],
)
```

**Use cases:**
- Header detection: stop at double newline (model tends to echo transaction data after headers)
- Receipt extraction: stop at `---END---` sentinel placed in the prompt

**YAML:**
```yaml
nodes:
  detect_headers:
    template: |
      List column headers, one per line.
      After the last header, write ---END--- on its own line.
    stop: ["---END---"]
    edges:
      ok: select_strategy
```

#### 5. VllmBackend.generate() Update

The backend's `generate()` method dispatches on `GenerationParams` via `match`:

```python
# models/backends/vllm_backend.py
class VllmBackend:
    def generate(
        self,
        image: Image,
        prompt: str,
        params: GenerationParams,
    ) -> GenerateResult:
        """Build SamplingParams from GenerationParams via match dispatch."""
        base_kwargs = {
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
        }

        match params:
            case GenerationParams(output_schema=dict() as schema):
                base_kwargs["structured_output"] = StructuredOutputsParams(
                    json=schema
                )
            case GenerationParams(stop=list() as stops) if stops:
                base_kwargs["stop"] = stops
            case _:
                pass

        if params.logprobs is not None:
            base_kwargs["logprobs"] = params.logprobs

        sampling = SamplingParams(**base_kwargs)
        output = self.model.chat(messages, sampling_params=sampling)

        token_logprobs = None
        if params.logprobs and output.outputs[0].logprobs:
            token_logprobs = [
                {tok.decoded_token: tok.logprob for tok in step.values()}
                for step in output.outputs[0].logprobs
            ]

        return GenerateResult(
            text=output.outputs[0].text,
            logprobs=token_logprobs,
        )
```

#### vLLM Feature Summary

| Feature | vLLM API | Graph Executor Use | Activation |
|---------|----------|-------------------|------------|
| Structured outputs | `StructuredOutputsParams(json=schema)` | `output_schema` per YAML node | Per-node opt-in |
| Prefix caching | `enable_prefix_caching=True` | Automatic — shared image prefix | Engine-level flag |
| Logprobs | `SamplingParams(logprobs=N)` | Confidence scoring, retry trigger | Per-node opt-in |
| Stop sequences | `SamplingParams(stop=[...])` | Early termination, cleaner parsing | Per-node opt-in |
| Paged attention | Default (already on) | Handles variable-length multi-turn | No change |
| Tensor parallelism | `tensor_parallel_size=N` | Multi-GPU inference | Engine-level (already configured) |

---

## Migration Path (4 phases)

### Phase 1: Introduce Data Structures (no behavior change)

1. Create `common/extraction_types.py` with `NodeResult`, `WorkflowState`, `WorkflowTrace`, `ExtractionSession`.
2. Have `UnifiedBankExtractor.extract_bank_statement()` return `ExtractionSession` instead of `(schema_fields, metadata)` tuple.
   - Internal: build `NodeResult` for each step it already executes.
   - `ExtractionSession.final_fields` = current `ExtractionResult.to_schema_dict()`.
   - Keep `ExtractionResult` as internal implementation detail.
3. Update `stages/extract.py` `_extract_bank_with_adapter()` to call `session.to_record()`.
4. Update `stages/extract.py` `_extract_standard()` to wrap single-turn result in `ExtractionSession` too.
5. **Clean stage unchanged** — still reads `raw_response`, calls `handler.handle()`.

**Validation:** Pipeline produces identical output. Diff `raw_extractions.jsonl` before/after (ignoring new `nodes`/`trace` fields).

### Phase 2: Graph Executor + YAML Graphs

1. Restructure prompt YAMLs from flat `key: {prompt: ...}` to `key: {nodes: {...}}` format. Single-turn docs get one node with `edges: {ok: done}`.
2. Create `common/graph_executor.py` with the `match`-based graph walker (~200 lines).
3. Create `common/extraction_types.py` with `GenerationParams`, `GenerateResult`, `NodeResult`, `WorkflowState`, `WorkflowTrace`, `ExtractionSession`.
4. Create `common/turn_parsers.py` — extract bank-specific parsers from `UnifiedBankExtractor`.
5. Add `ParseError` exception + retry-with-reflection logic in the graph walker.
6. Add `type: validator` nodes for `balance_sequence` check.
7. Replace `hybrid_parse_response` JSON detection with `match`-based `_detect_response_format`.
8. Wire `GraphExecutor` into `stages/extract.py` — replaces both `_extract_bank_with_adapter()` and `_extract_standard()`.
9. `UnifiedBankExtractor` becomes a thin wrapper or is retired.
10. **vLLM backend update**: Extend `VllmBackend.generate()` to accept `GenerationParams`, dispatch via `match` to build `SamplingParams` with structured outputs / logprobs / stop as needed.
11. **Enable APC**: Add `enable_prefix_caching=True` to `VllmSpec` / `LLM()` kwargs in `models/model_loader.py`.
12. Add `output_schema` to invoice/receipt YAML nodes for structured output (constrained decoding eliminates parsing for fixed-schema docs).

**Structural pattern matching touchpoints:**
- `GraphExecutor.run()` — definition dispatch (workflow vs graph vs legacy)
- `_execute_node()` — node type dispatch (router vs validator vs model call)
- `_execute_node()` — generation mode dispatch (`GenerationParams` with `output_schema` vs free-text)
- `_evaluate_router()` — `ColumnMapping` dataclass matching
- `_detect_response_format()` — JSON shape detection
- `VllmBackend.generate()` — `GenerationParams` → `SamplingParams` dispatch

**Validation:** Same pipeline output. Bank extraction F1 unchanged. Retry counts visible in trace. APC hit rate visible in vLLM engine logs.

### Phase 3: Multi-Turn Standard Docs

With the framework in place, add multi-node graphs for standard docs where it helps:

- **Invoice**: Node 1 = extract line items, Node 2 = extract header fields (split focus for better accuracy).
- **Receipt**: Node 1 = extract all, Validator = verify totals match line items → retry on mismatch.
- New doc types can define arbitrary graphs in YAML without Python changes.

**Validation:** Improved or equal accuracy on invoices/receipts.

### Phase 4: Transaction Linking (Cross-Image Workflows)

Promote the notebook prototype (`transaction-linking` branch, `staged_transaction_linking.ipynb`) into the production pipeline.

#### What the notebook does today (hardcoded):

1. Pairs receipt images with bank statement images via ground truth CSV.
2. **Stage 1** — receipt image only: `model.chat(receipt_pv, stage1_prompt)` → parse `--- RECEIPT N ---` blocks into `{STORE, DATE, TOTAL}` per receipt.
3. **Stage 2a** — bank statement image: `model.chat(bank_pv, stage2a_prompt)` → column headers → `ColumnMatcher.match()` → `ColumnMapping`.
4. **Stage 2b** — bank statement image + text injection: format `purchases_text` from Stage 1 results + column names from Stage 2a into `stage2_template` → `model.chat()` → parse `--- RECEIPT N ---` match blocks.
5. Post-processing: `enforce_amount_gate()` overrides match status when receipt total != bank debit. Dedup receipts with identical store/date/total.
6. Evaluation via `evaluate_transaction_linking.ipynb` — F1 on 6 fields per pair.

#### Migration steps:

1. **YAML workflow definition**: Move the 3-stage prompt sequence into `prompts/staged_transaction_linking.yaml` under a `workflows:` key with node graph (see Section A2 above).

2. **New parsers**:
   - `ReceiptListParser` — parse `--- RECEIPT N ---` blocks into list of `{STORE, DATE, TOTAL}` dicts + formatted text for injection. Extracted from notebook's `parse_stage1_response()`.
   - `TransactionMatchParser` — parse match response blocks into `{MATCHED_TRANSACTION, TRANSACTION_DATE, TRANSACTION_AMOUNT, ...}`. Extracted from notebook's inline parsing.

3. **Validator + post-processing hooks**:
   - `amount_gate` validator node — enforce `receipt_total == bank_debit`.
   - `dedup` post-processing — remove duplicate receipts.
   - Declared in YAML, dispatched via pattern matching in `GraphExecutor`.

4. **Pairing logic**: New `stages/pair.py` stage (or section in `stages/extract.py`):
   - Reads `classifications.jsonl` + a pairing CSV/config that maps receipt images to bank statement images.
   - For each pair, calls `GraphExecutor.run()` with `images={"receipt": path1, "bank_statement": path2}`.
   - Writes results to `raw_extractions.jsonl` with `document_type: "TRANSACTION_LINK"`.

5. **Field schema**: Register `transaction_link` in `field_definitions.yaml` with:
   - Scored fields: `RECEIPT_DATE`, `RECEIPT_DESCRIPTION`, `RECEIPT_TOTAL`, `BANK_TRANSACTION_DATE`, `BANK_TRANSACTION_DESCRIPTION`, `BANK_TRANSACTION_DEBIT`
   - Metadata fields (captured, not scored): `MATCHED_TRANSACTION`, `CONFIDENCE`, `MISMATCH_TYPE`, `REASONING`

6. **Evaluation**: Wire into `stages/evaluate.py` by treating `TRANSACTION_LINK` as another document type.

#### What changes vs notebook:

| Notebook | Pipeline |
|----------|----------|
| Hardcoded `model.chat()` calls | `GraphExecutor.run()` with YAML workflow |
| Inline prompt loading | `PromptCatalog.get_prompt()` from YAML |
| Inline parsing functions | Registered `TurnParser` implementations |
| `enforce_amount_gate()` in loop | `type: validator` node in graph |
| Manual CSV read/write | `stages/io.py` JSONL read/write |
| `ColumnMatcher.match()` inline | Reuses bank `header_list` parser + `detect_headers` node |
| One notebook, one model load | Same model, shared across all extract stage work |
| No retry on parse failure | Retry-with-reflection on every model-call node |

**Validation:** Transaction linking accuracy matches or exceeds notebook baseline. Run both back-to-back on the same image pairs and diff results.

---

## What Stays the Same

- **Clean stage** — always `handler.handle(raw_response, expected_fields)`. Zero changes.
- **ResponseHandler** — parse → clean → validate chain untouched.
- **PromptCatalog** — same API, just reads richer YAML structure.
- **Model backends** — `generate_fn` accepts `GenerationParams` dataclass (additive change); vLLM backend gains structured outputs, APC, logprobs, stop sequences via `match` dispatch on `GenerationParams`.
- **Evaluation** — reads `cleaned_extractions.jsonl`, format unchanged.

## What Changes

| Component | Current | After |
|-----------|---------|-------|
| Bank extraction return | `(schema_fields, metadata)` tuple | `ExtractionSession` dataclass |
| Standard extraction return | `dict` with `raw_response` key | `ExtractionSession` dataclass |
| Transaction linking | Hardcoded notebook with inline `model.chat()` | `GraphExecutor.run()` with YAML workflow |
| Extract stage dispatch | `if bank: ... else: ...` | `GraphExecutor.run()` for all |
| Strategy selection | Python `if/elif` in `UnifiedBankExtractor` | `type: router` nodes with `match` on `ColumnMapping` |
| Parse failure handling | Silent fallback or `NOT_FOUND` | Self-Refine: retry with reflection prompt |
| Validation | `BalanceCorrector` post-hoc fixup | `type: validator` nodes with graph edges back to extraction |
| Per-turn parsing | Methods on `ResponseParser` class | Named `TurnParser` implementations raising `ParseError` |
| Prompt YAML | Flat `key: {prompt: ...}` | `key: {nodes: {...}}` graph + `workflows:` for cross-image |
| Debug artifact | `metadata["raw_responses"]` dict | `nodes` + `trace` in JSONL (with retries, edges, timing) |
| Post-processing | `enforce_amount_gate()` in notebook loop | Declarative YAML `post_processing` with `match` dispatch |
| Observability | None | `WorkflowTrace` with node path, edge decisions, retry counts |
| vLLM generate | `SamplingParams(max_tokens, temperature=0)` | `GenerationParams` → `match` dispatch → structured/logprobs/stop |
| vLLM prefix caching | Disabled | `enable_prefix_caching=True` (APC for multi-turn same-image) |
| vLLM structured output | Not used | `StructuredOutputsParams(json=schema)` for fixed-schema nodes |
| Confidence scoring | Model self-report (`CONFIDENCE: HIGH`) | Token logprobs + threshold-based retry trigger |

## Structural Pattern Matching Summary

Eleven `match` sites across the engine and backend:

| Site | Phase | Matches on | Replaces |
|------|-------|-----------|----------|
| Definition dispatch | 2 | `{"inputs": ..., "nodes": ...}` vs `{"nodes": ...}` vs `{"template": ...}` | `if "inputs" in definition` type checks |
| Node execution | 2 | `{"type": "router"}` vs `{"type": "validator"}` vs `{"template": ...}` | `if node_def.get("type") == "router"` chains |
| Generation mode | 2 | `GenerationParams(output_schema=dict())` vs `GenerationParams()` | `if "output_schema" in node_def` |
| Strategy router | 2 | `ColumnMapping` dataclass attributes | `if mapping.has_balance and mapping.debit` booleans |
| Response format | 2 | JSON structure shape (dict vs list vs multi-turn) | `isinstance` + nested `if` in `hybrid_parse_response` |
| vLLM SamplingParams | 2 | `GenerationParams(output_schema=...)` / `(stop=...)` / `(logprobs=...)` | `if params.output_schema is not None` chains |
| Image resolution | 4 | `{"image": str(ref)}` per node def | `node_def.get("image", "primary")` |
| Post-processing | 4 | `{"type": "amount_gate"}` / `{"type": "dedup"}` / etc. | `if step["type"] == "amount_gate"` chains |
| Validator dispatch | 2 | `{"check": "balance_sequence"}` etc. | `if check_name == "balance_sequence"` |
| Confidence routing | 4 | `GenerationParams(logprobs=int())` with threshold check | `if confidence < threshold` ad-hoc |
| Backend dispatch | 2 | `GenerationParams` dataclass → `SamplingParams` builder | Flat `if/elif` in `VllmBackend.generate()` |

**Where match/case is NOT used** (and why):
- **Clean stage** — single `handler.handle()` call, no branching.
- **Per-field cleaning** — field type routing is string-prefix lookup, better as dict dispatch.
- **YAML loading** — already clean dict access via `PromptCatalog`.
- **Graph traversal loop** — simple `while current != "done"` with dict lookup, no polymorphism needed.
- **vLLM engine config** — `LLM()` constructor kwargs are flat; `match` adds no value over keyword args.

## Risk Mitigation

- **Phase 1 is zero-risk** — just wrapping existing returns in dataclasses. No logic changes.
- **Phase 2 parser extraction** — each parser already exists as a method. Moving to a class is mechanical.
- **Retry cycles** — `max_graph_steps` circuit breaker prevents infinite loops. `max_retries` per node caps individual retry counts.
- **Phase 4 notebook parity** — run the notebook and the pipeline on the same pairs, diff results field-by-field. The notebook stays as a regression reference until pipeline results match.
- **No framework dependency** — `GraphExecutor` is ~200 lines of Python with `match` dispatch. No LangGraph, CrewAI, or other external dependency. Auditable, debuggable, no version lock-in.
- **Rollback** — each phase is independently deployable. Phase 1 can ship without 2, 3, or 4. Phase 4 (transaction linking) is fully independent of Phase 3 (multi-turn standard docs).
