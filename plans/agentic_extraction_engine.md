# Agentic Extraction Engine

## References

- [Agentic Workflows: Emerging Architectures (Vellum AI, 2026)](https://www.vellum.ai/blog/agentic-workflows-emerging-architectures-and-design-patterns) — Level 1/2/3 taxonomy, ReAct, Self-Refine, Reflexion patterns
- [Agentic Document Workflows (LlamaIndex, 2026)](https://www.llamaindex.ai/blog/introducing-agentic-document-workflows) — cross-document coordination, state preservation, validation
- [2026 Guide to Agentic Workflow Architectures (StackAI)](https://www.stackai.com/blog/the-2026-guide-to-agentic-workflow-architectures) — observability, retry/escalation, production checklist
- [PEP 634 — Structural Pattern Matching](https://peps.python.org/pep-0634/) — Python 3.10+ match/case syntax
- [vLLM Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — `StructuredOutputsParams(json=schema)`, xgrammar/guidance/outlines backends
- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/automatic_prefix_caching.html) — hash-based KV block caching for shared prefixes
- [vLLM SamplingParams](https://docs.vllm.ai/en/latest/api/params/sampling_params.html) — `logprobs`, `stop`, `temperature`, `structured_output`

## Constraint

**All changes are additive.** Existing code is not modified. New capabilities live in
new files alongside the existing pipeline. The current bank extraction
(`UnifiedBankExtractor`), standard extraction (`process_document_aware`), stages
(`extract.py`, `clean.py`), response handler, prompt YAMLs, and vLLM backend
are all untouched.

## Status Quo

- **Bank statements**: Multi-turn via `UnifiedBankExtractor`. Works. Not touched.
- **Standard docs** (invoice, receipt): Single-turn via `processor.process_document_aware()`. Works. Not touched.
- **Transaction linking**: 3-stage cross-image pipeline hardwired in a notebook (`transaction-linking` branch, `staged_transaction_linking.ipynb`). Not productionized.
- **Pipeline contract**: Extract stage produces `raw_response` string, clean stage parses via `ResponseHandler.handle()`.

## Scope

Two workloads, both new:

1. **Transaction linking** (cross-image) — receipt + bank statement image pair, 3-node graph with data injection between nodes. Promotes the notebook prototype to a production stage.
2. **Single-image multi-turn** — split extraction into multiple focused model calls on the same image (e.g., invoice headers + line items, receipt extraction + total validation). New workflow definitions that opt in to multi-turn without changing the existing single-turn path.

## Classification

Per the [Vellum taxonomy](https://www.vellum.ai/blog/agentic-workflows-emerging-architectures-and-design-patterns), both workloads are **Level 2 Router Workflows** — model outputs drive routing (column detection -> strategy, parse success -> next node), but execution is predefined.

---

## New Files

All new code lives in new files. No existing file is modified.

| New file | Purpose |
|----------|---------|
| `common/extraction_types.py` | Dataclasses: `GenerationParams`, `GenerateResult`, `NodeResult`, `WorkflowState`, `WorkflowTrace`, `ExtractionSession` |
| `common/graph_executor.py` | Graph walker engine (~200 lines) |
| `common/turn_parsers.py` | `TurnParser` protocol + concrete parsers + `ParseError` |
| `prompts/workflows/*.yaml` | Workflow graph definitions (one YAML per workflow) |
| `stages/link.py` | Transaction linking stage (pairs images, runs graph, writes JSONL) |
| `stages/graph_extract.py` | Single-image multi-turn extraction stage (opt-in alternative to `extract.py`) |

---

## Data Structures

### A. Workflow Graph Definition (YAML)

Each workflow declares a **directed graph of nodes** with named edges.
Workflow YAMLs live in `prompts/workflows/` — separate from existing prompt files.

#### Single-Image Multi-Turn Example

```yaml
# prompts/workflows/invoice_multi_turn.yaml
name: invoice_multi_turn
description: Split invoice extraction into header + line item passes.

nodes:
  extract_headers:
    template: |
      Extract the header fields from this invoice image.
      For each field, write FIELD_NAME: value on its own line.
      Fields: INVOICE_NUMBER, INVOICE_DATE, DUE_DATE, VENDOR_NAME,
      VENDOR_ADDRESS, CUSTOMER_NAME, SUBTOTAL, TAX, TOTAL_AMOUNT
    max_tokens: 1024
    parser: field_value
    stop: ["\n\n\n"]
    max_retries: 1
    reflection: |
      Could not parse field values from your response:
      {error}
      Use exactly the FIELD_NAME: value format, one per line.
    edges:
      ok: extract_line_items
      parse_failed: extract_headers

  extract_line_items:
    template: |
      Now extract the line items from this invoice image.
      For each line item, write:
      ITEM_DESCRIPTION: ...
      ITEM_QUANTITY: ...
      ITEM_UNIT_PRICE: ...
      ITEM_AMOUNT: ...
      Separate items with a blank line.
    max_tokens: 2048
    parser: field_value
    edges:
      ok: validate_totals

  validate_totals:
    type: validator
    check: invoice_total_matches_line_items
    edges:
      ok: done
      failed: done    # capture mismatch, don't re-extract
```

#### Cross-Image Workflow: Transaction Linking

```yaml
# prompts/workflows/transaction_link.yaml
name: transaction_link
description: >
  Match receipts to bank statement debit transactions.
  Stage 1: extract receipt details from receipt image.
  Stage 2a: detect column headers from bank statement image.
  Stage 2b: match receipts to bank transactions.

inputs:
  - name: receipt
    type: image
  - name: bank_statement
    type: image

nodes:
  extract_receipts:
    image: receipt
    template: |
      You are given a single image of a scanned page that may contain
      MULTIPLE receipts.

      For each receipt found, respond in the following format:

      --- RECEIPT 1 ---
      STORE: the store or merchant name
      DATE: the purchase date (DD/MM/YYYY)
      TOTAL: the total amount including currency symbol (e.g. $83.48)

      --- RECEIPT 2 ---
      (same fields)

      Continue for all receipts found on the page.
    max_tokens: 500
    parser: receipt_list
    max_retries: 1
    reflection: |
      Could not parse receipt blocks from your response:
      {error}
      Use exactly the --- RECEIPT N --- format with STORE, DATE, TOTAL fields.
    edges:
      ok: detect_headers
      parse_failed: extract_receipts

  detect_headers:
    image: bank_statement
    template: |
      You are given one image: a bank statement.

      Find the transaction table and list its column headers exactly as they
      appear, in order from left to right.

      Respond as a numbered list, one header per line:
      1. First Column Name
      2. Second Column Name
      3. ...

      Only list column headers -- do not include any transaction data.
    max_tokens: 200
    parser: header_list
    stop: ["\n\n\n"]
    max_retries: 1
    reflection: |
      Could not parse column headers from your response:
      {error}
      List headers as a numbered list, one per line.
    edges:
      ok: match_to_statement
      parse_failed: detect_headers

  match_to_statement:
    image: bank_statement
    template: |
      You are given one image: a bank statement with a transaction table.

      The following purchases were extracted from receipts in a prior step:

      {purchases_text}

      Your task: for EACH purchase listed above, identify the specific debit
      transaction in the bank statement that corresponds to that purchase.

      Matching strategy (in priority order):
      1. AMOUNT is the strongest signal: find rows where the "{debit_column}"
         column matches the purchase total.
      2. Then check the "{description_column}" column -- the store/merchant
         name should appear there (possibly truncated or abbreviated).
      3. Then confirm the "{date_column}" column is the same as the purchase
         date, or up to 3 days later (bank processing delay).

      Row integrity: all extracted fields must come from the SAME row.

      For each purchase, respond in the following format:

      --- RECEIPT 1 ---
      MATCHED_TRANSACTION: FOUND or NOT_FOUND
      TRANSACTION_DATE: the date of the matched transaction (DD/MM/YYYY)
      TRANSACTION_AMOUNT: the dollar amount of the matched transaction
      TRANSACTION_DESCRIPTION: the description from the bank statement
      RECEIPT_STORE: the store/merchant name from the purchase
      RECEIPT_TOTAL: the total amount from the purchase
      CONFIDENCE: HIGH, MEDIUM, or LOW
      REASONING: brief explanation of why this transaction matches
    inject:
      purchases_text: "extract_receipts.formatted_text"
      date_column: "detect_headers.column_mapping.date"
      description_column: "detect_headers.column_mapping.description"
      debit_column: "detect_headers.column_mapping.debit"
    max_tokens: 2000
    parser: transaction_match
    logprobs: 5
    edges:
      ok: validate_amounts

  validate_amounts:
    type: validator
    check: amount_gate
    edges:
      ok: done
      failed: done    # override match status, don't re-extract

post_processing:
  - type: dedup
    field: RECEIPT_DESCRIPTION
```

**Key design elements:**

- **`image` key per node** — names which input image the model sees. Absent for
  single-image workflows (defaults to sole image). Required for cross-image.

- **`inject`** uses dot-path references — `extract_receipts.formatted_text` resolves
  from accumulated `WorkflowState`. Data flows between nodes across image boundaries.

- **`reflection` + `max_retries`** — the [Self-Refine pattern](https://www.vellum.ai/blog/agentic-workflows-emerging-architectures-and-design-patterns):
  on parse failure, append error context and retry instead of silent fallback.

- **`type: validator` nodes** — no model call, just check accumulated state.
  `validate_amounts` routes to `done` on failure (overrides, doesn't re-extract).

- **`post_processing`** — declarative hooks after the graph reaches `done`.

- **`done` is the reserved terminal** — graph exits and produces `final_fields`.

### B. Runtime State (Dataclasses)

```python
# common/extraction_types.py  (NEW file — zero deps on existing modules)

@dataclass
class GenerationParams:
    """Per-node generation parameters built from YAML node def.

    Carries vLLM-specific options alongside standard config.
    The generate_fn wrapper dispatches via match on these fields.
    """
    max_tokens: int = 4096
    temperature: float = 0.0
    stop: list[str] | None = None
    output_schema: dict | None = None   # JSON schema -> vLLM StructuredOutputsParams
    logprobs: int | None = None         # top-k logprobs per token

@dataclass
class GenerateResult:
    """Response from a single model call."""
    text: str
    logprobs: list[dict[str, float]] | None = None

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
    edge_taken: str                 # which edge was followed

@dataclass
class WorkflowState:
    """Typed state accumulated across graph execution.

    Implements the MetaGPT pattern: structured outputs between nodes
    prevent garbage propagation through the graph.
    """
    node_results: dict[str, NodeResult]

    def get(self, dot_path: str) -> Any:
        """Resolve a dot-path reference (e.g., 'detect_headers.column_mapping.balance')."""
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
        return node_key in self.node_results

@dataclass
class WorkflowTrace:
    """Observability record for full graph execution."""
    nodes_visited: list[str]
    edges_taken: list[tuple[str, str, str]]     # (from, edge, to)
    retries: dict[str, int]
    total_model_calls: int
    total_elapsed: float

@dataclass
class ExtractionSession:
    """Complete graph execution for one image (or image pair)."""
    image_path: str
    image_name: str
    document_type: str
    node_results: list[NodeResult]
    final_fields: dict[str, str]
    strategy: str                                # last extraction node key
    trace: WorkflowTrace
    input_images: dict[str, str] | None = None   # cross-image only

    @property
    def raw_response(self) -> str:
        """Flat FIELD: value string — same contract as existing pipeline."""
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
```

### C. Inter-Stage Serialization (JSONL)

```jsonc
// raw_extractions.jsonl — transaction linking example
{
  "image_name": "pair_001",
  "document_type": "TRANSACTION_LINK",
  "raw_response": "RECEIPT_STORE: ...\nRECEIPT_DATE: ...\nRECEIPT_TOTAL: ...\n...",
  "input_images": {"receipt": "/data/receipt_001.png", "bank_statement": "/data/bank_001.png"},
  "nodes": [
    {"key": "extract_receipts", "image_ref": "receipt", "elapsed": 1.8, "attempt": 1, "edge_taken": "ok"},
    {"key": "detect_headers", "image_ref": "bank_statement", "elapsed": 1.2, "attempt": 1, "edge_taken": "ok"},
    {"key": "match_to_statement", "image_ref": "bank_statement", "elapsed": 3.1, "attempt": 1, "edge_taken": "ok"},
    {"key": "validate_amounts", "image_ref": "", "elapsed": 0.01, "attempt": 1, "edge_taken": "ok"}
  ],
  "trace": {
    "nodes_visited": ["extract_receipts", "detect_headers", "match_to_statement", "validate_amounts"],
    "edges_taken": [
      ["extract_receipts", "ok", "detect_headers"],
      ["detect_headers", "ok", "match_to_statement"],
      ["match_to_statement", "ok", "validate_amounts"],
      ["validate_amounts", "ok", "done"]
    ],
    "retries": {},
    "total_model_calls": 3
  },
  "strategy": "match_to_statement",
  "processing_time": 6.11,
  "error": null
}
```

- `raw_response` is the flat parseable text (clean stage reads this).
- `nodes` + `trace` are debug/observability artifacts (clean stage ignores them).
- `input_images` tracks both images for the cross-image workflow.

---

## Execution Engine

### `common/graph_executor.py` (NEW file)

The engine walks the YAML-defined graph using structural pattern matching
for exhaustive, auditable dispatch. ~200 lines, no framework dependency.

```python
class GraphExecutor:
    """Walk a YAML-defined node graph against a model.

    Consumes the existing model infrastructure via a generate_fn callback.
    Does not modify any existing code — it is a new consumer.

    Implements:
    - Self-Refine pattern: retry with reflection on parse failure
    - MetaGPT pattern: typed state between nodes
    - Observability: full WorkflowTrace
    """

    def __init__(
        self,
        generate_fn: Callable[[Image, str, GenerationParams], GenerateResult],
        parsers: dict[str, TurnParser],
        *,
        default_max_tokens: int = 4096,
        max_graph_steps: int = 20,
    ) -> None: ...

    def run(
        self,
        document_type: str,
        definition: dict,
        **kwargs,
    ) -> ExtractionSession:
        """Entry point — dispatch via match on definition shape."""
        match definition:
            case {"inputs": list(), "nodes": dict(nodes), **rest}:
                # Cross-image workflow
                return self._walk(nodes, kwargs["images"], document_type, rest)
            case {"nodes": dict(nodes), **rest}:
                # Single-image multi-turn
                images = {"primary": kwargs["image_path"]}
                return self._walk(nodes, images, document_type, rest)
            case _:
                raise ValueError(f"Unrecognized workflow definition: {type(definition)}")
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
            raise RuntimeError(f"Graph exceeded {self.max_graph_steps} steps")
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

    # Build ExtractionSession from accumulated state + trace
    ...
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

            # Resolve inject placeholders from accumulated state
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

            # Model call
            attempt = retries.get(key, 0) + 1
            start = time.time()
            result = self.generate_fn(image, prompt, gen_params)
            elapsed = time.time() - start

            # Dispatch on generation mode via match
            match gen_params:
                case GenerationParams(output_schema=dict()):
                    # Structured output — response guaranteed valid JSON
                    parsed = json.loads(result.text)
                    edge = "ok"
                case _:
                    # Free-text parsing with Self-Refine retry
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

            # Attach logprobs for downstream confidence scoring
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

```python
# common/turn_parsers.py  (NEW file)

class TurnParser(Protocol):
    def parse(self, raw_response: str, context: WorkflowState) -> dict[str, Any]: ...

class ParseError(Exception):
    """Raised when a parser cannot extract structured data."""
    pass

# Concrete implementations:
class HeaderListParser(TurnParser):         # parse numbered column header list
class FieldValueParser(TurnParser):         # parse FIELD: value lines
class ReceiptListParser(TurnParser):        # parse --- RECEIPT N --- blocks
class TransactionMatchParser(TurnParser):   # parse match response blocks
```

Parser implementations extract logic from the existing notebook (`parse_stage1_response()`,
inline match parsing) and from `hybrid_parse_response()`. The existing code is not
modified — parsers are new standalone classes that duplicate the parsing algorithms.

### generate_fn Wrapper

The `GraphExecutor` accepts a `generate_fn` callback. This is a thin wrapper around the
existing model infrastructure — it adapts the existing `VllmBackend.generate()` or
`processor.generate()` to the `GenerationParams` / `GenerateResult` interface.

```python
# common/graph_generate.py  (NEW file)

def make_generate_fn(
    backend: VllmBackend,
) -> Callable[[Image, str, GenerationParams], GenerateResult]:
    """Wrap existing VllmBackend into the generate_fn interface.

    Dispatches GenerationParams to vLLM SamplingParams via match.
    Does NOT modify VllmBackend — wraps it.
    """
    def generate(image: Image, prompt: str, params: GenerationParams) -> GenerateResult:
        base_kwargs: dict[str, Any] = {
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
        }

        match params:
            case GenerationParams(output_schema=dict() as schema):
                from vllm.sampling_params import StructuredOutputsParams
                base_kwargs["structured_output"] = StructuredOutputsParams(json=schema)
            case GenerationParams(stop=list() as stops) if stops:
                base_kwargs["stop"] = stops
            case _:
                pass

        if params.logprobs is not None:
            base_kwargs["logprobs"] = params.logprobs

        sampling = SamplingParams(**base_kwargs)

        # Call existing backend — no modification to VllmBackend
        output = backend.model.chat(
            build_messages(image, prompt),
            sampling_params=sampling,
        )

        token_logprobs = None
        if params.logprobs and output.outputs[0].logprobs:
            token_logprobs = [
                {tok.decoded_token: tok.logprob for tok in step.values()}
                for step in output.outputs[0].logprobs
            ]

        return GenerateResult(text=output.outputs[0].text, logprobs=token_logprobs)

    return generate
```

---

## vLLM Advantages

These vLLM features are consumed by the new engine. The existing vLLM setup
(`VllmSpec`, `model_loader.py`, `VllmBackend`) is not modified — the
`make_generate_fn` wrapper accesses the underlying `LLM` object directly.

### 1. Structured Outputs (Constrained Decoding)

`StructuredOutputsParams(json=schema)` in `SamplingParams` guarantees valid JSON.
Eliminates parsing and retries for nodes with known output structure.

**vLLM API** (0.8+):
```python
from vllm.sampling_params import StructuredOutputsParams

sampling = SamplingParams(
    max_tokens=4096,
    structured_output=StructuredOutputsParams(
        json={"type": "object", "properties": {...}, "required": [...]}
    ),
)
```

Backends: xgrammar (default, fastest), guidance, outlines. Token-by-token enforcement.

**Where it helps:**
- Transaction match response (fixed structure per receipt block)
- Invoice/receipt extraction (fixed field set)

**Where it does NOT help:**
- Header detection (unknown column names, variable count)
- Receipt list extraction (variable number of receipts)

YAML opt-in per node via `output_schema:` key. The `_execute_node` match on
`GenerationParams(output_schema=dict())` skips the parser entirely.

### 2. Automatic Prefix Caching (APC)

Multi-turn workflows on the same image share a long common prefix (system prompt +
image tokens). APC caches KV states so turns 2+ skip redundant prefill.

**Activation** — one engine-level flag (additive config change):
```python
# Could be added to VllmSpec or passed as env var
llm = LLM(model=..., enable_prefix_caching=True, ...)
```

No code changes needed in the graph executor or generate wrapper — vLLM detects
shared prefixes automatically. Turns 2+ on the same image ~30-50% faster.

**Biggest wins:**
- Transaction linking: Stage 2a and 2b both hit the bank statement image
- Retry-with-reflection: same image + prompt prefix, only reflection suffix differs
- Invoice multi-turn: both passes hit the same invoice image

### 3. Logprobs for Confidence Scoring

Per-token logprobs provide a real confidence signal, replacing the model's
self-reported `CONFIDENCE: HIGH/LOW`.

**Use cases:**
- Transaction match: average logprob of `MATCHED_TRANSACTION: FOUND` tokens
- Validator input: weight borderline amount gate decisions
- Retry trigger: `confidence_threshold` in YAML → `low_confidence` edge

YAML opt-in per node: `logprobs: 5`. Logprobs attach to
`NodeResult.parsed["_logprobs"]` for downstream access.

### 4. Stop Sequences

Terminate generation early via `SamplingParams(stop=[...])`.
Reduces wasted tokens and prevents hallucination beyond expected boundaries.

**Use cases:**
- Header detection: stop at `---END---` (model tends to echo transaction data)
- Receipt extraction: stop after last receipt block

YAML opt-in per node: `stop: ["---END---"]`.

### vLLM Feature Summary

| Feature | vLLM API | Graph Executor Use | Existing Code Changed? |
|---------|----------|-------------------|----------------------|
| Structured outputs | `StructuredOutputsParams(json=schema)` | `output_schema` per node | No — wrapper builds SamplingParams |
| Prefix caching | `enable_prefix_caching=True` | Automatic for multi-turn | No — engine-level flag only |
| Logprobs | `SamplingParams(logprobs=N)` | Confidence scoring, retry | No — wrapper passes through |
| Stop sequences | `SamplingParams(stop=[...])` | Early termination | No — wrapper passes through |
| Paged attention | Default (already on) | Multi-turn variable length | Already active |
| Tensor parallelism | `tensor_parallel_size=N` | Multi-GPU inference | Already configured |

---

## New Stages

### `stages/link.py` — Transaction Linking Stage

New stage that pairs receipt + bank statement images, runs the `transaction_link`
workflow via `GraphExecutor`, and writes results to JSONL.

```python
# stages/link.py  (NEW file)

def run(
    pairs_path: Path,         # CSV/JSONL mapping receipt -> bank statement images
    image_dir: Path,
    output_path: Path,
    *,
    workflow_path: Path = Path("prompts/workflows/transaction_link.yaml"),
    model_type: str = "internvl3",
    config_path: Path | None = None,
) -> Path:
    """Stage: Transaction linking via graph executor.

    Reads image pairs, runs cross-image workflow, writes raw_extractions.jsonl.
    Compatible with existing clean + evaluate stages downstream.
    """
    pairs = read_pairs(pairs_path)
    workflow_def = load_workflow(workflow_path)

    # Load model via existing infrastructure (no changes to model loading)
    model, tokenizer = load_model(config)
    generate_fn = make_generate_fn(backend)

    parsers = build_parser_registry()
    executor = GraphExecutor(generate_fn, parsers)

    with StreamingJsonlWriter(output_path) as writer:
        for pair in pairs:
            session = executor.run(
                document_type="TRANSACTION_LINK",
                definition=workflow_def,
                images={
                    "receipt": str(image_dir / pair["receipt"]),
                    "bank_statement": str(image_dir / pair["bank_statement"]),
                },
            )
            writer.write(session.to_record())

    return output_path
```

### `stages/graph_extract.py` — Single-Image Multi-Turn Stage

New opt-in stage for multi-turn extraction on single images. Runs alongside
the existing `stages/extract.py` — not a replacement.

```python
# stages/graph_extract.py  (NEW file)

def run(
    classifications_path: Path,
    image_dir: Path,
    output_path: Path,
    *,
    workflow_dir: Path = Path("prompts/workflows"),
    model_type: str = "internvl3",
) -> Path:
    """Stage: Multi-turn extraction via graph executor.

    For each classified image, looks up a workflow YAML by document type.
    If no workflow exists for a doc type, falls back to single-node extraction.

    Writes raw_extractions.jsonl compatible with existing clean stage.
    """
    ...
```

Both stages produce `raw_extractions.jsonl` in the same format as the existing
`stages/extract.py`. The downstream clean and evaluate stages work unchanged.

---

## Implementation Phases

### Phase 1: Foundation (no model calls)

New files only. No behavior change. No existing code touched.

1. Create `common/extraction_types.py` — all dataclasses.
2. Create `common/turn_parsers.py` — `TurnParser` protocol, `ParseError`, stub implementations.
3. Create `common/graph_executor.py` — graph walker with `match`-based dispatch.
4. Create `common/graph_generate.py` — `make_generate_fn` wrapper.
5. Unit tests for graph executor with mock `generate_fn`.

**Validation:** Tests pass. No existing tests affected.

### Phase 2: Transaction Linking

Promote the notebook prototype to a production stage.

1. Create `prompts/workflows/transaction_link.yaml` — full workflow definition.
2. Implement `ReceiptListParser` — from notebook's `parse_stage1_response()`.
3. Implement `TransactionMatchParser` — from notebook's match parsing.
4. Implement `HeaderListParser` — parse numbered column header lists.
5. Implement `amount_gate` validator and `dedup` post-processor.
6. Create `stages/link.py` — pairing logic + `GraphExecutor.run()`.
7. Register `transaction_link` in `field_definitions.yaml` with scored + metadata fields.

**Validation:** Run `stages/link.py` on the same image pairs as the notebook.
Diff results field-by-field. Accuracy matches or exceeds notebook baseline.

### Phase 3: Single-Image Multi-Turn

Add multi-turn workflows for standard doc types.

1. Create `prompts/workflows/invoice_multi_turn.yaml` — header + line item split.
2. Create `prompts/workflows/receipt_validated.yaml` — extract + total validation.
3. Implement `FieldValueParser` for standard `FIELD: value` parsing.
4. Implement `invoice_total_matches_line_items` validator.
5. Create `stages/graph_extract.py` — single-image multi-turn stage.

**Validation:** Compare accuracy against existing single-turn extraction on the
same images. Multi-turn should match or improve.

---

## Structural Pattern Matching Summary

Nine `match` sites across the new engine:

| Site | Phase | Matches on | Purpose |
|------|-------|-----------|---------|
| Definition dispatch | 1 | `{"inputs": ..., "nodes": ...}` vs `{"nodes": ...}` | Cross-image vs single-image workflow |
| Node execution | 1 | `{"type": "router"}` vs `{"type": "validator"}` vs `{"template": ...}` | Node type dispatch |
| Generation mode | 1 | `GenerationParams(output_schema=dict())` vs default | Structured output vs free-text parsing |
| Strategy router | 2 | `ColumnMapping` dataclass attributes | Bank column strategy selection |
| Response format | 1 | JSON shape (dict vs list vs multi-turn dict) | Auto-detect response structure |
| SamplingParams build | 1 | `GenerationParams(output_schema=...)` / `(stop=...)` / `(logprobs=...)` | vLLM parameter dispatch in wrapper |
| Validator dispatch | 2 | `check_name` string | Route to validator implementation |
| Post-processing | 2 | `{"type": "dedup"}` / `{"type": "amount_gate"}` | Declarative post-processing hooks |
| Confidence routing | 3 | `GenerationParams(logprobs=int())` + threshold | Logprob-based retry trigger |

**Where match/case is NOT used:**
- Graph traversal loop — simple `while current != "done"` with dict lookup.
- Per-field cleaning — string-prefix lookup, better as dict dispatch.
- Workflow YAML loading — straightforward `yaml.safe_load()`.
- vLLM engine config — `LLM()` constructor kwargs are flat.

---

## What Is NOT Touched

| Existing component | Status |
|-------------------|--------|
| `stages/extract.py` | Unchanged — continues handling bank + standard extraction |
| `stages/clean.py` | Unchanged — reads `raw_response`, calls `handler.handle()` |
| `stages/evaluate.py` | Unchanged — reads `cleaned_extractions.jsonl` |
| `common/unified_bank_extractor.py` | Unchanged — bank extraction continues as-is |
| `common/response_handler.py` | Unchanged — parse/clean/validate chain |
| `common/prompt_catalog.py` | Unchanged — existing prompt loading |
| `models/backends/vllm_backend.py` | Unchanged — wrapper accesses `backend.model` directly |
| `models/model_loader.py` | Unchanged — `VllmSpec` untouched |
| `prompts/internvl3_prompts.yaml` | Unchanged — existing prompts stay as-is |
| `prompts/llama_prompts.yaml` | Unchanged |
| `config/run_config.yml` | Unchanged |

## Risk Mitigation

- **Zero blast radius** — all new files. If the engine has bugs, existing pipeline
  continues to work. Delete the new files to fully revert.
- **Parser duplication** — new parsers duplicate algorithms from the notebook and
  existing code rather than importing/modifying them. Slight code duplication is
  the cost of the additive constraint; it ensures the existing code path is stable.
- **Retry cycles** — `max_graph_steps=20` circuit breaker prevents infinite loops.
  `max_retries` per node caps individual retry counts.
- **No framework dependency** — `GraphExecutor` is ~200 lines of Python with `match`
  dispatch. No LangGraph, CrewAI, or other external dependency.
- **Phase independence** — Phase 2 (transaction linking) and Phase 3 (multi-turn)
  are independent. Either can ship without the other.
- **Notebook as regression reference** — the transaction linking notebook stays as-is.
  Run both side-by-side on the same pairs to validate.
