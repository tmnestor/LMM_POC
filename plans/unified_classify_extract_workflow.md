# Unified Classify + Extract YAML Workflow

## Context

The pipeline currently runs as 4 separate stages:
```
Stage 1 (classify.py) → classifications.jsonl → Stage 2 (extract.py) → raw_extractions.jsonl → Stage 3 → Stage 4
```

Classification uses its own model call, prompt loading, and response parsing -- all in Python, completely outside the GraphExecutor engine. We want to express classification as a YAML graph node so the entire image pipeline (classify → route by type → extract → post-process) is a single declarative workflow.

**Goal**: A unified YAML workflow where the first node classifies the document, a router branches by detected type, and type-specific extraction subgraphs run to completion. Stage 1 is eliminated for the graph path; `stages/extract.py` drives the whole thing.

**Constraint**: Additive. `stages/classify.py` and the existing `bank_extract.yaml` stay untouched. New unified workflow runs alongside via an opt-in flag.

---

## Current vs Proposed Flow

### Current (4 stages)
```
Image → Stage 1 (classify.py, GPU) → classifications.jsonl
     → Stage 2 (extract.py, GPU)   → raw_extractions.jsonl
     → Stage 3 (clean.py, CPU)     → cleaned_extractions.jsonl
     → Stage 4 (evaluate.py, CPU)  → evaluation_results.jsonl
```

### Proposed (3 stages, graph path)
```
Image → Stage 2 (extract.py --graph-unified, GPU) → raw_extractions.jsonl
     → Stage 3 (clean.py, CPU)                    → cleaned_extractions.jsonl
     → Stage 4 (evaluate.py, CPU)                  → evaluation_results.jsonl
```

Stage 2 discovers images directly (like classify.py does today), runs the unified graph per image (classify + extract in one workflow), and writes `raw_extractions.jsonl` in the same schema.

---

## Unified Graph Structure

```
classify_document → route_by_type
  → is_bank_statement → detect_headers → select_bank_strategy
        → has_balance_debit   → extract_balance       → bank_post_process → done
        → has_amount          → extract_amount         → bank_post_process → done
        → has_debit_or_credit → extract_debit_credit   → bank_post_process → done
        → default             → extract_schema_fallback                    → done
  → is_receipt   → extract_receipt → done
  → is_invoice   → extract_invoice → done
  → default      → extract_receipt → done   (safety fallback)
```

One YAML file. One `GraphExecutor.run()` call per image. Classification + extraction happen in the same model session.

---

## Files

| File | Action | Purpose |
|------|--------|---------|
| `prompts/workflows/unified_extract.yaml` | NEW | Unified classify + extract workflow graph |
| `common/turn_parsers.py` | EXTEND | Add `ClassificationParser` |
| `common/graph_executor.py` | EXTEND | Add `route_by_type` router logic; make `document_type` derivable from graph |
| `stages/extract.py` | EXTEND | Add `--graph-unified` flag, image discovery, unified graph path |
| `tests/test_unified_graph.py` | NEW | GPU-free tests with mock generate_fn |
| `scripts/run_graph_unified.sh` | NEW | 3-stage pipeline script (no classify stage) |

---

## Implementation

### Step 1: ClassificationParser

Add to `common/turn_parsers.py`:

```python
class ClassificationParser:
    """Parse document classification response using detection YAML config."""

    def parse(self, raw_response: str, context: WorkflowState) -> dict[str, Any]:
        from common.prompt_catalog import PromptCatalog

        catalog = PromptCatalog()
        detection_config = catalog.get_detection_config()

        doc_type = _match_document_type(
            raw_response,
            detection_config["type_mappings"],
            detection_config.get("fallback_keywords", {}),
            detection_config["settings"]["fallback_type"],
        )

        return {"DOCUMENT_TYPE": doc_type, "_raw_classification": raw_response}
```

The `_match_document_type()` helper is a standalone copy of the matching logic from `orchestrator._parse_document_type_response()` (~25 lines):
1. Lowercase + strip the response
2. Check `type_mappings` dict (case-insensitive substring match)
3. Check `fallback_keywords` (first keyword match wins)
4. Return `fallback_type` as last resort

Register as `"classification"` in `build_parser_registry()`.

### Step 2: Extend GraphExecutor Router

Add document-type routing to `_evaluate_router()`:

```python
# Document-type routing (from classify_document node)
if state.has("classify_document"):
    doc_type = state.get("classify_document.DOCUMENT_TYPE")
    if isinstance(doc_type, str):
        type_lower = doc_type.lower()
        for edge_name in edges:
            # Match edge names like "is_bank_statement", "is_receipt", "is_invoice"
            if edge_name.startswith("is_") and edge_name[3:] == type_lower:
                return edge_name

# Existing column-based routing (for bank sub-strategy)
if state.has("detect_headers"):
    ...  # unchanged
```

The router tries document-type edges first, then column-based edges. This is backward-compatible -- existing workflows that don't have a `classify_document` node skip this block entirely.

### Step 3: Make document_type Derivable from Graph

In `GraphExecutor.run()`, after the graph walk completes:

```python
# If the graph classified the document, use that type instead of the input param
if state.has("classify_document"):
    detected_type = state.get("classify_document.DOCUMENT_TYPE")
    if detected_type:
        document_type = detected_type
```

This lets `_build_final_fields(document_type, ...)` use the graph-derived type. The `document_type` parameter to `run()` becomes a fallback default rather than the only source.

### Step 4: YAML Workflow

Create `prompts/workflows/unified_extract.yaml`:

```yaml
name: unified_extract
description: >
  Unified classify + extract workflow. Classifies the document type,
  routes to type-specific extraction, and post-processes results.

nodes:
  classify_document:
    template: |
      What type of business document is this?

      Answer with one of:
      - INVOICE (includes bills, quotes, estimates)
      - RECEIPT (includes purchase receipts)
      - BANK_STATEMENT (includes credit card statements)
    max_tokens: 50
    parser: classification
    edges:
      next: route_by_type

  route_by_type:
    type: router
    edges:
      is_bank_statement: detect_headers
      is_receipt: extract_receipt
      is_invoice: extract_invoice
      default: extract_receipt

  # --- Receipt extraction (single turn) ---
  extract_receipt:
    template: |
      Extract ALL data from this receipt image. Respond in exact format below with actual values or NOT_FOUND.

      DOCUMENT_TYPE: RECEIPT
      BUSINESS_ABN: NOT_FOUND
      SUPPLIER_NAME: NOT_FOUND
      ...
    max_tokens: 2048
    parser: field_value
    edges:
      next: done

  # --- Invoice extraction (single turn) ---
  extract_invoice:
    template: |
      Extract ALL data from this invoice image. Respond in exact format below with actual values or NOT_FOUND.

      DOCUMENT_TYPE: INVOICE
      BUSINESS_ABN: NOT_FOUND
      SUPPLIER_NAME: NOT_FOUND
      ...
    max_tokens: 2048
    parser: field_value
    edges:
      next: done

  # --- Bank statement subgraph (from bank_extract.yaml) ---
  detect_headers:
    template: |
      Look at the transaction table in this bank statement image.
      ...
    max_tokens: 512
    parser: header_list
    edges:
      next: select_bank_strategy

  select_bank_strategy:
    type: router
    edges:
      has_balance_debit: extract_balance
      has_amount: extract_amount
      has_debit_or_credit: extract_debit_credit
      default: extract_schema_fallback

  extract_balance:
    # ... (same as bank_extract.yaml)
    edges:
      next: bank_post_process

  extract_amount:
    # ... (same as bank_extract.yaml)
    edges:
      next: bank_post_process

  extract_debit_credit:
    # ... (same as bank_extract.yaml)
    edges:
      next: bank_post_process

  extract_schema_fallback:
    # ... (same as bank_extract.yaml)
    edges:
      next: done

  bank_post_process:
    type: validator
    check: bank_post_process
    edges:
      pass: done
```

Full prompt templates copied from `internvl3_prompts.yaml` (receipt, invoice) and `bank_extract.yaml` (bank subgraph). The YAML is self-contained.

### Step 5: Stage Integration

Extend `stages/extract.py`:

1. Add `--graph-unified/--no-graph-unified` flag (default `False`).

2. When `--graph-unified`:
   - Discover images from `--data-dir` (reuse image discovery from `stages/classify.py`)
   - Load `unified_extract.yaml` workflow
   - Create `GraphExecutor` with the model's `generate_fn`
   - For each image:
     - Run `executor.run(document_type="UNKNOWN", definition=definition, image_path=path)`
     - The graph classifies the document, routes to extraction, returns `ExtractionSession`
     - Write `session.to_record()` to `raw_extractions.jsonl`
   - No `classifications.jsonl` needed -- classification is embedded in the graph

3. When `--graph-unified` is active, the `--classifications` argument becomes optional (not needed). Image paths come from `--data-dir` directly.

4. Output format stays identical to current Stage 2: `raw_extractions.jsonl` with `image_name`, `image_path`, `document_type`, `raw_response`, `processing_time`, `prompt_used`, `error`.

### Step 6: Pipeline Script

Create `scripts/run_graph_unified.sh` -- a 3-stage script:

```bash
# No Stage 1 -- classification happens inside the graph
# Stage 2: Extract (GPU) with --graph-unified
python -m stages.extract \
    --data-dir "${DATA_DIR}" \
    --output-dir "${ARTIFACTS}/raw_extractions.jsonl" \
    --graph-unified

# Stage 3: Clean (CPU)
python -m stages.clean \
    --input "${ARTIFACTS}/raw_extractions.jsonl" \
    --output-dir "${ARTIFACTS}/cleaned_extractions.jsonl"

# Stage 4: Evaluate (CPU)
python -m stages.evaluate \
    --input "${ARTIFACTS}/cleaned_extractions.jsonl" \
    --ground-truth "${GROUND_TRUTH}" \
    --output-dir "${ARTIFACTS}"
```

### Step 7: Tests

Create `tests/test_unified_graph.py`:

- Mock `generate_fn` returning canned classification + extraction responses
- Test classification parsing: "receipt" → RECEIPT, "bank statement" → BANK_STATEMENT, etc.
- Test router routes correctly by document type
- Test receipt path: classify → route → extract_receipt → done
- Test invoice path: classify → route → extract_invoice → done
- Test bank path: classify → route → detect_headers → select_strategy → extract → post-process → done
- Test fallback: garbage classification response → default → extract_receipt
- Test `document_type` in final fields matches the classified type (not the input param)
- Test full session serialization via `session.to_record()`

---

## What Gets Reused

| Component | From | How |
|-----------|------|-----|
| `_parse_document_type_response()` logic | `orchestrator.py` | Extracted as standalone `_match_document_type()` in `turn_parsers.py` |
| `type_mappings` + `fallback_keywords` | `document_type_detection.yaml` | Loaded at parse time via `PromptCatalog` |
| Bank subgraph nodes | `bank_extract.yaml` | Copied into unified YAML (same prompts) |
| Receipt/invoice prompts | `internvl3_prompts.yaml` | Copied into unified YAML |
| `BalanceDescriptionParser` | `turn_parsers.py` | Already exists |
| `AmountDescriptionParser` | `turn_parsers.py` | Already exists |
| `HeaderListParser` | `turn_parsers.py` | Already exists |
| `FieldValueParser` | `turn_parsers.py` | Already exists |
| `bank_post_process` validator | `bank_post_process.py` | Already exists |
| Image discovery | `stages/classify.py` | Import `_discover_images()` or duplicate (~15 lines) |
| Column-based routing | `graph_executor.py` | Already exists in `_evaluate_router()` |

---

## Key Design Decisions

1. **One YAML per model** -- The prompt templates are model-specific (internvl3 vs llama). Start with `unified_extract.yaml` using InternVL3 prompts. Add `unified_extract_llama.yaml` later if needed.

2. **`document_type` param becomes fallback** -- `run(document_type="UNKNOWN")` is valid. If the graph has a `classify_document` node, its output overrides the param. Backward-compatible with existing workflows that pass a known type.

3. **Single image only** -- The unified workflow handles one image at a time (same as current Stage 2 loop). No batch inference in the graph -- that's a separate optimization.

4. **Prompt copied, not referenced** -- Prompts are embedded in the YAML rather than `$ref`-ing external files. Self-contained is easier to reason about and version.

5. **`--graph-unified` is opt-in** -- Default pipeline behavior unchanged. The 4-stage pipeline with separate classify/extract stages remains the default.

---

## Verification

1. **Unit tests**: `pytest tests/test_unified_graph.py` -- GPU-free, mock generate_fn
2. **Regression**: `pytest tests/` -- all existing tests still pass
3. **Lint**: `ruff check --fix && ruff format && mypy --ignore-missing-imports`
4. **Integration** (on remote): Run `scripts/run_graph_unified.sh` on evaluation data. Compare F1 scores against 4-stage baseline (`scripts/run_baseline_bank.sh`). Both go through same clean/evaluate -- scores should match.

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Classification accuracy differs when model is "warmed up" with extraction context | Unlikely -- classification is the first turn, no prior context. But measure F1 against separate Stage 1. |
| Large YAML file (~300 lines with all prompts) | Acceptable. Self-contained is worth the size. |
| Router edge naming collision between doc-type and column-type edges | Doc-type edges use `is_` prefix (`is_bank_statement`), column edges use `has_` prefix (`has_balance_debit`). No collision. |
| `--graph-unified` makes `--classifications` optional | Use mutually exclusive flag group or just skip reading classifications when `--graph-unified` is set. |
