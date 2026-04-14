# Plan: KFP Staged Pipeline Decomposition

## Problem

The current pipeline runs as a single monolithic step (`run_batch_inference`) in KFP:

```
entrypoint.sh  ->  cli.py  ->  load model + detect + extract + clean + evaluate + report
```

This means:
- **No intermediate artifacts** -- if extraction succeeds but evaluation has a bug, you re-run everything including the expensive GPU inference
- **No cacheability** -- KFP can cache stage outputs and skip unchanged stages; a monolith defeats this
- **No stage-level parallelism** -- classification could fan out to N extraction workers
- **No observability** -- you can't inspect "what did the model detect?" independently of "what did it extract?"
- **GPU time wasted** -- evaluation, cleaning, and reporting are CPU-only but hold the GPU allocation for the full duration

The target is independent KFP tasks with file-based handoff:

```
classify  ->  [classifications.jsonl]  ->  extract  ->  [extractions.jsonl]  ->  clean  ->  [cleaned.jsonl]  ->  evaluate
```

Each artifact (JSONL file) is intrinsically valuable and inspectable.

---

## Current Data Flow

Tracing what actually flows between phases in `DocumentPipeline._run_pipeline()`:

### Phase 1: Detection

**Input**: `list[str]` (image paths)
**Output**: `list[DetectionResult]` -- one per image

```python
@dataclass(frozen=True, slots=True)
class DetectionResult:
    image_path: str
    image_name: str
    document_type: str                  # "INVOICE", "RECEIPT", "BANK_STATEMENT"
    classification_info: dict[str, Any] # full model output (confidence, raw_response, prompt_used)
```

### Phase 2: Extraction

**Input**: `list[str]` (image paths) + `list[DetectionResult]` (from Phase 1)
**Output**: `list[ExtractionOutput]` -- one per image

```python
@dataclass(frozen=True, slots=True)
class ExtractionOutput:
    image_path: str
    image_name: str
    document_type: str
    extracted_data: dict[str, str]      # {"TOTAL_AMOUNT": "$42.00", "INVOICE_DATE": "15/03/2026", ...}
    processing_time: float
    prompt_used: str
    error: str | None = None
```

**Note**: `extracted_data` here is the **raw parsed** output -- it has been through `ResponseHandler.handle()` which runs parse -> clean -> validate. So extraction and cleaning are currently fused.

### Phase 3: Evaluation

**Input**: `list[ExtractionOutput]` (from Phase 2) + ground truth CSV
**Output**: `list[ImageResult]` with `evaluation: dict` containing F1 scores

---

## Proposed Stage Boundaries

### Stage 1: `classify` (GPU)

**Runs on**: GPU node
**Input**: Image directory path
**Output**: `classifications.jsonl` -- one JSON object per line

```jsonl
{"image_path": "/data/img_001.png", "image_name": "img_001.png", "document_type": "INVOICE", "confidence": 0.95, "raw_response": "INVOICE", "prompt_used": "detection"}
{"image_path": "/data/img_002.png", "image_name": "img_002.png", "document_type": "BANK_STATEMENT", "confidence": 0.88, "raw_response": "BANK STATEMENT", "prompt_used": "detection"}
```

**What it does**: Loads model, runs detection on all images, writes results, releases GPU.

**Artifact value**: "What document types does the model see?" -- useful for data quality assessment, routing decisions, and debugging misclassifications without re-running extraction.

### Stage 2: `extract` (GPU)

**Runs on**: GPU node
**Input**: `classifications.jsonl` + image directory
**Output**: `raw_extractions.jsonl` -- one JSON object per line

```jsonl
{"image_name": "img_001.png", "document_type": "INVOICE", "raw_response": "DOCUMENT_TYPE: INVOICE\nSUPPLIER_NAME: Acme Corp\n...", "processing_time": 3.2, "prompt_used": "invoice", "error": null}
{"image_name": "img_002.png", "document_type": "BANK_STATEMENT", "raw_response": "DOCUMENT_TYPE: BANK_STATEMENT\n...", "processing_time": 45.1, "prompt_used": "unified_bank_flat", "error": null}
```

**What it does**: Loads model, reads classifications, runs type-specific extraction (including multi-turn bank), writes **raw model responses**, releases GPU.

**Key decision**: This stage writes the **raw model response string**, not the parsed/cleaned dict. This is the critical change -- it separates GPU inference from CPU post-processing.

**Artifact value**: "What did the model actually say?" -- the raw response is the most valuable debugging artifact. You can re-parse, re-clean, and re-evaluate without touching the GPU.

### Stage 3: `clean` (CPU)

**Runs on**: CPU node (no GPU needed)
**Input**: `raw_extractions.jsonl`
**Output**: `cleaned_extractions.jsonl`

```jsonl
{"image_name": "img_001.png", "document_type": "INVOICE", "extracted_data": {"DOCUMENT_TYPE": "INVOICE", "SUPPLIER_NAME": "Acme Corp", "TOTAL_AMOUNT": "$42.00", ...}, "field_count": 14, "extracted_fields_count": 11}
{"image_name": "img_002.png", "document_type": "BANK_STATEMENT", "extracted_data": {"DOCUMENT_TYPE": "BANK_STATEMENT", "TRANSACTION_DATES": "01/03/2026 | 05/03/2026", ...}, "field_count": 5, "extracted_fields_count": 5}
```

**What it does**: Reads raw responses, runs `ResponseHandler.handle()` (parse -> clean -> validate) on each, writes structured field dicts.

**Artifact value**: "What structured data did we extract?" -- the primary business deliverable. Can be fed into downstream systems, compared across model versions, or used for prompt engineering iteration.

### Stage 4: `evaluate` (CPU)

**Runs on**: CPU node (no GPU needed)
**Input**: `cleaned_extractions.jsonl` + ground truth CSV
**Output**: `evaluation_results.jsonl` + summary CSVs + reports

```jsonl
{"image_name": "img_001.png", "document_type": "INVOICE", "overall_accuracy": 0.92, "median_f1": 0.95, "field_scores": {"SUPPLIER_NAME": 1.0, "TOTAL_AMOUNT": 1.0, "INVOICE_DATE": 0.0, ...}}
```

**What it does**: Reads cleaned extractions, compares against ground truth, computes F1 scores, generates analytics/visualizations/reports.

**Artifact value**: "How good is the extraction?" -- the evaluation output. Comparing evaluation artifacts across model versions or prompt versions is the primary use case.

---

## What Needs to Change in the Codebase

### New: `stages/` package

```
stages/
    __init__.py
    classify.py         # Stage 1: detection
    extract.py          # Stage 2: raw extraction
    clean.py            # Stage 3: parse + clean + validate
    evaluate.py         # Stage 4: evaluation + reporting
    io.py               # JSONL read/write helpers
```

Each stage module exposes:
- A `run()` function callable from Python (for testing, notebooks)
- A `typer` CLI command (for `entrypoint.sh` dispatch)

### `stages/io.py` -- shared JSONL I/O

```python
import json
from pathlib import Path
from typing import Any

def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]
```

### `stages/classify.py` -- Stage 1

```python
def run(
    image_dir: Path,
    output_path: Path,
    *,
    model_type: str = "internvl3",
    config_path: Path | None = None,
) -> Path:
    """Detect document types for all images, write classifications.jsonl."""
    # Load model (GPU)
    # For each image: orchestrator.detect_and_classify_document()
    # Write DetectionResult dicts to JSONL
    # Release GPU
    return output_path
```

### `stages/extract.py` -- Stage 2

```python
def run(
    classifications_path: Path,
    image_dir: Path,
    output_path: Path,
    *,
    model_type: str = "internvl3",
    config_path: Path | None = None,
) -> Path:
    """Extract fields from classified images, write raw_extractions.jsonl."""
    # Load model (GPU)
    # Read classifications.jsonl
    # For each image: run extraction, capture RAW model response string
    # Write raw responses to JSONL (NOT parsed/cleaned)
    # Release GPU
    return output_path
```

**Critical**: This requires the orchestrator to expose the raw response string separately from the parsed dict. Currently `process_document_aware()` returns an `ExtractionResult` where `extracted_data` is already parsed+cleaned. We need to either:

1. Add a `raw_response` field to the extraction output (it's already in the TypedDict definition but gets overwritten), or
2. Split the orchestrator's `process_single_image()` into `generate_raw()` + `parse_and_clean()` (cleaner but more work)

### `stages/clean.py` -- Stage 3

```python
def run(
    raw_extractions_path: Path,
    output_path: Path,
) -> Path:
    """Parse, clean, and validate raw model responses. No GPU needed."""
    # Read raw_extractions.jsonl
    # For each record: ResponseHandler.handle(raw_response, expected_fields)
    # Write cleaned_extractions.jsonl
    return output_path
```

This is the stage that benefits most from the `ResponseHandler` refactor we just completed -- it's a pure function of `(raw_response, field_list) -> dict[str, str]` with no GPU dependency.

### `stages/evaluate.py` -- Stage 4

```python
def run(
    cleaned_extractions_path: Path,
    ground_truth_csv: Path,
    output_dir: Path,
) -> Path:
    """Evaluate cleaned extractions against ground truth. No GPU needed."""
    # Read cleaned_extractions.jsonl
    # Load ground truth CSV
    # Run ExtractionEvaluator on each record
    # Generate analytics, visualizations, reports
    # Write evaluation_results.jsonl + CSVs
    return output_dir
```

### Modified: `entrypoint.sh`

The `case` statement gains new task names:

```bash
case "${KFP_TASK:-}" in
  run_batch_inference)
    # Legacy monolithic path (unchanged, for backward compat)
    python3 ./cli.py "${CLI_ARGS[@]}" || exit $?
    ;;
  classify)
    python3 -m stages.classify "${CLI_ARGS[@]}" || exit $?
    ;;
  extract)
    python3 -m stages.extract "${CLI_ARGS[@]}" || exit $?
    ;;
  clean)
    python3 -m stages.clean "${CLI_ARGS[@]}" || exit $?
    ;;
  evaluate)
    python3 -m stages.evaluate "${CLI_ARGS[@]}" || exit $?
    ;;
  *)
    log "FATAL: Unknown KFP_TASK '${KFP_TASK}'"
    exit 1
    ;;
esac
```

### Unchanged: `cli.py`

The existing monolithic `cli.py` stays as-is. It's the development/debug entry point. The staged pipeline is an alternative execution path for KFP production.

---

## Orchestrator Changes Required

The main code change is exposing the **raw model response** from extraction, before parsing/cleaning.

Currently in `DocumentOrchestrator.process_single_image()`:

```python
raw_response = self._generate_with_oom_recovery(image, prompt, params)
extracted_data = self._response_handler.handle(raw_response, active_fields)  # parse+clean+validate fused
return {"extracted_data": extracted_data, "raw_response": raw_response, ...}
```

The `raw_response` is already captured -- it just needs to be preserved through the pipeline instead of being discarded. The `ExtractionResult` TypedDict already has a `raw_response` field. The change is ensuring `DocumentPipeline` propagates it into `ExtractionOutput` and the JSONL writer captures it.

Similarly for `extract_batch()` -- the batch path returns `raw_response` per image but it gets dropped when building the result dict.

**Estimated changes**:
- `ExtractionOutput` in `batch_types.py`: add `raw_response: str = ""` field
- `DocumentPipeline._extract_all()`: populate `raw_response` from orchestrator output
- `stages/extract.py`: write `raw_response` to JSONL
- `stages/clean.py`: read `raw_response`, run `ResponseHandler.handle()`, write cleaned data

---

## KFP Pipeline YAML

The KFP manifest would define the stage DAG:

```yaml
workflow_definition:
  tasks:
    classify:
      type: gpu
      command: classify
      outputs:
        - classifications_jsonl
    extract:
      type: gpu
      command: extract
      inputs:
        - classify.classifications_jsonl
      outputs:
        - raw_extractions_jsonl
    clean:
      type: cpu
      command: clean
      inputs:
        - extract.raw_extractions_jsonl
      outputs:
        - cleaned_extractions_jsonl
    evaluate:
      type: cpu
      command: evaluate
      inputs:
        - clean.cleaned_extractions_jsonl
        - ground_truth_csv
      outputs:
        - evaluation_results_jsonl
        - reports
```

**Cost benefit**: The `clean` and `evaluate` stages run on CPU nodes (much cheaper than GPU nodes). Currently they hold a GPU allocation for their entire duration.

---

## Effort Estimate

| Component | Lines | Complexity | Notes |
|-----------|-------|------------|-------|
| `stages/io.py` | ~20 | Trivial | JSONL read/write |
| `stages/classify.py` | ~80 | Low | Thin wrapper around orchestrator detection |
| `stages/extract.py` | ~120 | Medium | Must handle bank vs standard routing, raw response capture |
| `stages/clean.py` | ~60 | Low | Pure function: ResponseHandler.handle() per record |
| `stages/evaluate.py` | ~100 | Low | Wrapper around ExtractionEvaluator + reporting |
| `batch_types.py` change | ~5 | Trivial | Add raw_response to ExtractionOutput |
| `document_pipeline.py` change | ~10 | Low | Propagate raw_response through pipeline |
| `entrypoint.sh` change | ~20 | Trivial | New case branches |
| **Total** | **~415** | | |

---

## Design Decision: Two Separate GPU Stages

**Decision**: `classify` and `extract` are **two separate GPU stages** that each load the model independently.

**Rationale**: At production scale (10,000 images per run), model load time (~60s x2 = ~2 min) is negligible against total inference time (~hours). The benefits of separation far outweigh the cost:

- **Classification artifact at scale is intrinsically valuable** -- at 10K images, knowing the document type distribution, misclassification rate, and confidence distribution *before* spending GPU hours on extraction is critical for go/no-go decisions
- **KFP caching** -- if images haven't changed, KFP skips classification entirely and jumps to extraction with cached classifications
- **Failure isolation** -- if extraction OOMs on image 8,500 of 10,000, you don't re-classify all 10,000
- **Resource sizing** -- classification is fast (~1-2s/image, single pass); extraction is slow (~3-5s for standard, ~60-90s for bank). Different stages can request different GPU time limits and memory

---

## Open Questions

1. **JSONL vs Parquet?** JSONL is human-readable and simple. Parquet is more efficient for large volumes and integrates better with pandas/analytics. Could support both via a `--format` flag. At 10K images, JSONL files are still small (~10-50 MB) so this is not urgent.

2. **Should the monolithic `cli.py` path also use the staged internals?** i.e., `cli.py` calls `stages.classify.run()` then `stages.extract.run()` etc. internally, just without file I/O between stages. This keeps one code path but adds complexity.

3. **Bank statement extraction spans detect+extract** (the multi-turn pipeline does its own sub-detection in Turn 0). Should bank statements be a special case that runs both stages atomically, or should Turn 0 header detection be folded into the classify stage?

4. **Extraction stage restart from failure** -- at 10K images, the extract stage should support resumption. If it fails at image 8,500, it should be able to read `raw_extractions.jsonl` (with 8,499 records), skip those images, and continue from 8,500. This requires the JSONL writer to flush per-record (not buffer the full batch).
