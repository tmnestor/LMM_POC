# Batch Processor Refactor: Hybrid A+D

## Goal

Decompose the 1,302 LOC `BatchDocumentProcessor` in `common/batch_processor.py` by:

1. Extracting `ExtractionEvaluator` as a standalone class (Design A)
2. Adding typed phase dataclasses as contracts between phases (Design D)
3. Collapsing the two duplicate code paths into one loop
4. Adding a `create_batch_pipeline()` factory for ergonomic construction

## Branch

`refactor/batch-processor-decompose`  (from `feature/multi-gpu`)

## Files Changed

| File | Action | Description |
|---|---|---|
| `common/extraction_evaluator.py` | **NEW** | ~250 LOC. Owns ground truth loading, math enhancement, debit filtering, F1 scoring, metric aggregation |
| `common/batch_types.py` | **NEW** | ~80 LOC. Typed dataclasses: `DetectionResult`, `ExtractionOutput`, `ImageResult`, `BatchResult`, `BatchStats` |
| `common/batch_processor.py` | **MODIFY** | Refactor into one loop, inject `ExtractionEvaluator`, use typed dataclasses, add `create_batch_pipeline()` factory |
| `cli.py` | **MODIFY** | Switch to `create_batch_pipeline()`, use `BatchResult.as_tuple()` |
| `common/multi_gpu.py` | **MODIFY** | Accept `BatchResult` from `run_batch_processing()` (via `.as_tuple()` bridge) |

## Step-by-Step Implementation

### Step 1: Create `common/batch_types.py`

Typed contracts between phases. All frozen dataclasses:

```python
@dataclass(frozen=True, slots=True)
class DetectionResult:
    image_path: str
    image_name: str
    document_type: str
    classification_info: dict[str, Any]

@dataclass(frozen=True, slots=True)
class ExtractionOutput:
    image_path: str
    image_name: str
    document_type: str
    extracted_data: dict[str, str]
    processing_time: float
    prompt_used: str
    skip_math_enhancement: bool = False
    error: str | None = None

@dataclass(frozen=True)
class BatchStats:
    configured_batch_size: int
    avg_detection_batch: float
    avg_extraction_batch: float
    num_detection_calls: int
    num_extraction_calls: int

@dataclass(frozen=True)
class ImageResult:
    image_name: str
    image_path: str
    document_type: str
    extraction_result: dict[str, Any]
    evaluation: dict[str, Any]
    processing_time: float
    prompt_used: str
    timestamp: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Backward-compatible dict for BatchAnalytics/BatchReporter/BatchVisualizer."""
        ...

@dataclass(frozen=True)
class BatchResult:
    results: list[ImageResult]
    processing_times: list[float]
    document_types_found: dict[str, int]
    stats: BatchStats

    def results_as_dicts(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.results]

    def as_tuple(self) -> tuple[list[dict], list[float], dict[str, int]]:
        """Drop-in replacement for the current 3-tuple return."""
        return (self.results_as_dicts(), self.processing_times, self.document_types_found)
```

### Step 2: Create `common/extraction_evaluator.py`

Move these methods out of `BatchDocumentProcessor`:
- `_evaluate_extraction()` (lines 778-994) -> `ExtractionEvaluator.evaluate()`
- `_filter_debit_transactions()` (lines 996-1115) -> `ExtractionEvaluator._filter_debit_transactions()`
- `_lookup_ground_truth()` (lines 183-201) -> `ExtractionEvaluator._lookup_ground_truth()`

Class shape:

```python
class ExtractionEvaluator:
    def __init__(
        self,
        ground_truth_csv: str | None,
        field_definitions: dict[str, list[str]],
        *,
        enable_math_enhancement: bool = False,
        evaluation_method: str = "order_aware_f1",
    ) -> None:
        """Load ground truth once. None = inference-only."""
        ...

    def evaluate(
        self,
        extraction: ExtractionOutput,
    ) -> dict[str, Any]:
        """Score one image's extraction. Returns empty dict when no GT available."""
        ...

    @property
    def has_ground_truth(self) -> bool: ...
```

Ground truth is loaded once in `__init__` (fail-fast). `SimpleModelEvaluator` becomes an internal detail. Bank math enhancement and debit filtering stay inside this class.

### Step 3: Refactor `common/batch_processor.py`

**3a.** Add private methods `_detect_all()` and `_extract_all()` that each handle batch-vs-sequential internally:

```python
def _detect_all(
    self, image_paths: list[str], batch_size: int, verbose: bool
) -> list[DetectionResult]:
    """Detect all images. Uses batch_detect_documents when available, else sequential."""
    ...

def _extract_all(
    self,
    image_paths: list[str],
    detections: list[DetectionResult],
    batch_size: int,
    verbose: bool,
) -> list[ExtractionOutput]:
    """Extract all images. Batched for standard, sequential for bank."""
    ...
```

**3b.** Replace `_process_batch_two_phase` + `_process_batch_sequential` with one `_run_pipeline()`:

```python
def _run_pipeline(self, image_paths, batch_size, verbose) -> BatchResult:
    detections = self._detect_all(image_paths, batch_size, verbose)
    extractions = self._extract_all(image_paths, detections, batch_size, verbose)

    results = []
    for ext in extractions:
        evaluation = self.evaluator.evaluate(ext)
        results.append(ImageResult(
            image_name=ext.image_name,
            image_path=ext.image_path,
            document_type=ext.document_type,
            extraction_result={"extracted_data": ext.extracted_data, ...},
            evaluation=evaluation,
            processing_time=ext.processing_time,
            prompt_used=ext.prompt_used,
            timestamp=datetime.now().isoformat(),
            error=ext.error,
        ))

    return BatchResult(results=results, ...)
```

**3c.** Update `process_batch()` to call `_run_pipeline()` and return the legacy 3-tuple (backward compat):

```python
def process_batch(self, image_paths, verbose=True, progress_interval=5):
    result = self._run_pipeline(image_paths, self._resolve_batch_size(), verbose)
    self.batch_stats = asdict(result.stats)
    return result.as_tuple()
```

**3d.** Add `create_batch_pipeline()` factory:

```python
def create_batch_pipeline(
    model, prompt_config, ground_truth_csv, *,
    console=None, enable_math_enhancement=False,
    bank_adapter=None, field_definitions=None, batch_size=None,
) -> "BatchDocumentProcessor":
    """Same params as the current constructor. Returns configured processor."""
    ...
```

**3e.** Remove `_evaluate_extraction`, `_filter_debit_transactions`, `_lookup_ground_truth`, `_process_batch_two_phase`, `_process_batch_sequential` (all moved or collapsed).

### Step 4: Update `cli.py`

Minimal diff — swap constructor call for factory, everything else unchanged:

```python
# Before:
batch_processor = BatchDocumentProcessor(model=processor, ...)
batch_results, processing_times, doc_types = batch_processor.process_batch(...)

# After:
batch_processor = create_batch_pipeline(model=processor, ...)
batch_results, processing_times, doc_types = batch_processor.process_batch(...)
```

### Step 5: Update `common/multi_gpu.py`

Same pattern — the `run_batch_processing` function it calls returns the same 3-tuple. No changes needed if `process_batch()` signature is preserved.

### Step 6: Lint + type check

- `ruff check --fix` + `ruff format` on all changed files
- `mypy --ignore-missing-imports` on all changed files

## Verification

Since this is a refactor (no behavior change), verification is:

1. **Structural**: All imports resolve, types check, linting passes
2. **Behavioral**: Run the full evaluation pipeline on the remote machine with the same images and confirm identical F1 scores before/after
3. **Backward compat**: `process_batch()` returns the same 3-tuple shape, `batch_stats` dict has the same keys, downstream consumers (`BatchAnalytics`, `BatchReporter`, `BatchVisualizer`, `print_accuracy_by_document_type`) work unchanged

## What This Does NOT Change

- Model loading, registry, SDPA patching
- Bank statement extraction logic (adapter, extractor, calculator)
- Prompt loading
- GPU memory management
- Detection/extraction model calls (just reorganized)
- The `print_accuracy_by_document_type` free function stays in `batch_processor.py`

## Migration Path

The `process_batch()` method continues to return the legacy 3-tuple. Callers can optionally switch to the new typed API:

```python
# New API (when ready):
result: BatchResult = batch_processor.run(image_paths)
result.results[0].evaluation["median_f1"]
```

The legacy `process_batch()` can be deprecated later once all callers migrate.
