# Plan: Type-Aware GPU Partition

## Problem

With 4 GPUs and 195 images (34 bank statements, 161 standard), even shuffled partitioning can give one GPU disproportionately more bank statements (~90s each vs ~10s for standard docs). Result: one GPU finishes 8+ minutes after the others.

## Goal

Run detection on one GPU first (Phase 0), then distribute images so each GPU gets an equal share of bank statements. The wall-clock time becomes limited by the slowest GPU, so equalizing bank statement count minimizes total time.

## Approach: Phase 0 Detection + Round-Robin by Type

### Current flow (multi_gpu.py)

```
partition images -> load models -> process chunks in parallel
                                   (each chunk does: detect -> extract -> evaluate)
```

### New flow

```
load ONE model -> Phase 0: detect ALL images on GPU 0
-> type-aware partition (bank statements dealt round-robin)
-> load remaining models
-> process chunks in parallel (skip detection, extract + evaluate only)
```

### What changes

#### File: `common/multi_gpu.py`

1. **New method `_type_aware_partition()`** (~20 lines)
   - Takes `images: list[Path]` and `classifications: list[dict]`
   - Separates bank statements from standard docs
   - Deals bank statements round-robin across N GPU buckets
   - Fills remaining slots with standard docs (contiguous split)
   - Prints distribution summary per GPU
   - Returns `list[list[Path]]` (same shape as `_partition_images`)
   - Also returns `list[list[dict]]` — the pre-computed classifications per chunk

2. **New method `_run_phase0_detection()`** (~15 lines)
   - Takes `gpu_stack` (GPU 0's loaded model) and all `images`
   - Calls `processor.batch_detect_documents()` or sequential fallback
   - Returns `list[dict]` — classification info for every image
   - Prints timing: `Phase 0 detection: X.Xs for N images`

3. **Modify `run()` method** to support pre-classified mode
   - When `type_aware=True`:
     - Load model on GPU 0 first
     - Run Phase 0 detection on GPU 0
     - Call `_type_aware_partition()` instead of `_partition_images()`
     - Load models on remaining GPUs
     - Pass pre-classifications to `_process_chunk()`
   - When `type_aware=False`: existing flow unchanged

4. **Modify `_process_chunk()`** to accept optional pre-classifications
   - New param: `pre_classifications: list[dict] | None = None`
   - Passes them through to `run_batch_processing()`

#### File: `cli.py`

5. **Modify `run_batch_processing()`** to accept optional pre-classifications
   - New param: `pre_classifications: list[dict] | None = None`
   - Passes to `BatchDocumentProcessor` or sets on it before `process_batch()`

#### File: `common/batch_processor.py`

6. **Modify `_process_batch_two_phase()`** to skip Phase 1 when pre-classifications provided
   - New param or attribute: `pre_classifications: list[dict] | None`
   - If set, skip Phase 1 detection entirely, use pre-classifications directly
   - Phase 2 extraction proceeds as normal

### Constructor change

```python
class MultiGPUOrchestrator:
    def __init__(self, config, num_gpus, *, shuffle=False, type_aware=False):
```

- `type_aware=False` by default (production/inference mode: no Phase 0 overhead)
- `type_aware=True` for evaluation: pays ~15s detection cost on one GPU, saves minutes of imbalance
- `shuffle` is ignored when `type_aware=True` (type-aware supersedes shuffle)

### Partition algorithm

```python
def _type_aware_partition(self, images, classifications):
    bank_indices = [i for i, c in enumerate(classifications)
                    if c["document_type"].upper() == "BANK_STATEMENT"]
    standard_indices = [i for i in range(len(images)) if i not in set(bank_indices)]

    n = min(self.num_gpus, len(images))
    buckets = [[] for _ in range(n)]

    # Round-robin bank statements
    for i, idx in enumerate(bank_indices):
        buckets[i % n].append(idx)

    # Fill with standard docs (contiguous split)
    chunk_size = math.ceil(len(standard_indices) / n)
    for gpu_id in range(n):
        start = gpu_id * chunk_size
        end = min(start + chunk_size, len(standard_indices))
        buckets[gpu_id].extend(standard_indices[start:end])

    # Convert indices to image lists
    chunks = [[images[i] for i in bucket] for bucket in buckets]
    chunk_classifications = [[classifications[i] for i in bucket] for bucket in buckets]
    return chunks, chunk_classifications
```

### Example distribution (195 images, 34 bank, 4 GPUs)

| GPU | Bank | Standard | Total |
|-----|------|----------|-------|
| 0   | 9    | 40       | 49    |
| 1   | 9    | 40       | 49    |
| 2   | 8    | 41       | 49    |
| 3   | 8    | 40       | 48    |

Max bank difference: 1 image (~90s) vs current 5+ images (~450s+).

## Files Modified

| File | Change | Lines |
|------|--------|-------|
| `common/multi_gpu.py` | `type_aware` flag, `_type_aware_partition()`, `_run_phase0_detection()`, modified `run()` | ~60 added |
| `cli.py` | Pass `type_aware` to orchestrator, `pre_classifications` through `run_batch_processing()` | ~10 modified |
| `common/batch_processor.py` | Skip Phase 1 when `pre_classifications` provided | ~10 modified |
| `README.md` | Document `type_aware` flag (already added) | ~30 added |

## Toggle

```python
# cli.py — enable for evaluation, disable for production
orchestrator = MultiGPUOrchestrator(config, resolved_gpus, type_aware=True)
```

## Throughput timing

`start_time` (used for `self.inference_elapsed`) must be set **after** Phase 0 detection completes. Phase 0 is a classification-only pass that won't exist in production, so it must not inflate the throughput denominator.

## Trade-offs

- **Cost**: ~15s extra for Phase 0 detection on one GPU (excluded from throughput timing)
- **Benefit**: Eliminates 5-8 minute GPU imbalance with bank-heavy workloads
- **Risk**: Phase 0 detection must match what per-GPU detection would produce. Since it uses the same model and prompts, this is guaranteed.
- **No impact on single-GPU**: `type_aware` only applies to multi-GPU path
