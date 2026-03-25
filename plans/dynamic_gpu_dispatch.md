# Plan: Dynamic GPU Work-Queue Dispatch

## Problem

The current multi-GPU approach pre-partitions images into fixed chunks (one per GPU). Even with type-aware partitioning, GPUs finish at different times because bank statement processing time varies (60-120s per image). A GPU that draws several slow bank statements sits idle at the end while others have long finished.

**Root cause**: Static partitioning commits each GPU to a fixed workload before processing starts. No rebalancing is possible at runtime.

## Goal

Replace static partitioning with a shared work queue. Each GPU pulls the next image when it finishes the current one. GPUs that get fast images naturally process more of them; GPUs stuck on slow bank statements simply process fewer total images. Wall-clock time converges to `total_work / num_gpus` — optimal load balancing with zero overhead.

## Why This Works Now

Average batch size is always < 4. The batched detection/extraction pipeline (`batch_chat`) provides negligible speedup at these sizes. This means switching to single-image dispatch per GPU costs almost nothing in per-image throughput, while eliminating multi-minute GPU idle time.

## Approach: Thread-Safe Queue + Per-GPU Worker Loop

### Current flow (multi_gpu.py)

```
partition images into N chunks → load N models → process chunks in parallel
                                                  (each chunk: detect batch → extract batch)
```

### New flow

```
load N models → fill shared queue with all images → N worker threads pull images one at a time
                                                     (each image: detect → extract → store result)
```

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Shared Work Queue                    │
│  [(0, img_a), (1, img_b), (2, img_c), ...]          │
└──────────┬──────────────┬──────────────┬─────────────┘
           │              │              │
     ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
     │  GPU 0    │ │  GPU 1    │ │  GPU 2    │   ...
     │  model    │ │  model    │ │  model    │
     │  bank_adp │ │  bank_adp │ │  bank_adp │
     │           │ │           │ │           │
     │  loop:    │ │  loop:    │ │  loop:    │
     │   pull()  │ │   pull()  │ │   pull()  │
     │   detect  │ │   detect  │ │   detect  │
     │   extract │ │   extract │ │   extract │
     │   store   │ │   store   │ │   store   │
     └───────────┘ └───────────┘ └───────────┘
           │              │              │
           ▼              ▼              ▼
     ┌─────────────────────────────────────────────────┐
     │         Results Array (pre-allocated)            │
     │  results[0] = {...}, results[1] = {...}, ...     │
     └─────────────────────────────────────────────────┘
```

### What changes

#### File: `common/multi_gpu.py`

1. **New method `_worker_loop()`** (~30 lines)
   - Takes: `gpu_id`, `processor`, `bank_adapter`, `work_queue`, `results`, `timings`, `doc_types_lock`, `doc_types_found`, `console`
   - Loop: `queue.get_nowait()` → detect → extract → store in `results[idx]`
   - Each image goes through the same detect→route→extract path as `_process_image()` in batch_processor.py
   - Thread-safe: each worker writes to its own slot in `results[idx]`, `doc_types_found` protected by a `threading.Lock`
   - Bank statements routed to per-GPU `BankStatementAdapter` instance

2. **New method `_create_bank_adapter()`** (~10 lines)
   - Creates a `BankStatementAdapter` for a given processor (mirrors logic in `cli.py:run_batch_processing()`)
   - Each GPU thread gets its own adapter (not shared — avoids thread-safety issues)

3. **Modify `run()` method** — replace the `ThreadPoolExecutor.submit(_process_chunk)` block:
   - Phase 1: Load models on all GPUs (unchanged)
   - Phase 2: Fill `queue.Queue` with `(index, image_path)` tuples
   - Phase 3: Launch N worker threads, each running `_worker_loop()`
   - Phase 4: Join threads, collect results in original order
   - Throughput timing wraps Phase 2-4

4. **Remove**: `_partition_images()`, `_shuffle_images()`, `_type_aware_partition()`, `_run_phase0_detection()`, `_process_chunk()`
   - All partitioning logic becomes unnecessary
   - `shuffle` and `type_aware` constructor params removed

5. **Keep**: `_merge_results()` may need minor adjustment — or replace with simpler result collection since results are already indexed

#### File: `cli.py`

6. **Simplify orchestrator call site**
   - Remove `shuffle=True, type_aware=True` — no longer needed
   - `MultiGPUOrchestrator(config, resolved_gpus)` is sufficient

7. **`run_batch_processing()`** — unchanged (single-GPU path still uses `BatchDocumentProcessor`)

#### File: `common/batch_processor.py`

8. **No changes** — single-GPU path is untouched. The multi-GPU path bypasses `BatchDocumentProcessor` entirely, using processor methods directly.

### Worker loop pseudocode

```python
def _worker_loop(
    self,
    gpu_id: int,
    processor: Any,
    bank_adapter: BankStatementAdapter | None,
    work_queue: queue.Queue,
    results: list[dict | None],
    timings: list[float | None],
    doc_types_found: dict[str, int],
    doc_types_lock: threading.Lock,
) -> None:
    """Pull images from shared queue until empty."""
    import torch
    torch.cuda.set_device(gpu_id)
    count = 0

    while True:
        try:
            idx, image_path = work_queue.get_nowait()
        except queue.Empty:
            break

        img_start = time.time()

        # Detect
        classification = processor.detect_and_classify_document(
            str(image_path), verbose=False
        )
        doc_type = classification["document_type"]

        # Track document types (thread-safe)
        with doc_types_lock:
            doc_types_found[doc_type] = doc_types_found.get(doc_type, 0) + 1

        # Extract (route bank statements to adapter)
        if doc_type.upper() == "BANK_STATEMENT" and bank_adapter is not None:
            schema_fields, metadata = bank_adapter.extract_bank_statement(
                str(image_path)
            )
            extraction_result = {
                "extracted_data": schema_fields,
                "raw_response": metadata.get("raw_responses", {}).get("turn1", ""),
                "field_list": list(schema_fields.keys()),
                "metadata": metadata,
            }
        else:
            extraction_result = processor.process_document_aware(
                str(image_path), classification, verbose=False
            )

        # Store result at original index (no lock needed — unique slot)
        results[idx] = {
            "image_file": image_path.name,
            "document_type": doc_type,
            "extraction_result": extraction_result,
        }
        timings[idx] = time.time() - img_start
        count += 1
        work_queue.task_done()

    console.print(f"  [green]GPU {gpu_id}: processed {count} images[/green]")
```

### Evaluation

Evaluation (ground truth comparison) currently happens per-image inside `BatchDocumentProcessor._evaluate_single_image()`. Two options:

**Option A (simpler)**: Run evaluation as a post-processing step after all workers finish. Load ground truth once, iterate results, call `SimpleModelEvaluator.evaluate_extraction()` per image. This keeps workers focused on GPU-bound work.

**Option B**: Each worker also evaluates (CPU-only, fast). Requires passing ground truth data and field definitions to workers. Slightly more complex but gives real-time progress.

**Recommendation**: Option A — evaluation is CPU-only and fast (~0.01s/image). Running it sequentially after all GPU work is done simplifies workers and avoids threading ground truth state.

### Progress reporting

Current: Rich progress bar updated per-batch inside `BatchDocumentProcessor`.

New: A monitor thread or periodic check from the main thread reports queue depth:
```
Processing: 142/195 images remaining (GPU 0: 18, GPU 1: 15, GPU 2: 17, GPU 3: 14 processed)
```

Alternatively, keep it simple — just print when each GPU finishes (as we do now), plus a final summary.

## What does NOT change

- Single-GPU path — still uses `BatchDocumentProcessor` with batched detection/extraction
- Model loading — still sequential, one model per GPU
- `BankStatementAdapter` / `UnifiedBankExtractor` — unchanged, just instantiated per-GPU
- CLI flags — `--gpus`, `--batch-size` (ignored in multi-GPU), `--model`, etc.
- Evaluation metrics — same `SimpleModelEvaluator`, same F1 calculation

## Example scenario (195 images, 34 bank, 4 GPUs)

### Current (type-aware partition)
| GPU | Bank | Standard | Est. time |
|-----|------|----------|-----------|
| 0   | 9    | 40       | 9×90 + 40×10 = 1210s |
| 1   | 9    | 40       | 1210s |
| 2   | 8    | 41       | 1130s |
| 3   | 8    | 40       | 1120s |

Wall-clock: ~1210s (limited by GPUs 0/1)

### Dynamic dispatch
All 4 GPUs pull from the same queue. When GPU 2 finishes a 10s receipt, it grabs the next image immediately — it doesn't wait for GPU 0 to finish its 90s bank statement. Each GPU processes roughly equal total work time:
- ~4850s total work / 4 GPUs ≈ 1213s per GPU

Wall-clock: ~1213s — but critically, **no GPU sits idle**. The variance comes from the last image, not from partition imbalance.

## Migration path

1. Implement `_worker_loop()` and queue-based dispatch in `multi_gpu.py`
2. Remove partitioning methods (`_partition_images`, `_shuffle_images`, `_type_aware_partition`, `_run_phase0_detection`)
3. Remove `shuffle`/`type_aware` params from constructor
4. Add post-processing evaluation step (reuse `SimpleModelEvaluator`)
5. Update `cli.py` call site
6. Update `README.md` — remove shuffle/type-aware sections, document work-queue approach
7. Test on 4×A10G with 195 images

## Trade-offs

| Aspect | Current (type-aware) | Dynamic dispatch |
|--------|---------------------|-----------------|
| Load balancing | Near-optimal for bank count, not for processing time variance | Optimal — GPUs never idle |
| Batch inference | Preserves batched detection/extraction | Single-image only (negligible cost at avg batch < 4) |
| Complexity | Phase 0 + partition + skip logic | Simpler — just a queue and worker loop |
| Phase 0 overhead | ~15s detection on GPU 0 | None |
| Code to remove | — | ~120 lines (partition/shuffle/phase0) |
| Code to add | — | ~60 lines (worker loop + queue setup) |

## Risks

- **Bank adapter thread safety**: Each GPU gets its own `BankStatementAdapter` instance — no shared state. The underlying model is also per-GPU. Should be safe.
- **Console output interleaving**: Multiple threads printing simultaneously. Mitigate with `threading.Lock` around console writes, or accept minor interleaving (current approach already has this with `ThreadPoolExecutor`).
- **Queue overhead**: `queue.Queue.get_nowait()` is effectively free compared to GPU inference time.
