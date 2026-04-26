# Plan: vLLM Native Data Parallel (Option B)

**Goal**: Run InternVL3.5-8B as DP=N, TP=1 on 4x A10G (PCIe-only), avoiding
the `/dev/shm` stall that TP causes on this cluster.

**Branch**: `agentic-extraction-engine`

---

## Architecture Decision

**Pattern**: Mirror `MultiGPUOrchestrator` but with `multiprocessing.Process`
instead of `ThreadPoolExecutor`.

Why processes, not threads:
- vLLM engines are heavyweight (own CUDA contexts, schedulers, worker threads)
- Each process gets `CUDA_VISIBLE_DEVICES` pinned to one GPU
- No GIL concern — each process is independent
- Matches vLLM's own offline DP example exactly

**Scope**: Only the stages pipeline (`stages/classify.py`, `stages/extract.py`).
The old `cli.py` path keeps using `MultiGPUOrchestrator` for HF models unchanged.

---

## Implementation Steps

### Step 1: Add config fields

**Files**: `common/pipeline_config.py`, `config/run_config.yml`

Add to `PipelineConfig` dataclass:
```python
data_parallel_size: int | None = None  # None = use num_gpus for DP when vLLM
```

Add YAML parsing in `load_yaml_config()`:
```python
flat_config["data_parallel_size"] = raw_config["processing"].get("data_parallel_size")
```

Add env var in `load_env_config()`:
```python
f"{ENV_PREFIX}DATA_PARALLEL_SIZE": ("data_parallel_size", int),
```

YAML section:
```yaml
processing:
  data_parallel_size: null  # null = auto (num_gpus for vLLM, ignored for HF)
```

---

### Step 2: Create `VllmDPOrchestrator`

**File**: `common/vllm_dp.py` (new)

This is the core of the plan. A process-based orchestrator that:

1. Partitions images into N chunks (reuse `_partition_images` logic from `multi_gpu.py`)
2. Spawns N `multiprocessing.Process` workers
3. Each worker:
   - Sets `CUDA_VISIBLE_DEVICES=gpu_id`
   - Imports vLLM and constructs `LLM(tensor_parallel_size=1, ...)`
   - Receives its image chunk + config via constructor args (not pickle)
   - Runs extraction (calls a `worker_fn` callback)
   - Puts results on a `multiprocessing.Queue`
4. Main process collects results from queue, merges in original order

```python
"""vLLM data-parallel orchestrator.

Spawns N independent vLLM engine processes, each pinned to one GPU
via CUDA_VISIBLE_DEVICES. No NCCL, no /dev/shm, no inter-process
communication during inference.
"""

import math
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


def _partition_images(images: list[Path], n: int) -> list[list[Path]]:
    """Split images into n contiguous chunks."""
    n = min(n, len(images))
    chunk_size = math.ceil(len(images) / n)
    return [images[i : i + chunk_size] for i in range(0, len(images), chunk_size)]


def run_dp(
    *,
    num_gpus: int,
    images: list[Path],
    worker_fn: str,       # dotted import path, e.g. "common.vllm_dp._classify_worker"
    worker_kwargs: dict[str, Any],  # serializable kwargs passed to each worker
) -> list[dict[str, Any]]:
    """Launch N workers, collect merged results in original image order.

    Args:
        num_gpus: Number of GPUs / DP ranks.
        images: Full image list (will be partitioned).
        worker_fn: Dotted path to a top-level function with signature:
            (gpu_id: int, image_paths: list[str], **worker_kwargs) -> list[dict]
        worker_kwargs: Extra kwargs forwarded to worker_fn (must be picklable).

    Returns:
        Merged list of result dicts in original image order.
    """
    chunks = _partition_images(images, num_gpus)
    actual_gpus = len(chunks)

    console.print(
        f"\n[bold cyan]vLLM DP: distributing {len(images)} images "
        f"across {actual_gpus} GPUs (TP=1 per GPU)[/bold cyan]"
    )

    result_queue: mp.Queue = mp.Queue()
    procs: list[mp.Process] = []

    for gpu_id, chunk in enumerate(chunks):
        p = mp.Process(
            target=_worker_wrapper,
            args=(gpu_id, chunk, worker_fn, worker_kwargs, result_queue),
        )
        p.start()
        procs.append(p)

    # Collect results (blocks until all workers finish)
    gpu_results: dict[int, list[dict]] = {}
    for _ in range(actual_gpus):
        gpu_id, results = result_queue.get()
        gpu_results[gpu_id] = results
        console.print(f"  [green]GPU {gpu_id} finished: {len(results)} images[/green]")

    for p in procs:
        p.join()

    # Merge in GPU order (preserves original image order since chunks are contiguous)
    merged: list[dict[str, Any]] = []
    for gpu_id in range(actual_gpus):
        merged.extend(gpu_results[gpu_id])

    return merged


def _worker_wrapper(
    gpu_id: int,
    image_chunk: list[Path],
    worker_fn_path: str,
    worker_kwargs: dict[str, Any],
    result_queue: mp.Queue,
) -> None:
    """Subprocess entry point. Pins GPU, imports worker_fn, runs it."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Import the worker function by dotted path
    module_path, fn_name = worker_fn_path.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    fn = getattr(mod, fn_name)

    results = fn(
        gpu_id=gpu_id,
        image_paths=[str(p) for p in image_chunk],
        **worker_kwargs,
    )
    result_queue.put((gpu_id, results))
```

**Design notes**:
- `worker_fn` is a dotted import path (not a callable) because it must be
  importable in the subprocess — no closures over unpicklable objects.
- Each worker imports vLLM fresh in its own process — no shared state.
- `CUDA_VISIBLE_DEVICES` is the only GPU pinning needed. vLLM sees one GPU
  and uses `tensor_parallel_size=1` naturally.
- No `VLLM_DP_RANK` / `VLLM_DP_MASTER` env vars — those are for vLLM's
  internal coordinator which we don't need for independent engines.

> **Note**: This is technically a hybrid of Option B and C from the
> investigation doc. We use independent processes (Option C's shape) but
> follow vLLM's recommended offline DP pattern (Option B's spirit). The
> result is the same: N independent TP=1 engines, zero NCCL, zero `/dev/shm`.

---

### Step 3: Write worker functions

**File**: `common/vllm_dp.py` (append) or `common/vllm_dp_workers.py` (new)

Two worker functions, one per GPU stage:

#### a) Classify worker

```python
def classify_worker(
    gpu_id: int,
    image_paths: list[str],
    *,
    model_path: str,
    model_type: str,
    config_dict: dict[str, Any],
) -> list[dict[str, Any]]:
    """Worker: load vLLM engine, classify each image, return records."""
    from vllm import LLM, SamplingParams
    # ... build LLM, load prompts, run detection per image ...
    # Return list of {"image": stem, "doc_type": type, ...} dicts
```

#### b) Extract worker (graph-based)

```python
def extract_worker(
    gpu_id: int,
    image_paths: list[str],
    *,
    model_path: str,
    model_type: str,
    workflow_name: str,
    config_dict: dict[str, Any],
) -> list[dict[str, Any]]:
    """Worker: load vLLM engine, run graph extraction per image, return records."""
    # 1. Construct LLM(tensor_parallel_size=1, ...)
    # 2. Build generate_fn via make_vllm_generate_fn(engine)
    # 3. Load workflow YAML, build GraphExecutor
    # 4. For each image: executor.run(image, generate_fn) -> record
    # 5. Return list of extraction records
```

**Key constraint**: Worker functions must be importable top-level functions
(not methods, not closures). All arguments must be picklable (strings, dicts,
lists — no model objects, no Path objects across process boundary).

---

### Step 4: Integrate into stages

**File**: `stages/classify.py`

In `run()`, after discovering images and before loading model:

```python
if is_vllm_model(config.model_type) and resolved_gpus > 1:
    from common.vllm_dp import run_dp
    records = run_dp(
        num_gpus=resolved_gpus,
        images=images,
        worker_fn="common.vllm_dp_workers.classify_worker",
        worker_kwargs={
            "model_path": str(config.model_path),
            "model_type": config.model_type,
            "config_dict": config.to_dict(),  # or relevant subset
        },
    )
    # Write records and return
    ...
else:
    # Existing single-GPU / HF multi-GPU path
    ...
```

**File**: `stages/extract.py`

Same pattern in `_run_unified()` (used by `--graph-robust` and `--graph-unified`):

```python
if is_vllm_model(config.model_type) and resolved_gpus > 1:
    from common.vllm_dp import run_dp
    records = run_dp(
        num_gpus=resolved_gpus,
        images=images,
        worker_fn="common.vllm_dp_workers.extract_worker",
        worker_kwargs={
            "model_path": str(config.model_path),
            "model_type": config.model_type,
            "workflow_name": workflow_name,
            "config_dict": config.to_dict(),
        },
    )
    ...
```

**Helper** (in `models/registry.py` or `common/pipeline_ops.py`):
```python
def is_vllm_model(model_type: str) -> bool:
    """Check if a model type uses vLLM backend."""
    return model_type in _vllm_registry
```

---

### Step 5: GPU count resolution for vLLM DP

**File**: `stages/classify.py`, `stages/extract.py`

Add GPU resolution logic (similar to `cli.py` lines 414-456 but simpler):

```python
def _resolve_gpu_count(config) -> int:
    """Resolve effective GPU count for vLLM DP."""
    if config.data_parallel_size is not None:
        return config.data_parallel_size
    if config.num_gpus > 0:
        return config.num_gpus
    # Auto-detect
    import torch
    return torch.cuda.device_count()
```

---

### Step 6: Force TP=1 for DP mode in model_loader

**File**: `models/model_loader.py`

In `build_vllm_loader()`, when DP mode is active, force `tensor_parallel_size=1`:

```python
# Current (line 427):
"tensor_parallel_size": tp_size,

# New:
"tensor_parallel_size": 1 if dp_mode else tp_size,
```

However, in the DP architecture above, each worker sets
`CUDA_VISIBLE_DEVICES` to a single GPU, so `build_vllm_loader()` will
auto-detect 1 GPU and set `tp_size=1` naturally. **No change needed** if
we rely on `CUDA_VISIBLE_DEVICES` masking.

This step may be a no-op. Verify during implementation.

---

### Step 7: Update run scripts

**File**: `scripts/run_graph_robust.sh`, `scripts/run_graph_bank.sh`

No script changes needed if `num_gpus` in `run_config.yml` is already set
(currently `num_gpus: 2`). The stages will auto-detect vLLM + multi-GPU
and use DP.

Optional: add `--data-parallel-size` CLI flag to stages for explicit control.

---

## File Change Summary

| File | Change | Size |
|---|---|---|
| `common/vllm_dp.py` | **New** — DP orchestrator + worker wrapper | ~100 lines |
| `common/vllm_dp_workers.py` | **New** — classify_worker, extract_worker | ~150 lines |
| `common/pipeline_config.py` | Add `data_parallel_size` field + YAML/env parsing | ~10 lines |
| `stages/classify.py` | DP dispatch (if vllm + multi-gpu) | ~15 lines |
| `stages/extract.py` | DP dispatch in `_run_unified()` | ~15 lines |
| `models/registry.py` | Add `is_vllm_model()` helper | ~5 lines |
| `config/run_config.yml` | Add `data_parallel_size` field | ~1 line |

**Unchanged**:
- `common/graph_executor.py` — callback-based, model-agnostic
- `common/graph_generate.py` — `make_vllm_generate_fn()` works as-is
- `common/unified_bank_extractor.py` — uses generate callback
- `models/orchestrator.py` — DocumentOrchestrator unchanged
- `models/model_loader.py` — `CUDA_VISIBLE_DEVICES` masking handles TP=1
- `common/multi_gpu.py` — HF path stays as-is
- `cli.py` — old entrypoint unchanged

---

## Verification Checklist (on production)

1. **NCCL check** — confirm no NCCL init at TP=1:
   ```bash
   NCCL_DEBUG=INFO python -m stages.extract --graph-robust ... 2>&1 | grep "NCCL INFO"
   ```
   Expected: no output.

2. **`/dev/shm` check** — no growth:
   ```bash
   ls -la /dev/shm  # before and during
   ```

3. **Throughput benchmark** — 29-image synthetic dataset:
   - 1 GPU baseline (single engine)
   - 4 GPU DP (4 independent engines)
   - Target: 3-4x speedup

4. **Accuracy parity** — F1 on DP=4 must match DP=1 (same prompts, same model, deterministic if `do_sample=false`).

5. **Prefix caching** — verify KV reuse on multi-turn bank extractions:
   ```bash
   VLLM_LOGGING_LEVEL=DEBUG ... | grep "prefix cache"
   ```

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| 8B model doesn't fit on A10G 24GB at TP=1 | Lower `gpu_memory_utilization` to 0.85, reduce `max_model_len` |
| Worker crash doesn't propagate cleanly | Add timeout + `p.exitcode` check in `run_dp()` |
| Results arrive out of order | Queue returns `(gpu_id, results)` — merge by gpu_id |
| mp.Queue can't handle large result dicts | Results are small (text fields per image) — not a concern |
| vLLM engine startup is slow x4 | Sequential startup is fine — model weights are read-only, no contention |

---

## Commit Sequence

1. `feat: add data_parallel_size to PipelineConfig`
2. `feat: add VllmDPOrchestrator with process-based worker spawning`
3. `feat: add classify and extract DP worker functions`
4. `feat: integrate vLLM DP dispatch into stages/classify and stages/extract`
5. `test: verify DP=4 on production with 29-image synthetic dataset`
