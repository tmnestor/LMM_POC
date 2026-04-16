# Plan: Fix kfp OOM -- `_release_gpu_memory()` on wrong GPU in worker threads

## Correction

My first diagnosis was wrong. The user is running the **monolithic `cli.py`
path**, and 4 independent model copies ARE being loaded in parallel (via
`MultiGPUOrchestrator`) -- same as `feature/multi-gpu`. The stages are not
involved in the OOM.

The OOM is happening within the multi-GPU parallel path. So the fix must
be in the common inference code, not the stages.

---

## Problem

`kfp` OOMs on 4x A10G (24 GB each) when `feature/multi-gpu` does not, using
the same `cli.py` entrypoint and same `MultiGPUOrchestrator` path.

## Root Cause Hypothesis

### What `kfp` added that `feature/multi-gpu` doesn't have

`common/gpu_memory.py` is a **kfp-only file** -- it doesn't exist on
`feature/multi-gpu` (which uses the older `robust_gpu_memory.py`).

`common/document_pipeline.py` on kfp calls `_release_gpu_memory()` at
three sites (lines 386, 437, 536):

```python
def _release_gpu_memory() -> None:
    try:
        from .gpu_memory import release_memory
        release_memory(threshold_gb=1.0)
    except ImportError:
        pass
```

`release_memory()` (from `common/gpu_memory.py:47`):

```python
def release_memory(*, threshold_gb: float = 1.0) -> None:
    # Loop over ALL GPUs to compute GLOBAL fragmentation
    total_alloc = 0.0
    total_reserved = 0.0
    for gpu_id in range(torch.cuda.device_count()):
        total_alloc += torch.cuda.memory_allocated(gpu_id) / (1024**3)
        total_reserved += torch.cuda.memory_reserved(gpu_id) / (1024**3)

    fragmentation = total_reserved - total_alloc
    if fragmentation <= threshold_gb:
        return

    gc.collect()
    torch.cuda.empty_cache()       # operates on current device
    torch.cuda.synchronize()       # operates on current device
```

### Why this breaks under MultiGPUOrchestrator

`common/multi_gpu.py:85` calls `torch.cuda.set_device(gpu_id)` exactly once
per GPU **during Phase 1 model loading in the main thread**. Phase 2 spawns
a `ThreadPoolExecutor` and dispatches `_process_chunk()` per GPU -- the
worker function has **no `set_device()` call**:

```python
@staticmethod
def _process_chunk(gpu_stack, images, field_definitions):
    gpu_config, _model_ctx, processor = gpu_stack
    return run_batch(gpu_config, processor, images, field_definitions)
```

PyTorch's current device is a **thread-local** setting. Worker threads
inherit whatever state the main thread had last. When Phase 1 ends, the
main thread's current device is `cuda:3` (the last set_device in the loop).

So when each of 4 worker threads calls `_release_gpu_memory()`:
- `torch.cuda.memory_allocated(gpu_id)` returns correct per-GPU values (it
  takes an explicit index) -- the **global fragmentation check is correct**
- `torch.cuda.empty_cache()` runs on the thread's **current device**, which
  is `cuda:3` for all four workers
- `torch.cuda.synchronize()` same -- serializes on GPU 3

**Result**: every worker's `empty_cache()` call targets GPU 3. Memory
fragmentation on GPUs 0, 1, 2 is never released, even though the
fragmentation check keeps firing (threshold is computed globally). Over
many batches this drives GPUs 0-2 into OOM. GPU 3 is over-synchronized,
serializing the parallel work.

On `feature/multi-gpu` none of this fires because `_release_gpu_memory()`
simply doesn't exist there.

### Why only now

Commit `94537f1 "fix: add GPU cleanup after standard batches"` added the
`_release_gpu_memory()` call after the standard-extraction batch loop
(`document_pipeline.py:386`). Before that commit, cleanup only happened
between bank statements (sequential, single-threaded after the batch phase
ends) where current-device-is-wrong is less catastrophic. Once it started
firing inside the parallel batch loop, every batch calls `empty_cache()`
on the wrong GPU in three of four worker threads.

---

## Goals

1. Fix the multi-GPU OOM so `kfp` matches `feature/multi-gpu` throughput.
2. Keep the memory-release behaviour (which is valuable single-GPU) but
   make it correct under threaded multi-GPU.
3. Do not regress the bank statement cleanup (which actually needs to free
   memory between slow multi-turn rounds).

## Non-Goals

- No changes to `MultiGPUOrchestrator`'s public API.
- No changes to the stages (they're unrelated to this OOM).
- No changes to `common/gpu_memory.py`'s public interface.
- No change to bank statement cleanup semantics (it runs after the batch
  phase in the main thread, so it's already correct).

---

## Design

### Option 1 (Recommended) -- pin current device in worker threads

**Smallest, safest fix.** Set the CUDA device at the top of
`MultiGPUOrchestrator._process_chunk()`:

```python
@staticmethod
def _process_chunk(gpu_stack, images, field_definitions):
    gpu_config, _model_ctx, processor = gpu_stack

    # Pin worker thread to this GPU so memory-release APIs
    # (empty_cache, synchronize) target the correct device.
    if gpu_config.device_map and gpu_config.device_map.startswith("cuda:"):
        import torch
        gpu_id = int(gpu_config.device_map.split(":")[1])
        torch.cuda.set_device(gpu_id)

    return run_batch(gpu_config, processor, images, field_definitions)
```

This is the **minimum change** that makes `empty_cache()` / `synchronize()`
behave correctly inside worker threads. It does not alter fragmentation
logic, the `release_memory()` threshold, or the call sites.

### Option 2 -- make `release_memory()` per-device and multi-GPU aware

Pass the target device into `release_memory()` and only clean that device:

```python
def release_memory(*, threshold_gb: float = 1.0, device: str | None = None):
    """Release fragmented memory on a specific GPU (or all if device is None)."""
```

Then update `_release_gpu_memory()` in `document_pipeline.py` to pass the
current device, and call sites to pass the GPU the orchestrator is pinned
to. Cleaner long-term but requires plumbing device identity through the
pipeline.

### Option 3 -- skip cleanup in multi-GPU mode

Gate `_release_gpu_memory()` behind a "single-GPU only" flag. Simplest
short-term mitigation but loses the fragmentation protection that the
commit added in the first place.

### Recommendation

Ship **Option 1** first (3-line change, minimum risk). Track Option 2 as a
follow-up cleanup. Skip Option 3 -- we want the protection on.

Option 1 also benefits any future memory-cleanup code that runs inside
worker threads, because the thread is now correctly pinned end-to-end.

---

## Testing Strategy

**Revised for 2xL4 dev machine availability (prod offline until user is back).**
The thread-local device bug reproduces with N>=2 GPUs, so 2xL4 is sufficient
to validate the hypothesis and the fix. Prod 4x A10G remains the final
acceptance gate for production-scale throughput/stress but not for
correctness.

### Local (macOS, no GPU)

#### Test 1: Unit test for device pinning

`tests/test_multi_gpu_device_pin.py`:
- Mock `torch.cuda.set_device` and assert it's called with correct gpu_id
  derived from `gpu_config.device_map`
- Test that `_process_chunk` works when `device_map` is `"cuda:0"`,
  `"cuda:1"`, `"cuda:2"`, `"cuda:3"`
- Test that it no-ops when `device_map` is `"auto"` or `"cpu"` (single-GPU
  fallback)

Run: `pytest tests/test_multi_gpu_device_pin.py -v`

#### Test 2: Existing test suite still passes

Run full test suite to catch any regressions:
```bash
pytest -x
ruff check . --fix --ignore ARG001,ARG002,F841
ruff format .
mypy . --ignore-missing-imports
```

### Dev (2xL4) -- PRIMARY validation (user has access now)

With 2 GPUs the bug reproduces: pre-fix, both workers' `empty_cache()`
calls target the main thread's last-set device (cuda:1), so cuda:0
fragmentation grows unbounded while cuda:1 is cleaned normally. The
signature is distinctive and diagnosable even at 2-GPU scale.

#### Test 3a: Reproduce on pre-fix commit (baseline OOM / memory pattern)

```bash
# On 2xL4 dev machine, on a commit BEFORE the fix
git checkout 9b5307f

# trace_gpu_memory.sh was added AFTER 9b5307f, so extract it from the
# branch tip into /tmp without switching trees.
git show kfp:scripts/trace_gpu_memory.sh > /tmp/trace_gpu_memory.sh
chmod +x /tmp/trace_gpu_memory.sh

# Terminal A: start memory trace
/tmp/trace_gpu_memory.sh /tmp/trace_pre.csv 2 &
TRACE_PID=$!

# Terminal A (continued): run a batch large enough to show fragmentation
# (30+ images recommended so you see at least 3-4 batch boundaries)
python3 cli.py \
  --model internvl3 \
  --data-dir evaluation_data/images_30 \
  --output-dir /tmp/kfp_pre \
  --num-gpus 0   # auto-detect = 2

kill $TRACE_PID
```

**Pass criteria for the hypothesis**:
- `/tmp/trace_pre.csv` shows GPU 0 `mem_used` trending monotonically
  upward (no downward steps between batches)
- `/tmp/trace_pre.csv` shows GPU 1 `mem_used` with periodic downward
  steps (empty_cache firing correctly on cuda:1)
- If run is long enough: OOM on GPU 0, NOT on GPU 1

If GPU 1 shows the growth instead, the hypothesis is inverted (worker
threads actually default to cuda:0, not cuda:1). The fix still applies
-- it explicitly pins each worker -- but we should update the plan's
explanation.

#### Test 3b: Post-fix validation

```bash
# Apply the fix (or checkout the fix commit); kfp tip works.
git checkout kfp   # or: git checkout 3a13651

# From here, scripts/trace_gpu_memory.sh is present in the tree.
./scripts/trace_gpu_memory.sh /tmp/trace_post.csv 2 &
TRACE_PID=$!

python3 cli.py \
  --model internvl3 \
  --data-dir evaluation_data/images_30 \
  --output-dir /tmp/kfp_post \
  --num-gpus 0

kill $TRACE_PID
```

**Pass criteria**:
- Both GPUs show bounded memory footprints with similar downward-step
  patterns at batch boundaries
- No OOM
- Same field-value output as pre-fix (use a small subset where pre-fix
  completed before OOM)

#### Test 3c: Visualize the memory trace (optional)

```python
import pandas as pd
import matplotlib.pyplot as plt

for label, path in [("pre", "/tmp/trace_pre.csv"),
                    ("post", "/tmp/trace_post.csv")]:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for gpu_id in sorted(df["gpu_id"].unique()):
        sub = df[df["gpu_id"] == gpu_id]
        plt.plot(sub["timestamp"], sub["mem_used_mib"],
                 label=f"{label} GPU {gpu_id}")
plt.legend(); plt.ylabel("MiB used"); plt.savefig("/tmp/gpu_mem.png")
```

Expected: pre-post diff is dramatic on GPU 0; GPU 1 is similar in both.

#### Test 3d: Parity diff -- kfp post-fix vs feature/multi-gpu on 2xL4

```bash
# Run 20 images through both branches on same machine, compare CSVs
git checkout feature/multi-gpu
python3 cli.py --model internvl3 -d evaluation_data/images_20 \
  -o /tmp/baseline --num-gpus 0

git checkout <fix-commit>
python3 cli.py --model internvl3 -d evaluation_data/images_20 \
  -o /tmp/fix --num-gpus 0

diff /tmp/baseline/extraction_results.csv /tmp/fix/extraction_results.csv
```

**Pass criteria**: zero diffs (modulo timestamps and processing_time).

### Sandbox (1x L40) -- optional single-GPU sanity check

Phase 1 validation -- confirm single-GPU path is unaffected.

#### Test 3: Smoke test, 5 images via cli.py

```bash
python3 cli.py \
  --model internvl3 \
  --data-dir evaluation_data/images_5 \
  --output-dir /tmp/kfp_fix_test \
  --num-gpus 1
```

**Pass criteria**:
- No OOM
- Same field-value output as pre-fix run on same images
- Log shows `MultiGPUOrchestrator` is NOT invoked (num_gpus=1)
- Completes in expected wall-clock time (within 5% of pre-fix)

### Prod (4x A10G or 4x L4) -- final acceptance (deferred until user back)

#### Test 4: Reproduce the OOM on pre-fix commit

```bash
git checkout 9b5307f   # current HEAD, pre-fix
python3 cli.py --model internvl3 --data-dir <full_dataset> --num-gpus 0
```

**Expected**: OOM. Record:
- Which GPU OOMs first (should be 0, 1, or 2 -- NOT 3)
- How many images completed before OOM
- `nvidia-smi` snapshot showing GPU 3 with low memory, others pinned high

This confirms the hypothesis. If GPU 3 OOMs first, the hypothesis is
wrong and we need to reinvestigate.

#### Test 5: Fix validates -- same run on fix commit

```bash
git checkout <fix-commit>
python3 cli.py --model internvl3 --data-dir <full_dataset> --num-gpus 0
```

**Pass criteria**:
- No OOM
- All images process through to completion
- `nvidia-smi` during run shows all 4 GPUs with similar memory footprint
  (rather than 3 saturated + 1 low)
- Wall-clock time within 10% of `feature/multi-gpu` on same data

#### Test 6: Parity against feature/multi-gpu

Run the same dataset through both:
- `feature/multi-gpu` (known good baseline)
- `kfp` with fix applied

Diff the `extraction_results.csv` files. Field values should match modulo
non-determinism in attention (temperature=0 should be deterministic, so
exact match expected).

**Pass criteria**: zero field-value diffs across the full dataset.

#### Test 7: Stress test -- 3 consecutive runs

Kick off 3 full runs back-to-back without restarting Python. Memory
fragmentation on GPUs 0-2 should now be released at each batch boundary,
so run 3 should finish cleanly.

**Pass criteria**: all 3 runs complete without OOM; `nvidia-smi` between
runs shows clean memory state.

---

## Fallback / Mitigation Plan

If Option 1 doesn't fix it, the hypothesis is wrong. Next steps:

1. Add debug logging inside `release_memory()` to log the current device
   and which GPU was actually cleaned. Re-run on prod, examine logs.
2. Try Option 3 (disable `_release_gpu_memory()` in multi-GPU mode) as a
   bisect step. If this fixes OOM, we've confirmed the cleanup is at fault
   but our device-pinning theory was wrong.
3. Consider whether `torch.cuda.synchronize()` in the worker is serializing
   the pipeline-parallel forward pass in InternVL3 (which itself uses
   multiple streams).

---

## Rollout

1. **Local (done)**: Unit test + Option 1 fix applied, lint/format/mypy/tests all green
2. **2xL4 dev (now)**:
   - Test 3a -- pre-fix memory trace (baseline)
   - Test 3b -- post-fix memory trace (validates)
   - Test 3c -- visualize diff (optional but highly informative)
   - Test 3d -- parity vs feature/multi-gpu
3. **Commit + push** once 2xL4 tests pass -- the hypothesis is validated
   and the fix is proven at 2-GPU scale
4. **Prod 4x when available (deferred)**:
   - Test 4 -- reproduce OOM on pre-fix at 4-GPU scale
   - Test 5 -- validate fix at 4-GPU scale
   - Test 6 -- parity vs feature/multi-gpu at full scale
   - Test 7 -- stress: 3 consecutive runs
5. **Commit message** should reference the root cause (worker threads
   inheriting main thread's current device) so future readers understand
   what's subtle here.

---

## Effort Estimate

| Component | Lines | Complexity |
|-----------|-------|------------|
| `multi_gpu.py::_process_chunk` device pin | ~6 | Trivial |
| `tests/test_multi_gpu_device_pin.py` | ~60 | Low |
| **Total** | **~66** | |

---

## Open Questions

1. **Is `device_map` reliably of the form `"cuda:N"` in worker stacks?**
   Confirm by reading `multi_gpu.py:78` -- it uses `replace(self.config,
   device_map=f"cuda:{gpu_id}")`, so yes.

2. **Does InternVL3's backend need its own `set_device()` pin?** Its
   `.chat()` / `.batch_chat()` code might not explicitly set device on
   tensors; if it relies on current device, Option 1 fixes that too as a
   side benefit. If it already dispatches to the correct device via tensor
   placement, no impact. Either way, Option 1 doesn't make anything worse.

3. **Should `common/gpu_memory.py::release_memory()` take a device
   argument (Option 2)?** Track as a follow-up; not blocking for the OOM
   fix.

4. **Do we also need to pin device in the stages?** The stages don't use
   `MultiGPUOrchestrator` yet, so not for this fix. If/when stages gain
   multi-GPU, the same pinning pattern applies.
