# InternVL3.5-8B vLLM Attention Backend Benchmark

## Objective

Measure throughput and accuracy of **InternVL3.5-8B** served via **vLLM** on the
**wildreceipt** benchmark, comparing two attention backends:

1. `VLLM_ATTENTION_BACKEND=FLASHINFER`
2. `VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1` *(exact enum TBC — see Preflight §1)*

This is the GQA configuration that broke `XFORMERS` paged decode earlier in
this branch, so the comparison is specifically about which of the remaining
policy-safe backends actually works on dense-GQA decode and which is faster.

## Scope (LOCKED)

In scope:
- Model: **InternVL3.5-8B only** (`--model internvl3-vllm`)
- Dataset: **wildreceipt only**
- Engine: **vLLM only** (no HF transformers / no SDPA monkey-patch)
- Backends: **FLASHINFER** and **TRITON** only

Explicit non-goals:
- No SROIE, no bank-statement, no private evaluation_data
- No 14B / 38B
- No HF transformers path
- No `XFORMERS` backend (already known-broken on GQA paged decode)
- No env yaml edits — per-invocation env var override only
- No harness changes — `benchmark_wildreceipt.py` already reports F1 +
  `elapsed_seconds` + images/min

## Hardware

- **Dev**: 2× NVIDIA L4 (22.9 GiB each, SM 8.9 Ada)
- **Prod**: 4× A10G or 4× L4 (all 22–24 GiB-class Ampere/Ada)

Runs in this plan execute on **dev only**. Prod runs come later and are not
part of this plan.

## Prerequisites (assumed — do not re-verify here)

- `vllm-0.11.2` conda env already built from `vllm_env.yaml` (Python 3.12)
- `models/registry.py` InternVL3 vLLM loader has
  `gpu_memory_utilization=0.85` and `max_num_seqs=8` (commit `ed82ec8`)
- `../data/wildreceipt/image_files/` exists with test images on dev machine
- Correct `--data-dir` is `../data/wildreceipt` (not `data/wildreceipt`)

## Preflight (one-off, dev machine)

### 1. Resolve the Triton backend enum name

vLLM has renamed this backend across minor versions. The only authoritative
source is the installed package:

```bash
conda activate vllm-0.11.2
python -c "from vllm.platforms.interface import _Backend; print(sorted(b.name for b in _Backend))"
```

Paste the output into this plan at the placeholder below and pin the exact
string used for all Triton runs.

**Pinned Triton enum**: `<TBC — fill in from command above>`

If neither `TRITON_ATTN_VLLM_V1` nor any `TRITON_*` value appears in the enum
list, stop and report back — Triton is not available in this vLLM build and
the comparison reduces to FLASHINFER vs nothing.

### 2. Per-invocation env var takes precedence over conda activation var

`vllm_env.yaml` pins `VLLM_ATTENTION_BACKEND=XFORMERS` as a conda activation
variable. Shell precedence: a command-line `VAR=value command` override wins
over the activation export in the same shell. Confirm once:

```bash
VLLM_ATTENTION_BACKEND=FLASHINFER python -c "import os; print(os.environ['VLLM_ATTENTION_BACKEND'])"
# expect: FLASHINFER
```

If this prints `XFORMERS`, the conda activation is `export -f`-ing the var
and we need a different override strategy (e.g. `unset VLLM_ATTENTION_BACKEND
&& VLLM_ATTENTION_BACKEND=...`).

### 3. Confirm the registered model key

```bash
python -c "from models.registry import _REGISTRY; print([k for k in _REGISTRY if 'internvl3' in k])"
```

Expect `internvl3-vllm` in the list. That's the key the run commands below
use.

## Smoke tests (5 images per backend)

Fast sanity check before committing to full runs. Catches backend load
failures, Triton JIT warmup crashes, or OOMs without wasting a full eval.

```bash
# FLASHINFER smoke
VLLM_ATTENTION_BACKEND=FLASHINFER \
python benchmark_wildreceipt.py \
    --model internvl3-vllm \
    --data-dir ../data/wildreceipt \
    --max-images 5 \
    --output-dir output/wildreceipt/smoke_flashinfer

# TRITON smoke (use pinned enum from Preflight §1)
VLLM_ATTENTION_BACKEND=<PINNED_TRITON_ENUM> \
python benchmark_wildreceipt.py \
    --model internvl3-vllm \
    --data-dir ../data/wildreceipt \
    --max-images 5 \
    --output-dir output/wildreceipt/smoke_triton
```

Pass criteria for each smoke run:
- Model loads (no OOM, no `OperatorNotFound`)
- 5 images processed without crash
- vLLM startup log mentions the selected backend (grep for
  `Using .* backend` or `attention backend` in stderr)
- F1 > 0 (non-empty extractions)

If the Triton smoke fails with a kernel-compile or attr error, stop and
report — do not proceed to full runs.

## Full runs (all wildreceipt test images)

Only after both smoke tests pass.

```bash
# FLASHINFER full run
VLLM_ATTENTION_BACKEND=FLASHINFER \
python benchmark_wildreceipt.py \
    --model internvl3-vllm \
    --data-dir ../data/wildreceipt \
    --output-dir output/wildreceipt/flashinfer \
    2>&1 | tee logs/wildreceipt_flashinfer.log

# TRITON full run
VLLM_ATTENTION_BACKEND=<PINNED_TRITON_ENUM> \
python benchmark_wildreceipt.py \
    --model internvl3-vllm \
    --data-dir ../data/wildreceipt \
    --output-dir output/wildreceipt/triton \
    2>&1 | tee logs/wildreceipt_triton.log
```

Notes:
- Triton's first run has a ~20–60 s one-time JIT kernel compile. The smoke
  run above primes the JIT cache, so the full run's `elapsed_seconds` is
  clean (compiled kernels cached to `~/.cache/triton/`).
- Run backends back-to-back in the **same shell session** to keep conda env
  + GPU state identical. Don't run them on different machines.
- `nvidia-smi dmon -s um -d 5 -o T` in a separate terminal while each run
  executes captures memory/utilization timeseries if peak memory data is
  wanted. Optional — the JSON output already has enough for the primary
  comparison.

## Metrics captured

`benchmark_wildreceipt.py` already writes both:
- `output/wildreceipt/<backend>/results.json`
- `output/wildreceipt/<backend>/results.csv`

Each contains per-run:
- `total_images`, `total_fields`
- `overall_f1`, per-field F1 breakdown
- `elapsed_seconds`
- implied `images_per_minute = total_images / elapsed_seconds * 60`
- model type, data dir, timestamp

No harness changes needed.

## Comparison deliverable

After both full runs complete, produce a single markdown table at
`results/internvl3_8b_vllm_backend_comparison.md`:

| Backend | Images | F1 (overall) | Elapsed (s) | Images/min | Notes |
|---|---|---|---|---|---|
| FLASHINFER | … | … | … | … | |
| TRITON (`<enum>`) | … | … | … | … | |

Plus a per-field F1 comparison if the overall numbers are close (< 1 F1
point apart). If the Triton smoke failed, record that and stop — the
deliverable is "FLASHINFER is the only policy-safe backend that works".

## Acceptance

This plan is done when:
1. Both smoke tests passed (or Triton failed with a clear error recorded).
2. Both full runs (or the remaining one) produced `results.json`.
3. The comparison markdown is committed to `results/`.
4. A one-paragraph conclusion in the comparison file states which backend
   is the dev-machine winner and whether anything blocks moving the same
   comparison to 4× A10G / 4× L4 prod.

## Out of scope — do not touch as part of this plan

- Changing `gpu_memory_utilization` or `max_num_seqs` in `registry.py`
  (the 0.85 / 8 values are the fix we just pushed and are part of the
  baseline, not a variable)
- Adding `TORCH_SDPA` or any third backend
- Adding a GPU-memory-aware backend auto-selector
- Editing `vllm_env.yaml` to change the default backend
- Running the same comparison on any other model / dataset
