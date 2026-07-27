# Sandbox setup: `google/gemma-4-12B-it` + nightly vLLM

Operational runbook for making `gemma4-12b-unified-vllm` runnable on the sandbox. Two independent
prerequisites: a **nightly vLLM engine** (the harder one) and the **~24 GB model download**.

Target box: 1xL40S (~44-48 GiB). The model registration and config already exist on branch
`feature/gemma4` — this document only covers getting the environment and weights in place.

## Why a nightly engine is required

`google/gemma-4-12B-it` is the *Unified* variant: `model_type: gemma4_unified`, architecture
`Gemma4UnifiedForConditionalGeneration`. It is encoder-free — raw image patches and audio waveforms are
projected straight into the LLM embedding space, with no vision tower.

Support landed in `vllm-project/vllm#44429` and **has not shipped in a stable release**. Releases
≤ 0.22.1 cannot load it: the engine fails at startup with an unknown-architecture error that has nothing
to do with this repo's code. There is no workaround short of upgrading the engine.

| Fact | Value |
|---|---|
| Repo | `google/gemma-4-12B-it` |
| Parameters | 11.95B |
| Download size | ~23.95 GB (single unsharded `model.safetensors` = 23,919,549,408 bytes) |
| Licence | Apache-2.0, **not gated** — no token, no click-through |
| Fits | Single 40 GB+ GPU unquantised (~24 GB BF16) |
| Engine | vLLM **nightly** or the pinned `vllm/vllm-openai:gemma4-unified` image |

---

## Part 1 — Survey the box first

Nothing below is destructive; run it before deciding which route to take.

```bash
# What vLLM is actually installed? (the env yaml does NOT pin a version)
conda activate LMM_POC_VLLM
python -c "import vllm; print(vllm.__version__)"

# CUDA toolkit version — this decides whether the pip route is safe (see Part 2)
nvcc --version

# GPU and free VRAM
nvidia-smi

# Space for the weights (~24 GB, plus headroom)
df -h /home/jovyan/nfs_share/models

# Is any Gemma checkpoint already on the share?
ls /home/jovyan/nfs_share/models | grep -i gemma
```

If `vllm.__version__` is already a nightly/dev build that post-dates PR #44429, skip Part 2 entirely.

---

## Part 2 — Nightly vLLM

### ⚠️ Do not install into `LMM_POC_VLLM`

That env runs the validated InternVL3.5-8B production path. `conda_envs/vllm_env.yml` installs vLLM
**unpinned** (`pip install vllm --no-cache-dir` in its post-install notes), so there is no version floor
protecting it — upgrading in place would silently replace the engine the 91.8% baseline was measured on,
with no way back except rebuilding the env. Use a separate environment.

### ⚠️ CUDA toolkit mismatch is the likely failure

`conda_envs/vllm_env.yml` pins torch to the **cu124** index with the note *"must match nvcc toolkit
(12.4), not driver (13.0)"*. vLLM's default nightly wheel is built against **CUDA 12.9**. Installing it
will pull a torch built for 12.9 against a 12.4 toolkit.

This is the main reason to prefer the container route below.

### Route A — pinned Docker image (recommended)

The container ships its own CUDA runtime, so the toolkit mismatch cannot arise, and the stable conda env
is untouched.

```bash
docker pull vllm/vllm-openai:gemma4-unified
```

Run it with the model share mounted and the GPU exposed; serve or exec into it for offline inference.
This is the route the vLLM Gemma 4 recipe recommends for this model.

### Route B — separate conda env with a nightly wheel

Only if Docker is unavailable on the box. Per the project convention, create a **new env yaml** rather
than installing ad hoc — copy `conda_envs/vllm_env.yml` to `conda_envs/vllm_nightly_env.yml`, change
`name:` to `LMM_POC_VLLM_NIGHTLY`, then:

```bash
conda env create -f conda_envs/vllm_nightly_env.yml
conda activate LMM_POC_VLLM_NIGHTLY

# Nightly wheel (default variant is CUDA 12.9 — see the toolkit warning above)
uv pip install -U vllm --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/nightly
```

To pin an exact commit instead of a moving nightly — preferable for anything you want to reproduce later:

```bash
export VLLM_COMMIT=<sha>
uv pip install vllm --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT}
```

Verify the architecture is registered before going further:

```bash
python -c "import vllm; print(vllm.__version__)"
python -c "from vllm.model_executor.models.registry import ModelRegistry as R; \
print([a for a in R.get_supported_archs() if 'emma' in a])"
```

`Gemma4UnifiedForConditionalGeneration` must appear in that list. If it does not, the wheel predates the
PR and nothing downstream will work.

---

## Part 3 — Download the model

Public and ungated, so no authentication step. The `hf` CLI comes with `huggingface_hub`
(`pip install -U huggingface_hub` if missing; `huggingface-cli` is the deprecated alias).

```bash
# Confirm the file list and size before committing to ~24 GB
hf download google/gemma-4-12B-it --dry-run
```

Expect 9 files totalling ~23.9 GB. Then:

```bash
hf download google/gemma-4-12B-it \
  --local-dir /home/jovyan/nfs_share/models/gemma-4-12B-it
```

That path is the one already listed in `config/run_config.yml` under
`bootstrap.model.default_paths`. `--local-dir` writes real files (not cache symlinks) and keeps a
`.cache/huggingface/` metadata folder at the destination, so **re-running the same command resumes** —
which matters because a single 23.9 GB file means one interrupted transfer has a lot to redo.

If `$HOME` is small or read-only, keep temp/cache writes on the share:

```bash
HF_HOME=/home/jovyan/nfs_share/.hf_home hf download google/gemma-4-12B-it \
  --local-dir /home/jovyan/nfs_share/models/gemma-4-12B-it
```

**No outbound internet from the box?** The repo is public, so there is no credential to move — download
on a machine that has egress and copy it across:

```bash
hf download google/gemma-4-12B-it --local-dir ./gemma-4-12B-it
rsync -av --progress ./gemma-4-12B-it/ <box>:/home/jovyan/nfs_share/models/gemma-4-12B-it/
```

### Verify the download

A truncated `model.safetensors` is the most likely silent problem — the byte count is the check.
(`jq` is installed on the box.)

```bash
cd /home/jovyan/nfs_share/models/gemma-4-12B-it

ls -l model.safetensors                              # expect exactly 23919549408 bytes
jq -r '.architectures[0], .model_type' config.json   # Gemma4UnifiedForConditionalGeneration / gemma4_unified
jq '.vision_config.num_soft_tokens' config.json      # expect 280 — the budget knob we override
jq 'has("vision_soft_tokens_per_image")' config.json # expect false (that field belongs to the 31B)
```

The last two matter: this model's soft-token budget lives at `vision_config.num_soft_tokens`, **not** the
`vision_soft_tokens_per_image` / `vision_config.default_output_length` fields the dense 31B uses. The YAML
already reflects that. If these checks disagree with the above, stop and re-check the config before
running — a mismatch means the override silently no-ops and the engine quietly runs at 280.

---

## Part 4 — Switch the pipeline to it

In `config/run_config.yml`, three coupled edits:

```yaml
bootstrap:
  model:
    type: gemma4-12b-unified-vllm
    path: /home/jovyan/nfs_share/models/gemma-4-12B-it

inference:
  tiling:
    pre_tiling:
      enabled: false      # REQUIRED — Gemma sizes images via its own soft-token budget
```

Pre-tiling must be off: it crops InternVL 448-px tiles, which this model does not want, and with
`limit_mm_per_prompt: 1` vLLM would silently drop all but the first crop. The loader fails fast with a
diagnostic if you forget, so a missed edit is loud rather than silent.

Everything else (`max_model_len`, `gpu_memory_utilization`, soft-token budget) is already set under
`inference.vllm.models.gemma4-12b-unified-vllm`.

---

## Part 5 — Smoke it

Always through the entrypoint — never a standalone `python -m stages/...`, or the model path, tiling and
GPU-memory environment will be wrong and the measurement invalid.

```bash
KFP_TASK=run_info_extract bash entrypoint.sh
```

Confirm from the logs:

1. Engine loads, `tp=1`.
2. No `<think>` blocks in raw output (enable `inference.tracing.raw_prompts: true` for one run to inspect).
3. The image part precedes the text part in the prompt.
4. Accuracy and s/image, compared against an InternVL3.5-8B run **from the same day and same code** —
   historical numbers are not a valid comparison.

## Troubleshooting

| Symptom | Cause |
|---|---|
| Unknown/unsupported architecture at engine load | vLLM predates PR #44429 — Part 2 not done, or done in the wrong env |
| `ImportError` / torch CUDA mismatch after pip install | cu129 nightly wheel against the 12.4 toolkit — use Route A |
| Loads but behaves as if budget is 280 | `hf_overrides` not landing; re-check the Part 3 `jq` output |
| Silently reads only part of the page | `pre_tiling.enabled` still true, or `limit_mm_per_prompt` > 1 |
| `<think>` blocks in output | Thinking is opt-in on this model via a `<\|think\|>` token in the system prompt; check nothing is injecting a system prompt |

## Rollback

Set `bootstrap.model.type: internvl3-vllm`, restore its path, and set
`inference.tiling.pre_tiling.enabled: true`. Nothing in the Gemma work changes InternVL behaviour, and
the stable `LMM_POC_VLLM` env is untouched provided Part 2's warning was respected.
