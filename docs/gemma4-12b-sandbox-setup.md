# Sandbox setup: `google/gemma-4-12B-it` + nightly vLLM

Operational runbook for making `gemma4-12b-unified-vllm` runnable on the sandbox. Two independent
prerequisites: a **nightly vLLM engine** (the harder one) and the **~24 GB model download**.

Target box: 1xL40S (~44-48 GiB). The model registration and config already exist on branch
`feature/gemma4` — this document only covers getting the environment and weights in place.

**Most of this runs on a CPU-only node.** Only the engine load and the smoke need the GPU, so the
preparation can all be done while waiting for a GPU allocation:

| Step | CPU-only node | Needs the L40S |
| --- | --- | --- |
| Part 2 — nightly vLLM install | ✅ | |
| Part 2 — arch-registry check | ✅ (pure Python import) | |
| Part 3 — download + `jq` verification | ✅ (network + disk) | |
| Part 4 — `run_config.yml` edits | ✅ | |
| Part 5 — engine load and smoke | | ✅ |

The ~24 GB download is the long pole, so start it first — it costs nothing to run while an allocation is
pending, and it takes the wait off the critical path.

## Why a nightly engine is required

`google/gemma-4-12B-it` is the *Unified* variant: `model_type: gemma4_unified`, architecture
`Gemma4UnifiedForConditionalGeneration`. It is encoder-free — raw image patches and audio waveforms are
projected straight into the LLM embedding space, with no vision tower.

Support landed in `vllm-project/vllm#44429` and **has not shipped in a stable release**. Releases
≤ 0.22.1 cannot load it: the engine fails at startup with an unknown-architecture error that has nothing
to do with this repo's code. There is no workaround short of upgrading the engine.

**This is already settled, not something to discover on the box.** The environment in use,
`conda_envs/vllm_env2.yaml`, pins **`vllm==0.19.0`** — below the threshold. The 12B cannot run on the
current stack.

> **The 31B does NOT need a nightly.** `vllm==0.19.0` already carries `compressed-tensors` as a
> transitive dependency, which is the quantisation path `google/gemma-4-31B-it-qat-w4a16-ct` uses
> (`quant_method: compressed-tensors`, `format: pack-quantized`). So if the goal is to get *a* Gemma 4
> running for comparison, the **31B W4A16 is runnable today on `vllm_env2`** and only needs its weights
> downloaded. The engine upgrade is a 12B-specific cost. Worth weighing before spending time on Part 2.

| Fact | Value |
|---|---|
| Repo | `google/gemma-4-12B-it` |
| Parameters | 11.95B |
| Download size | ~23.95 GB (single unsharded `model.safetensors` = 23,919,549,408 bytes) |
| Licence | Apache-2.0, **not gated** — no token, no click-through |
| Fits | Single 40 GB+ GPU unquantised (~24 GB BF16) |
| Engine | vLLM **nightly** in a separate conda env (no Docker in the sandbox, so the container route is unavailable) |

---

## Part 1 — Survey the box first

Nothing below is destructive; run it before deciding which route to take.

```bash
# Confirm the installed engine matches the pin (expect 0.19.0)
conda activate vllm_env2
python -c "import vllm; print(vllm.__version__)"

# GPU, driver and free VRAM
nvidia-smi

# Space for the weights (~24 GB, plus headroom)
df -h /home/jovyan/nfs_share/models

# Is any Gemma checkpoint already on the share?
ls /home/jovyan/nfs_share/models | grep -i gemma
```

If `vllm.__version__` reports something other than `0.19.0`, the env has drifted from
`conda_envs/vllm_env2.yaml` — find out why before continuing. If it is already a nightly/dev build that
post-dates PR #44429, skip Part 2 entirely.

---

## Part 2 — Nightly vLLM

### ⚠️ Never upgrade `vllm_env2` in place

That env runs the validated InternVL3.5-8B production path, and it is a **tightly matched stack**:

```
vllm==0.19.0  torch==2.10.0  torchvision==0.25.0  torchaudio==2.10.0
flashinfer-python==0.6.6  flashinfer-cubin==0.6.6
```

The yaml's own note: these are *"exact `==` pins; must install together from the same CUDA build or vLLM
will fail to load"*. Bumping `vllm` alone would break that matching — and rebuilding it is the only way
back to the engine the 91.8% baseline was measured on. Create a separate environment.

### There is no Docker in the AI Sandbox

The vLLM Gemma 4 recipe recommends the pinned `vllm/vllm-openai:gemma4-unified` image, and that is **not
available to us** — the sandbox has no Docker. The nightly pip wheel below is therefore the only route,
with no container fallback if it goes wrong. That raises the stakes on two things: keep `vllm_env2`
pristine (the separate env is now the *only* protection for the working stack), and pin a commit rather
than a moving nightly so a working setup can be rebuilt.

### The route — separate conda env with a nightly wheel

Per the project convention, add a **new env yaml** rather than installing ad hoc. Copy
`conda_envs/vllm_env2.yaml` to `conda_envs/vllm_nightly_env.yaml` and change `name:` to `vllm_nightly`.

**Do not carry the exact pins across.** `torch==2.10.0` and `flashinfer==0.6.6` are matched to vLLM
0.19.0; a nightly will want its own versions and the old pins will conflict. Strip the `vllm`, `torch*`
and `flashinfer*` lines from the copy and let the nightly resolve them:

```bash
conda env create -f conda_envs/vllm_nightly_env.yaml
conda activate vllm_nightly
```

**`uv` is required and is NOT installed on the sandbox.** This is not a style preference — vLLM's docs
state that installing from the nightly index with plain `pip` is *"not supported, because pip combines
packages from `--extra-index-url` and the default index, choosing only the latest version"*, and
`--torch-backend=auto` (which detects the CUDA driver and picks the matching torch wheel) is a uv-only
flag with no pip equivalent. Install uv into the env — an ordinary pip package, no root needed:

```bash
pip install uv

# --torch-backend=auto lets the wheel pick its own matching torch build.
# --python is belt-and-braces: uv is venv-oriented and can refuse to act on a
# conda env, or resolve to the wrong interpreter.
uv pip install -U vllm --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/nightly \
  --python "$CONDA_PREFIX/bin/python"
```

If PyPI is unreachable, the standalone installer is
`curl -LsSf https://astral.sh/uv/install.sh | sh` (installs to `~/.local/bin`).

CUDA compatibility is not expected to be a problem: the current stack already runs **cu129** wheels
(0.19.0's default build), which is also what the nightly defaults to, and per NVIDIA's minor-version
compatibility policy CUDA 12.x is cross-compatible within 12.x on the 13.0-capable driver. Note the env
sets `VLLM_ATTENTION_BACKEND=FLASHINFER`; if the nightly's bundled FlashInfer differs, let it resolve
rather than forcing 0.6.6.

To pin an exact commit instead of a moving nightly — strongly preferred for anything whose result you
want to reproduce or report:

```bash
export VLLM_COMMIT=<sha>
uv pip install vllm --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT} \
  --python "$CONDA_PREFIX/bin/python"
```

**Plain-pip fallback, only if installing `uv` is blocked.** Requires the full wheel URL — package names
do not work against the nightly index — plus installing torch separately from the correct index and
resolving the rest by hand:

```bash
pip install -U https://wheels.vllm.ai/<commit>/vllm-<version>+g<hash>-cp38-abi3-manylinux_2_28_x86_64.whl
```

The filename changes every build, so this is fiddly and hard to reproduce. Prefer `pip install uv`.

### If the nightly will not resolve

With no container fallback, these are the remaining options, cheapest first:

1. **Try an older nightly.** Pick a commit shortly after PR #44429 landed rather than today's `main` —
   less drift from the 0.19.0-era dependency set, and it still has the architecture.
2. **Loosen resolution:** add `--index-strategy unsafe-best-match` so the resolver can mix the nightly
   index with PyPI. Use only if a straight install deadlocks on a dependency conflict.
3. **Stop and report.** If the nightly needs a torch that the driver or the rest of the stack can't
   accommodate, that is a real finding: the 12B is not reachable on this sandbox without infrastructure
   work. The **31B W4A16 runs on the existing engine today** — say so and fall back to it rather than
   burning days on the engine.

Building vLLM from source is possible but is a multi-hour compile with its own CUDA toolchain
requirements; treat it as out of scope for an evaluation.

**Known-good build (verified on the sandbox 2026-07-28):**

```
vllm 0.23.1rc1.dev1458+ge222c33f2
```

That build registers `Gemma4UnifiedForConditionalGeneration`. The short commit is `e222c33f2` — the
`wheels.vllm.ai` index wants the full 40-char SHA, so resolve it against the vLLM repo if you need the
pinned-commit form to rebuild this env.

Since 0.23.x exists, **check whether a stable 0.23.x release already carries this architecture** before
treating a nightly as permanent. A pinned stable release is far better for reproducibility, and is
effectively a precondition for running this model under KFP.

Verify the architecture is registered before going further:

```bash
python -c "import vllm; print(vllm.__version__)"
python -c "from vllm.model_executor.models.registry import ModelRegistry as R; \
print([a for a in R.get_supported_archs() if 'emma' in a])"
```

`Gemma4UnifiedForConditionalGeneration` must appear in that list. If it does not, the wheel predates the
PR and nothing downstream will work.

**A Triton warning during this check is expected on a CPU-only node and can be ignored:**

```
Triton is installed but 0 active driver(s) found (expected 1). Disabling Triton
```

It just means the process saw no GPU driver. The arch check is a pure Python import, so it is valid
regardless — this is exactly why Part 2 can be completed before a GPU allocation arrives.

Before the Part 5 smoke (not before the download), confirm the GPU is actually visible:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

`False 0` means no GPU in this session. Note the engine will get as far as vLLM's init and fail there —
the loader resolves tensor-parallel size from `nvidia-smi` and falls back to 1, so a CPU-node attempt
does not fail on GPU *counting*. Don't read that failure as a problem with the model or the config.

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

**`jq` is NOT installed on the AI Sandbox** (it is present on the remote GPU/cluster box; don't assume it
here). Use python, which is always available inside the conda env:

```bash
cd /home/jovyan/nfs_share/models/gemma-4-12B-it

ls -l model.safetensors    # expect exactly 23919549408 bytes

python -c "import json; c=json.load(open('config.json')); print('arch:', c['architectures'][0]); print('model_type:', c['model_type']); print('num_soft_tokens:', c['vision_config'].get('num_soft_tokens')); print('has vision_soft_tokens_per_image:', 'vision_soft_tokens_per_image' in c)"
```

Expected:

```
arch: Gemma4UnifiedForConditionalGeneration
model_type: gemma4_unified
num_soft_tokens: 280
has vision_soft_tokens_per_image: False
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
| Unknown/unsupported architecture at engine load | vLLM predates PR #44429 — Part 2 not done, or running in `vllm_env2` (0.19.0) instead of the nightly env |
| `ImportError` / undefined symbol after pip install | torch/flashinfer pins carried over from the 0.19.0 yaml — remove them and let the nightly resolve |
| `bash: uv: command not found` | `uv` is not on the sandbox — `pip install uv` into the nightly env first (see Part 2) |
| uv installs into the wrong place, or refuses to run | Pass `--python "$CONDA_PREFIX/bin/python"` so it targets the active conda env |
| Nightly install "succeeds" but vLLM is still 0.19.0 | pip was used against the nightly index — it silently prefers the default index's latest release. Use uv |
| Loads but behaves as if budget is 280 | `hf_overrides` not landing; re-check the Part 3 `jq` output |
| Silently reads only part of the page | `pre_tiling.enabled` still true, or `limit_mm_per_prompt` > 1 |
| `<think>` blocks in output | Thinking is opt-in on this model via a `<\|think\|>` token in the system prompt; check nothing is injecting a system prompt |

## Rollback

Set `bootstrap.model.type: internvl3-vllm`, restore its path, and set
`inference.tiling.pre_tiling.enabled: true`. Nothing in the Gemma work changes InternVL behaviour, and
`vllm_env2` is untouched provided Part 2's warning was respected — which is the whole reason for the
separate nightly env.
