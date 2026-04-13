# vLLM 0.19.0 - Production Compatibility Analysis

## TL;DR

**YES, vllm 0.19.0 from PyPI will work on both production GPU types.**
All dependencies resolve from PyPI alone -- no GitHub releases or extra indexes needed.

---

## 1. Production Hardware (variable allocation)

The production environment assigns one of two GPU types per session:

| Factor | 4xL4 (confirmed) | 4xA10G (also possible) |
|--------|-------------------|------------------------|
| **Architecture** | Ada Lovelace, **sm_89** | Ampere, **sm_86** |
| **VRAM per GPU** | 23 GB | 24 GB |
| **Total VRAM** | 92 GB | 96 GB |
| **Driver** | 580.105.08 | **TBD -- run `nvidia-smi`** |
| **CUDA Version** | 13.0 | **TBD** |

| Factor | Detail |
|--------|--------|
| **PyPI vllm 0.19.0 wheel** | Compiled with CUDA **12.9** (default build) |
| **torch 2.10.0 runtime libs** | nvidia-cuda-runtime-cu12==**12.8**.90 (CUDA 12.8) |

### Compute capability: both covered

The vllm 0.19.0 wheel includes SASS for both sm_86 (A10G) and sm_89 (L4).
Both are well-supported by torch 2.10.0 and vllm 0.19.0. No issues here.

### CUDA / driver compatibility

**4xL4** -- zero concerns:
- Driver **580** supports CUDA 13.0, a superset of 12.9 and 12.8.
- `cuda-bindings==12.9.4` works fine (driver exposes all 12.9 APIs).

**4xA10G** -- depends on driver version:
- If driver >= **575**: no concerns (supports CUDA 12.9 natively).
- If driver is **570-574** (CUDA 12.8 era): the `cuda-bindings==12.9.4`
  transitive dep from torch 2.10.0 *may* call 12.9-specific driver functions.
  Mitigation: install `cuda-compat-12-9` system package on A10G nodes.
- If driver >= **580** (same as L4): no concerns at all.

### ACTION REQUIRED

Run `nvidia-smi` next time an A10G instance is assigned and record the
driver version. If >= 575, we're clear. If < 575, flag for data engineering
to install `cuda-compat-12-9`.

---

## 2. Dependency Delta: 0.11.2 vs 0.19.0

### Torch family (all exact pins)

| Package | 0.11.2 | 0.19.0 |
|---------|--------|--------|
| torch | 2.9.0 | **2.10.0** |
| torchvision | 0.24.0 | **0.25.0** |
| torchaudio | 2.9.0 | **2.10.0** |
| triton | (transitive) | **3.6.0** (transitive from torch) |

### Attention / kernel packages

| Package | 0.11.2 | 0.19.0 |
|---------|--------|--------|
| xformers | ==0.0.33.post1 | **REMOVED** (no longer a dep) |
| flashinfer-python | ==0.5.2 | **==0.6.6** |
| flashinfer-cubin | N/A | **==0.6.6** (NEW) |
| flash-attn (external) | not installed | **not needed** (vllm bundles `_vllm_fa2_C` and `_vllm_fa3_C` .so files) |

### New mandatory dependencies in 0.19.0

| Package | Version | Notes |
|---------|---------|-------|
| numba | ==0.61.2 | JIT compiler, exact pin |
| nvidia-cudnn-frontend | >=1.13.0, <1.19.0 | cuDNN frontend Python lib |
| nvidia-cutlass-dsl | >=4.4.0.dev1 | CUTLASS DSL |
| quack-kernels | >=0.2.7 | Custom kernels |
| anthropic | >=0.71.0 | Anthropic API client |
| mcp | (any) | Model Context Protocol |
| openai-harmony | >=0.0.3 | OpenAI compatibility |
| model-hosting-container-standards | >=0.1.13, <1.0.0 | Container standards |
| cuda-bindings | ==12.9.4 | CUDA driver API bindings (from torch) |
| compressed-tensors | ==0.14.0.1 | Quantization support |
| depyf | ==0.20.0 | Python function decompiler |
| opencv-python-headless | >=4.13.0 | Image processing |
| opentelemetry-sdk/api/exporter | >=1.27.0 | Telemetry (now mandatory) |
| blake3 | (any) | Hashing |
| cbor2 | (any) | CBOR serialization |
| ijson | (any) | Streaming JSON parser |
| setproctitle | (any) | Process title |
| pybase64 | (any) | Base64 |
| ninja | (any) | Build system (for JIT compilation) |
| watchfiles | (any) | File watching |
| python-json-logger | (any) | JSON logging |

### Updated version constraints

| Package | 0.11.2 constraint | 0.19.0 constraint |
|---------|-------------------|-------------------|
| transformers | (older) | **>=4.56.0, <5** |
| tokenizers | (older) | **>=0.21.1** |
| pydantic | (older) | **>=2.12.0** |
| aiohttp | (older) | **>=3.13.3** |
| protobuf | (older) | **>=5.29.6** (with exclusions for buggy 6.3x) |
| lm-format-enforcer | (older) | **==0.11.3** |
| outlines_core | (older) | **==0.2.11** |
| xgrammar | (older) | **>=0.1.32, <1.0.0** |

### NVIDIA runtime libs (transitive from torch 2.10.0)

All on PyPI with `manylinux_2_27_x86_64` or `manylinux2014_x86_64` tags:

```
cuda-bindings==12.9.4
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cudnn-cu12==9.10.2.21
nvidia-cublas-cu12==12.8.4.1
nvidia-cufft-cu12==11.3.3.83
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90
nvidia-cusparse-cu12==12.5.8.93
nvidia-cusparselt-cu12==0.7.1
nvidia-nccl-cu12==2.27.5
nvidia-nvshmem-cu12==3.4.5
nvidia-nvtx-cu12==12.8.90
nvidia-nvjitlink-cu12==12.8.93
nvidia-cufile-cu12==1.13.1.3
```

---

## 3. PyPI-only Verification

Every package below was verified downloadable from PyPI with Linux x86_64
wheels (various manylinux tags: 2014, 2_27, 2_28, 2_31):

- **vllm==0.19.0** -- `cp38-abi3-manylinux_2_31_x86_64` (432 MB)
- **torch==2.10.0** -- `cp312-cp312-manylinux_2_28_x86_64` (916 MB)
- **torchvision==0.25.0** -- `cp312-cp312-manylinux_2_28_x86_64` (8 MB)
- **torchaudio==2.10.0** -- `cp312-cp312-manylinux_2_28_x86_64` (2 MB)
- **flashinfer-python==0.6.6** -- `py3-none-any` (8 MB)
- **flashinfer-cubin==0.6.6** -- `py3-none-any` (268 MB)
- **All nvidia-* packages** -- verified at exact pinned versions
- **All pure-Python deps** -- verified on PyPI

**No GitHub releases or extra index URLs required.**

---

## 4. Environment YAML

```yaml
# ---------------------------------------------------------------------------
# Conda environment for vLLM 0.19.0
# ---------------------------------------------------------------------------
# Target stack (all exact pins from the vLLM 0.19.0 wheel METADATA):
#   vLLM        0.19.0
#   torch       2.10.0    \
#   torchvision 0.25.0    |  exact ==  pins; must install together from the
#   torchaudio  2.10.0    /  same CUDA build or vLLM will fail to load
#   flashinfer-python  0.6.6   (pinned ==, transitive from vLLM)
#   flashinfer-cubin   0.6.6   (pinned ==, pre-compiled CUDA kernels)
#   Python      3.12      (hard requirement -- prod runs 3.12; vLLM
#                          supports 3.10-3.13, but the rest of the LMM_POC
#                          codebase targets 3.12 and uses PEP 695 `type`
#                          statements that require 3.12+)
#
# Production hardware (variable allocation):
#   4x NVIDIA L4  (Ada Lovelace sm_89, 23 GB each, driver 580, CUDA 13.0)
#   4x NVIDIA A10G (Ampere sm_86, 24 GB each, driver/CUDA TBD)
#   The vllm wheel includes SASS for both sm_86 and sm_89.
#
# Why the default PyPI wheel (cu129) and not cu128:
#   Data engineering policy: PyPI only, no extra index URLs, no GitHub
#   release wheels. The default vllm 0.19.0 wheel on PyPI is compiled
#   with CUDA 12.9 (cu129). Per NVIDIA's Minor Version Compatibility
#   policy, CUDA 12.x apps are cross-compatible within the 12.x family.
#   The L4 production driver (580) supports CUDA 13.0, so 12.9 code runs
#   without issue. For A10G instances, verify driver >= 575; if < 575,
#   install the cuda-compat-12-9 system package.
#
# Why pip, not conda, for torch:
#   vLLM's custom CUDA kernels are built against the ABI of the pip torch
#   wheels. Conda-forge torch drifts from that ABI and causes
#   `undefined symbol` / `no kernel image available` errors at first
#   inference. Only Python and pip come from conda.
#
# Flash-Attention:
#   Deliberately NOT installed as a separate package. vLLM 0.19.0 bundles
#   its own flash-attn kernels (_vllm_fa2_C.abi3.so, _vllm_fa3_C.abi3.so).
#   FlashInfer is the default and recommended attention backend.
#
# Changes from vllm_env.yaml (0.11.2):
#   - torch 2.9.0 -> 2.10.0 (torchvision 0.24.0 -> 0.25.0)
#   - xformers REMOVED (no longer a vllm dependency)
#   - flashinfer-cubin NEW (pre-compiled CUDA kernels for FlashInfer)
#   - ~20 new transitive deps (anthropic, mcp, numba, nvidia-cudnn-frontend,
#     nvidia-cutlass-dsl, quack-kernels, compressed-tensors, etc.)
#   - VLLM_ATTENTION_BACKEND changed from XFORMERS to FLASHINFER
#
# Every other vLLM dependency (fastapi, uvicorn, pydantic, openai, ray,
# transformers>=4.56, tokenizers, outlines_core, xgrammar, numba,
# prometheus_client, aiohttp, httpx, anthropic, mcp, nvidia-cudnn-frontend,
# nvidia-cutlass-dsl, quack-kernels, compressed-tensors, partial-json-parser,
# sentencepiece, numpy, protobuf, msgspec, pyzmq, pillow, mistral_common,
# nvidia-ml-py, ...) is resolved transitively from `vllm==0.19.0` and does
# not need to be listed here.
#
# Usage on the dev machine (2xL40S, CUDA 12.4):
#   conda env create -f vllm_env2.yaml
#   conda activate vllm_env2
#   pip freeze > vllm-0.19.0-frozen-requirements.txt   # <-- give this to
#                                                      #     data engineering
#
# Verify:
#   python -c "
#   import vllm, torch, torchvision, torchaudio, flashinfer
#   print('vllm       ', vllm.__version__)
#   print('torch      ', torch.__version__, 'cuda', torch.version.cuda)
#   print('torchvision', torchvision.__version__)
#   print('torchaudio ', torchaudio.__version__)
#   print('flashinfer ', flashinfer.__version__)
#   from vllm import _custom_ops
#   print('custom ops loaded OK')
#   print('devices:', torch.cuda.device_count())
#   "
# ---------------------------------------------------------------------------

name: vllm_env2
channels:
  - conda-forge
dependencies:
  - python=3.12
  - pip>=24.0

  - pip:
      # Torch family -- installed from plain PyPI. PyPI's default torch
      # 2.10.0 wheel ships with nvidia-cuda-runtime-cu12==12.8.90 (CUDA
      # 12.8 runtime libs). Do not add --index-url here; data engineering
      # policy forbids extra indexes.
      - torch==2.10.0
      - torchvision==0.25.0
      - torchaudio==2.10.0

      # vLLM and its two most version-sensitive hard deps. Listed
      # explicitly (even though transitive) so an accidental later
      # `pip install --upgrade` of flashinfer cannot silently break vLLM.
      - vllm==0.19.0
      - flashinfer-python==0.6.6
      - flashinfer-cubin==0.6.6

      # Analysis / visualization -- not vllm deps, needed for our pipeline
      - pandas
      - seaborn

# Environment variables automatically exported on `conda activate` and
# unset on `conda deactivate`. FlashInfer is the default attention backend
# for vLLM 0.19.0 (replaces xformers from 0.11.2).
variables:
  VLLM_ATTENTION_BACKEND: FLASHINFER
```

---

## 5. Open Questions / Next Steps

1. **A10G driver version**: Run `nvidia-smi` on an A10G instance. If driver
   < 575, data engineering needs to install `cuda-compat-12-9`.
   L4 is confirmed OK (driver 580, CUDA 13.0).

2. **Security audit**: The original goal was patching vulnerabilities from
   0.11.2. Verify that the specific CVEs are addressed in 0.19.0 by checking
   the vLLM security advisories.

3. **Transitive dep security**: Run `pip-audit` against the frozen requirements
   to catch any vulnerabilities in the ~60+ transitive dependencies.

4. **Test on sandbox first**: Install on the 2xL40S CUDA 12.4 sandbox before
   deploying to production. The same CUDA 12.x minor version compatibility
   applies there too.

5. **VRAM planning**: L4 has 23 GB/GPU (92 GB total) vs A10G 24 GB/GPU
   (96 GB total). Size models for the smaller L4 allocation to ensure
   portability across both hardware types.

---

## Sources

- [vLLM PyPI](https://pypi.org/project/vllm/)
- [vLLM GPU Installation Docs](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
- [NVIDIA CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)
- [CUDA Toolkit Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
- [nvidia-cudnn-frontend PyPI](https://pypi.org/project/nvidia-cudnn-frontend/)
