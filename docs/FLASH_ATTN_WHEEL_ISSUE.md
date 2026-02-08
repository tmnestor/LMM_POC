# Flash Attention Wheel Issue - Request to Data Engineering

## Current Environment

```
PyTorch: 2.9.1+cu128
CUDA available: True
CUDA version: 12.8
GPU: NVIDIA L4
Python: 3.12
```

---

## Why the Generic Wheel Will Not Work

Flash Attention is a compiled CUDA extension — not pure Python. Its wheel contains pre-compiled C++ and CUDA kernels that are **binary-coupled** to three specific components of the runtime environment:

| Component | Required value | Why it must match |
|-----------|---------------|-------------------|
| **Python version** | 3.12 (`cp312`) | The compiled `.so` files link against a specific CPython ABI. A wheel built for Python 3.11 cannot load in Python 3.12. |
| **PyTorch version** | 2.9.1 (`torch291`) | Flash Attention calls PyTorch's internal C++ API (`libtorch`). These internal symbols change between PyTorch releases — a wheel compiled against PyTorch 2.6 will produce `undefined symbol` errors on PyTorch 2.9. |
| **CUDA version** | 12.8 (`cu128`) | The CUDA kernels are compiled for specific CUDA toolkit versions. Mismatched versions produce load failures or silent correctness issues. |

A mismatch in **any one** of these produces `ImportError` at import time.

### The Generic Wheel Problem

The generic wheel available on PyPI has platform tags `py3-none-any`:

```
flash_attn-X.X.X-py3-none-any.whl   <-- WILL NOT WORK
```

This wheel **does not contain compiled CUDA kernels**. It is only the Python wrapper code. Installing it produces:

```
ModuleNotFoundError: No module named 'flash_attn_2_cuda'
```

### Pre-built Wheels from GitHub Releases

The flash-attn project publishes ~200 prebuilt wheel variants ([Dao-AILab/flash-attention Releases](https://github.com/Dao-AILab/flash-attention/releases)) covering common combinations. However, our specific combination of **Python 3.12 + PyTorch 2.9.1 + CUDA 12.8** may not have a matching pre-built wheel. Even when a pre-built wheel exists, binary regressions have been reported (see References).

### What a Correct Wheel Looks Like

A correctly compiled wheel has **platform-specific tags** matching our environment:

```
flash_attn-X.X.X-cp312-cp312-linux_x86_64.whl   <-- CORRECT
```

Where:
- `cp312` = compiled against CPython 3.12 ABI
- `linux_x86_64` = compiled for this specific platform

---

## Request to Data Engineering

We need a flash-attn wheel compiled from source against our exact environment. The build must run on a machine with an NVIDIA GPU and CUDA 12.8 toolkit installed.

### Build Environment Requirements

| Requirement | Value |
|------------|-------|
| Python | 3.12 |
| PyTorch | 2.9.1+cu128 |
| CUDA Toolkit | 12.8 |
| GPU | Any NVIDIA GPU (L4, A10G, etc.) |
| OS | Linux x86_64 |

### Build Steps

```bash
# 1. Activate the target conda environment (must have Python 3.12 + PyTorch 2.9.1+cu128)
conda activate lmm_poc_env

# 2. Verify the environment matches
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); import sys; print(f'Python: {sys.version}')"
# Expected output:
#   PyTorch: 2.9.1+cu128
#   CUDA: 12.8
#   Python: 3.12.x

# 3. Install build dependencies
pip install ninja packaging wheel setuptools

# 4. Clone or download flash-attn source
pip download flash-attn --no-binary :all: --no-deps -d /tmp/flash-attn-src
cd /tmp/flash-attn-src
tar xzf flash_attn-*.tar.gz
cd flash_attn-*

# 5. Build from source (FLASH_ATTENTION_FORCE_BUILD is critical)
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS=4    # adjust based on available CPU/memory
python setup.py bdist_wheel

# 6. Verify the output wheel has correct platform tags
ls dist/
# Expected: flash_attn-X.X.X-cp312-cp312-linux_x86_64.whl
# MUST NOT be: flash_attn-X.X.X-py3-none-any.whl

# 7. Copy to shared storage
cp dist/flash_attn-*-cp312-cp312-linux_x86_64.whl /efs/shared/flash-attn/
```

### Critical Notes

- **`FLASH_ATTENTION_FORCE_BUILD=TRUE` must be set** — without it, `setup.py` downloads the pre-built wheel from GitHub instead of compiling from source.
- The build takes **10-30 minutes** as it compiles CUDA kernels.
- The output wheel in `dist/` **must** have platform-specific tags (`cp312-cp312-linux_x86_64`), not `py3-none-any`.
- The build **must** run on a machine with a GPU — the CUDA compiler needs GPU headers.

---

## Verification After Install

Once the correct wheel is available:

```bash
# Install the wheel
pip install /efs/shared/flash-attn/flash_attn-*-cp312-cp312-linux_x86_64.whl --no-cache-dir

# Verify it loads correctly
python -c "
import flash_attn
print(f'flash-attn: {flash_attn.__version__}')
from flash_attn import flash_attn_func
print('CUDA kernels loaded successfully')
print('SUCCESS')
"
```

---

## References

- [Dao-AILab/flash-attention Releases](https://github.com/Dao-AILab/flash-attention/releases) — Pre-built wheel variants
- [flash-attn Issue #1783](https://github.com/Dao-AILab/flash-attention/issues/1783) — Binary compatibility failure with PyTorch version mismatches
- [flash-attn Issue #1644](https://github.com/Dao-AILab/flash-attention/issues/1644) — PyTorch ABI changes breaking existing flash-attn wheels
- [flash-attn Issue #1717](https://github.com/Dao-AILab/flash-attention/issues/1717) — Undefined symbol errors in flash-attn wheel builds
- [PyTorch Issue #51039](https://github.com/pytorch/pytorch/issues/51039) — PyTorch's `_GLIBCXX_USE_CXX11_ABI=0` wheel policy
- [flash-attn Installation & Setup (DeepWiki)](https://deepwiki.com/Dao-AILab/flash-attention/1.1-installation-and-setup) — Wheel selection process and ABI compatibility details
