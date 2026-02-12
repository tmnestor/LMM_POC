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

### Pre-built Wheel from GitHub Releases

The flash-attn project publishes ~200 prebuilt wheel variants ([Dao-AILab/flash-attention Releases](https://github.com/Dao-AILab/flash-attention/releases)) covering common combinations. A matching pre-built wheel **does exist** for our environment:

```
flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl  (253 MB)
```

| Tag | Value | Meaning |
|-----|-------|---------|
| `cu12` | CUDA 12 | Compatible with our CUDA 12.8 toolkit |
| `torch2.9` | PyTorch 2.9 | Matches our PyTorch 2.9.1 |
| `cxx11abiTRUE` | CXX11 ABI enabled | Must match PyTorch's ABI setting (see verification below) |
| `cp312` | CPython 3.12 | Matches our Python 3.12 |
| `linux_x86_64` | Platform | Correct architecture |

**CXX11 ABI verification** — the pre-built wheel requires `cxx11abiTRUE`. Confirmed on our environment (2026-02-12):
```bash
python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
# Output: True  ✓
```

> **This pre-built wheel is the recommended approach** — it does not require `nvcc` or the CUDA toolkit to be installed, as the CUDA kernels are already compiled.

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

Our GPU environment does not have internet access. We need Data Engineering to download a file from GitHub and place it on shared storage.

### Recommended: Pre-built Wheel (No nvcc Required)

A pre-built wheel matching our environment is available on GitHub. This is the preferred approach — it does not require `nvcc` or the CUDA toolkit, as the CUDA kernels are already compiled.

**Download URL:**

```
https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

**Download command (from a machine with internet access):**

```bash
# Download the pre-built wheel (~253 MB)
wget -O /efs/shared/flash-attn/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
```

**Verify the download:**

```bash
# Should be ~253 MB
ls -lh /efs/shared/flash-attn/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# Verify it's a valid zip (wheels are zip files)
file /efs/shared/flash-attn/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
# Expected: Zip archive data
```

**Pre-requisite check** — CXX11 ABI compatibility has been verified (2026-02-12):

```bash
conda activate lmm_poc_env
python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
# Output: True  ✓  — pre-built wheel is compatible
```

---

### Fallback: Build from Source (Requires nvcc)

If the CXX11 ABI check prints `False`, or if the pre-built wheel fails at import time, a from-source build is required. This requires `nvcc` (the CUDA compiler) to be installed on the build machine.

> **Note**: Our current GPU environment does **not** have `nvcc` installed (`/usr/local/cuda/bin/nvcc` is missing). Building from source requires either installing the CUDA toolkit (`conda install -c nvidia cuda-nvcc cuda-toolkit`) or running on a machine that has it.

#### Source Tarball

The source tarball has already been downloaded to shared storage:

```
/efs/shared/flash-attn/flash_attn-2.8.3.tar.gz  (8.4 MB)
```

<details>
<summary>How the source tarball was obtained (resolved issues)</summary>

The initial `pip download --no-binary :all:` command failed because pip's PEP 517 build isolation tried to download `setuptools>=40.8.0` from the Artifactory mirror, which doesn't carry it. Adding `--no-build-isolation` would fix this:

```bash
pip download flash-attn --no-binary :all: --no-deps --no-build-isolation -d /efs/shared/flash-attn/
```

Direct `curl`/`wget` to the Artifactory API endpoint returned 403 Forbidden (requires authenticated access). The working approach used a `.netrc` file for credential-based download:

```bash
cat > ~/.netrc << 'EOF'
machine artifactory.ctz.atocnet.gov.au
login <username>
password <password_or_api_key>
EOF
chmod 600 ~/.netrc

curl -L --netrc \
  -o /efs/shared/flash-attn/flash_attn-2.8.3.tar.gz \
  "https://artifactory.ctz.atocnet.gov.au/artifactory/api/pypi/sdpaapdl-pypi-rhel9-develop-local/flash-attn/2.8.3/flash_attn-2.8.3.tar.gz"

rm ~/.netrc
```

> **Security note**: Never pass credentials directly on the command line (e.g. `curl -u user:pass`). The plaintext password is exposed in shell history, process listings (`ps aux`), and audit logs. Use `.netrc` with `chmod 600` or environment variables via `read -s` instead.

</details>

#### Build Steps

```bash
# 1. Activate the target conda environment (must have Python 3.12 + PyTorch 2.9.1+cu128)
conda activate lmm_poc_env

# 2. Verify the environment matches
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); import sys; print(f'Python: {sys.version}')"
# Expected output:
#   PyTorch: 2.9.1+cu128
#   CUDA: 12.8
#   Python: 3.12.x

# 3. Ensure nvcc is available
nvcc --version
# If missing: conda install -c nvidia cuda-nvcc cuda-toolkit

# 4. Install build dependencies
pip install ninja packaging wheel setuptools

# 5. Extract the source tarball from shared storage
cd /efs/shared/flash-attn
tar xzf flash_attn-2.8.3.tar.gz
cd flash_attn-2.8.3

# 6. Build from source (FLASH_ATTENTION_FORCE_BUILD is critical)
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS=4    # adjust based on available CPU/memory
python setup.py bdist_wheel

# 7. Verify the output wheel has correct platform tags
ls dist/
# Expected: flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
# MUST NOT be: flash_attn-2.8.3-py3-none-any.whl

# 8. Copy to shared storage
cp dist/flash_attn-*-cp312-cp312-linux_x86_64.whl /efs/shared/flash-attn/
```

#### Critical Notes

- **`FLASH_ATTENTION_FORCE_BUILD=TRUE` must be set** — without it, `setup.py` downloads the pre-built wheel from GitHub instead of compiling from source.
- The build takes **10-30 minutes** as it compiles CUDA kernels.
- The output wheel in `dist/` **must** have platform-specific tags (`cp312-cp312-linux_x86_64`), not `py3-none-any`.
- The build requires `nvcc` — the CUDA compiler from the CUDA toolkit.

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
