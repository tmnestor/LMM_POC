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

We need the `nvcc` CUDA compiler to build flash-attn from source. The source tarball and generic wheel are already on EFS — the only missing component is `nvcc`.

### What We Need: cuda-nvcc-dev via conda-forge

Our GPU environment does not have `nvcc` at the standard path (`/usr/local/cuda/bin/nvcc`), and the pip package `nvidia-cuda-nvcc-cu12` does not include the `nvcc` binary (only `ptxas`).

The `nvcc` compiler is available via **conda-forge** as `cuda-nvcc-dev_linux-64`, but our GPU environment cannot access conda-forge directly. We need Data Engineering to make this package available through the Artifactory conda mirror.

**Package details:**

| Package | Version | Channel | Build |
|---------|---------|---------|-------|
| `cuda-nvcc-dev_linux-64` | 12.8.93 | conda-forge | `he91c749_0` |
| `cuda-nvcc-dev_linux-64` | 12.8.61 | conda-forge | `he91c749_0` |

Either version is suitable (both match our CUDA 12.8 toolkit). Version 12.8.93 is preferred as it's the latest 12.8.x release.

**Once available in the Artifactory conda mirror**, we can install it ourselves:

```bash
conda activate lmm_poc_env
conda install cuda-nvcc-dev_linux-64=12.8.93
nvcc --version  # verify
```

### Why We Need nvcc

We need to build flash-attn from source because:

1. **The generic wheel (`py3-none-any`) does not work** — it lacks compiled CUDA kernels, producing `ModuleNotFoundError: No module named 'flash_attn_2_cuda'`
2. **The pre-built wheel on GitHub is inaccessible** — policy prevents downloading from GitHub
3. **The pip `nvidia-cuda-nvcc-cu12` package is incomplete** — it installs `ptxas` but not the `nvcc` binary
4. **The source tarball is already on EFS** — `/efs/shared/flash-attn/flash_attn-2.8.3.tar.gz` (8.4 MB), ready to build

Once `nvcc` is available, we can compile flash-attn ourselves (10-30 minute build).

<details>
<summary>Attempted approaches (for reference)</summary>

| Approach | Result |
|----------|--------|
| Generic wheel from PyPI/Artifactory | `ModuleNotFoundError: No module named 'flash_attn_2_cuda'` — no compiled CUDA kernels |
| Pre-built wheel from GitHub releases | Policy prevents Data Engineering from downloading GitHub artifacts |
| `pip install nvidia-cuda-nvcc-cu12==12.8.61` | Installs `ptxas` only, no `nvcc` binary |
| `pip download --no-binary :all:` (source tarball) | Initial failure due to build-isolation; resolved via `.netrc` authenticated download |
| Build from source with `python setup.py bdist_wheel` | `FileNotFoundError: '/usr/local/cuda/bin/nvcc'` — nvcc not installed |

</details>

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

### Build Steps (Once nvcc Is Available)

```bash
# 1. Activate the target conda environment
conda activate lmm_poc_env

# 2. Install nvcc from conda-forge (requires Artifactory conda mirror access)
conda install cuda-nvcc-dev_linux-64=12.8.93

# 3. Verify nvcc is available
nvcc --version
# If not on PATH:
export CUDA_HOME=$CONDA_PREFIX
export PATH="$CUDA_HOME/bin:$PATH"

# 4. Verify the environment matches
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); import sys; print(f'Python: {sys.version}')"
# Expected output:
#   PyTorch: 2.9.1+cu128
#   CUDA: 12.8
#   Python: 3.12.x

# 5. Install build dependencies
pip install ninja packaging wheel setuptools

# 6. Extract the source tarball from shared storage
# cd /efs/shared/flash-attn
# tar xzf flash_attn-2.8.3.tar.gz
# cd flash_attn-2.8.3
cd /efs/shared/flash-attn/flash_attn-2.8.3

# 7. Build from source (FLASH_ATTENTION_FORCE_BUILD is critical)
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS=4    # adjust based on available CPU/memory
python setup.py bdist_wheel

# 8. Verify the output wheel has correct platform tags
ls dist/
# Expected: flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
# MUST NOT be: flash_attn-2.8.3-py3-none-any.whl

# 9. Install the compiled wheel
pip install dist/flash_attn-*-cp312-cp312-linux_x86_64.whl --no-cache-dir
```

### Critical Notes

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
