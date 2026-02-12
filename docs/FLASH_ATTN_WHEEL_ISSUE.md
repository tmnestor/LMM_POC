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

### What We Need from Data Engineering

Our GPU environment does not have proxy forwarding / internet access, so we cannot run `pip download` directly. We need the flash-attn source tarball downloaded to shared storage.

#### Download Command

```bash
# Run this from a machine with internet access:
pip download flash-attn --no-binary :all: --no-deps --no-build-isolation -d /efs/shared/flash-attn/
```

> **Note on `--no-build-isolation`**: Without this flag, pip creates an isolated virtual environment and attempts to download build dependencies (e.g. `setuptools>=40.8.0`) from the package index. On environments where the Artifactory mirror does not carry `setuptools`, this fails with:
> ```
> ERROR: No matching distribution found for setuptools>=40.8.0
> ```
> The `--no-build-isolation` flag tells pip to use the setuptools already installed in the current environment, bypassing this issue. This is safe because we only need to **download** the source tarball — we are not building here.

#### Fallback A: Recover Tarball from pip's Cache

The failed `pip download` attempt already downloaded the tarball successfully before failing at the build-dependency stage. The error output confirms this:

```
Using cached https://artifactory.ctz.atocnet.gov.au/artifactory/api/pypi/sdpaapdl-pypi-rhel9-develop-local/flash-attn/2.8.3/flash_attn-2.8.3.tar.gz (8.4 MB)
```

The 8.4 MB tarball is sitting in pip's local cache. Retrieve it directly:

```bash
# 1. Find pip's cache directory
pip cache dir
# Typical output: /home/jovyan/.cache/pip

# 2. Search for the cached flash-attn tarball
find $(pip cache dir) -name "flash_attn*" -type f

# 3. Copy it to shared storage
cp <path_from_step_2> /efs/shared/flash-attn/flash_attn-2.8.3.tar.gz
```

If `pip cache` commands are not available (older pip), search the cache manually:

```bash
# pip HTTP cache stores files with hashed names — search by size (~8.4 MB)
find ~/.cache/pip -size +8M -size -9M -type f

# Or search all pip cache contents for the tarball
find ~/.cache/pip -name "*.tar.gz" -type f
```

#### Fallback B: Direct Download from Artifactory (with Authentication)

The Artifactory PyPI API endpoint requires authenticated access. Direct `curl`/`wget` returns 403 Forbidden unless credentials are provided.

> **Security note**: Never pass credentials directly on the command line (e.g. `curl -u user:pass`). While HTTPS encrypts the connection, the plaintext password is exposed in shell history (`~/.bash_history`), process listings (`ps aux`), and system audit logs.

**Option 1 — Artifactory API key (preferred):**

```bash
# Store API key in a variable (not in shell history with leading space):
 read -s -p "API Key: " ARTIFACTORY_KEY && echo

curl -L -H "X-JFrog-Art-Api:${ARTIFACTORY_KEY}" \
  -o /efs/shared/flash-attn/flash_attn-2.8.3.tar.gz \
  "https://artifactory.ctz.atocnet.gov.au/artifactory/api/pypi/sdpaapdl-pypi-rhel9-develop-local/flash-attn/2.8.3/flash_attn-2.8.3.tar.gz"

unset ARTIFACTORY_KEY
```

**Option 2 — .netrc file:**

```bash
# Create a .netrc file with restricted permissions
cat > ~/.netrc << 'EOF'
machine artifactory.ctz.atocnet.gov.au
login <username>
password <password_or_api_key>
EOF
chmod 600 ~/.netrc

# curl reads credentials from .netrc automatically
curl -L --netrc \
  -o /efs/shared/flash-attn/flash_attn-2.8.3.tar.gz \
  "https://artifactory.ctz.atocnet.gov.au/artifactory/api/pypi/sdpaapdl-pypi-rhel9-develop-local/flash-attn/2.8.3/flash_attn-2.8.3.tar.gz"

# Clean up afterward
rm ~/.netrc
```

> **Note**: pip authenticates to Artifactory automatically via its index configuration (pip.conf or `--index-url`). Direct `curl`/`wget` does not inherit this — credentials must be provided explicitly. Check with your Artifactory admin for API key access if needed.

#### Verify the Download

Whichever method succeeds, verify the tarball before proceeding to the build:

```bash
# Should be ~8.4 MB
ls -lh /efs/shared/flash-attn/flash_attn-2.8.3.tar.gz

# Verify it's a valid gzip tarball
file /efs/shared/flash-attn/flash_attn-2.8.3.tar.gz
# Expected: gzip compressed data

# Verify contents include source code and CUDA kernels
tar tzf /efs/shared/flash-attn/flash_attn-2.8.3.tar.gz | head -10
# Expected: flash_attn-2.8.3/setup.py, flash_attn-2.8.3/csrc/, etc.
```

Once the source tarball is in `/efs/shared/flash-attn/`, we can complete the build ourselves (Steps 1-7 below).

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

# 4. Extract the source tarball from shared storage
cd /efs/shared/flash-attn
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
