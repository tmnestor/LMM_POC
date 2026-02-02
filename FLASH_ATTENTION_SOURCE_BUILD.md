# Flash Attention 2 Source Build Guide

## Problem

Flash Attention 2 prebuilt wheels (and standard `pip install flash-attn --no-build-isolation`) fail with an ABI mismatch on JupyterHub/containerized environments:

```
ImportError: flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so: undefined symbol:
_ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
```

### Root Cause

Two independent issues combine to cause this:

1. **Prebuilt wheel download**: flash-attn's `setup.py` downloads a prebuilt wheel from GitHub releases instead of compiling from source, even when you run `pip install flash-attn --no-build-isolation`. This is controlled by `FLASH_ATTENTION_FORCE_BUILD` (defaults to `FALSE`).

2. **CXX11 ABI mismatch**: PyTorch pip wheels are compiled with `_GLIBCXX_USE_CXX11_ABI=0` (old ABI), but the system's `g++` defaults to ABI=1 (new ABI via `/usr/include/c++/.../bits/c++config.h`). Flash-attn's `setup.py` does **not** pass `-D_GLIBCXX_USE_CXX11_ABI=0` to its compiler flags, and PyTorch's `BuildExtension` fails to inject it automatically.

### Affected Environments

- JupyterHub containers with NVIDIA GPUs
- Any Linux environment where `g++` defaults to CXX11 ABI=1 and PyTorch uses ABI=0
- Confirmed on: Python 3.12, PyTorch 2.6.0+cu124, CUDA 12.4, NVIDIA L4 GPUs

## Prerequisites

Before starting, verify your environment:

```bash
# Check PyTorch version and ABI
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA (PyTorch): {torch.version.cuda}')
print(f'CXX11 ABI: {torch._C._GLIBCXX_USE_CXX11_ABI}')
print(f'Device: {torch.cuda.get_device_name(0)}')
"

# Check nvcc is available (required for source build)
which nvcc && nvcc --version

# Check system ABI default
echo '#include <bits/c++config.h>' | g++ -dM -E -x c++ - | grep GLIBCXX_USE_CXX11_ABI
```

**Required outputs:**
- `CXX11 ABI: False` (PyTorch uses old ABI)
- `nvcc` must be available (e.g., `/usr/local/cuda-12.4/bin/nvcc`)
- System header shows `#define _GLIBCXX_USE_CXX11_ABI 1` (confirms mismatch)

If PyTorch reports `CXX11 ABI: True`, you do not need this guide -- standard `pip install flash-attn --no-build-isolation` should work.

## Build Steps

### Step 1: Uninstall existing flash-attn

```bash
pip uninstall flash-attn -y
```

### Step 2: Install ninja (build accelerator)

```bash
pip install ninja
```

### Step 3: Download and extract source

```bash
cd /tmp && rm -rf fa_build && mkdir fa_build && cd fa_build
curl -L -o flash_attn-2.8.3.tar.gz "https://pypi.io/packages/source/f/flash-attn/flash_attn-2.8.3.tar.gz"
tar xzf flash_attn-2.8.3.tar.gz
cd flash_attn-2.8.3
```

### Step 4: Patch setup.py with ABI flag

Two patches are needed -- one for C++ compilation and one for CUDA (nvcc) compilation:

**Patch 1 -- C++ compiler flags (line ~220):**
```bash
sed -i 's/compiler_c17_flag=\["-O3", "-std=c++17"\]/compiler_c17_flag=["-O3", "-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=0"]/' setup.py
```

**Patch 2 -- NVCC compiler flags (line ~199):**
```bash
sed -i 's/nvcc_flags = \[/nvcc_flags = [\n    "-Xcompiler", "-D_GLIBCXX_USE_CXX11_ABI=0",/' setup.py
```

**Verify both patches:**
```bash
grep "GLIBCXX_USE_CXX11_ABI=0" setup.py
```

Expected output (2 lines):
```
    "-Xcompiler", "-D_GLIBCXX_USE_CXX11_ABI=0",
    compiler_c17_flag=["-O3", "-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=0"]
```

### Step 5: Build from source

The `FLASH_ATTENTION_FORCE_BUILD=TRUE` environment variable is critical -- without it, `setup.py` downloads a prebuilt wheel from GitHub instead of compiling.

```bash
FLASH_ATTENTION_FORCE_BUILD=TRUE python setup.py bdist_wheel
```

This compiles all CUDA kernels and takes several minutes. You will see individual `.cu` file compilations scrolling by.

### Step 6: Install the built wheel

```bash
pip install dist/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl --no-cache-dir
```

### Step 7: Verify

```bash
python -c "import flash_attn; print(f'flash-attn: {flash_attn.__version__}'); print('SUCCESS')"
```

Expected output:
```
flash-attn: 2.8.3
SUCCESS
```

### Step 8: Re-register Jupyter kernel (if applicable)

```bash
python -m ipykernel install --user --name LMM_POC_IVL3.5 --display-name "Python (LMM_POC_IVL3.5)"
```

Then restart the kernel and Run All on the notebook.

## Quick Reference (All Commands)

For copy-paste convenience, the complete sequence:

```bash
# Uninstall and prepare
pip uninstall flash-attn -y
pip install ninja

# Download source
cd /tmp && rm -rf fa_build && mkdir fa_build && cd fa_build
curl -L -o flash_attn-2.8.3.tar.gz "https://pypi.io/packages/source/f/flash-attn/flash_attn-2.8.3.tar.gz"
tar xzf flash_attn-2.8.3.tar.gz
cd flash_attn-2.8.3

# Patch ABI flags
sed -i 's/compiler_c17_flag=\["-O3", "-std=c++17"\]/compiler_c17_flag=["-O3", "-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=0"]/' setup.py
sed -i 's/nvcc_flags = \[/nvcc_flags = [\n    "-Xcompiler", "-D_GLIBCXX_USE_CXX11_ABI=0",/' setup.py

# Verify patches
grep "GLIBCXX_USE_CXX11_ABI=0" setup.py

# Build from source (takes several minutes)
FLASH_ATTENTION_FORCE_BUILD=TRUE python setup.py bdist_wheel

# Install built wheel
pip install dist/flash_attn-*.whl --no-cache-dir

# Verify
python -c "import flash_attn; print(f'flash-attn: {flash_attn.__version__}'); print('SUCCESS')"

# Re-register kernel
python -m ipykernel install --user --name LMM_POC_IVL3.5 --display-name "Python (LMM_POC_IVL3.5)"
```

## Troubleshooting

### `nvcc` not found

The CUDA toolkit must be installed. Check:
```bash
ls /usr/local/cuda*/bin/nvcc
```

If present but not on PATH:
```bash
export PATH=/usr/local/cuda-12.4/bin:$PATH
```

If not installed and you cannot install it (no sudo), you cannot build from source. In this case, try finding the exact matching prebuilt wheel from [flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases).

### Build fails with compiler errors

Ensure the CUDA toolkit version matches PyTorch's CUDA version:
```bash
nvcc --version          # System CUDA
python -c "import torch; print(torch.version.cuda)"  # PyTorch CUDA
```

These should match (e.g., both CUDA 12.4).

### Same `undefined symbol` error after build

Verify the installed `.so` file uses the correct ABI:
```bash
nm -D $(python -c "import flash_attn_2_cuda; print(flash_attn_2_cuda.__file__)") | grep -c "__cxx11"
```

If the count is high, the ABI patch was not applied. Re-check `grep "GLIBCXX_USE_CXX11_ABI=0" setup.py` before building.

### `/tmp` cleared between steps

If your environment clears `/tmp` on disconnect (e.g., JupyterHub containers), consider using a persistent directory instead:
```bash
mkdir -p ~/nfs_share/flash_attn_build
cd ~/nfs_share/flash_attn_build
# ... then follow from Step 3
```

## Version Compatibility Matrix

| flash-attn | PyTorch | CUDA | Python | Status |
|-----------|---------|------|--------|--------|
| 2.8.3 | 2.6.0+cu124 | 12.4 | 3.12 | Confirmed working with ABI patch |

## Why Standard Install Fails

The standard installation path `pip install flash-attn --no-build-isolation` fails because:

1. flash-attn's `setup.py` calls `get_wheel_url()` which constructs a URL to a prebuilt wheel on GitHub releases
2. It downloads this prebuilt wheel instead of compiling from source
3. The prebuilt wheel, despite being labeled `cxx11abiFALSE`, has binary incompatibilities with the specific PyTorch build
4. Even when forced to build from source, `setup.py` does not pass `-D_GLIBCXX_USE_CXX11_ABI=0` to the compiler, despite PyTorch's `BuildExtension` supposedly handling this automatically
5. The system's `g++` defaults to `_GLIBCXX_USE_CXX11_ABI=1`, so without an explicit override, the compiled binary uses the wrong ABI
