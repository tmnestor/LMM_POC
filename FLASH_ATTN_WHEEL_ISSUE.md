# Flash Attention Wheel Issue - Action Required

## Background: Why Flash Attention Must Match Your Exact Environment

Flash Attention is a compiled CUDA extension — not pure Python. Its wheel contains pre-compiled C++ and CUDA kernels that are **binary-coupled** to three specific components of the runtime environment:

| Component | Why it must match |
|-----------|-------------------|
| **Python version** | The compiled `.so` files link against a specific CPython ABI (e.g. `cp311`). A wheel built for Python 3.12 cannot load in Python 3.11. |
| **PyTorch version** | Flash Attention calls PyTorch's internal C++ API (`libtorch`). These internal symbols change between PyTorch releases, so a wheel compiled against PyTorch 2.5 will produce `undefined symbol` errors on PyTorch 2.6. |
| **CUDA version** | The CUDA kernels are compiled for specific CUDA toolkit versions. Mismatched CUDA runtime/toolkit versions produce load failures or silent correctness issues. |

A mismatch in **any one** of these produces `ImportError: undefined symbol` at import time.

### The C++11 ABI Problem

On top of version matching, there is an **ABI (Application Binary Interface) flag** issue:

- **PyTorch pip wheels** are compiled with the legacy C++ ABI: `_GLIBCXX_USE_CXX11_ABI=0`
- **Linux system compilers** (`g++`) default to the new ABI: `_GLIBCXX_USE_CXX11_ABI=1`

When flash-attn is compiled from source without explicitly setting `ABI=0`, the resulting `.so` files use the new ABI but try to link against PyTorch's old-ABI symbols — producing the `undefined symbol` error. This is specific to our **Python 3.11** environment; Python 3.12 does not exhibit this issue.

The flash-attn project publishes ~200 prebuilt wheel variants to cover different combinations ([Dao-AILab/flash-attention Releases](https://github.com/Dao-AILab/flash-attention/releases)), but when no matching wheel exists — or the matching wheel itself has a binary regression — a **patched source build** is required.

### References

- [flash-attn Issue #1783](https://github.com/Dao-AILab/flash-attention/issues/1783) — Binary compatibility failure with flash_attn 2.8.2 + PyTorch 2.6.0+cu124 on Python 3.11 (our exact scenario)
- [flash-attn Issue #1644](https://github.com/Dao-AILab/flash-attention/issues/1644) — PyTorch ABI changes breaking existing flash-attn wheels
- [flash-attn Issue #1717](https://github.com/Dao-AILab/flash-attention/issues/1717) — Undefined symbol errors in flash-attn 2.8 wheel builds
- [PyTorch Issue #51039](https://github.com/pytorch/pytorch/issues/51039) — Long-standing discussion on PyTorch's `_GLIBCXX_USE_CXX11_ABI=0` wheel policy
- [flash-attn Installation & Setup (DeepWiki)](https://deepwiki.com/Dao-AILab/flash-attention/1.1-installation-and-setup) — Wheel selection process and ABI compatibility details

---

## Current State

The wheel at `/efs/shared/flash-attn/` is **not a source-compiled build**:

```
flash_attn-2.8.3-py3-none-any.whl   <-- prebuilt (WRONG)
flash_attn-2.8.3.tar.gz             <-- source tarball (unused)
```

## The Problem

The wheel filename `flash_attn-2.8.3-py3-none-any.whl` has platform tags `py3-none-any`, which indicates it is the **prebuilt wheel downloaded from GitHub/PyPI** — not a wheel compiled from source with the ABI patch.

Installing this wheel will produce the same `undefined symbol` ABI mismatch error:

```
ImportError: flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so: undefined symbol:
_ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
```

**Note:** This ABI mismatch is specific to Python 3.11. Python 3.12 does not have this issue.

### How to verify

```bash
unzip -p /efs/shared/flash-attn/flash_attn-2.8.3-py3-none-any.whl \
  flash_attn-2.8.3.dist-info/WHEEL
```

This will show `Tag: py3-none-any`, confirming it is not a source build.

### What a correct source build looks like

A correctly compiled wheel has **platform-specific tags**:

```
flash_attn-2.8.3-cp311-cp311-linux_x86_64.whl   <-- source-built (CORRECT)
```

Where:
- `cp311` = compiled against CPython 3.11 ABI
- `linux_x86_64` = compiled for this specific platform

## Required Action

The source tarball (`flash_attn-2.8.3.tar.gz`) is already in place. Steps 4 and 5 from the [build guide](FLASH_ATTENTION_SOURCE_BUILD.md) need to be completed:

```bash
# Extract the source tarball
cd /efs/shared/flash-attn
tar xzf flash_attn-2.8.3.tar.gz
cd flash_attn-2.8.3

# Step 4: Patch setup.py with ABI=0 flag (two patches)
sed -i 's/compiler_c17_flag=\["-O3", "-std=c++17"\]/compiler_c17_flag=["-O3", "-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=0"]/' setup.py
sed -i 's/nvcc_flags = \[/nvcc_flags = [\n    "-Xcompiler", "-D_GLIBCXX_USE_CXX11_ABI=0",/' setup.py

# Verify BOTH patches applied (should show 2 matches)
grep -c "GLIBCXX_USE_CXX11_ABI=0" setup.py

# Step 5: Build from source (FLASH_ATTENTION_FORCE_BUILD is critical)
export FLASH_ATTENTION_FORCE_BUILD=TRUE
python setup.py bdist_wheel

# The correctly compiled wheel will be in dist/
ls dist/
# Expected: flash_attn-2.8.3-cp311-cp311-linux_x86_64.whl
```

### Important

- `FLASH_ATTENTION_FORCE_BUILD=TRUE` **must** be set — without it, `setup.py` downloads the prebuilt wheel from GitHub instead of compiling.
- The `grep` check must show **2 matches** (one for C++ flags, one for NVCC flags).
- The build takes several minutes as it compiles CUDA kernels.
- The output wheel in `dist/` **must** have platform-specific tags (`cp3xx-cp3xx-linux_x86_64`), not `py3-none-any`.

## After the Build

Once the correct wheel is in `/efs/shared/flash-attn/`, we can install with:

```bash
pip install /efs/shared/flash-attn/flash_attn-2.8.3-cp311-cp311-linux_x86_64.whl --no-cache-dir
python -c "import flash_attn; print(f'flash-attn: {flash_attn.__version__}'); print('SUCCESS')"
```
