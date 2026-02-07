# Flash Attention Wheel Issue - Action Required

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
ImportError: flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so: undefined symbol:
_ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
```

### How to verify

```bash
unzip -p /efs/shared/flash-attn/flash_attn-2.8.3-py3-none-any.whl \
  flash_attn-2.8.3.dist-info/WHEEL
```

This will show `Tag: py3-none-any`, confirming it is not a source build.

### What a correct source build looks like

A correctly compiled wheel has **platform-specific tags**:

```
flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl   <-- source-built (CORRECT)
```

Where:
- `cp312` = compiled against CPython 3.12 ABI
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
# Expected: flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
```

### Important

- `FLASH_ATTENTION_FORCE_BUILD=TRUE` **must** be set — without it, `setup.py` downloads the prebuilt wheel from GitHub instead of compiling.
- The `grep` check must show **2 matches** (one for C++ flags, one for NVCC flags).
- The build takes several minutes as it compiles CUDA kernels.
- The output wheel in `dist/` **must** have platform-specific tags (`cp3xx-cp3xx-linux_x86_64`), not `py3-none-any`.

## After the Build

Once the correct wheel is in `/efs/shared/flash-attn/`, we can install with:

```bash
pip install /efs/shared/flash-attn/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl --no-cache-dir
python -c "import flash_attn; print(f'flash-attn: {flash_attn.__version__}'); print('SUCCESS')"
```
