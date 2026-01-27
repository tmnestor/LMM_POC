# Flash Attention 2 Diagnostics for AWS G6 Instances

Guide for diagnosing and resolving Flash Attention 2 installation issues on AWS G6 production machines (NVIDIA L4 / Ada Lovelace).

---

## GPU Compatibility

AWS G6 instances use NVIDIA L4 GPUs (Ada Lovelace architecture, compute capability 8.9). Flash Attention 2 requires compute capability >= 8.0, so the **hardware is fully supported**.

| Requirement | G6 (L4) Status |
|-------------|:--------------:|
| GPU Architecture (Ada Lovelace) | Supported |
| Compute Capability 8.9 | Supported (needs >= 8.0) |
| fp16 | Supported |
| bf16 | Supported |
| Head dimensions up to 256 | Supported |
| Head dim > 192 backward pass | Not supported (requires A100/H100) |

---

## Software Requirements

| Dependency | Minimum Version |
|------------|:--------------:|
| Python | >= 3.9 |
| PyTorch | >= 2.2 |
| CUDA Toolkit | >= 12.0 |
| ninja | Any (critical for build speed) |
| packaging | Any |
| gcc/g++ | Compatible with CUDA version |

---

## Diagnostic Commands

### 1. GPU and CUDA Hardware

```bash
# Confirm GPU model and driver version
nvidia-smi

# Check GPU name, compute capability, and driver version
nvidia-smi --query-gpu=gpu_name,compute_cap,driver_version --format=csv
```

### 2. CUDA Toolkit

```bash
# Check if nvcc is available (most common failure point)
nvcc --version

# Check CUDA_HOME is set
echo $CUDA_HOME

# Check if CUDA is in PATH
which nvcc

# Find CUDA installations on the system
ls -la /usr/local/cuda*
```

### 3. Python and PyTorch

```bash
# Python version (needs >= 3.9)
python --version

# PyTorch version and CUDA support (needs >= 2.2)
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### 4. Build Tools

```bash
# Check ninja (critical - without it, compilation takes ~2 hours)
ninja --version
echo $?  # Should return 0

# Check gcc/g++ version
gcc --version

# Check pip and packaging
pip show packaging
```

### 5. Flash Attention Status

```bash
# Check if flash-attn is already installed
pip show flash-attn

# Test if it actually loads
python -c "import flash_attn; print(f'flash-attn version: {flash_attn.__version__}')"

# Check if transformers can see it
python -c "from transformers.utils import is_flash_attn_2_available; print(f'FA2 available: {is_flash_attn_2_available()}')"
```

---

## All-in-One Diagnostic Script

Run this as a single block to get a full report:

```bash
echo "=== Flash Attention 2 Diagnostics ==="
echo ""
echo "--- GPU ---"
nvidia-smi --query-gpu=gpu_name,compute_cap,driver_version --format=csv 2>/dev/null || echo "nvidia-smi FAILED"
echo ""
echo "--- CUDA Toolkit ---"
echo "CUDA_HOME: ${CUDA_HOME:-NOT SET}"
nvcc --version 2>/dev/null || echo "nvcc NOT FOUND - CUDA toolkit missing or not in PATH"
echo ""
echo "--- Python/PyTorch ---"
python -c "
import sys
print(f'Python: {sys.version}')
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'CUDA version (torch): {torch.version.cuda}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'Compute capability: {torch.cuda.get_device_capability(0)}')
except ImportError:
    print('PyTorch: NOT INSTALLED')
" 2>/dev/null
echo ""
echo "--- Build Tools ---"
ninja --version 2>/dev/null && echo "ninja: OK" || echo "ninja: NOT FOUND"
gcc --version 2>/dev/null | head -1 || echo "gcc: NOT FOUND"
echo ""
echo "--- Flash Attention ---"
python -c "
try:
    import flash_attn
    print(f'flash-attn: {flash_attn.__version__}')
except ImportError:
    print('flash-attn: NOT INSTALLED')
try:
    from transformers.utils import is_flash_attn_2_available
    print(f'FA2 via transformers: {is_flash_attn_2_available()}')
except Exception as e:
    print(f'transformers check failed: {e}')
" 2>/dev/null
echo ""
echo "=== End Diagnostics ==="
```

---

## Common Failures and Fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `nvcc not found` | CUDA toolkit not installed | `sudo apt install nvidia-cuda-toolkit` or set `CUDA_HOME` |
| `CUDA_HOME not set` | Environment not configured | `export CUDA_HOME=/usr/local/cuda` |
| `ninja not found` | Build tool missing | `pip install ninja` |
| `PyTorch < 2.2` | Old PyTorch | `pip install torch>=2.2` |
| Build takes 2+ hours | ninja missing | Install ninja first, then rebuild |
| `undefined symbol` error | ABI mismatch | Rebuild with `pip install flash-attn --no-build-isolation` |

---

## Installation

Once all prerequisites are confirmed:

```bash
# Ensure ninja is installed first
pip install ninja

# Install flash-attn (--no-build-isolation ensures ABI compatibility with installed PyTorch)
pip install flash-attn --no-build-isolation
```

### Low-Memory Build

If the machine has less than 96GB RAM with many CPU cores, limit parallel jobs:

```bash
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

---

## References

- [Flash Attention GitHub](https://github.com/Dao-AILab/flash-attention)
- [Flash Attention Installation and Setup](https://deepwiki.com/Dao-AILab/flash-attention/1.1-installation-and-setup)
- [flash-attn on PyPI](https://pypi.org/project/flash-attn/)
- [AWS EC2 G6 Instances](https://aws.amazon.com/ec2/instance-types/g6/)

---

*Document created: January 2026*
