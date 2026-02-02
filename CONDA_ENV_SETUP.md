# Conda Environment Setup

## Create Shared Environment

```bash
# Create new environment with Python 3.12 in shared location
conda create --prefix /efs/shared/venvs/lmm_poc python=3.12 -y

# Set group permissions (run after creation)
chmod -R g+rwX /efs/shared/venvs/lmm_poc
# Optional: set group ownership if needed
# chgrp -R <your-group-name> /efs/shared/venvs/lmm_poc

# Activate (use full path for prefix-based environments)
conda activate /efs/shared/venvs/lmm_poc

# Install PyTorch 2.3.1 with CUDA 12.1
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# Install transformers
pip install transformers==4.57.3

# Re-apply group permissions after installing packages
chmod -R g+rwX /efs/shared/venvs/lmm_poc
```

## Activation for Group Members

```bash
# All group members activate using the full path
conda activate /efs/shared/venvs/lmm_poc
```

## Verify Installation

```bash
python -c "
import torch
import transformers
print(f'Python: {__import__(\"sys\").version.split()[0]}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA (PyTorch): {torch.version.cuda}')
print(f'CXX11 ABI: {torch._C._GLIBCXX_USE_CXX11_ABI}')
print(f'Transformers: {transformers.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
"
```

## Expected Output

```
Python: 3.12.x
PyTorch: 2.3.1+cu121
CUDA (PyTorch): 12.1
CXX11 ABI: False
Transformers: 4.57.3
CUDA available: True
Device: NVIDIA L4
```

## Notes

- Environment uses `--prefix` for shared location instead of `-n` for named environment
- Group members must use the full path `/efs/shared/venvs/lmm_poc` to activate
- Re-run `chmod -R g+rwX` after installing additional packages
