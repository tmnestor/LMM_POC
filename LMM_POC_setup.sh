#!/bin/bash

# LMM_POC_setup.sh - LMM POC Model Notebooks Setup Script
# Usage: source LMM_POC_setup.sh
#
# This script sets up the environment for running vision model comparison notebooks:
# - Creates conda environment from environment_ivl35.yml
# - Registers Jupyter kernel: LMM_POC_IVL3.5
# - Validates model dependencies and hardware

# Symlink SSH keys from persistent NFS storage to ephemeral home
NFS_SSH_DIR="$HOME/nfs_share/tod_2026/.ssh"
if [ -d "$NFS_SSH_DIR" ]; then
    mkdir -p "$HOME/.ssh"
    ln -sf "$NFS_SSH_DIR/id_ed25519" "$HOME/.ssh/id_ed25519"
    ln -sf "$NFS_SSH_DIR/id_ed25519.pub" "$HOME/.ssh/id_ed25519.pub"
    chmod 700 "$HOME/.ssh"
    chmod 600 "$NFS_SSH_DIR/id_ed25519"
    chmod 644 "$NFS_SSH_DIR/id_ed25519.pub"
    echo "✅ SSH keys symlinked from NFS persistent storage"
else
    echo "⚠️ No SSH keys found at $NFS_SSH_DIR"
    echo "   Generate with: ssh-keygen -t ed25519 -C \"your@email\" -f $NFS_SSH_DIR/id_ed25519 -N \"\""
fi

# Configure git to use SSH instead of HTTPS for GitHub
if [ -f "$HOME/.ssh/id_ed25519" ]; then
    echo "🔑 Setting up git SSH authentication..."
    
    # Set git remote to use SSH if currently using HTTPS
    CURRENT_REMOTE=$(git remote get-url origin 2>/dev/null || echo "")
    if [[ "$CURRENT_REMOTE" == https://github.com/* ]]; then
        SSH_REMOTE=$(echo "$CURRENT_REMOTE" | sed 's|https://github.com/|git@github.com:|')
        git remote set-url origin "$SSH_REMOTE"
        echo "✅ Updated git remote from HTTPS to SSH: $SSH_REMOTE"
    elif [[ "$CURRENT_REMOTE" == git@github.com:* ]]; then
        echo "✅ Git already configured for SSH: $CURRENT_REMOTE"
    fi
    
    # Test SSH connection
    if ssh -T git@github.com -o StrictHostKeyChecking=no -o ConnectTimeout=10 2>&1 | grep -q "successfully authenticated"; then
        echo "✅ SSH authentication to GitHub working"
    else
        echo "⚠️ SSH authentication test failed - you may need to add the SSH key to GitHub"
        echo "   Add this key to GitHub: https://github.com/settings/ssh/new"
        [ -f "/home/jovyan/.ssh/id_ed25519.pub" ] && echo "   Public key:" && cat /home/jovyan/.ssh/id_ed25519.pub
    fi
fi

# Default configuration
DEFAULT_DIR="$HOME/nfs_share/tod_2026/LMM_POC"
DEFAULT_ENV="LMM_POC_IVL3.5"
DEFAULT_YML="environment_ivl35.yml"

# Parse arguments
WORK_DIR=${1:-$DEFAULT_DIR}
CONDA_ENV=${2:-$DEFAULT_ENV}

# Print header
echo "========================================================"
echo "🔬 Unified Vision Document Processing System"
echo "🚀 Setting up environment: $CONDA_ENV"
echo "========================================================"

# Change to working directory
if [ -d "$WORK_DIR" ]; then
    cd "$WORK_DIR"
    echo "✅ Changed directory to: $(pwd)"
else
    echo "❌ Error: Directory $WORK_DIR does not exist"
    echo "   Expected: LMM_POC project directory"
    return 1
fi

# Initialize conda
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
    echo "✅ Conda initialized"
    
    # Try to activate the conda environment
    if conda activate "$CONDA_ENV" 2>/dev/null; then
        echo "✅ Activated conda environment: $CONDA_ENV"
    else
        echo "⚠️ Conda environment '$CONDA_ENV' not found"
        echo "   Creating environment from $DEFAULT_YML..."

        if [ -f "$DEFAULT_YML" ]; then
            echo "📦 Installing dependencies..."
            if conda env create -f "$DEFAULT_YML"; then
                echo "✅ Environment created successfully"
                conda activate "$CONDA_ENV"
                echo "✅ Activated new environment: $CONDA_ENV"
            else
                echo "❌ Failed to create environment from $DEFAULT_YML"
                echo "   Available environments:"
                conda env list
                return 1
            fi
        else
            echo "❌ $DEFAULT_YML not found in current directory"
            return 1
        fi
    fi
else
    echo "❌ Error: Conda initialization file not found"
    return 1
fi

# Register conda environment as Jupyter kernel
KERNEL_NAME="$CONDA_ENV"
KERNEL_DISPLAY="Python (LMM_POC_IVL3.5)"
if [ -n "$KERNEL_NAME" ] && jupyter kernelspec list 2>/dev/null | grep -qw "$KERNEL_NAME"; then
    echo "✅ Jupyter kernel '$KERNEL_DISPLAY' already registered"
else
    echo "📦 Registering Jupyter kernel: $KERNEL_DISPLAY..."
    if python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY"; then
        echo "✅ Jupyter kernel registered: $KERNEL_DISPLAY"
    else
        echo "❌ Failed to register Jupyter kernel"
        echo "   Try manually: python -m ipykernel install --user --name $KERNEL_NAME --display-name \"$KERNEL_DISPLAY\""
    fi
fi

# Set up PYTHONPATH for package access (no pip install needed)
# Append to existing PYTHONPATH to avoid overwriting other paths, but avoid duplicates
CURRENT_DIR="$(pwd)"
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH="$CURRENT_DIR"
    echo "✅ Set PYTHONPATH to: $CURRENT_DIR"
elif [[ ":$PYTHONPATH:" != *":$CURRENT_DIR:"* ]]; then
    export PYTHONPATH="$CURRENT_DIR:$PYTHONPATH"
    echo "✅ Added project to PYTHONPATH: $CURRENT_DIR"
    echo "   Full PYTHONPATH: $PYTHONPATH"
else
    echo "✅ Project already in PYTHONPATH: $CURRENT_DIR"
    echo "   Current PYTHONPATH: $PYTHONPATH"
fi

# Detect hardware environment
echo ""
echo "🔍 Hardware Detection:"
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo "   GPUs detected: $GPU_COUNT"
    echo "   GPU memory: ${GPU_MEMORY}MB"
    
    if [ "$GPU_MEMORY" -lt 20000 ]; then
        echo "   💡 Limited GPU memory detected"
        echo "      Recommended: Use 8-bit quantization in notebooks"
        echo "      (This is already configured in the notebooks)"
    fi
else
    echo "   CPU-only environment detected"
    echo "   💡 Notebooks will run on CPU (slower but functional)"
fi

# Check key dependencies
echo ""
echo "🔍 Verifying installation:"
echo "   📦 Checking dependencies:"

python -c "import torch; print(f'   ✅ PyTorch: {torch.__version__}')" 2>/dev/null || echo "   ❌ PyTorch not available"
python -c "import transformers; print(f'   ✅ Transformers: {transformers.__version__}')" 2>/dev/null || echo "   ❌ Transformers not available"
python -c "import PIL; print('   ✅ PIL (Pillow): available')" 2>/dev/null || echo "   ❌ PIL not available"
python -c "import torchvision; print('   ✅ Torchvision: available')" 2>/dev/null || echo "   ❌ Torchvision not available"

# Check CUDA availability
if python -c "import torch; print(f'   ✅ CUDA available: {torch.cuda.is_available()}')" 2>/dev/null; then
    python -c "import torch; print(f'   ✅ CUDA devices: {torch.cuda.device_count()}')" 2>/dev/null
else
    echo "   ⚠️ CUDA not available - notebooks will run on CPU"
fi

echo ""
echo "📋 Current Environment:"
echo "   - Working directory: $(pwd)"
echo "   - Python: $(which python)"
echo "   - Conda environment: $CONDA_ENV"

# Configure colored prompt with conda environment and git branch
# Colors: Green for conda env, Cyan for git branch, Yellow for directory, White for $
git_branch() {
    git branch 2>/dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/(\1)/'
}

export PS1='\[\e[0;32m\](${CONDA_DEFAULT_ENV})\[\e[0m\] \[\e[0;36m\]$(git_branch)\[\e[0m\] \[\e[0;33m\]\w\[\e[0m\] \[\e[0;37m\]\$\[\e[0m\] '

alias gsync='git checkout -- . && git pull && reset'
echo "✅ Setup complete! You can now run the vision model comparison notebooks."
echo "✅ Colored prompt configured: (conda-env) (git-branch) directory $"