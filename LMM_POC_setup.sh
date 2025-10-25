#!/bin/bash

# LMM_POC_setup.sh - LMM POC Model Notebooks Setup Script
# Usage: source LMM_POC_setup.sh [work_dir] [conda_env]
#
# This script automates environment setup for vision model comparison on remote GPU machines (H200/V100):
# - Configures SSH authentication for passwordless git operations
# - Creates/activates conda environment from environment.yml
# - Sets up PYTHONPATH for development imports (no pip install needed)
# - Detects hardware (GPU/CPU) and validates dependencies
# - Configures colored prompt showing conda environment and git branch
#
# IMPORTANT: Use 'source' not 'bash' so environment variables persist in your shell

# ============================================================================
# SECTION 1: SSH Key Setup
# ============================================================================
# Set correct permissions for SSH keys (required for git operations)
[ -f "/home/jovyan/.ssh/id_ed25519" ] && chmod 600 /home/jovyan/.ssh/id_ed25519      # Private key: read-only by owner
[ -f "/home/jovyan/.ssh/id_ed25519.pub" ] && chmod 644 /home/jovyan/.ssh/id_ed25519.pub  # Public key: readable by all

# Automatically convert HTTPS git remotes to SSH for passwordless operations
if [ -f "/home/jovyan/.ssh/id_ed25519" ]; then
    echo "🔑 Setting up git SSH authentication..."

    # Check current git remote URL
    CURRENT_REMOTE=$(git remote get-url origin 2>/dev/null || echo "")

    # Convert HTTPS URLs to SSH format (https://github.com/user/repo → git@github.com:user/repo)
    if [[ "$CURRENT_REMOTE" == https://github.com/* ]]; then
        SSH_REMOTE=$(echo "$CURRENT_REMOTE" | sed 's|https://github.com/|git@github.com:|')
        git remote set-url origin "$SSH_REMOTE"
        echo "✅ Updated git remote from HTTPS to SSH: $SSH_REMOTE"
    elif [[ "$CURRENT_REMOTE" == git@github.com:* ]]; then
        echo "✅ Git already configured for SSH: $CURRENT_REMOTE"
    fi

    # Test SSH connection to GitHub (verifies key is added to GitHub account)
    if ssh -T git@github.com -o StrictHostKeyChecking=no -o ConnectTimeout=10 2>&1 | grep -q "successfully authenticated"; then
        echo "✅ SSH authentication to GitHub working"
    else
        echo "⚠️ SSH authentication test failed - you may need to add the SSH key to GitHub"
        echo "   Add this key to GitHub: https://github.com/settings/ssh/new"
        [ -f "/home/jovyan/.ssh/id_ed25519.pub" ] && echo "   Public key:" && cat /home/jovyan/.ssh/id_ed25519.pub
    fi
fi

# ============================================================================
# SECTION 2: Configuration & Arguments
# ============================================================================
# Default paths for remote GPU machines (H200/V100)
DEFAULT_DIR="$HOME/nfs_share/tod/LMM_POC"
DEFAULT_ENV="unified_vision_processor"

# Parse command-line arguments (allows custom paths if needed)
# Usage: source LMM_POC_setup.sh [/custom/path] [custom_env_name]
WORK_DIR=${1:-$DEFAULT_DIR}    # Use argument 1, or default directory
CONDA_ENV=${2:-$DEFAULT_ENV}   # Use argument 2, or default environment name

# ============================================================================
# SECTION 3: Startup Banner
# ============================================================================
echo "========================================================"
echo "🔬 Unified Vision Document Processing System"
echo "🚀 Setting up environment: $CONDA_ENV"
echo "========================================================"

# ============================================================================
# SECTION 4: Directory Navigation
# ============================================================================
# Navigate to project directory
if [ -d "$WORK_DIR" ]; then
    cd "$WORK_DIR"
    echo "✅ Changed directory to: $(pwd)"
else
    echo "❌ Error: Directory $WORK_DIR does not exist"
    echo "   Expected: unified_vision_processor project directory"
    return 1
fi

# ============================================================================
# SECTION 5: Conda Environment Management
# ============================================================================
# Initialize conda (required before using conda commands)
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
    echo "✅ Conda initialized"

    # Try to activate the conda environment
    if conda activate "$CONDA_ENV" 2>/dev/null; then
        echo "✅ Activated conda environment: $CONDA_ENV"
    else
        # Environment doesn't exist - create it automatically from environment.yml
        echo "⚠️ Conda environment '$CONDA_ENV' not found"
        echo "   Creating environment from environment.yml..."

        if [ -f "environment.yml" ]; then
            echo "📦 Installing dependencies (this may take a few minutes)..."
            if conda env create -f environment.yml; then
                echo "✅ Environment created successfully"
                conda activate "$CONDA_ENV"
                echo "✅ Activated new environment: $CONDA_ENV"
            else
                echo "❌ Failed to create environment from environment.yml"
                echo "   Available environments:"
                conda env list
                return 1
            fi
        else
            echo "❌ environment.yml not found in current directory"
            return 1
        fi
    fi
else
    echo "❌ Error: Conda initialization file not found"
    return 1
fi

# ============================================================================
# SECTION 6: Python Path Configuration
# ============================================================================
# Add project directory to PYTHONPATH for development imports (no pip install needed)
# Enables: from common.config import EXTRACTION_FIELDS
CURRENT_DIR="$(pwd)"
if [ -z "$PYTHONPATH" ]; then
    # PYTHONPATH is empty - set it
    export PYTHONPATH="$CURRENT_DIR"
    echo "✅ Set PYTHONPATH to: $CURRENT_DIR"
elif [[ ":$PYTHONPATH:" != *":$CURRENT_DIR:"* ]]; then
    # PYTHONPATH exists but doesn't contain project directory - prepend it
    export PYTHONPATH="$CURRENT_DIR:$PYTHONPATH"
    echo "✅ Added project to PYTHONPATH: $CURRENT_DIR"
    echo "   Full PYTHONPATH: $PYTHONPATH"
else
    # Project directory already in PYTHONPATH - no action needed
    echo "✅ Project already in PYTHONPATH: $CURRENT_DIR"
    echo "   Current PYTHONPATH: $PYTHONPATH"
fi

# ============================================================================
# SECTION 7: Hardware Detection
# ============================================================================
# Detect GPU/CPU environment and provide optimization recommendations
echo ""
echo "🔍 Hardware Detection:"
if command -v nvidia-smi >/dev/null 2>&1; then
    # GPU detected - get count and VRAM
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo "   GPUs detected: $GPU_COUNT"
    echo "   GPU memory: ${GPU_MEMORY}MB"

    # Provide optimization recommendations based on VRAM
    if [ "$GPU_MEMORY" -lt 20000 ]; then
        echo "   💡 Limited GPU memory detected (V100 16GB typical)"
        echo "      Recommended: Use 8-bit quantization in notebooks"
        echo "      (This is already configured in the notebooks)"
    fi
else
    # No GPU detected - CPU-only mode
    echo "   CPU-only environment detected"
    echo "   💡 Notebooks will run on CPU (slower but functional)"
fi

# ============================================================================
# SECTION 8: Dependency Verification
# ============================================================================
# Quick health check for critical dependencies
echo ""
echo "🔍 Verifying installation:"
echo "   📦 Checking dependencies:"

# Test critical Python packages
python -c "import torch; print(f'   ✅ PyTorch: {torch.__version__}')" 2>/dev/null || echo "   ❌ PyTorch not available"
python -c "import transformers; print(f'   ✅ Transformers: {transformers.__version__}')" 2>/dev/null || echo "   ❌ Transformers not available"
python -c "import PIL; print('   ✅ PIL (Pillow): available')" 2>/dev/null || echo "   ❌ PIL not available"
python -c "import torchvision; print('   ✅ Torchvision: available')" 2>/dev/null || echo "   ❌ Torchvision not available"

# Check CUDA availability within PyTorch
if python -c "import torch; print(f'   ✅ CUDA available: {torch.cuda.is_available()}')" 2>/dev/null; then
    python -c "import torch; print(f'   ✅ CUDA devices: {torch.cuda.device_count()}')" 2>/dev/null
else
    echo "   ⚠️ CUDA not available - notebooks will run on CPU"
fi

# ============================================================================
# SECTION 9: Environment Summary
# ============================================================================
echo ""
echo "📋 Current Environment:"
echo "   - Working directory: $(pwd)"
echo "   - Python: $(which python)"
echo "   - Conda environment: $CONDA_ENV"

# ============================================================================
# SECTION 10: Colored Prompt Configuration
# ============================================================================
# Configure shell prompt to display: (conda-env) (git-branch) directory $
# Colors: Green for conda env, Cyan for git branch, Yellow for directory, White for $
# Helper function to extract current git branch name
git_branch() {
    git branch 2>/dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/(\1)/'
}

# Set PS1 prompt with color codes:
# \[\e[0;32m\] = Green   | \[\e[0;36m\] = Cyan   | \[\e[0;33m\] = Yellow
# \[\e[0;37m\] = White   | \[\e[0m\]    = Reset color
export PS1='\[\e[0;32m\](${CONDA_DEFAULT_ENV})\[\e[0m\] \[\e[0;36m\]$(git_branch)\[\e[0m\] \[\e[0;33m\]\w\[\e[0m\] \[\e[0;37m\]\$\[\e[0m\] '

# ============================================================================
# SECTION 11: Useful Aliases
# ============================================================================
# Quick sync: Discard local changes, pull from remote, clear terminal
alias gsync='git checkout -- . && git pull && reset'
# ============================================================================
# Setup Complete
# ============================================================================
echo "✅ Setup complete! You can now run the vision model comparison notebooks."
echo "✅ Colored prompt configured: (conda-env) (git-branch) directory $"
echo ""
echo "💡 Tip: Add 'source ~/nfs_share/tod/LMM_POC/LMM_POC_setup.sh' to ~/.bashrc for automatic setup on login"