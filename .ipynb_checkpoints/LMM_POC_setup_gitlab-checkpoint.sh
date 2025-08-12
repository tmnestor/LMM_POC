#!/bin/bash

# LMM_POC_setup_gitlab.sh - Vision Model Notebooks Setup Script for Corporate GitLab
# Usage: source LMM_POC_setup_gitlab.sh
#
# This script sets up the environment for running vision model comparison notebooks:
# - Creates conda environment from environment.yml
# - Sets up Jupyter kernel for notebook execution
# - Validates model dependencies and hardware
# - Configures SSH for corporate GitLab access

# Corporate GitLab Configuration - MODIFY THESE FOR YOUR ORGANIZATION
GITLAB_DOMAIN="gitlab.yourcompany.com"  # Replace with your corporate GitLab domain
GITLAB_PORT="22"  # Change to custom port if needed (e.g., 2222)

# Set permissions for SSH keys (check multiple key types)
for keytype in id_ecdsa id_ed25519 id_rsa; do
    [ -f "/home/jovyan/.ssh/$keytype" ] && chmod 600 "/home/jovyan/.ssh/$keytype"
    [ -f "/home/jovyan/.ssh/$keytype.pub" ] && chmod 644 "/home/jovyan/.ssh/$keytype.pub"
done

# Configure git to use SSH instead of HTTPS for Corporate GitLab
# Check for any available SSH key (ECDSA, Ed25519, or RSA)
SSH_KEY_FOUND=""
for keytype in id_ecdsa id_ed25519 id_rsa; do
    if [ -f "/home/jovyan/.ssh/$keytype" ]; then
        SSH_KEY_FOUND="$keytype"
        break
    fi
done

if [ -n "$SSH_KEY_FOUND" ]; then
    echo "🔑 Setting up git SSH authentication for Corporate GitLab..."
    echo "   Using SSH key: $SSH_KEY_FOUND"
    
    # Set git remote to use SSH if currently using HTTPS
    CURRENT_REMOTE=$(git remote get-url origin 2>/dev/null || echo "")
    if [[ "$CURRENT_REMOTE" == https://$GITLAB_DOMAIN/* ]]; then
        SSH_REMOTE=$(echo "$CURRENT_REMOTE" | sed "s|https://$GITLAB_DOMAIN/|git@$GITLAB_DOMAIN:|")
        git remote set-url origin "$SSH_REMOTE"
        echo "✅ Updated git remote from HTTPS to SSH: $SSH_REMOTE"
    elif [[ "$CURRENT_REMOTE" == git@$GITLAB_DOMAIN:* ]]; then
        echo "✅ Git already configured for SSH: $CURRENT_REMOTE"
    fi
    
    # Configure SSH for corporate GitLab if using custom port
    SSH_CONFIG="/home/jovyan/.ssh/config"
    if [ "$GITLAB_PORT" != "22" ]; then
        if ! grep -q "Host $GITLAB_DOMAIN" "$SSH_CONFIG" 2>/dev/null; then
            echo "🔧 Adding GitLab SSH configuration for custom port..."
            mkdir -p /home/jovyan/.ssh
            cat >> "$SSH_CONFIG" << EOF
Host $GITLAB_DOMAIN
    Port $GITLAB_PORT
    User git
    IdentityFile /home/jovyan/.ssh/$SSH_KEY_FOUND
EOF
            chmod 600 "$SSH_CONFIG"
            echo "✅ SSH config updated for $GITLAB_DOMAIN:$GITLAB_PORT"
        fi
    fi
    
    # Test SSH connection to corporate GitLab
    echo "🔍 Testing SSH connection to Corporate GitLab..."
    if ssh -T git@$GITLAB_DOMAIN -o StrictHostKeyChecking=no -o ConnectTimeout=10 2>&1 | grep -q "Welcome to GitLab"; then
        echo "✅ SSH authentication to Corporate GitLab working"
    else
        echo "⚠️ SSH authentication test failed"
        echo "   Troubleshooting steps:"
        echo "   1. Add SSH key to GitLab: https://$GITLAB_DOMAIN/-/profile/keys"
        echo "   2. Ensure VPN connection to corporate network"
        echo "   3. Verify GitLab domain and port settings in this script"
        echo "   4. Check with IT if firewall blocks SSH to GitLab"
        if [ -n "$SSH_KEY_FOUND" ] && [ -f "/home/jovyan/.ssh/$SSH_KEY_FOUND.pub" ]; then
            echo "   Your public key ($SSH_KEY_FOUND):"
            cat "/home/jovyan/.ssh/$SSH_KEY_FOUND.pub"
        fi
    fi
fi

# Default configuration for unified vision processor
DEFAULT_DIR="$HOME/nfs_share/tod/LMM_POC"
DEFAULT_ENV="unified_vision_processor"

# Parse arguments
WORK_DIR=${1:-$DEFAULT_DIR}
CONDA_ENV=${2:-$DEFAULT_ENV}

# Print header
echo "========================================================"
echo "🔬 Unified Vision Document Processing System (GitLab)"
echo "🚀 Setting up environment: $CONDA_ENV"
echo "🏢 Corporate GitLab: $GITLAB_DOMAIN"
echo "========================================================"

# Change to working directory
if [ -d "$WORK_DIR" ]; then
    cd "$WORK_DIR"
    echo "✅ Changed directory to: $(pwd)"
else
    echo "❌ Error: Directory $WORK_DIR does not exist"
    echo "   Expected: unified_vision_processor project directory"
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

# Set up PYTHONPATH for package access (no pip install needed)
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
echo "📋 Corporate GitLab Configuration:"
echo "   🏢 GitLab Domain: $GITLAB_DOMAIN"
echo "   🔌 SSH Port: $GITLAB_PORT"
echo "   📝 To update model paths, edit common/config.py:"
echo "      - CURRENT_INTERNVL3_MODEL = 'InternVL3-8B' or 'InternVL3-2B'"
echo "      - CURRENT_DEPLOYMENT = 'AISandbox' or 'efs'"

echo ""
echo "🚀 Quick Start:"
echo "   1. Configure model paths in common/config.py"
echo "   2. Run evaluation: python llama_keyvalue.py"
echo "   3. Or launch Jupyter: jupyter notebook"
echo "   4. Git operations now work with Corporate GitLab"

echo ""
echo "📋 Current Environment:"
echo "   - Working directory: $(pwd)"
echo "   - Python: $(which python)"
echo "   - Conda environment: $CONDA_ENV"
echo "   - GitLab: $GITLAB_DOMAIN"

echo ""
echo "========================================================"
echo "🚀 Vision Model Notebooks Ready for Corporate GitLab!"
echo "========================================================"
echo "Remember to run with 'source' to preserve environment:"
echo "source LMM_POC_setup_gitlab.sh [environment_name]"
echo ""
echo "Corporate GitLab Notes:"
echo "- SSH key must be added to $GITLAB_DOMAIN/-/profile/keys"
echo "- Ensure VPN connection if required"
echo "- Contact IT if SSH connections are blocked"
echo "========================================================"