#!/bin/bash

# unified_setup.sh - Vision Model Notebooks Setup Script
# Usage: source unified_setup.sh
#
# This script sets up the environment for running vision model comparison notebooks:
# - Creates conda environment from environment.yml
# - Sets up Jupyter kernel for notebook execution
# - Validates model dependencies and hardware

# Set permissions for SSH and Kaggle (if they exist)
[ -f "/home/jovyan/.ssh/id_ed25519" ] && chmod 600 /home/jovyan/.ssh/id_ed25519

# Configure git to use SSH instead of HTTPS for GitHub
if [ -f "/home/jovyan/.ssh/id_ed25519" ]; then
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

# Default configuration for unified vision processor
DEFAULT_DIR="$HOME/nfs_share/tod/LMM_POC"
DEFAULT_ENV="unified_vision_processor"

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




# # Default configuration
# DEFAULT_ENV="unified_vision_processor"
# CONDA_ENV=${1:-$DEFAULT_ENV}

# # Print header
# echo "========================================================"
# echo "🔬 Vision Model Comparison Notebooks"
# echo "🚀 Setting up environment: $CONDA_ENV"
# echo "========================================================"

# # Get current directory
# NOTEBOOK_DIR="$(pwd)"
# echo "✅ Working directory: $NOTEBOOK_DIR"

# # Initialize conda
# if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
#     source "/opt/conda/etc/profile.d/conda.sh"
#     echo "✅ Conda initialized"
    
#     # Try to activate the conda environment
#     if conda activate "$CONDA_ENV" 2>/dev/null; then
#         echo "✅ Activated conda environment: $CONDA_ENV"
#     else
#         echo "⚠️ Conda environment '$CONDA_ENV' not found"
#         echo "   Creating environment from environment.yml..."
        
#         if [ -f "environment.yml" ]; then
#             echo "📦 Installing dependencies (this may take a few minutes)..."
#             if conda env create -f environment.yml; then
#                 echo "✅ Environment created successfully"
#                 conda activate "$CONDA_ENV"
#                 echo "✅ Activated new environment: $CONDA_ENV"
                
#                 # Register Jupyter kernel
#                 echo "🔧 Registering Jupyter kernel..."
#                 if python -m ipykernel install --user --name "$CONDA_ENV" --display-name "Python (Vision Notebooks)"; then
#                     echo "✅ Jupyter kernel registered: Python (Vision Notebooks)"
#                 else
#                     echo "⚠️ Failed to register Jupyter kernel"
#                 fi
#             else
#                 echo "❌ Failed to create environment from environment.yml"
#                 echo "   Available environments:"
#                 conda env list
#                 return 1
#             fi
#         else
#             echo "❌ environment.yml not found in current directory"
#             return 1
#         fi
#     fi
# else
#     echo "❌ Error: Conda initialization file not found"
#     echo "   Expected: /opt/conda/etc/profile.d/conda.sh"
#     return 1
# fi

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
echo "📋 Notebook Information:"
echo "   📁 Notebooks directory: $NOTEBOOK_DIR/notebooks/"
echo "   🖼️  Sample image: $NOTEBOOK_DIR/sample_data/synthetic_invoice_014.png"
echo ""
echo "   Available notebooks:"
echo "   1️⃣  llama_VQA.ipynb          - Llama visual question answering"
echo "   2️⃣  internvl3_VQA.ipynb      - InternVL3 visual question answering"
echo "   3️⃣  llama_keyvalue.ipynb     - Llama structured field extraction"
echo "   4️⃣  internvl3_keyvalue.ipynb - InternVL3 structured field extraction"

echo ""
echo "⚠️  IMPORTANT: Update Model Paths"
echo "   Before running notebooks, edit the model_id paths in each notebook:"
echo "   - Llama: '/path/to/Llama-3.2-11B-Vision-Instruct'"
echo "   - InternVL3: '/path/to/InternVL3-2B'"
echo "   - Sample image path: 'sample_data/synthetic_invoice_014.png'"

echo ""
echo "🚀 Quick Start:"
echo "   1. Update model paths in notebooks"
echo "   2. Launch Jupyter: jupyter notebook"
echo "   3. Select kernel: Python (Vision Notebooks)"
echo "   4. Run any notebook to test your models"

echo ""
echo "📋 Current Environment:"
echo "   - Working directory: $NOTEBOOK_DIR"
echo "   - Python: $(which python)"
echo "   - Conda environment: $CONDA_ENV"

echo ""
echo "========================================================"
echo "🚀 Vision Model Notebooks Ready!"
echo "========================================================"
echo "Remember to run with 'source' to preserve environment:"
echo "source unified_setup.sh [environment_name]"
echo ""
echo "Next steps:"
echo "1. Edit model paths in notebooks/"
echo "2. Launch: jupyter notebook"
echo "3. Test with sample_data/synthetic_invoice_014.png"
echo "========================================================"