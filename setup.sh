#!/bin/bash

# LMM_POC Environment Setup Script
# This script sets up environment variables and activates the conda environment
# for the vision-language model document extraction project.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 LMM_POC Environment Setup${NC}"
echo "=================================="

# Get the directory where this script is located
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Script is being executed directly
    SCRIPT_DIR="$( cd "$( dirname "${0}" )" &> /dev/null && pwd )"
else
    # Script is being sourced
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
fi

# If SCRIPT_DIR detection failed, use current directory as fallback
if [ -z "$SCRIPT_DIR" ] || [ ! -d "$SCRIPT_DIR" ]; then
    SCRIPT_DIR="$PWD"
fi

PROJECT_ROOT="$SCRIPT_DIR"

echo -e "${BLUE}📁 Project root: ${PROJECT_ROOT}${NC}"

# Verify project root contains expected files
if [ ! -f "${PROJECT_ROOT}/common/config.py" ]; then
    echo -e "${RED}❌ ERROR: Invalid project root - common/config.py not found${NC}"
    echo -e "${YELLOW}💡 Expected project structure not found at: ${PROJECT_ROOT}${NC}"
    echo -e "${BLUE}💡 Make sure to run this script from the LMM_POC project directory${NC}"
    if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
        return 1  # Return instead of exit when sourced
    else
        exit 1
    fi
fi

# ============================================================================
# CONDA ENVIRONMENT SETUP
# ============================================================================

CONDA_ENV_NAME="unified_vision_processor"
CONDA_ENV_PATH="/opt/homebrew/Caskroom/miniforge/base/envs/${CONDA_ENV_NAME}"

echo -e "\n${BLUE}🐍 Conda Environment Setup${NC}"
echo "Environment: ${CONDA_ENV_NAME}"
echo "Path: ${CONDA_ENV_PATH}"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}❌ ERROR: conda not found${NC}"
    echo -e "${YELLOW}💡 Please install conda (miniforge) first:${NC}"
    echo "   brew install --cask miniforge"
    exit 1
fi

# Check if environment exists
if [ ! -d "$CONDA_ENV_PATH" ]; then
    echo -e "${YELLOW}⚠️  Conda environment not found: ${CONDA_ENV_NAME}${NC}"
    echo -e "${BLUE}Creating environment from environment.yml...${NC}"
    
    if [ -f "${PROJECT_ROOT}/environment.yml" ]; then
        conda env create -f "${PROJECT_ROOT}/environment.yml"
        echo -e "${GREEN}✅ Environment created successfully${NC}"
    else
        echo -e "${RED}❌ ERROR: environment.yml not found${NC}"
        echo -e "${YELLOW}💡 Create environment manually:${NC}"
        echo "   conda create -n ${CONDA_ENV_NAME} python=3.11 -y"
        echo "   conda activate ${CONDA_ENV_NAME}"
        echo "   pip install torch transformers accelerate bitsandbytes"
        exit 1
    fi
else
    echo -e "${GREEN}✅ Conda environment exists${NC}"
fi

# ============================================================================
# ENVIRONMENT VARIABLES CONFIGURATION
# ============================================================================

echo -e "\n${BLUE}🌍 Environment Variables Setup${NC}"

# Determine environment profile - check for existing env var first, then detect
if [ -n "$LMM_ENVIRONMENT" ]; then
    echo -e "${BLUE}🔧 Using pre-set LMM_ENVIRONMENT: $LMM_ENVIRONMENT${NC}"
elif [[ $(hostname) == *"jovyan"* ]] || [[ $(hostname) == *"jupyter"* ]]; then
    LMM_ENVIRONMENT="testing"
    echo -e "${BLUE}🧪 Detected Jupyter/H200 environment - using 'testing' profile${NC}"
elif [[ "$PWD" == *"/efs/"* ]] || [[ "$PWD" == *"v100"* ]]; then
    LMM_ENVIRONMENT="production"
    echo -e "${BLUE}🚀 Detected EFS/V100 environment - using 'production' profile${NC}"
else
    LMM_ENVIRONMENT="development"
    echo -e "${BLUE}🏠 Using 'development' profile for local environment${NC}"
fi

# Set environment variables
export LMM_ENVIRONMENT="$LMM_ENVIRONMENT"

# Model paths (customize these based on your setup)
# Set model paths based on environment
case "$LMM_ENVIRONMENT" in
    "development")
        # Local development paths - use environment profile defaults
        echo -e "${BLUE}🏠 Using environment profile defaults for model paths${NC}"
        # Don't set model path environment variables - let config.py use profile defaults
        ;;
    "testing")
        # H200 GPU testing environment
        export LLAMA_MODEL_PATH="/home/jovyan/nfs_share/models/Llama-3.2-11B-Vision-Instruct"
        export INTERNVL3_MODEL_PATH="/home/jovyan/nfs_share/models/InternVL3-8B"
        echo -e "${BLUE}🧪 Set model paths for H200 testing environment${NC}"
        ;;
    "production")
        # V100 production environment
        export LLAMA_MODEL_PATH="/efs/shared/PTM/Llama-3.2-11B-Vision-Instruct"
        export INTERNVL3_MODEL_PATH="/efs/shared/PTM/InternVL3-8B"
        echo -e "${BLUE}🚀 Set model paths for V100 production environment${NC}"
        ;;
esac

# Data and output paths
export GROUND_TRUTH_PATH="${PROJECT_ROOT}/evaluation_data/evaluation_ground_truth.csv"
export OUTPUT_DIR="${PROJECT_ROOT}/output"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}✅ Environment variables set:${NC}"
echo "   LMM_ENVIRONMENT: $LMM_ENVIRONMENT"
echo "   GROUND_TRUTH_PATH: $GROUND_TRUTH_PATH"
echo "   OUTPUT_DIR: $OUTPUT_DIR"

# Show model paths if they are set
if [ -n "$LLAMA_MODEL_PATH" ]; then
    echo "   LLAMA_MODEL_PATH: $LLAMA_MODEL_PATH"
fi
if [ -n "$INTERNVL3_MODEL_PATH" ]; then
    echo "   INTERNVL3_MODEL_PATH: $INTERNVL3_MODEL_PATH"
fi

# ============================================================================
# CONDA ACTIVATION SCRIPT
# ============================================================================

echo -e "\n${BLUE}🔄 Setting up conda activation${NC}"

# Source conda setup if available
CONDA_BASE=$(conda info --base 2>/dev/null || echo "/opt/homebrew/Caskroom/miniforge/base")
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    echo -e "${GREEN}✅ Conda initialization script sourced${NC}"
else
    echo -e "${YELLOW}⚠️  Conda initialization script not found at expected location${NC}"
fi

# ============================================================================
# COMPLETION MESSAGE
# ============================================================================

echo -e "\n${GREEN}🎉 Setup Complete!${NC}"
echo "=================================="
echo -e "${BLUE}To activate the environment manually:${NC}"
echo "   source setup.sh"
echo "   conda activate $CONDA_ENV_NAME"
echo ""
echo -e "${BLUE}To run the project:${NC}"
echo "   python llama_keyvalue.py        # Run Llama evaluation"
echo "   python internvl3_keyvalue.py    # Run InternVL3 evaluation"
echo ""
echo -e "${BLUE}To check configuration:${NC}"
echo "   python -c \"from common.config import show_current_config; show_current_config()\""

# ============================================================================
# ACTIVATION (if sourced)
# ============================================================================

# Check if script is being sourced (not executed)
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo -e "\n${BLUE}🚀 Activating conda environment...${NC}"
    conda activate "$CONDA_ENV_NAME" 2>/dev/null || {
        echo -e "${YELLOW}⚠️  Direct activation failed. Using full path...${NC}"
        conda activate "$CONDA_ENV_PATH" 2>/dev/null || {
            echo -e "${RED}❌ Failed to activate conda environment${NC}"
            echo -e "${YELLOW}💡 Try manually:${NC}"
            echo "   conda activate $CONDA_ENV_NAME"
            return 1
        }
    }
    
    # Verify activation
    if [[ "$CONDA_DEFAULT_ENV" == "$CONDA_ENV_NAME" ]]; then
        echo -e "${GREEN}✅ Successfully activated: $CONDA_DEFAULT_ENV${NC}"
        
        # Test import
        python -c "from common.config import show_current_config; show_current_config()" 2>/dev/null && {
            echo -e "${GREEN}✅ Configuration loaded successfully${NC}"
        } || {
            echo -e "${YELLOW}⚠️  Configuration test failed - check dependencies${NC}"
        }
    else
        echo -e "${RED}❌ Environment activation verification failed${NC}"
    fi
else
    echo -e "\n${YELLOW}💡 To activate the environment:${NC}"
    echo "   source setup.sh"
fi

# Export functions for convenience
export PROJECT_ROOT
export CONDA_ENV_NAME
export CONDA_ENV_PATH