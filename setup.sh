#!/bin/bash

# LMM_POC Environment Setup Script - YAML-First Architecture
# This script activates the conda environment and validates YAML configurations
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
# YAML CONFIGURATION VALIDATION
# ============================================================================

echo -e "\n${BLUE}📋 YAML Configuration Validation${NC}"

# Check for critical YAML configuration files
YAML_FILES=(
    "config/fields.yaml"
    "prompts/prompt_config.yaml"
    "prompts/llama_single_pass_high_performance.yaml"
    "prompts/internvl3_single_pass_v4.yaml"
    "prompts/document_type_detection.yaml"
    "config/document_metrics.yaml"
)

echo -e "${BLUE}Checking YAML configuration files...${NC}"
YAML_MISSING=0

for yaml_file in "${YAML_FILES[@]}"; do
    if [ -f "${PROJECT_ROOT}/${yaml_file}" ]; then
        echo -e "  ${GREEN}✅${NC} ${yaml_file}"
    else
        echo -e "  ${RED}❌${NC} ${yaml_file} - NOT FOUND"
        YAML_MISSING=$((YAML_MISSING + 1))
    fi
done

if [ $YAML_MISSING -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Warning: ${YAML_MISSING} YAML configuration file(s) missing${NC}"
    echo -e "${YELLOW}💡 The YAML-first architecture requires these configuration files${NC}"
fi

# Check ground truth data
GROUND_TRUTH_PATH="${PROJECT_ROOT}/evaluation_data/ground_truth.csv"
if [ -f "$GROUND_TRUTH_PATH" ]; then
    echo -e "${GREEN}✅ Ground truth data found: ground_truth.csv${NC}"
else
    echo -e "${YELLOW}⚠️  Ground truth not found at: $GROUND_TRUTH_PATH${NC}"
fi

# Create output directory if it doesn't exist
OUTPUT_DIR="${PROJECT_ROOT}/output"
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}✅ Output directory ready: $OUTPUT_DIR${NC}"

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
echo -e "${BLUE}📋 YAML-First Configuration Active${NC}"
echo ""
echo -e "${BLUE}To activate the environment:${NC}"
echo "   source setup.sh"
echo "   conda activate $CONDA_ENV_NAME"
echo ""
echo -e "${BLUE}To run evaluations:${NC}"
echo "   python llama_keyvalue.py        # Run Llama evaluation"
echo "   python internvl3_keyvalue.py    # Run InternVL3 evaluation"
echo "   python llama_document_aware.py  # Run document-aware extraction"
echo ""
echo -e "${BLUE}To run notebooks:${NC}"
echo "   jupyter notebook llama_document_aware.ipynb"
echo ""
echo -e "${BLUE}Configuration locations (YAML-first):${NC}"
echo "   📁 Field Schema: config/fields.yaml (simplified!)"
echo "   📁 Prompts: prompts/prompt_config.yaml"
echo "   📁 Metrics: config/document_metrics.yaml"
echo "   📁 Model paths: Set in common/config.py (not env vars)"
echo ""
echo -e "${YELLOW}💡 Note: Model paths are configured in common/config.py${NC}"
echo -e "${YELLOW}   NOT through environment variables (YAML-first strategy)${NC}"

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
        
        # Test YAML configuration loading
        python -c "
from pathlib import Path
import sys

# Test unified schema (single source of truth)
try:
    from common.unified_schema import DocumentTypeFieldSchema
    schema = DocumentTypeFieldSchema()
    detection_config = schema.load_detection_prompts()
    print('✅ Unified schema with document detection loaded successfully')
    print(f'   Supported document types: {len(detection_config["supported_types"])}')
except Exception as e:
    print(f'⚠️  Unified schema initialization failed: {e}')
    sys.exit(1)

# Test schema loader
try:
    from common.unified_schema import DocumentTypeFieldSchema
    schema = DocumentTypeFieldSchema()
    print(f'✅ Schema loaded: {schema.get_field_count()} fields from config/fields.yaml')
    print(f'✅ Document types: {len(schema.get_supported_document_types())} supported')
except Exception as e:
    print(f'⚠️  Schema loader failed: {e}')
    sys.exit(1)

# Verify model paths in config.py (not environment variables)
try:
    from common.config import LLAMA_MODEL_PATH, INTERNVL3_MODEL_PATH
    print('✅ Model paths configured in common/config.py')
except Exception as e:
    print(f'⚠️  Model path configuration issue: {e}')
" 2>/dev/null && {
            echo -e "${GREEN}✅ YAML-first configuration validated${NC}"
        } || {
            echo -e "${YELLOW}⚠️  Configuration validation failed - check YAML files${NC}"
        }
    else
        echo -e "${RED}❌ Environment activation verification failed${NC}"
    fi
else
    echo -e "\n${YELLOW}💡 To activate the environment:${NC}"
    echo "   source setup.sh"
fi

# Export only essential paths (not model paths - those are in YAML/config)
export PROJECT_ROOT
export CONDA_ENV_NAME
export CONDA_ENV_PATH

# Reminder about YAML-first configuration
echo -e "\n${BLUE}📋 Remember: All configuration is in YAML files${NC}"
echo -e "${BLUE}   Model paths: common/config.py${NC}"
echo -e "${BLUE}   Prompts: prompts/prompt_config.yaml${NC}"
echo -e "${BLUE}   Schema: config/schema_v4.yaml${NC}"