# LMM_POC Environment Setup Guide

## Quick Start

To set up the environment and activate the project:

```bash
# Navigate to project directory
cd /path/to/LMM_POC

# Run setup script (will set environment variables and activate conda)
source setup.sh
```

## What setup.sh Does

### 🐍 **Conda Environment Management**
- Checks for existing `unified_vision_processor` conda environment
- Creates environment from `environment.yml` if needed
- Activates the environment automatically when sourced

### 🌍 **Environment Variable Configuration**
- Sets `LMM_ENVIRONMENT` based on detected system:
  - **development** - Local Mac development (default)
  - **testing** - H200 GPU environment (Jupyter/jovyan hostname)  
  - **production** - V100 environment (EFS paths)

### 📁 **Path Configuration**
- Sets up appropriate model paths for each environment:
  - **Development**: Uses environment profile defaults  
  - **Testing**: `/home/jovyan/nfs_share/models/`
  - **Production**: `/efs/shared/PTM/`
- Configures data paths:
  - `GROUND_TRUTH_PATH`: Points to project's ground truth CSV
  - `OUTPUT_DIR`: Creates and sets project output directory

### ✅ **Validation**
- Verifies project structure exists
- Tests conda environment activation
- Validates configuration loading

## Environment Profiles

The project supports multiple environment profiles configured in `common/config.py`:

| Profile | Description | Use Case |
|---------|-------------|----------|
| `development` | Local development environment | Mac M1 development |
| `testing` | H200 GPU testing environment | High-spec model testing |
| `production` | V100 production deployment | Final deployment target |
| `aisandbox` | AISandbox deployment (legacy) | Backward compatibility |
| `efs` | EFS deployment (legacy) | Backward compatibility |

## Manual Setup (Alternative)

If you prefer manual setup:

```bash
# 1. Activate conda environment
conda activate unified_vision_processor

# 2. Set environment variables
export LMM_ENVIRONMENT=development
export GROUND_TRUTH_PATH=$(pwd)/evaluation_data/evaluation_ground_truth.csv  
export OUTPUT_DIR=$(pwd)/output

# 3. Create output directory
mkdir -p output

# 4. Verify setup
python -c "from common.config import show_current_config; show_current_config()"
```

## Customization

### Custom Model Paths
To use custom model paths, set environment variables before running setup:

```bash
export LLAMA_MODEL_PATH="/path/to/your/llama/model"
export INTERNVL3_MODEL_PATH="/path/to/your/internvl3/model"
source setup.sh
```

### Different Environment Profile
To force a specific environment profile:

```bash
export LMM_ENVIRONMENT=production
source setup.sh
```

## Troubleshooting

### Environment Not Found
```bash
# If conda environment doesn't exist:
conda env create -f environment.yml
source setup.sh
```

### Permission Issues
```bash
# Make sure setup.sh is executable:
chmod +x setup.sh
```

### Path Issues
```bash
# Run from project root directory:
cd /path/to/LMM_POC
source setup.sh
```

### Configuration Test
```bash
# Test that everything is working:
python -c "from common.config import show_current_config; show_current_config()"
```

## Usage Examples

### After Setup - Run Evaluations
```bash
# Run Llama evaluation
python llama_keyvalue.py

# Run InternVL3 evaluation  
python internvl3_keyvalue.py

# Compare extraction strategies
python compare_grouping_strategies.py
```

### Check Current Configuration
```bash
python -c "from common.config import show_current_config; show_current_config()"
```

### Switch Environments
```bash
# Switch to production environment
export LMM_ENVIRONMENT=production
python -c "from common.config import switch_environment('production')"
```

## Files Created by Setup

- `output/` - Directory for evaluation results
- Environment variables set in current shell session
- Conda environment activated

## Integration with IDE

The setup script works well with IDEs:

1. **VS Code**: Run `source setup.sh` in integrated terminal
2. **PyCharm**: Set environment variables in run configurations
3. **Jupyter**: Setup script works in Jupyter terminal

## Best Practices

1. **Always source the script**: Use `source setup.sh`, not `./setup.sh`
2. **Run from project root**: Ensures correct path detection
3. **Check conda environment**: Verify `unified_vision_processor` is active
4. **Test configuration**: Run config test after setup

---

**Quick Setup**: `source setup.sh` from project directory  
**Support**: Check `CLAUDE.md` for additional configuration options