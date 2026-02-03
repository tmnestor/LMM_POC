# InternVL3.5-8B Document Extraction

This project provides two ways to run document field extraction using InternVL3.5-8B:

1. **Jupyter Notebook** (`ivl3_5_8b.ipynb`) - Interactive development and experimentation
2. **CLI Script** (`ivl3_cli.py`) - Production pipelines and automation

## Quick Start

### Option 1: Jupyter Notebook (Interactive)

```bash
# Activate environment
conda activate vision_notebooks

# Launch Jupyter
jupyter notebook ivl3_5_8b.ipynb

# In notebook: Kernel > Restart & Run All
```

### Option 2: CLI Script (Production)

```bash
# Activate environment
conda activate vision_notebooks

# Run with ground truth (evaluation mode)
python ivl3_cli.py \
  --data-dir ./evaluation_data/travel \
  --output-dir ./output \
  --ground-truth ./evaluation_data/travel/ground_truth_travel.csv

# Run without ground truth (inference-only mode)
python ivl3_cli.py \
  --data-dir ./production_images \
  --output-dir ./results
```

## Installation

### 1. Create Conda Environment

```bash
# Create environment from YAML
conda env create -f environment_ivl35.yml

# OR use the unified environment
conda env create -f environment.yml

# Activate
conda activate vision_notebooks
```

### 2. Install Flash Attention (Optional, Recommended)

```bash
# Required for H200/A100/L40S GPUs
pip install flash-attn --no-build-isolation
```

### 3. Model Download

Download InternVL3.5-8B from Hugging Face:

```bash
# Using huggingface-cli
huggingface-cli download OpenGVLab/InternVL3_5-8B --local-dir /models/InternVL3_5-8B

# Or using git-lfs
git lfs install
git clone https://huggingface.co/OpenGVLab/InternVL3_5-8B /models/InternVL3_5-8B
```

**Common model locations:**
- `/home/jovyan/nfs_share/models/InternVL3_5-8B` (JupyterHub)
- `/models/InternVL3_5-8B` (Docker/Kubernetes)
- `./models/InternVL3_5-8B` (Local development)

### 4. GPU Requirements

| GPU | VRAM | Recommended Settings |
|-----|------|---------------------|
| H200/H100 | 80GB | `--dtype bfloat16 --flash-attn --max-tiles 11` |
| A100 | 40-80GB | `--dtype bfloat16 --flash-attn --max-tiles 11` |
| L40S | 48GB | `--dtype bfloat16 --flash-attn --max-tiles 11` |
| V100 | 32GB | `--dtype float32 --no-flash-attn --max-tiles 14` |
| V100 | 16GB | `--dtype float32 --no-flash-attn --max-tiles 6` |

## Running the Pipeline

### Jupyter Notebook

**When to use:** Development, experimentation, debugging, visualization review

1. Open `ivl3_5_8b.ipynb`
2. Edit the CONFIG cell to set paths:
   ```python
   CONFIG = {
       'MODEL_PATH': '/path/to/InternVL3_5-8B',
       'DATA_DIR': './evaluation_data/travel',
       'GROUND_TRUTH': './evaluation_data/travel/ground_truth_travel.csv',
       # ... other settings
   }
   ```
3. Run all cells: `Kernel > Restart & Run All`

### CLI Script

**When to use:** Kubeflow pipelines, batch processing, automation, CI/CD

#### Basic Commands

```bash
# Evaluation mode (with ground truth)
python ivl3_cli.py \
  --data-dir ./evaluation_data/travel \
  --output-dir ./output \
  --ground-truth ./evaluation_data/travel/ground_truth_travel.csv

# Inference-only mode (no ground truth)
python ivl3_cli.py \
  --data-dir ./production_images \
  --output-dir ./results

# Quick test run (2 images, no visualizations)
python ivl3_cli.py \
  --data-dir ./test_images \
  --output-dir ./tmp \
  --max-images 2 \
  --no-viz \
  --no-reports

# Using config file
python ivl3_cli.py --config run_config.yaml

# V100 configuration
python ivl3_cli.py \
  --data-dir ./data \
  --output-dir ./output \
  --max-tiles 14 \
  --no-flash-attn \
  --dtype float32

# Process specific document types
python ivl3_cli.py \
  --data-dir ./mixed_documents \
  --output-dir ./output \
  --document-types "invoice,receipt"
```

#### CLI Options Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--data-dir`, `-d` | Path | **Required** | Directory containing images |
| `--output-dir`, `-o` | Path | **Required** | Output directory for results |
| `--config`, `-c` | Path | None | YAML configuration file |
| `--model-path`, `-m` | Path | Auto-detect | Path to InternVL3.5-8B model |
| `--ground-truth`, `-g` | Path | None | Ground truth CSV (omit for inference) |
| `--max-images` | int | None | Limit number of images (None = all) |
| `--document-types` | str | None | Filter: "invoice,receipt,bank_statement" |
| `--bank-v2/--no-bank-v2` | bool | True | V2 bank statement extraction |
| `--balance-correction/--no-balance-correction` | bool | True | Balance validation for bank statements |
| `--max-tiles` | int | 11 | Image tiles (H200: 11, V100: 14) |
| `--flash-attn/--no-flash-attn` | bool | True | Flash Attention 2 |
| `--dtype` | str | bfloat16 | Torch dtype (bfloat16, float16, float32) |
| `--no-viz` | flag | False | Skip visualization generation |
| `--no-reports` | flag | False | Skip report generation |
| `--verbose/-v`, `--quiet/-q` | bool | True | Verbose output |
| `--version`, `-V` | flag | False | Show version and exit |

## Configuration

### Priority Order (Highest to Lowest)

1. CLI arguments
2. Config file (`--config`)
3. Environment variables (`IVL_*` prefix)
4. Built-in defaults

### YAML Config File

Create a `run_config.yaml` file:

```yaml
# Model configuration
model:
  path: /models/InternVL3_5-8B
  max_tiles: 11
  flash_attn: true
  dtype: bfloat16

# Data paths
data:
  dir: ./evaluation_data/travel
  ground_truth: ./evaluation_data/travel/ground_truth_travel.csv
  max_images: null  # null = process all
  document_types: null  # null = all types

# Output settings
output:
  dir: ./output
  skip_visualizations: false
  skip_reports: false

# Processing options
processing:
  bank_v2: true
  balance_correction: true
  verbose: true
```

Run with config:
```bash
python ivl3_cli.py --config run_config.yaml
```

Override specific settings:
```bash
python ivl3_cli.py --config run_config.yaml --max-images 5 --no-viz
```

### Environment Variables

All settings can be configured via environment variables with the `IVL_` prefix:

| Environment Variable | CLI Equivalent |
|---------------------|----------------|
| `IVL_DATA_DIR` | `--data-dir` |
| `IVL_OUTPUT_DIR` | `--output-dir` |
| `IVL_MODEL_PATH` | `--model-path` |
| `IVL_GROUND_TRUTH` | `--ground-truth` |
| `IVL_MAX_IMAGES` | `--max-images` |
| `IVL_MAX_TILES` | `--max-tiles` |
| `IVL_FLASH_ATTN` | `--flash-attn` (true/false) |
| `IVL_DTYPE` | `--dtype` |
| `IVL_BANK_V2` | `--bank-v2` (true/false) |
| `IVL_VERBOSE` | `--verbose` (true/false) |

Example usage in Kubeflow/containers:
```bash
export IVL_MODEL_PATH=/models/InternVL3_5-8B
export IVL_MAX_TILES=11
export IVL_FLASH_ATTN=true

python ivl3_cli.py --data-dir ./data --output-dir ./output
```

## Hardware Configurations

### H200/A100 (Recommended)

```bash
python ivl3_cli.py \
  --data-dir ./data \
  --output-dir ./output \
  --dtype bfloat16 \
  --flash-attn \
  --max-tiles 11
```

**Features:**
- Native bfloat16 support
- Flash Attention 2 for optimized inference
- 11 tiles for high-resolution OCR
- ~2-3 seconds per image

### V100 (Legacy)

```bash
python ivl3_cli.py \
  --data-dir ./data \
  --output-dir ./output \
  --dtype float32 \
  --no-flash-attn \
  --max-tiles 14
```

**Features:**
- float32 required (no bfloat16 support)
- No Flash Attention (not supported)
- 14 tiles compensates for precision loss
- ~5-8 seconds per image

## Output Files

The pipeline generates the following output structure:

```
output_dir/
├── batch_results/
│   └── batch_results_20240215_143022.json     # Raw extraction results
├── csv/
│   ├── batch_20240215_143022_results.csv      # Per-image results
│   ├── batch_20240215_143022_summary.csv      # Summary statistics
│   ├── batch_20240215_143022_doctype_stats.csv # By document type
│   └── internvl3_5_8b_batch_results_*.csv     # Model comparison format
├── visualizations/
│   ├── dashboard_20240215_143022.png          # 2x2 performance dashboard
│   └── field_accuracy_heatmap_*.png           # Field-level accuracy
└── reports/
    └── batch_report_20240215_143022.md        # Markdown report
```

### CSV Files

**results.csv** - Per-image extraction results:
- `file`: Image filename
- `document_type`: Detected document type
- `accuracy`: Overall accuracy (F1 score)
- `processing_time`: Time in seconds
- Per-field columns with extracted values

**summary.csv** - Aggregate statistics:
- Mean, median, min, max accuracy
- Processing throughput (images/minute)
- Field-level statistics

**doctype_stats.csv** - Accuracy by document type:
- Per-type accuracy and count
- Processing time by type

### Visualizations

**dashboard.png** - 2x2 performance overview:
- Accuracy distribution histogram
- Processing time distribution
- Accuracy vs. time scatter plot
- Document type comparison

**field_accuracy_heatmap.png** - Field extraction accuracy:
- Heat map showing accuracy per field
- Color-coded: green (>90%), yellow (70-90%), red (<70%)

## Document Types Supported

| Document Type | Key Fields |
|--------------|------------|
| **Invoice** | BUSINESS_ABN, INVOICE_DATE, INVOICE_NUMBER, GST_AMOUNT, TOTAL_AMOUNT, SUBTOTAL, LINE_ITEMS |
| **Receipt** | BUSINESS_ABN, RECEIPT_DATE, RECEIPT_NUMBER, GST_AMOUNT, TOTAL_AMOUNT, PAYMENT_METHOD |
| **Bank Statement** | STATEMENT_DATE_RANGE, ACCOUNT_BALANCE, OPENING_BALANCE, CLOSING_BALANCE, TRANSACTION_DETAILS |
| **Travel Expense** | PASSENGER_NAME, TRAVEL_MODE, DEPARTURE_LOCATION, ARRIVAL_LOCATION, TRAVEL_DATE, BOOKING_REFERENCE |

## Troubleshooting

### Model Loading Errors

**Error:** `Model path not found`
```
FATAL: Model path not found: /models/InternVL3_5-8B
```

**Solution:** Specify correct model path:
```bash
python ivl3_cli.py --model-path /correct/path/to/InternVL3_5-8B ...
```

### GPU Memory Issues

**Error:** `CUDA out of memory`

**Solutions:**
1. Reduce max tiles: `--max-tiles 6`
2. Use float16 instead of bfloat16: `--dtype float16`
3. Process fewer images at once: `--max-images 10`

### Flash Attention Errors

**Error:** `Flash Attention not available`

**Solution:** Disable flash attention for V100 or CPU:
```bash
python ivl3_cli.py --no-flash-attn --dtype float32 ...
```

### Missing Dependencies

**Error:** `ModuleNotFoundError: No module named 'typer'`

**Solution:** Update environment:
```bash
conda env update -f environment.yml --prune
# OR
pip install typer
```

### No Images Found

**Error:** `FATAL: No images found in: ./data`

**Solution:** Check image extensions. Supported formats:
- `.png`, `.jpg`, `.jpeg`
- `.tiff`, `.tif`
- `.bmp`, `.webp`

## Examples

### Evaluation Pipeline

Complete evaluation with ground truth comparison:

```bash
python ivl3_cli.py \
  --data-dir ./evaluation_data/invoice \
  --output-dir ./eval_results \
  --ground-truth ./evaluation_data/invoice/ground_truth.csv \
  --verbose
```

### Production Inference

High-throughput inference without evaluation:

```bash
python ivl3_cli.py \
  --data-dir /mnt/incoming_documents \
  --output-dir /mnt/extracted_data \
  --no-viz \
  --no-reports \
  --quiet
```

### Kubeflow Pipeline Step

```python
# kubeflow_component.py
import subprocess

def extract_documents(data_dir: str, output_dir: str, model_path: str):
    subprocess.run([
        "python", "ivl3_cli.py",
        "--data-dir", data_dir,
        "--output-dir", output_dir,
        "--model-path", model_path,
        "--no-viz",
        "--quiet"
    ], check=True)
```

### Docker Usage

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "ivl3_cli.py"]
```

```bash
docker run -v /data:/data -v /output:/output myimage \
  --data-dir /data \
  --output-dir /output \
  --model-path /models/InternVL3_5-8B
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Configuration/validation error |
| 2 | Model loading error |
| 3 | Processing error |

## Related Files

| File | Purpose |
|------|---------|
| `ivl3_5_8b.ipynb` | Interactive notebook |
| `ivl3_cli.py` | CLI script |
| `environment.yml` | Conda environment |
| `environment_ivl35.yml` | IVL-specific environment |
| `config/field_definitions.yaml` | Field definitions per document type |
| `prompts/internvl3_prompts.yaml` | Extraction prompts |
