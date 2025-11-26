# Llama Bank Statement Batch Extraction and Evaluation

Batch processing script for extracting and evaluating bank statement data using Llama-3.2-Vision with the independent single-turn approach.

## Overview

This script runs the full extraction pipeline on multiple bank statement images:
1. **Turn 0**: Header detection
2. **Turn 0.5**: Date format classification (date-per-row vs date-grouped)
3. **Turn 1**: Table extraction
4. **Balance validation**: Auto-correct debit/credit misalignment
5. **Python filtering**: Extract debit transactions only
6. **Evaluation**: Compare against ground truth with configurable scoring

## Fields Evaluated

| Field | Description |
|-------|-------------|
| `DOCUMENT_TYPE` | Always "BANK_STATEMENT" |
| `STATEMENT_DATE_RANGE` | Date range (e.g., "18 Mar 2024 - 14 Apr 2024") |
| `TRANSACTION_DATES` | Pipe-delimited dates |
| `LINE_ITEM_DESCRIPTIONS` | Pipe-delimited transaction descriptions |
| `TRANSACTION_AMOUNTS_PAID` | Pipe-delimited debit amounts |

## Usage

### Basic Usage

```bash
# Run with default settings (order_aware_f1 evaluation)
python llama_bank_statement_batch.py
```

### Evaluation Methods

```bash
# Order-aware F1 (default) - position-sensitive, strictest
python llama_bank_statement_batch.py --method order_aware_f1

# Position-agnostic F1 - set-based matching, more lenient
python llama_bank_statement_batch.py --method position_agnostic_f1

# KIEval - error-type differentiation (substitution, addition, deletion)
python llama_bank_statement_batch.py --method kieval

# Correlation-aware F1 - validates cross-list alignment
python llama_bank_statement_batch.py --method correlation
```

### Limiting Images

```bash
# Process only first 3 images (for testing)
python llama_bank_statement_batch.py --max-images 3

# Process 5 images with verbose output
python llama_bank_statement_batch.py --max-images 5 --verbose
```

### Dry Run

```bash
# Show what would be processed without running
python llama_bank_statement_batch.py --dry-run
```

### Combined Options

```bash
# Full example: 5 images, correlation scoring, verbose output
python llama_bank_statement_batch.py --max-images 5 --method correlation --verbose
```

## Output Files

All output files are saved to `output/` directory with timestamp:

| File | Description |
|------|-------------|
| `llama_bank_statement_batch_{timestamp}.csv` | Per-image results with extracted fields and F1 scores |
| `llama_bank_statement_evaluation_{timestamp}.json` | Full evaluation data with metadata |
| `llama_bank_statement_summary_{timestamp}.md` | Executive summary with accuracy statistics |

## Evaluation Method Comparison

| Method | Description | Use When |
|--------|-------------|----------|
| `order_aware_f1` | Items must match in value AND position | Order matters (default for transactions) |
| `position_agnostic_f1` | Items only need to match in value | Order doesn't matter |
| `kieval` | Measures "effort to fix" errors | Understanding error patterns |
| `correlation` | Validates dates↔descriptions↔amounts alignment | Bank statements (recommended) |

## Configuration

Edit `CONFIG` in the script to change default paths:

```python
CONFIG = {
    "DATA_DIR": Path("/path/to/evaluation_data"),
    "GROUND_TRUTH": Path("/path/to/ground_truth.csv"),
    "OUTPUT_BASE": Path("/path/to/output"),
    "MODEL_PATH": "/path/to/Llama-3.2-11B-Vision-Instruct",
    ...
}
```

## Example Output

```
============================================================
LLAMA BANK STATEMENT BATCH EXTRACTION
============================================================
Evaluation Method: order_aware_f1

📊 Loading ground truth...
✅ Found 12 bank statement images

[1/12] Processing: cba_date_grouped.png
  Turn 0: 5 headers detected
  Turn 0.5: Date-grouped format detected
  Turn 1: Table extraction complete
  Parsed: 10 rows, 10 debits
  ✅ Accuracy: 85.2%
  ⏱️  Time: 12.34s

...

============================================================
BATCH SUMMARY
============================================================
Total: 12 images
Successful: 12
Failed: 0

Accuracy Statistics:
  Average: 78.5%
  Min: 65.2%
  Max: 92.1%
```

## Requirements

- Python 3.11+
- PyTorch with CUDA
- transformers
- Llama-3.2-11B-Vision-Instruct model
- Ground truth CSV at configured path
