# Bank Statement Extraction Protocol

This document describes how the `*_bank_statement_batch.py` scripts extract and evaluate bank statement fields.

## Overview

The extraction pipeline uses a two-turn approach:
1. **Turn 0**: Detect column headers from the transaction table
2. **Turn 1**: Extract the full transaction table as markdown

Python post-processing then filters and structures the extracted data for evaluation.

## Extracted Fields

| Field | Description | Source |
|-------|-------------|--------|
| `STATEMENT_DATE_RANGE` | Earliest to latest date in transaction table | All rows (includes balance rows) |
| `TRANSACTION_DATES` | Pipe-delimited dates of debit transactions | Debit rows only |
| `LINE_ITEM_DESCRIPTIONS` | Pipe-delimited descriptions of debit transactions | Debit rows only |
| `TRANSACTION_AMOUNTS_PAID` | Pipe-delimited amounts of debit transactions | Debit rows only |

## Key Principle: Extract What You See

**Ground truth reflects what is visually present in the image, not semantic interpretation.**

- Models extract what they *see* in the document
- Business logic filtering (e.g., "only debits") is applied in Python post-processing
- Evaluation compares filtered output against filtered ground truth

## Extraction Pipeline

### Step 1: Model Extracts All Visible Rows

The vision-language model extracts the complete transaction table, including:
- Opening Balance rows
- Closing Balance rows
- Brought Forward / Carried Forward rows (NAB format)
- Credit transactions (deposits, refunds)
- Debit transactions (purchases, withdrawals)

### Step 2: Python Parses Markdown Table

```python
all_rows = parse_markdown_table(model_response)
```

`all_rows` contains every row the model extracted from the table.

### Step 3: Python Filters to Debit Transactions

```python
debit_rows = filter_debit_transactions(all_rows, debit_col, desc_col)
```

The `filter_debit_transactions()` function excludes:
- Rows without a debit amount
- Non-transaction rows (identified by `is_non_transaction_row()`)

### Step 4: Non-Transaction Row Detection

```python
def is_non_transaction_row(row, desc_col):
    """Excludes balance rows from debit filtering."""
    desc = row.get(desc_col, "").strip().upper()

    if "OPENING BALANCE" in desc:
        return True
    if "CLOSING BALANCE" in desc:
        return True
    if "BROUGHT FORWARD" in desc:
        return True
    if "CARRIED FORWARD" in desc:
        return True

    return False
```

### Step 5: Schema Field Extraction

```python
schema_fields = extract_schema_fields(
    debit_rows,      # Filtered debit transactions
    date_col,
    desc_col,
    debit_col,
    all_rows=all_rows  # All rows for date range
)
```

**Critical distinction:**
- `STATEMENT_DATE_RANGE`: Calculated from `all_rows` (full date span visible in table)
- Other fields: Calculated from `debit_rows` (filtered transactions only)

## Date Range Calculation

```python
# Uses ALL rows (includes balance rows)
rows_for_range = all_rows if all_rows is not None else debit_rows
all_dates = [row.get(date_col, "").strip() for row in rows_for_range]
all_dates = [d for d in all_dates if d]  # Filter empty
date_range = f"{all_dates[0]} - {all_dates[-1]}"
```

This ensures `STATEMENT_DATE_RANGE` captures the full visible date span, including:
- Opening Balance date (often the earliest)
- Closing Balance date (often the latest)

## Ground Truth Format

Ground truth in `ground_truth_bank.csv` follows the same logic:

| Field | Ground Truth Contains |
|-------|----------------------|
| `STATEMENT_DATE_RANGE` | Full date span (includes balance row dates) |
| `TRANSACTION_DATES` | Debit transaction dates only |
| `LINE_ITEM_DESCRIPTIONS` | Debit descriptions only |
| `TRANSACTION_AMOUNTS_PAID` | Debit amounts only |

### Date Format Matching

Ground truth dates use the **exact format visible in each image**:

| Image | Date Format | Example |
|-------|-------------|---------|
| cba_date_grouped.png | DD Mon YYYY | 18 Mar 2024 |
| cba_debit_credit.png | DD Mon (no year) | 20 May |
| westpac_debit_credit.png | DD Mon YY | 28 Nov 23 |
| image_003.png | DD/MM/YYYY | 03/05/2025 |

## Evaluation Methods

The scripts support multiple evaluation methods via `--method`:

### order_aware_f1 (default)
- Items must match in **value AND position**
- Strictest evaluation
- Best for: Verifying extraction order matches document order

### position_agnostic_f1
- Items only need to match in **value**
- Set-based comparison
- Best for: When order doesn't matter

### kieval
- Measures "effort to fix" errors
- Differentiates: substitution, addition, deletion
- Best for: Understanding error patterns

### correlation
- Validates cross-field alignment (dates <-> descriptions <-> amounts)
- Best for: Bank statements where row correlation matters

## Example: cba_date_grouped.png

**What the model sees:**
```
| Date | Transaction | Debit | Credit | Balance |
|------|-------------|-------|--------|---------|
| 15 Mar 2024 | OPENING BALANCE | | | $3,156.28 CR |
| 18 Mar 2024 | TELSTRA MOBILE SERVICES | 48.50 | | $3,107.78 CR |
| 19 Mar 2024 | WOOLWORTHS 2847 | 127.35 | | $2,980.43 CR |
| ... | ... | ... | ... | ... |
| 14 Apr 2024 | CHEMIST WAREHOUSE | 54.64 | | $2,847.92 CR |
```

**Extracted fields:**
- `STATEMENT_DATE_RANGE`: `15 Mar 2024 - 14 Apr 2024` (includes Opening Balance date)
- `TRANSACTION_DATES`: `18 Mar 2024 | 19 Mar 2024 | ... | 14 Apr 2024` (debits only)
- `LINE_ITEM_DESCRIPTIONS`: `TELSTRA MOBILE SERVICES... | WOOLWORTHS 2847... | ...`
- `TRANSACTION_AMOUNTS_PAID`: `$48.50 | $127.35 | ... | $54.64`

## CLI Usage

```bash
# Basic usage
python llama_bank_statement_batch.py

# With evaluation method
python llama_bank_statement_batch.py --method correlation

# Limit images for testing
python llama_bank_statement_batch.py --max-images 3 --verbose

# Log output to file
python llama_bank_statement_batch.py --log-file output/run.log

# Dry run (show what would be processed)
python llama_bank_statement_batch.py --dry-run
```

## Scripts

| Script | Model |
|--------|-------|
| `llama_bank_statement_batch.py` | Llama-3.2-11B-Vision |
| `ivl3_8b_bank_statement_batch.py` | InternVL3-8B |
| `ivl3_5_8b_bank_statement_batch.py` | InternVL3.5-8B |

All scripts follow the same extraction protocol and produce comparable results.
