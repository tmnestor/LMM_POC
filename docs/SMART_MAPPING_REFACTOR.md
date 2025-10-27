# Smart Mapping Refactor - Transaction Extraction System

**Date:** 2025-10-27
**Version:** 1.0
**Author:** Tod Nestor

## Overview

This document describes the comprehensive refactor of the bank statement transaction extraction system to implement a **smart header mapping approach** with a **four-stage processing pipeline**.

## Problem Statement

### Original Challenge
- Different banks use different column names for the same semantic fields
- "Description" might be called "Transaction Details", "Particulars", "Details", "Narrative", etc.
- "Debit" might be called "Withdrawal", "Debit Amount", "Money Out", etc.
- Vision-Language Models (VLMs) struggle when prompts use generic field names that don't match the actual column headers in the image

### Solution Requirements
1. Classify document type (Mobile App vs Bank Statement) before processing
2. Extract actual column headers from bank statement images
3. Map detected headers to semantic field types using fuzzy matching
4. Generate dynamic extraction prompts using the actual column names from the image
5. Extract only Date, Description, and Debit columns in pipe-separated format

## Architecture

### Four-Stage Processing Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT IMAGE                          │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 1: DOCUMENT TYPE CLASSIFICATION                  │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Prompt: document_type_classifier.yaml             │  │
│  │ Output: "Mobile_APP" or "BANK_STATEMENT"          │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
           ┌──────────┴──────────┐
           │                     │
           ▼                     ▼
    Mobile_APP            BANK_STATEMENT
    (Skip extraction)           │
                                ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 2: HEADER EXTRACTION                             │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Prompt: bank_statement_structure_classifier.yaml  │  │
│  │ Output: "Date | Transaction | Debit | Credit..."  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 3: SMART HEADER MAPPING                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Module: common/header_mapping.py                  │  │
│  │ Function: map_headers_to_fields()                 │  │
│  │                                                    │  │
│  │ Fuzzy matching of detected headers to:            │  │
│  │  - DATE                                           │  │
│  │  - DESCRIPTION                                    │  │
│  │  - DEBIT                                          │  │
│  │  - CREDIT                                         │  │
│  │  - BALANCE                                        │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 4: TRANSACTION EXTRACTION                        │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Template: transaction_extraction_template.yaml    │  │
│  │ Dynamic Prompt Generation using mapped headers    │  │
│  │                                                    │  │
│  │ Output: Pipe-separated transactions               │  │
│  │   Date | Description | Debit                      │  │
│  │   01/06/2024 | ATM Withdrawal | 100.00            │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                    OUTPUT FILES                         │
│  - extracted_transactions.psv                           │
│  - complete_extraction_results.json                     │
└─────────────────────────────────────────────────────────┘
```

## Files Created

### 1. `common/header_mapping.py`
**Purpose:** Smart fuzzy matching module for mapping detected headers to semantic fields

**Key Functions:**

#### `fuzzy_match(text1: str, text2: str) -> float`
- Calculates similarity ratio between two strings (0.0 to 1.0)
- Uses Python's `difflib.SequenceMatcher`
- Case-insensitive matching

#### `find_best_match(target_keywords: list[str], available_headers: list[str], threshold: float = 0.4) -> str | None`
- Finds best matching header from available headers
- Returns None if no match exceeds threshold
- Supports multiple keyword variations per field

#### `map_headers_to_fields(headers_pipe_separated: str) -> dict[str, str | None]`
- **Input:** `"Date | Transaction Details | Debit | Credit | Balance"`
- **Output:**
  ```python
  {
      'DATE': 'Date',
      'DESCRIPTION': 'Transaction Details',
      'DEBIT': 'Debit',
      'CREDIT': 'Credit',
      'BALANCE': 'Balance'
  }
  ```

**Field Patterns Supported:**

| Semantic Field | Recognized Variations |
|----------------|----------------------|
| DATE | Date, Date of Transaction, Transaction Date, Dt, Day |
| DESCRIPTION | Description, Details, Transaction Details, Particulars, Transaction, Narrative, Remarks |
| DEBIT | Debit, Debit Amount, Debit ($), Withdrawal, Withdrawals, Money Out, Spent, Payments |
| CREDIT | Credit, Credit Amount, Credit ($), Deposit, Deposits, Money In, Received, Receipts |
| BALANCE | Balance, Running Balance, Closing Balance, Available Balance, Current Balance, Bal |

#### `validate_mapping(mapping: dict, required_fields: list[str] | None) -> tuple[bool, list[str]]`
- Validates that required fields have been mapped
- Returns (is_valid, missing_fields)
- Default required fields: ['DATE', 'DESCRIPTION']

#### `generate_extraction_instruction(mapping: dict, headers_pipe_separated: str) -> str`
- Generates dynamic extraction instruction using actual column names
- Uses mapped field names in the prompt
- Makes VLM prompts more accurate by referencing actual headers

**Example Output:**
```
Look at the transaction table in this bank statement.

The table has these columns: Date | Transaction Details | Debit | Credit | Balance

Extract ONLY these columns for each transaction row:
- Date (the transaction date)
- Transaction Details (the transaction description/details)
- Debit (the debit/withdrawal amount - leave blank if not present)

IMPORTANT:
- Read each row from top to bottom
- Skip the header row
- Extract EVERY transaction row you see
- For the Debit column, only extract if there is a value
```

### 2. `prompts/document_type_classifier.yaml`
**Purpose:** Binary classification between mobile app transactions and bank statements

**Structure:**
```yaml
name: "Document Type Classifier - Mobile App vs Bank Statement"
version: "1.0"
task: "classification"

instruction: |
  Analyze the image and determine if it shows a mobile banking app or
  a traditional bank statement.

  Look for these characteristics:

  MOBILE APP indicators:
  - Touch-friendly UI elements (buttons, tabs, search bars)
  - Navigation icons (back arrow, menu, filter icons)
  - Mobile status bar (time, battery, signal)
  - Interactive elements (search box, filter button)
  - Scrollable list interface
  - Category icons or merchant logos
  - Free-form transaction layout
  - Modern app design patterns

  BANK STATEMENT indicators:
  - Fixed table structure with rows and columns
  - Formal column headers (Date, Description, Amount, Balance)
  - PDF or printed document appearance
  - Grid-like layout with aligned columns
  - No mobile UI elements
  - Formal typography and spacing
  - Traditional document format

output_format: |
  Mobile_APP

  OR

  BANK_STATEMENT

  Output ONLY one of these two terms, nothing else.
```

### 3. `prompts/transaction_extraction_template.yaml`
**Purpose:** Defines output format for transaction extraction (instruction generated dynamically)

**Structure:**
```yaml
name: "Transaction Extraction - Date, Description, Debit"
version: "1.0"
task: "extraction"
description: "Template for extracting specific columns using detected column names"

output_format: |
  Return as pipe-separated values with exactly these headers:
  Date | Description | Debit

  For each transaction row, output one line in this format:
  {date_value} | {description_value} | {debit_amount}

  Rules:
  - Use " | " (space-pipe-space) as delimiter
  - If debit column is empty/blank for a row, leave the debit field empty
  - Only include debit amounts (withdrawals), not credits (deposits)
  - Extract ALL transaction rows from the table
  - Do NOT include the header row in your output
  - Do NOT add any explanatory text before or after the data

  Example output:
  Date | Description | Debit
  01/06/2024 | EFTPOS Purchase - WOOLWORTHS | 45.60
  01/06/2024 | ATM Withdrawal | 100.00
  02/06/2024 | Direct Debit - ELECTRICITY | 156.78
```

### 4. `notebooks/llama_structure_classifier.ipynb` (Refactored)

**New Structure: 11 Cells**

#### Cell 0: Imports
```python
import sys
from pathlib import Path
import torch, yaml
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration

# Add project root to path
sys.path.insert(0, '/home/jovyan/nfs_share/tod/LMM_POC')
from common.header_mapping import (
    generate_extraction_instruction,
    map_headers_to_fields,
    validate_mapping,
)
```

#### Cell 1: Load Classifier Prompts
Loads both `document_type_classifier.yaml` and `bank_statement_structure_classifier_simple.yaml`

#### Cell 2: Load Llama-3.2-Vision Model
Loads model with bfloat16 and device_map="auto"

#### Cell 3: Load Test Image
Loads bank statement image for testing

#### Cell 4: STAGE 1 - Document Type Classification
- Classifies as `Mobile_APP` or `BANK_STATEMENT`
- Stores result in `document_type` variable

#### Cell 5: STAGE 2 - Conditional Header Extraction
- **IF BANK_STATEMENT:** Extracts headers as pipe-separated string
- **IF Mobile_APP:** Skips extraction, sets `headers_pipe_separated = "N/A"`
- Stores result in `headers_pipe_separated` variable

#### Cell 6: Display Classification Results
Shows document type and headers detected

#### Cell 7: Save Classification Results
Saves to JSON and text files

#### Cell 8: STAGE 3 - Smart Header Mapping
- Maps headers using `map_headers_to_fields()`
- Validates required fields (DATE, DESCRIPTION, DEBIT)
- Displays mapping results
- Sets `can_extract` flag

#### Cell 9: STAGE 4 - Transaction Extraction
- Loads extraction template
- Generates dynamic prompt using `generate_extraction_instruction()`
- Extracts transactions using Llama-3.2-Vision
- Stores result in `extracted_transactions` variable

#### Cell 10: Display and Save Extracted Transactions
- Shows extracted transactions in pipe-separated format
- Counts transaction rows
- Saves to `.psv` file (pipe-separated values)
- Saves complete results to JSON

## Output Format

### Pipe-Separated Values (.psv)
**Codebase Standard:** All list-like outputs use pipe separation (" | ")

**Example Output:**
```
Date | Description | Debit
01/06/2024 | EFTPOS Purchase - WOOLWORTHS | 45.60
01/06/2024 | ATM Withdrawal | 100.00
02/06/2024 | Direct Debit - ELECTRICITY | 156.78
03/06/2024 | Online Transfer | 250.00
```

### Complete Results JSON
```json
{
  "image_path": "/path/to/image_003.png",
  "document_type": "BANK_STATEMENT",
  "headers_detected": "Date | Transaction Details | Debit | Credit | Balance",
  "field_mapping": {
    "DATE": "Date",
    "DESCRIPTION": "Transaction Details",
    "DEBIT": "Debit",
    "CREDIT": "Credit",
    "BALANCE": "Balance"
  },
  "transaction_count": 15,
  "transactions_psv": "Date | Description | Debit\n01/06/2024 | ATM Withdrawal | 100.00\n..."
}
```

## Testing Results

### Fuzzy Matching Tests

**Test Case 1:**
- Input: `"Date | Particulars | Withdrawals | Deposits | Running Balance"`
- Mapping:
  - ✅ DATE: Date
  - ✅ DESCRIPTION: Particulars
  - ✅ DEBIT: Withdrawals
  - Valid: True

**Test Case 2:**
- Input: `"Transaction Date | Description | Debit Amount | Credit Amount | Balance"`
- Mapping:
  - ✅ DATE: Transaction Date
  - ✅ DESCRIPTION: Description
  - ✅ DEBIT: Debit Amount
  - Valid: True

**Test Case 3:**
- Input: `"Dt | Details | Money Out | Money In | Bal"`
- Mapping:
  - ✅ DATE: Dt
  - ✅ DESCRIPTION: Details
  - ✅ DEBIT: Money Out
  - Valid: True

## Usage Instructions

### Running the Notebook (Remote H200/V100)

1. **Ensure files are copied to remote:**
   ```bash
   rsync -av common/header_mapping.py remote:/home/jovyan/nfs_share/tod/LMM_POC/common/
   rsync -av prompts/*.yaml remote:/home/jovyan/nfs_share/tod/LMM_POC/prompts/
   rsync -av notebooks/llama_structure_classifier.ipynb remote:/home/jovyan/nfs_share/tod/LMM_POC/notebooks/
   ```

2. **Open notebook on remote system**

3. **Restart kernel** (Important!)

4. **Run all cells in sequence** (Cells 0-10)

5. **Check outputs:**
   - `classification_results/extracted_transactions.psv`
   - `classification_results/complete_extraction_results.json`

### Batch Processing (Future Enhancement)

The smart mapping approach enables batch processing:

```python
from pathlib import Path
from common.header_mapping import map_headers_to_fields, generate_extraction_instruction

# Process multiple images
for image_path in Path("evaluation_data").glob("*.png"):
    # Stage 1: Classify document type
    document_type = classify_document(image_path)

    if document_type == "BANK_STATEMENT":
        # Stage 2: Extract headers
        headers = extract_headers(image_path)

        # Stage 3: Smart mapping
        mapping = map_headers_to_fields(headers)

        # Stage 4: Extract transactions
        instruction = generate_extraction_instruction(mapping, headers)
        transactions = extract_transactions(image_path, instruction)

        # Save results
        save_results(image_path, transactions)
```

## Benefits of Smart Mapping Approach

### 1. Robustness
- Handles different bank statement formats automatically
- Adapts to naming variations without manual intervention
- Fuzzy matching handles typos and slight variations

### 2. Accuracy
- VLM sees the actual column names from the image in the prompt
- Reduces confusion when prompt uses different terminology than the image
- Direct reference to visible headers improves extraction accuracy

### 3. Maintainability
- Single module (`header_mapping.py`) handles all mapping logic
- Easy to add new field pattern variations
- Centralized validation logic

### 4. Extensibility
- Easy to add new semantic fields (e.g., REFERENCE_NUMBER, LOCATION)
- Can adjust fuzzy match threshold per field type
- Template-based prompt generation allows customization

### 5. Debugging
- Clear mapping results show exactly which columns were matched
- Validation reports missing required fields
- JSON output includes complete mapping for audit trail

## Design Decisions

### Why Fuzzy Matching?
- Banks use inconsistent terminology
- Allows threshold-based matching (default 0.4 similarity)
- More flexible than exact string matching or regex

### Why Pipe-Separated Format?
- Codebase standard for list-like outputs
- Descriptions often contain commas, making CSV problematic
- More readable than CSV with escaped quotes
- Consistent with existing `headers_pipe_separated` convention

### Why Four-Stage Pipeline?
- **Separation of Concerns:** Each stage has a single responsibility
- **Early Exit:** Mobile apps skip extraction entirely
- **Error Isolation:** Failures at one stage don't crash the whole pipeline
- **Reusability:** Each stage can be used independently

### Why Dynamic Prompt Generation?
- Static prompts fail when column names don't match
- Dynamic prompts reference actual visible headers
- Improves VLM understanding and extraction accuracy

## Future Enhancements

### 1. Multi-Column Extraction
Extend to extract additional fields:
- Credits/Deposits
- Balance
- Reference numbers
- Transaction categories

### 2. Batch Processing Script
Create standalone script for processing multiple images:
```bash
python process_bank_statements.py --input evaluation_data/ --output results/
```

### 3. Confidence Scores
Add confidence scores to mapping results:
```python
{
    'DATE': ('Date', 1.0),  # Perfect match
    'DESCRIPTION': ('Particulars', 0.85),  # Good match
    'DEBIT': ('Withdrawals', 0.72)  # Acceptable match
}
```

### 4. Custom Field Patterns
Allow users to add custom field patterns via config:
```yaml
custom_patterns:
  MERCHANT:
    - "Merchant Name"
    - "Store"
    - "Vendor"
```

### 5. Mobile App Transaction Extraction
Develop mobile app-specific extraction logic:
- Different layout patterns
- No fixed table structure
- Icon-based categorization

## Known Limitations

### 1. Fuzzy Matching Threshold
- Default threshold (0.4) may need tuning for specific use cases
- Very short column names (e.g., "Dt" vs "Dr") can cause ambiguity

### 2. Multi-Line Headers
- Current parser assumes single-row headers
- Complex multi-line headers may require enhanced parsing

### 3. Merged Debit/Credit Columns
- Some statements have single "Amount" column with +/- signs
- Current approach assumes separate Debit/Credit columns

### 4. Date Format Variations
- Extracted dates are not normalized
- Different banks use different date formats
- Post-processing may be needed

## Migration Notes

### For Existing Code
If you have existing extraction scripts, update them to:

1. Import the new mapping module:
   ```python
   from common.header_mapping import map_headers_to_fields, generate_extraction_instruction
   ```

2. Add header extraction stage before transaction extraction

3. Use dynamic prompt generation instead of static prompts

4. Switch output format from CSV to pipe-separated

### Breaking Changes
- Output format changed from CSV to pipe-separated (.psv)
- Extraction now requires header detection stage first
- New dependency on `common/header_mapping.py` module

## Conclusion

The smart mapping refactor provides a robust, maintainable, and accurate approach to bank statement transaction extraction. By dynamically adapting to different bank statement formats and using actual column names in prompts, the system significantly improves extraction accuracy while remaining flexible enough to handle diverse document layouts.

The four-stage pipeline (Classification → Header Extraction → Smart Mapping → Transaction Extraction) provides clear separation of concerns and makes the system easier to debug, test, and extend.

---

**Questions or Issues?**
- Review test results in this document
- Check the fuzzy matching threshold if mappings seem incorrect
- Ensure all files are copied to remote system before running
- Verify notebook kernel was restarted after copying new files
