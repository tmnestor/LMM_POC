# InternVL3.5-8B Batch V2 Prompts Documentation

This document describes all prompts used by `notebooks_v2/ivl3_5_8b_batch_v2.ipynb` for document extraction.

## Overview

The notebook uses a **multi-turn, document-aware extraction pipeline** with the following flow:

```
Image → Turn 0 (Detection) → Route by Document Type → Extraction
                                    │
                                    ├── INVOICE/RECEIPT: Single-turn extraction
                                    │
                                    └── BANK_STATEMENT: Multi-turn extraction
                                            ├── Turn 1: Header detection
                                            └── Turn 2: Strategy-based extraction
```

## Prompt Sources

| Purpose | File | Used By |
|---------|------|---------|
| Document type detection | `prompts/document_type_detection.yaml` | All documents (Turn 0) |
| Invoice/Receipt extraction | `prompts/internvl3_prompts.yaml` | Invoices and receipts |
| Bank statement extraction | `config/bank_prompts.yaml` | Bank statements (V2 multi-turn) |

---

## Turn 0: Document Type Detection

**Source**: `prompts/document_type_detection.yaml` (key: `detection`)

**Purpose**: Classify the document type before selecting the appropriate extraction strategy.

### Prompt Text

```
What type of business document is this?

Answer with one of:
- INVOICE (includes bills, quotes, estimates)
- RECEIPT (includes purchase receipts)
- BANK_STATEMENT (includes credit card statements)
```

### Response Normalization

The system normalizes various response formats to canonical types:

| User Variations | Normalized To |
|-----------------|---------------|
| `invoice`, `tax invoice`, `bill`, `estimate`, `quote` | `INVOICE` |
| `receipt`, `purchase receipt`, `payment receipt` | `RECEIPT` |
| `bank statement`, `account statement`, `credit card statement` | `BANK_STATEMENT` |

---

## Invoice Extraction (Single-Turn)

**Source**: `prompts/internvl3_prompts.yaml` (key: `invoice`)

**Fields Extracted**: 14

### Prompt Text

```
Extract ALL data from this invoice image. Respond in exact format below with actual values or NOT_FOUND.

DOCUMENT_TYPE: INVOICE
BUSINESS_ABN: NOT_FOUND
SUPPLIER_NAME: NOT_FOUND
BUSINESS_ADDRESS: NOT_FOUND
PAYER_NAME: NOT_FOUND
PAYER_ADDRESS: NOT_FOUND
INVOICE_DATE: NOT_FOUND
LINE_ITEM_DESCRIPTIONS: NOT_FOUND
LINE_ITEM_QUANTITIES: NOT_FOUND
LINE_ITEM_PRICES: NOT_FOUND
LINE_ITEM_TOTAL_PRICES: NOT_FOUND
IS_GST_INCLUDED: NOT_FOUND
GST_AMOUNT: NOT_FOUND
TOTAL_AMOUNT: NOT_FOUND

Instructions:
- Find ABN: 11 digits like "12 345 678 901"
- Find supplier: Business name at top
- Find customer: "Bill To" section
- Find date: Use DD/MM/YYYY format
- Find line items: List with " | " separator
- Find amounts: Include $ symbol
- Replace NOT_FOUND with actual values
```

---

## Receipt Extraction (Single-Turn)

**Source**: `prompts/internvl3_prompts.yaml` (key: `receipt`)

**Fields Extracted**: 14

### Prompt Text

```
Extract ALL data from this receipt image. Respond in exact format below with actual values or NOT_FOUND.

DOCUMENT_TYPE: RECEIPT
BUSINESS_ABN: NOT_FOUND
SUPPLIER_NAME: NOT_FOUND
BUSINESS_ADDRESS: NOT_FOUND
PAYER_NAME: NOT_FOUND
PAYER_ADDRESS: NOT_FOUND
INVOICE_DATE: NOT_FOUND
LINE_ITEM_DESCRIPTIONS: NOT_FOUND
LINE_ITEM_QUANTITIES: NOT_FOUND
LINE_ITEM_PRICES: NOT_FOUND
LINE_ITEM_TOTAL_PRICES: NOT_FOUND
IS_GST_INCLUDED: NOT_FOUND
GST_AMOUNT: NOT_FOUND
TOTAL_AMOUNT: NOT_FOUND

Instructions:
- Find ABN: 11 digits like "12 345 678 901"
- Find store: Business name at top
- Find transaction date: Use DD/MM/YYYY format
- Find purchased items: List with " | " separator
- Find amounts: Include $ symbol
- Replace NOT_FOUND with actual values
```

---

## Bank Statement Extraction (V2 Multi-Turn)

When `USE_SOPHISTICATED_BANK_EXTRACTION=True` (default), bank statements use a sophisticated multi-turn extraction via `BankStatementAdapter`.

### Turn 1: Header Detection

**Source**: `config/bank_prompts.yaml` (key: `turn0_header_detection`)

**Purpose**: Identify the exact column names used in the transaction table.

```
Look at the transaction table in this bank statement image.

What are the exact column header names used in the transaction table?

List each column header exactly as it appears, in order from left to right.
Do not interpret or rename them - use the EXACT text from the image.
```

### Strategy Selection

Based on detected headers, the system automatically selects an extraction strategy:

| Detected Columns | Strategy | Use Case |
|------------------|----------|----------|
| Balance + Debit/Credit | `BALANCE_DESCRIPTION` | CBA-style with running balance |
| Amount (signed) | `AMOUNT_DESCRIPTION` | Single amount column (negative = debit) |
| Debit + Credit (no Balance) | `DEBIT_CREDIT_DESCRIPTION` | Separate debit/credit columns |

### Turn 2: Balance-Description Extraction

**Source**: `config/bank_prompts.yaml` (key: `turn1_balance_extraction`)

**Used When**: Statement has Balance + Debit/Credit columns

**Template** (with dynamic placeholders):

```
List all the balances in the {balance_col} column, including:
- Date from the Date Header of the balance
- {desc_col}
- {debit_col} Amount or "NOT_FOUND"
- {credit_col} Amount or "NOT_FOUND"

Format each balance entry like this:
1. **[Date]**
   - {desc_col}: [ALL rows of text for this transaction]
   - {debit_col}: [amount or NOT_FOUND]
   - {credit_col}: [amount or NOT_FOUND]
   - {balance_col}: [balance amount]

CRITICAL RULES:
1. List EVERY balance entry in order from top to bottom
2. EVERY balance entry has a date, either on the same row, or above
3. Include the FULL {desc_col} text - capture ALL rows that belong to this transaction
4. If amount is in {debit_col} column, put it there and use NOT_FOUND for {credit_col}
5. If amount is in {credit_col} column, put it there and use NOT_FOUND for {debit_col}
6. Do NOT skip any transactions
7. Use the DATE from the Date column ONLY - do NOT use "Value Date:" as the transaction date
8. Amounts in {balance_col} may include a CR suffix, but amounts in {debit_col} and {credit_col} NEVER have CR suffix
```

### Turn 2: Amount-Description Extraction

**Source**: `config/bank_prompts.yaml` (key: `turn1_amount_extraction`)

**Used When**: Statement has signed Amount column (negative = withdrawal)

**Template** (with dynamic placeholders):

```
List all transactions from this bank statement, including:
- Date
- {desc_col}
- {amount_col} (preserve the sign: negative = withdrawal, positive = deposit)
{balance_line}

Format each entry like this:
1. **[Date]**
   - {desc_col}: [ALL rows of text for this transaction]
   - {amount_col}: [amount with sign preserved]
   {balance_format}

CRITICAL RULES:
1. List EVERY transaction in order from top to bottom
2. EVERY entry has a date, either on the same row, or above
3. Include the FULL {desc_col} text - capture ALL rows, not abbreviated
4. PRESERVE the sign of amounts (negative = withdrawal, positive = deposit)
5. Do NOT skip any transactions
```

### Turn 2: Debit-Credit Extraction

**Source**: `config/bank_prompts.yaml` (key: `turn1_debit_credit_extraction`)

**Used When**: Statement has separate Debit/Credit columns but no Balance column

**Template** (with dynamic placeholders):

```
List all transactions from this bank statement, including:
- Date
- {desc_col}
- {debit_col} Amount or "NOT_FOUND"
- {credit_col} Amount or "NOT_FOUND"

Format each entry like this:
1. **[Date]**
   - {desc_col}: [description text]
   - {debit_col}: [amount or NOT_FOUND]
   - {credit_col}: [amount or NOT_FOUND]

CRITICAL RULES:
1. List EVERY transaction in order from top to bottom
2. EVERY entry has a date, either on the same row, or above
3. Include the FULL description text, not abbreviated
4. If amount is in {debit_col} column, put it there and use NOT_FOUND for {credit_col}
5. If amount is in {credit_col} column, put it there and use NOT_FOUND for {debit_col}
6. Do NOT skip any actual transactions - extract them ALL
7. SKIP "Opening balance" and "Closing balance" rows - these are NOT transactions
8. Use the DATE from the Date column ONLY
```

### Fallback: Schema-Based Extraction

**Source**: `config/bank_prompts.yaml` (key: `schema_fallback_extraction`)

**Used When**: Header detection fails (returns garbage like days of week, ATM locations)

```
Extract ALL data from this bank statement image. Respond in exact format below with actual values or NOT_FOUND.

DOCUMENT_TYPE: BANK_STATEMENT
STATEMENT_DATE_RANGE: NOT_FOUND
TRANSACTION_DATES: NOT_FOUND
LINE_ITEM_DESCRIPTIONS: NOT_FOUND
TRANSACTION_AMOUNTS_PAID: NOT_FOUND

Instructions:
- STATEMENT_DATE_RANGE: Find the statement period (e.g., "01/01/2024 - 31/01/2024")
- TRANSACTION_DATES: List ALL transaction dates separated by " | "
- LINE_ITEM_DESCRIPTIONS: List ALL transaction descriptions separated by " | "
- TRANSACTION_AMOUNTS_PAID: List ALL debit/withdrawal amounts separated by " | "

CRITICAL RULES:
1. Extract EVERY transaction from the statement
2. Keep items in the same order across all fields
3. Use " | " as separator between items
4. Include $ symbol for amounts
5. Replace NOT_FOUND with actual values
```

---

## Prompt Design Principles

The prompts in this notebook follow two critical rules:

### 1. Content-Generic Prompts

Prompts never reference specific transaction data, file names, or specific content from images.

**Correct**: `"Extract all transactions"`
**Wrong**: `"Extract the PIZZA HUT transaction for $97.95"`

### 2. Structure-Dynamic Prompts

Prompts adapt to detected column structure using dynamic placeholders.

**Correct**: `f"If amount is under the {debit_col} header..."`
**Wrong**: `"If amount is under the Debit header..."` (hardcoded)

This allows the same prompts to work across different bank statement formats (CBA, ANZ, Westpac, etc.) that use different column naming conventions.

---

## Configuration Options

### `CONFIG` Settings (in notebook Cell 4)

| Setting | Default | Description |
|---------|---------|-------------|
| `USE_SOPHISTICATED_BANK_EXTRACTION` | `True` | Enable multi-turn bank extraction |
| `ENABLE_BALANCE_CORRECTION` | `True` | Mathematical validation using balance deltas |
| `MAX_NEW_TOKENS` | `2000` | Maximum tokens for model response |
| `MAX_TILES` | `11` | Image tiles for dense OCR (InternVL3.5 max) |

### `PROMPT_CONFIG` (in notebook Cell 4)

```python
PROMPT_CONFIG = {
    'detection_file': 'prompts/document_type_detection.yaml',
    'detection_key': 'detection',
    'extraction_files': {
        'INVOICE': 'prompts/internvl3_prompts.yaml',
        'RECEIPT': 'prompts/internvl3_prompts.yaml',
        'BANK_STATEMENT': 'prompts/internvl3_prompts.yaml'  # Fallback only
    },
}
```

---

## Fields Extracted by Document Type

### Invoice/Receipt (14 fields)

| Field | Description |
|-------|-------------|
| `DOCUMENT_TYPE` | INVOICE or RECEIPT |
| `BUSINESS_ABN` | 11-digit Australian Business Number |
| `SUPPLIER_NAME` | Business issuing the invoice |
| `BUSINESS_ADDRESS` | Supplier address |
| `PAYER_NAME` | Customer/recipient name |
| `PAYER_ADDRESS` | Customer address |
| `INVOICE_DATE` | Document date (DD/MM/YYYY) |
| `LINE_ITEM_DESCRIPTIONS` | Item descriptions (pipe-separated) |
| `LINE_ITEM_QUANTITIES` | Quantities (pipe-separated) |
| `LINE_ITEM_PRICES` | Unit prices (pipe-separated) |
| `LINE_ITEM_TOTAL_PRICES` | Line totals (pipe-separated) |
| `IS_GST_INCLUDED` | true/false |
| `GST_AMOUNT` | GST amount with $ |
| `TOTAL_AMOUNT` | Total amount with $ |

### Bank Statement (5 fields for evaluation)

| Field | Description |
|-------|-------------|
| `DOCUMENT_TYPE` | BANK_STATEMENT |
| `STATEMENT_DATE_RANGE` | Period covered |
| `TRANSACTION_DATES` | All dates (pipe-separated) |
| `LINE_ITEM_DESCRIPTIONS` | Transaction descriptions (pipe-separated) |
| `TRANSACTION_AMOUNTS_PAID` | Debit amounts (pipe-separated) |

**Note**: `ACCOUNT_BALANCE` is extracted for mathematical validation but excluded from evaluation metrics.

---

## Version History

- **V2**: Multi-turn bank extraction with `BankStatementAdapter`
- **V1**: Single-turn extraction for all document types
