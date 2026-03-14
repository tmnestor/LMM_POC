# Evaluation Toolkit

Standalone evaluation toolkit for scoring document extraction models against ground truth data. Use this to measure your model's field-level accuracy using F1, KIEval, and correlation-aware metrics.

## Directory Structure

```
.
├── evaluate_model.ipynb              # Demo notebook — full evaluation workflow
├── common/
│   ├── __init__.py
│   ├── config.py                     # Field types, thresholds (delegates to field_definitions_loader)
│   ├── field_definitions_loader.py   # Loads field_definitions.yaml (simple config loader)
│   ├── evaluation_metrics.py         # Core scoring functions
│   ├── batch_analytics.py            # Results DataFrames and summary stats
│   ├── batch_reporting.py            # Markdown/JSON report generation
│   └── batch_visualizations.py       # Charts and dashboards
├── config/
│   └── field_definitions.yaml        # Document types, field lists, field type classifications
└── evaluation_data/
    ├── ground_truth.csv
    └── *.png (31 evaluation images)
```

## Setup

```bash
pip install pandas numpy matplotlib seaborn pyyaml rich python-Levenshtein
```

## Quick Start

Open `evaluate_model.ipynb` and follow the cells. The key steps are:

1. **Load ground truth** from the CSV
2. **Run your model** on the images (you provide this part)
3. **Score results** using `calculate_field_accuracy_with_method()`
4. **Generate reports** with `BatchAnalytics`, `BatchReporter`, and `BatchVisualizer`

## Ground Truth Format

The CSV (`evaluation_data/ground_truth.csv`) has one row per image:

| Column | Description |
|--------|-------------|
| `image_file` | Image filename (e.g., `receipt_001.png`, `bank_003.png`) |
| `DOCUMENT_TYPE` | `RECEIPT`, `INVOICE`, `BANK_STATEMENT`, or `TRAVEL_EXPENSE` |
| `BUSINESS_ABN` | 11-digit Australian Business Number |
| `BUSINESS_ADDRESS` | Complete supplier business address |
| `GST_AMOUNT` | GST/tax amount in dollars |
| `INVOICE_DATE` | Invoice/receipt date (DD/MM/YYYY format) |
| `IS_GST_INCLUDED` | Whether GST is shown in document (`true`/`false`) |
| `LINE_ITEM_DESCRIPTIONS` | Product/service names or transaction descriptions |
| `LINE_ITEM_QUANTITIES` | Quantities for each line item |
| `LINE_ITEM_PRICES` | Unit price per item |
| `LINE_ITEM_TOTAL_PRICES` | Total price per line (quantity x unit price) |
| `PAYER_ADDRESS` | Customer/payer address |
| `PAYER_NAME` | Customer/payer name |
| `STATEMENT_DATE_RANGE` | Bank statement period (DD/MM/YYYY - DD/MM/YYYY) |
| `SUPPLIER_NAME` | Business/company name providing goods/services |
| `TOTAL_AMOUNT` | Final total amount |
| `TRANSACTION_AMOUNTS_PAID` | Debit/withdrawal amounts |
| `TRANSACTION_DATES` | Bank statement transaction dates |
| `TRANSACTION_AMOUNTS_RECEIVED` | Credit/deposit amounts |
| `ACCOUNT_BALANCE` | Running balance after each transaction |
| `PASSENGER_NAME` | Name of person travelling (LASTNAME/FIRSTNAME format) |
| `TRAVEL_MODE` | Mode of travel (plane, train, bus, taxi, uber, ferry) |
| `TRAVEL_ROUTE` | Travel route with origin/destination cities |
| `TRAVEL_DATES` | Travel date(s) in DD Mon YYYY format |

Fields not present in a document are marked `NOT_FOUND`.

List fields (e.g., `LINE_ITEM_DESCRIPTIONS`) use pipe `|` as delimiter:
```
Car Wash | Coffee Large | Unleaded Petrol
```

## Expected Model Output Format

Your model should produce a `batch_results` list where each entry is a dictionary:

```python
{
    "image_name": "receipt_001.png",
    "document_type": "receipt",
    "processing_time": 2.5,
    "prompt_used": "your_prompt_name",
    "extracted_data": {
        "DOCUMENT_TYPE": "RECEIPT",
        "BUSINESS_ABN": "06 082 698 025",
        "SUPPLIER_NAME": "Liberty Oil",
        "TOTAL_AMOUNT": "$94.87",
        # ... all fields for this document type
    },
    "evaluation": {
        "overall_accuracy": 0.95,
        "fields_extracted": 14,
        "fields_matched": 13,
        "total_fields": 14,
        "field_accuracies": {
            "DOCUMENT_TYPE": {"accuracy": 1.0},
            "BUSINESS_ABN": {"accuracy": 1.0},
            # ...
        }
    }
}
```

## Evaluation Methods

All methods are accessed via a single router function:

```python
from common.evaluation_metrics import calculate_field_accuracy_with_method

result = calculate_field_accuracy_with_method(
    extracted_value, ground_truth_value, field_name, method="order_aware_f1"
)
# result["f1_score"] → float between 0.0 and 1.0
```

### Method summary

| Method | Strictness | Best for | Key idea |
| ------ | ---------- | -------- | -------- |
| `order_aware_f1` | Strict | Transactions, line items (default) | Position AND value must match |
| `f1` | Lenient | Unordered sets, categories | Value match only, order ignored |
| `kieval` | Moderate | Correction workload estimation | Counts edit operations to fix |
| `correlation` | Very strict | Cross-field coherence | Validates alignment across related lists |

### `order_aware_f1` — Position-Aware F1 (default)

Items must match in **both value and position**. A correct value at the wrong index counts as a mismatch.

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

| Extracted | Ground Truth | F1 | Why |
| --------- | ------------ | -- | --- |
| `A \| B \| C` | `A \| B \| C` | 100% | All positions match |
| `B \| A \| C` | `A \| B \| C` | 33.3% | Only position 3 matches |
| `A \| B` | `A \| B \| C` | 66.7% | 2 TP, 1 FN (missing C) |

A substitution error (wrong item at a position) counts as 1 FN only, not both FP+FN, to avoid double-penalising a single mistake.

### `f1` — Position-Agnostic F1

Same F1 formula, but matching is **set-based**: an extracted item can match any ground truth item regardless of position. Each ground truth item is matched at most once.

| Extracted | Ground Truth | order_aware_f1 | f1 |
| --------- | ------------ | -------------- | -- |
| `B \| A \| C` | `A \| B \| C` | 33.3% | 100% |
| `A \| A \| B` | `A \| B \| C` | 66.7% | 66.7% |

Use this when item order is not meaningful.

### `kieval` — KIEval Correction Cost

Measures **how much manual effort** is needed to correct the extraction. Decomposes errors into three correction operations:

```
Substitution = min(FP, FN)          # items to edit in place
Addition     = FN - Substitution    # missing items to add
Deletion     = FP - Substitution    # extra items to remove
Total Error  = Substitution + Addition + Deletion

KIEval Score = 1.0 - (Total Error / max(|extracted|, |ground_truth|))
```

**Example:**

| | Extracted | Ground Truth |
| - | --------- | ------------ |
| | `apple \| orange \| grape` | `apple \| banana \| cherry` |

- TP=1 (`apple`), FP=2 (`orange`, `grape`), FN=2 (`banana`, `cherry`)
- Substitution=2 (edit `orange`→`banana`, `grape`→`cherry`), Addition=0, Deletion=0
- Score = 1.0 - 2/3 = **33.3%**

### `correlation` — Correlation-Aware F1

Validates that **related list fields stay aligned** row-by-row. Individual field F1 can be high even when rows are shuffled — this metric catches that.

**Related field groups:**

| Document type | Related fields checked together |
| ------------- | ------------------------------- |
| Bank statement | `TRANSACTION_DATES`, `LINE_ITEM_DESCRIPTIONS`, `TRANSACTION_AMOUNTS_PAID` |
| Invoice/Receipt | `LINE_ITEM_DESCRIPTIONS`, `LINE_ITEM_QUANTITIES`, `LINE_ITEM_PRICES`, `LINE_ITEM_TOTAL_PRICES` |

**Scoring:**

1. **Standard F1** — position-agnostic F1 per field, averaged
2. **Alignment score** — fraction of rows where ALL fields in the group match at the same index
3. **Combined** = (Standard F1 + Alignment Score) / 2

### Field-type-specific comparison

Single-value fields use specialised matching before F1 logic:

| Field type | Matching rule | Examples |
| ---------- | ------------- | -------- |
| **Text** | Levenshtein similarity, ANLS 0.5 threshold | `SUPPLIER_NAME`, `BUSINESS_ADDRESS` |
| **Monetary** | Numeric comparison, 1% tolerance | `TOTAL_AMOUNT`, `GST_AMOUNT` |
| **Date** | Semantic parsing (month names, ranges, formats) | `INVOICE_DATE`, `STATEMENT_DATE_RANGE` |
| **ID/Number** | Exact match after stripping formatting | `BUSINESS_ABN` |
| **Boolean** | Case-insensitive (`true`/`false`/`yes`/`no`) | `IS_GST_INCLUDED` |

### NOT_FOUND handling

| Extracted | Ground Truth | Score |
| --------- | ------------ | ----- |
| `NOT_FOUND` | `NOT_FOUND` | 1.0 (both agree field is absent) |
| `NOT_FOUND` | any value | 0.0 (missed extraction) |
| any value | `NOT_FOUND` | 0.0 (false extraction) |

## Document Types and Fields

| Document Type | Field Count | Key Fields |
| ------------- | ----------- | ---------- |
| Invoice | 14 | ABN, supplier, line items, GST, total |
| Receipt | 14 | Same schema as invoice |
| Bank Statement | 5 | Date range, descriptions, transaction dates/amounts |
| Travel Expense | 9 | Passenger, travel mode/route/dates, GST, total, supplier |
