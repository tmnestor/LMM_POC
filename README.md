# Evaluation Toolkit

Standalone evaluation toolkit for scoring document extraction models against ground truth data. Use this to measure your model's field-level accuracy using F1, KIEval, and correlation-aware metrics.

## Directory Structure

```
.
├── evaluate_model.ipynb              # Demo notebook — full evaluation workflow
├── common/
│   ├── __init__.py
│   ├── config.py                     # Field type helpers and thresholds
│   ├── field_definitions_loader.py   # Loads field_definitions.yaml
│   ├── unified_schema.py             # Document type field schemas
│   ├── evaluation_metrics.py         # Core scoring functions
│   ├── batch_analytics.py            # Results DataFrames and summary stats
│   ├── batch_reporting.py            # Markdown/JSON report generation
│   └── batch_visualizations.py       # Charts and dashboards
├── config/
│   └── field_definitions.yaml        # Document types, field lists, field type classifications
└── evaluation_data/
    └── synthetic/
        ├── ground_truth_synthetic.csv
        └── *.png (9 sample images)
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

The CSV (`evaluation_data/synthetic/ground_truth_synthetic.csv`) has one row per image:

| Column | Description |
|--------|-------------|
| `image_file` | Image filename (e.g., `image_001.png`) |
| `DOCUMENT_TYPE` | `RECEIPT`, `INVOICE`, or `BANK_STATEMENT` |
| `BUSINESS_ABN` | 11-digit Australian Business Number |
| `SUPPLIER_NAME` | Business name |
| `TOTAL_AMOUNT` | Final total amount |
| `GST_AMOUNT` | GST/tax amount |
| ... | Other fields per document type |

Fields not present in a document are marked `NOT_FOUND`.

List fields (e.g., `LINE_ITEM_DESCRIPTIONS`) use pipe `|` as delimiter:
```
Car Wash | Coffee Large | Unleaded Petrol
```

## Expected Model Output Format

Your model should produce a `batch_results` list where each entry is a dictionary:

```python
{
    "image_name": "image_001.png",
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

The toolkit supports multiple scoring methods via `calculate_field_accuracy_with_method()`:

| Method | Description |
|--------|-------------|
| `order_aware_f1` | Position-aware F1 (default) — order matters for list fields |
| `f1` | Position-agnostic F1 — only values matter, not order |
| `kieval` | KIEval correction cost — "how much effort to fix?" |
| `correlation` | Cross-list validation — checks alignment across related fields |

## Document Types and Fields

| Document Type | Field Count | Key Fields |
|---------------|-------------|------------|
| Invoice | 14 | ABN, supplier, line items, GST, total |
| Receipt | 14 | Same schema as invoice |
| Bank Statement | 5 | Date range, descriptions, transaction dates/amounts |
