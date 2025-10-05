# Evaluation System Guide

**Document Version**: 1.0
**Last Updated**: 2025-10-06
**Purpose**: Comprehensive guide to the vision-language model evaluation system for business document extraction

---

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Ground Truth Data](#ground-truth-data)
4. [Evaluation Metrics](#evaluation-metrics)
5. [How to Use the Evaluation System](#how-to-use-the-evaluation-system)
6. [Understanding Evaluation Reports](#understanding-evaluation-reports)
7. [Advanced Usage](#advanced-usage)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The evaluation system measures **information extraction accuracy** for vision-language models (Llama-3.2-Vision and InternVL3) processing business documents. It compares model-extracted data against **ground truth** (known correct values) to calculate accuracy metrics.

### What It Does

- **Compares** extracted field values against ground truth data
- **Calculates** accuracy scores using field-specific comparison logic
- **Generates** detailed performance reports and visualizations
- **Supports** document-aware evaluation (invoices, receipts, bank statements)

### Key Components

```
evaluation_data/
  └── ground_truth.csv          # Known correct values for test images

common/
  ├── evaluation_metrics.py     # Core evaluation engine
  ├── simple_model_evaluator.py # Model comparison logic
  └── batch_processor.py        # Batch evaluation orchestration

output/
  ├── reports/                  # Generated evaluation reports
  └── visualizations/           # Performance comparison charts
```

---

## Core Concepts

### Ground Truth

**Ground truth** is the set of known correct values for each field in each test document. It serves as the benchmark against which model extractions are measured.

**Example**: For `image_001.png`, ground truth specifies:
- `SUPPLIER_NAME`: "Liberty Oil"
- `TOTAL_AMOUNT`: "$94.87"
- `INVOICE_DATE`: "05/08/2025"
- etc.

### Accuracy Score

Accuracy is calculated as a **float score from 0.0 to 1.0**:
- **1.0** = Perfect match
- **0.9** = Fuzzy text match (substring or 80%+ word overlap)
- **0.8** = Phone number with 80%+ correct digits
- **0.5** = Threshold for "correct" classification
- **0.0** = Complete mismatch

### Partial Credit

The system awards partial credit for near-matches:
- OCR errors with mostly correct values
- Formatting differences (dates, monetary values)
- Substring matches in text fields

This reflects real-world scenarios where minor variations shouldn't count as complete failures.

---

## Ground Truth Data

### CSV Structure

Ground truth data is stored in **`evaluation_data/ground_truth.csv`**:

```csv
image_file,DOCUMENT_TYPE,BUSINESS_ABN,SUPPLIER_NAME,TOTAL_AMOUNT,...
image_001.png,RECEIPT,06 082 698 025,Liberty Oil,$94.87,...
image_002.png,RECEIPT,29 466 483 258,Ampol Limited,$57.15,...
image_005.png,INVOICE,73 154 562 747,Aussie Office Supplies Pty Ltd,$4834.03,...
```

### Required Columns

1. **Image Identifier** (one of):
   - `image_file`
   - `filename`
   - `image_name`
   - `file`

2. **Field Columns**: Any fields you want to evaluate (e.g., `SUPPLIER_NAME`, `TOTAL_AMOUNT`, `INVOICE_DATE`)

### Field Types

The evaluation system recognizes these field types:

| Field Type | Examples | Comparison Logic |
|------------|----------|------------------|
| **Numeric IDs** | `BUSINESS_ABN` | Exact digit match (ignoring spaces/dashes) |
| **Monetary** | `TOTAL_AMOUNT`, `GST_AMOUNT` | Numeric comparison with 1% tolerance |
| **Dates** | `INVOICE_DATE` | Flexible format matching |
| **Boolean** | `IS_GST_INCLUDED` | Exact true/false match |
| **Lists** | `LINE_ITEM_DESCRIPTIONS` | Pipe-separated overlap calculation |
| **Transactions** | `TRANSACTION_DATES`, `TRANSACTION_AMOUNTS_PAID` | Structured comparison |
| **Phone** | `CONTACT_PHONE` | Digit-based with partial credit |
| **Text** | `SUPPLIER_NAME`, `PAYER_ADDRESS` | Fuzzy matching (substring, word overlap) |

### Creating Ground Truth

**Step 1**: Manually review test documents and record correct values

**Step 2**: Create/update `evaluation_data/ground_truth.csv`:

```csv
image_file,DOCUMENT_TYPE,SUPPLIER_NAME,TOTAL_AMOUNT,INVOICE_DATE
test_invoice.png,INVOICE,Acme Corp,$1234.56,15/03/2025
test_receipt.png,RECEIPT,Corner Store,$45.00,16/03/2025
```

**Step 3**: Use consistent formatting:
- **Dates**: DD/MM/YYYY format
- **Money**: Include `$` symbol and decimals
- **Lists**: Separate items with ` | ` (space-pipe-space)
- **Missing**: Use `NOT_FOUND` for fields that don't exist in document

---

## Evaluation Metrics

### Field-Level Accuracy

Calculated per field across all documents:

```python
field_accuracy = correct_extractions / total_fields_evaluated
```

**Example**:
- `SUPPLIER_NAME` extracted correctly in 8 out of 9 documents
- Field accuracy = 8/9 = 88.9%

### Image-Level Accuracy

Average accuracy across all fields in a single document:

```python
image_accuracy = sum(field_scores) / num_fields
```

**Example**: For `image_001.png` with 14 fields:
- 12 perfect matches (1.0 each)
- 1 fuzzy match (0.9)
- 1 mismatch (0.0)
- Image accuracy = (12 + 0.9 + 0) / 14 = 92.1%

### Overall Accuracy

Average of all per-image accuracies:

```python
overall_accuracy = sum(image_accuracies) / num_images
```

### Performance Metrics

- **Processing Time**: Seconds per document
- **Throughput**: Documents per minute
- **Perfect Documents**: Count of images with ≥99% accuracy

---

## How to Use the Evaluation System

### Method 1: Programmatic Evaluation (Recommended)

Use the evaluation system from Python code:

```python
from common.evaluation_metrics import load_ground_truth, evaluate_extraction_results

# Load ground truth
ground_truth = load_ground_truth(
    csv_path="evaluation_data/ground_truth.csv",
    show_sample=True,
    verbose=True
)

# After running extraction on images...
extraction_results = [
    {
        "image_name": "image_001.png",
        "extracted_data": {
            "SUPPLIER_NAME": "Liberty Oil",
            "TOTAL_AMOUNT": "$94.87",
            # ... more fields
        },
        "processing_time": 11.2
    },
    # ... more results
]

# Evaluate
evaluation = evaluate_extraction_results(
    extraction_results=extraction_results,
    ground_truth_map=ground_truth
)

# Access metrics
print(f"Overall Accuracy: {evaluation['overall_accuracy']:.1%}")
print(f"Perfect Documents: {evaluation['perfect_documents']}")
print(f"Best Field: {evaluation['summary_stats']['best_fields'][0]}")
```

### Method 2: Batch Processing with Evaluation

Use the `BatchDocumentProcessor` for integrated extraction + evaluation:

```python
from common.batch_processor import BatchDocumentProcessor

# Initialize processor
processor = BatchDocumentProcessor(
    model=model,
    processor=processor,
    prompt_config={"detection": "prompts/detection.yaml"},
    ground_truth_csv="evaluation_data/ground_truth.csv"
)

# Process batch with automatic evaluation
results, times, stats = processor.process_batch(
    image_paths=["evaluation_data/image_001.png", ...],
    verbose=True
)

# Results include accuracy metrics automatically
```

### Method 3: Notebook Evaluation

See `model_comparison.ipynb` for interactive evaluation workflow:

1. **Load models** (Llama and/or InternVL3)
2. **Process test images** through each model
3. **Compare results** side-by-side
4. **Generate reports** and visualizations

---

## Understanding Evaluation Reports

### Executive Report Format

Generated at `output/reports/executive_report_TIMESTAMP.md`:

```markdown
# Executive Model Comparison Report

**Generated**: 2025-10-01 00:25:56

## Performance Dashboard
![Executive Performance Comparison](../visualizations/executive_comparison_20251001_002554.png)

## Executive Summary

### Llama-3.2-Vision
- **Average Accuracy**: 61.6%
- **Average Processing Time**: 11.0 seconds
- **Throughput**: 5.5 documents per minute

### InternVL3-NonQuantized-2B
- **Average Accuracy**: 68.9%
- **Average Processing Time**: 10.5 seconds
- **Throughput**: 5.7 documents per minute

## Document Type Performance
| document_type   | InternVL3-2B | Llama-3.2 |
|:----------------|-------------:|----------:|
| invoice         | 73.8%        | 90.5%     |
| receipt         | 92.9%        | 81.0%     |
| bank_statement  | 40.0%        | 13.3%     |

## Key Findings
- **Accuracy Leader**: InternVL3-NonQuantized-2B
- **Best for Invoices**: Llama-3.2-Vision
- **Best for Receipts**: InternVL3-NonQuantized-2B
```

### Detailed Evaluation Results

Programmatic evaluation returns a comprehensive dictionary:

```python
{
    "overall_accuracy": 0.689,
    "overall_correct": 147.5,      # Sum of partial scores
    "overall_total": 189,           # Total fields evaluated

    "field_accuracies": {
        "SUPPLIER_NAME": {
            "accuracy": 0.92,
            "correct": 11.5,        # Float sum (partial credit)
            "total": 12
        },
        # ... more fields
    },

    "detailed_results": [
        {
            "image_name": "image_001.png",
            "overall_accuracy": 0.921,
            "fields": {
                "SUPPLIER_NAME": {
                    "extracted": "Liberty Oil",
                    "ground_truth": "Liberty Oil",
                    "correct": True,
                    "accuracy_score": 1.0
                },
                # ... more fields
            }
        },
        # ... more images
    ],

    "images_evaluated": 9,
    "perfect_documents": 2,
    "best_performing_image": "image_001.png",
    "best_performance_accuracy": 0.95,
    "worst_performing_image": "image_003.png",
    "worst_performance_accuracy": 0.13,

    "summary_stats": {
        "best_fields": [("INVOICE_DATE", {"accuracy": 0.98, ...}), ...],
        "worst_fields": [("TRANSACTION_DATES", {"accuracy": 0.15, ...}), ...],
        "avg_field_accuracy": 0.72
    }
}
```

---

## Advanced Usage

### Custom Field Comparison

Override default comparison logic for specific fields:

```python
from common.evaluation_metrics import calculate_field_accuracy

# The system automatically handles field types, but you can debug:
score = calculate_field_accuracy(
    extracted_value="$1,234.56",
    ground_truth_value="$1234.56",
    field_name="TOTAL_AMOUNT",
    debug=True  # Shows comparison steps
)
# Output: 1.0 (monetary values normalized)
```

### Document Type Filtering

Evaluate only specific document types:

```python
# Filter results before evaluation
invoice_results = [
    r for r in extraction_results
    if r['extracted_data'].get('DOCUMENT_TYPE', '').lower() == 'invoice'
]

invoice_evaluation = evaluate_extraction_results(
    extraction_results=invoice_results,
    ground_truth_map=ground_truth
)
```

### Custom Tolerance

Modify monetary tolerance in `evaluation_metrics.py`:

```python
# Default: 1% tolerance
tolerance = abs(ground_truth_num * 0.01) if ground_truth_num != 0 else 0.01

# For stricter evaluation, change to 0.1%:
tolerance = abs(ground_truth_num * 0.001) if ground_truth_num != 0 else 0.001
```

### Export to CSV

Save detailed results for external analysis:

```python
import pandas as pd

# Convert detailed results to DataFrame
rows = []
for result in evaluation['detailed_results']:
    for field, data in result['fields'].items():
        rows.append({
            'image': result['image_name'],
            'field': field,
            'extracted': data['extracted'],
            'ground_truth': data['ground_truth'],
            'correct': data['correct'],
            'score': data['accuracy_score']
        })

df = pd.DataFrame(rows)
df.to_csv('evaluation_details.csv', index=False)
```

---

## Troubleshooting

### No Ground Truth Found

**Error**: `⚠️ No ground truth found for image: image_010.png`

**Solutions**:
1. Check CSV has entry for `image_010.png`
2. Verify image identifier column name (`image_file`, `filename`, etc.)
3. Ensure exact filename match (case-sensitive)

### Low Accuracy on Known Good Extraction

**Symptoms**: Visual inspection shows correct extraction, but low accuracy score

**Debugging**:
```python
score = calculate_field_accuracy(
    extracted_value="Your Extracted Value",
    ground_truth_value="Ground Truth Value",
    field_name="FIELD_NAME",
    debug=True  # Shows step-by-step comparison
)
```

**Common Issues**:
- **Formatting**: `"$1,234"` vs `"$1234"` (should match - check for extra spaces)
- **Case sensitivity**: System normalizes to lowercase, but check quotes: `"ACME"` vs `"acme"`
- **List separators**: Use ` | ` (space-pipe-space), not `|` or `, `

### Field Not Evaluated

**Symptoms**: Field exists in extraction but not in evaluation results

**Check**:
1. Is field in ground truth CSV?
2. Is ground truth value `NOT_FOUND` or empty?
3. Is field in document-specific schema? (e.g., `TRANSACTION_DATES` only for bank statements)

### Partial Credit Not Working

**Issue**: Expected 0.9 for fuzzy match, got 0.0

**Verify**:
- Field type detection may be wrong (e.g., treating list as text)
- Check word overlap: `"Acme Corp Ltd"` vs `"Acme Corporation"` = 50% overlap (below 80% threshold)
- For lists: Check pipe separators are correct

---

## Best Practices

### 1. Consistent Ground Truth Format

✅ **Do**:
```csv
INVOICE_DATE,TOTAL_AMOUNT,LINE_ITEM_DESCRIPTIONS
15/03/2025,$1234.56,Item 1 | Item 2 | Item 3
```

❌ **Don't**:
```csv
INVOICE_DATE,TOTAL_AMOUNT,LINE_ITEM_DESCRIPTIONS
15-Mar-2025,1234.56,Item 1, Item 2, Item 3
```

### 2. Document Representative Test Set

- Include **variety**: Easy, medium, hard documents
- Cover **all document types**: Invoices, receipts, bank statements
- Test **edge cases**: Multi-page, poor quality, unusual layouts

### 3. Iterative Refinement

```
1. Run evaluation on test set
2. Analyze worst-performing fields
3. Review extraction failures manually
4. Update prompts or processing logic
5. Re-evaluate and compare
```

### 4. Track Metrics Over Time

Save evaluation results with timestamps:

```python
import json
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f"evaluation_history_{timestamp}.json", "w") as f:
    json.dump(evaluation, f, indent=2)
```

### 5. Use Visualization

Charts in `output/visualizations/` make trends obvious:
- Compare models side-by-side
- Spot document type weaknesses
- Track improvement over time

---

## Quick Reference

### Load Ground Truth
```python
from common.evaluation_metrics import load_ground_truth
gt = load_ground_truth("evaluation_data/ground_truth.csv", verbose=True)
```

### Evaluate Results
```python
from common.evaluation_metrics import evaluate_extraction_results
eval_results = evaluate_extraction_results(extraction_results, gt)
print(f"Accuracy: {eval_results['overall_accuracy']:.1%}")
```

### Field-Level Debug
```python
from common.evaluation_metrics import calculate_field_accuracy
score = calculate_field_accuracy("extracted", "ground_truth", "FIELD_NAME", debug=True)
```

### Generate Report
```python
from common.reporting import generate_executive_report
generate_executive_report(
    model_results={"Llama": llama_eval, "InternVL3": internvl_eval},
    output_dir="output/reports"
)
```

---

## Related Documentation

- **[PROMPT_TESTING_GUIDE.md](docs/PROMPT_TESTING_GUIDE.md)**: How to test and refine extraction prompts
- **[PIPELINE_FLOW.md](docs/PIPELINE_FLOW.md)**: Complete extraction pipeline architecture
- **[CLAUDE.md](CLAUDE.md)**: Project setup and environment configuration

---

**Questions or Issues?**
See [Troubleshooting](#troubleshooting) or check existing evaluation runs in `output/reports/`
