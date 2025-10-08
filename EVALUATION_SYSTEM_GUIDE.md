# Evaluation System Guide

**Document Version**: 1.1
**Last Updated**: 2025-10-08
**Purpose**: Comprehensive guide to the vision-language model evaluation system for business document extraction

**Version 1.1 Updates**:
- Added pipe normalization for multi-line text fields (addresses, names)
- Detailed explanation of word-based matching algorithm
- Added ABN/numeric ID format handling documentation
- Enhanced troubleshooting with pipe-related issues

---

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Ground Truth Data](#ground-truth-data)
4. [Custom Comparison Logic: Design Rationale](#custom-comparison-logic-design-rationale)
5. [Evaluation Metrics](#evaluation-metrics)
6. [How to Use the Evaluation System](#how-to-use-the-evaluation-system)
7. [Understanding Evaluation Reports](#understanding-evaluation-reports)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)
10. [Summary: Key Takeaways](#summary-key-takeaways)

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

### Field Types and Custom Comparison Logic

The evaluation system uses **custom comparison logic** (not sklearn) with field-specific scoring rules. Each field type has tailored comparison methods that award partial credit for near-matches, reflecting real-world OCR and extraction scenarios.

**IMPORTANT**: The system does **not** use sklearn's classification metrics. All accuracy calculations use custom fuzzy matching logic implemented in `common/evaluation_metrics.py`.

| Field Type | Examples | Comparison Logic | Scoring Rules |
|------------|----------|------------------|---------------|
| **Numeric IDs** | `BUSINESS_ABN` | Exact digit match (ignoring spaces/dashes) | 1.0 if digits match, 0.0 otherwise |
| **Monetary** | `TOTAL_AMOUNT`, `GST_AMOUNT` | Numeric comparison with tolerance | 1.0 if within 1% tolerance, 0.0 otherwise |
| **Dates** | `INVOICE_DATE` | Flexible format matching | 1.0 if all components match, 0.8 if 2+ components match, 0.0 otherwise |
| **Boolean** | `IS_GST_INCLUDED` | Exact true/false match | 1.0 if boolean values match, 0.0 otherwise |
| **Lists** | `LINE_ITEM_DESCRIPTIONS` | Pipe-separated overlap calculation | Ratio of matched items to max(extracted, ground_truth) |
| **Transactions** | `TRANSACTION_DATES`, `TRANSACTION_AMOUNTS_PAID` | Structured comparison | Ratio of matching transaction entries |
| **Phone** | `CONTACT_PHONE` | Digit-based with partial credit | 1.0 exact match, 0.8 if 80%+ digits match, 0.5 if 60%+ match |
| **Text** | `SUPPLIER_NAME`, `PAYER_ADDRESS`, `BUSINESS_ADDRESS` | Fuzzy matching (substring, word overlap) | 1.0 exact, 0.9 substring match, 0.8+ for 80%+ word overlap |
| **Document Type** | `DOCUMENT_TYPE` | Canonical type mapping | Maps variations to canonical types (e.g., "tax invoice" → "invoice") |

### Multi-Line Fields and Pipe Normalization

**New in v1.1**: The evaluation system now handles multi-line document fields correctly.

**Problem**: Some fields appear on multiple lines in source documents:
```
Document shows:        Ground Truth Should Be:
123 Main Street    →   "123 Main Street | Sydney NSW 2000"
Sydney NSW 2000
```

**Solution**: For **text fields only** (addresses, names), pipes `|` are normalized to spaces before comparison:
- Ground truth: `"123 Main St | Sydney NSW 2000"`
- Model extracts: `"123 Main St Sydney NSW 2000"`
- **Result**: ✅ Perfect match (1.0)

**Important**: This normalization **only applies to text fields**:
- ✅ **Text fields** (ADDRESS, NAME): Pipes normalized to spaces
- ❌ **List fields** (LINE_ITEM_DESCRIPTIONS): Pipes remain as delimiters
- ❌ **Transaction fields**: Pipes remain as delimiters

**Benefits**:
- Ground truth accurately reflects multi-line source documents
- Models can extract with or without line breaks
- No need to manually clean ground truth CSV

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

## Custom Comparison Logic: Design Rationale

### Why Custom Logic Instead of sklearn?

The evaluation system uses **custom fuzzy matching logic** rather than sklearn's binary classification for several critical reasons:

#### 1. **Partial Credit for Near-Matches**
**Business Need**: OCR and vision models often produce "almost correct" results due to image quality, font variations, or formatting differences.

**Example**:
- Ground truth: `"Acme Corporation Pty Ltd"`
- Extracted: `"Acme Corporation"`
- **Binary classification**: ❌ Incorrect (0.0)
- **Custom logic**: ✅ Partial credit (0.9 - substring match)

**Justification**: A 90% accurate extraction is valuable and should be scored differently from a completely wrong value. Binary metrics hide this nuance.

#### 2. **Field-Type-Specific Tolerance**
**Business Need**: Different field types have different error tolerances based on their business purpose.

**Examples**:
- **Monetary values**: 1% tolerance accounts for rounding ($100.00 vs $100.01 are functionally identical)
- **Phone numbers**: 80% digit match tolerance handles OCR errors in middle digits
- **Dates**: Component-based matching handles format variations (15/03/2025 vs 15-03-2025)
- **ABN/Tax IDs**: Zero tolerance - these must be exact for legal/regulatory compliance

**Justification**: sklearn's binary classification cannot encode domain-specific tolerance rules. A $1234.56 vs $1234.55 difference matters differently than a phone number with one wrong digit.

#### 3. **Structured Data Comparison**
**Business Need**: Some fields contain structured lists (line items, transactions) that require element-wise comparison.

**Example**:
- Ground truth transactions: `"$50.00 | $75.25 | $100.00"`
- Extracted: `"$50.00 | $75.25 | $99.99"`
- **Binary classification**: ❌ Incorrect (0.0)
- **Custom logic**: ✅ Partial credit (0.67 - 2/3 transactions match)

**Justification**: Successfully extracting 2 out of 3 transactions is better than 0 out of 3. The system should quantify this partial success.

#### 4. **NOT_FOUND Handling**
**Business Need**: Documents legitimately lack certain fields (e.g., receipts don't have invoice numbers).

**Example**:
- Ground truth: `"NOT_FOUND"` (field doesn't exist in document)
- Extracted: `"NOT_FOUND"` (model correctly identified absence)
- **Custom logic**: ✅ Correct (1.0)

**Justification**: Correctly identifying that a field is absent is as important as correctly extracting present fields. This is not a classification error—it's a correct prediction.

#### 5. **Normalization for OCR Artifacts**
**Business Need**: OCR introduces formatting noise (extra spaces, comma variations, case differences) that doesn't change semantic meaning.

**Example**:
- Ground truth: `"$1,234.56"`
- Extracted: `"$ 1234.56"` (extra space, missing comma)
- **Custom logic**: Normalizes both → `"1234.56"` → ✅ 1.0

**Justification**: These are presentation differences, not extraction errors. The semantic content is identical.

#### 6. **Word-Overlap for Text Fields**
**Business Need**: Business names and addresses often have minor variations that don't affect meaning.

**Example**:
- Ground truth: `"Aussie Office Supplies Pty Ltd"`
- Extracted: `"Aussie Office Supplies"`
- **Word overlap**: 3/5 words match (60%) → 0.6 score
- If 4/5 match → 0.8 score

**Justification**: Partial name matches are common in OCR. A supplier name with 80% word overlap is usually the correct entity, just with abbreviated legal suffixes.

### Comparison Logic Implementation

The custom logic is implemented in `common/evaluation_metrics.py:calculate_field_accuracy()` with the following workflow:

```python
def calculate_field_accuracy(extracted_value, ground_truth_value, field_name):
    # 1. Normalize both values (lowercase, remove formatting)
    # 2. Check for NOT_FOUND (both must agree)
    # 3. Apply field-type-specific comparison:
    #    - Numeric IDs: Extract digits only, exact match
    #    - Monetary: Parse numbers, apply 1% tolerance
    #    - Dates: Extract components, flexible matching
    #    - Phone: Compare digits with partial credit
    #    - Lists/Transactions: Element-wise comparison
    #    - Text: Substring or word overlap scoring
    # 4. Return float score (0.0 to 1.0)
```

### Why 1% Tolerance for Monetary Values?

**Rationale**:
- **Rounding differences**: OCR may read `$99.99` as `$100.00` due to image quality
- **Decimal precision**: Vision models sometimes round to nearest dollar
- **Business impact**: For a $10,000 invoice, $100 variance (1%) is typically acceptable for automated extraction
- **Stricter alternative**: Can be changed to 0.1% in code for high-precision requirements

**Implementation** (`evaluation_metrics.py:215`):
```python
tolerance = abs(ground_truth_num * 0.01) if ground_truth_num != 0 else 0.01
score = 1.0 if abs(extracted_num - ground_truth_num) <= tolerance else 0.0
```

### Numeric ID Fields (ABN, Tax IDs)

**Format doesn't matter - only digits are compared.**

For fields like `BUSINESS_ABN`, the evaluation system:
1. **Strips ALL non-digit characters** (spaces, dashes, etc.)
2. **Compares only the digits**
3. **Requires exact digit match** (0% tolerance for regulatory/legal IDs)

**Examples:**

| Extracted | Ground Truth | Digits Match? | Score |
|-----------|--------------|---------------|-------|
| `"06 082 698 025"` | `"06082698025"` | `06082698025` = `06082698025` | ✅ **1.0** |
| `"06-082-698-025"` | `"06 082 698 025"` | `06082698025` = `06082698025` | ✅ **1.0** |
| `"06082698025"` | `"06 082 698 025"` | `06082698025` = `06082698025` | ✅ **1.0** |
| `"06082698026"` | `"06082698025"` | `06082698026` ≠ `06082698025` | ❌ **0.0** |

**Rationale:**
- **Legal/regulatory compliance**: ABNs, Tax IDs must be exact for legal validity
- **Format variations**: Different systems display with different formatting
- **Zero tolerance**: A single wrong digit makes the ID invalid

**Implementation** (`evaluation_metrics.py:198-207`):
```python
if field_types.get(field_name) == "numeric_id":
    extracted_digits = re.sub(r"\D", "", extracted)  # Remove ALL non-digits
    ground_truth_digits = re.sub(r"\D", "", ground_truth)
    score = 1.0 if extracted_digits == ground_truth_digits else 0.0
```

### Word-Based Matching Explained

**What is word-based matching?**

Word-based matching is a fuzzy comparison method that breaks text into individual words and calculates how many words overlap. This is used for **text fields** (addresses, names) when exact and substring matches fail.

**How it works:**

```python
# Step 1: Split text into words
extracted = "Aussie Office Supplies Pty Ltd"
ground_truth = "Aussie Office Supplies Corporation Pty Ltd"

extracted_words = {"aussie", "office", "supplies", "pty", "ltd"}  # 5 words
ground_truth_words = {"aussie", "office", "supplies", "corporation", "pty", "ltd"}  # 6 words

# Step 2: Find matching words (set intersection)
matching_words = {"aussie", "office", "supplies", "pty", "ltd"}  # 5 words match

# Step 3: Calculate overlap ratio
overlap = len(matching_words) / len(ground_truth_words)
overlap = 5 / 6 = 0.833 (83.3%)

# Step 4: Apply threshold
if overlap >= 0.8:  # 80% threshold required
    score = 0.833  # Award partial credit
else:
    score = 0.0   # Below threshold
```

**Implementation** (`evaluation_metrics.py:368-375`):
```python
# Check word overlap for longer text
extracted_words = set(extracted_lower.split())  # Split on spaces
ground_truth_words = set(ground_truth_lower.split())

if ground_truth_words:
    overlap = len(extracted_words & ground_truth_words) / len(ground_truth_words)
    if overlap >= 0.8:
        return overlap  # Return the overlap ratio as score
```

**Examples:**

| Extracted | Ground Truth | Analysis | Score |
|-----------|--------------|----------|-------|
| `"Aussie Office Supplies Pty"` | `"Aussie Office Supplies Pty Ltd"` | 4/5 words = 80% | **0.80** |
| `"123 Main Street Sydney NSW"` | `"123 Main Street Sydney NSW 2000"` | Substring match | **0.90** |
| `"Aussie Office Supplies Ltd"` | `"Aussie Office Supplies Limited"` | 3/4 words = 75% | **0.0** (below 80%) |
| `"456 Collins St Melbourne"` | `"123 Main St Sydney"` | 1/4 words = 25% | **0.0** |

**Text Field Matching Cascade:**

Text fields use a cascading approach for maximum accuracy:

1. **Exact match** (after normalization) → Score: **1.0**
   - Example: `"acme corp"` = `"ACME Corp"` (case/punctuation ignored)

2. **Substring match** → Score: **0.9**
   - Example: `"Acme Corp"` is contained in `"Acme Corporation Ltd"`

3. **Word-based match** (≥80% overlap) → Score: **0.8-1.0**
   - Example: `"Acme Corp Pty"` vs `"Acme Corp Pty Ltd"` = 3/4 = 75% → **0.0**
   - Example: `"Acme Corp Pty Ltd"` vs `"Acme Corp Pty Ltd Australia"` = 4/5 = 80% → **0.80**

4. **No match** → Score: **0.0**

**Key characteristics:**

- ✅ **Order doesn't matter**: `"Sydney NSW 123 Main St"` = `"123 Main St Sydney NSW"`
- ✅ **Case doesn't matter**: `"ACME CORP"` = `"acme corp"`
- ✅ **Punctuation removed**: `"Acme, Corp."` = `"Acme Corp"`
- ✅ **Pipes normalized** (text fields only): `"A | B"` = `"A B"`
- ❌ **Abbreviations count as different**: `"Ltd"` ≠ `"Limited"`
- ❌ **Typos not handled**: `"Mian"` ≠ `"Main"`

### Why 80% Threshold for Text Matching?

**Rationale**:
- **OCR errors**: Common to drop articles, suffixes (Pty, Ltd, Inc)
- **Abbreviations**: Business names often abbreviated consistently
- **False positives**: Below 80%, matches become unreliable (e.g., only common words matching)
- **Empirical testing**: 80% word overlap empirically correlates with "same entity"
- **Balance**: Allows 1-2 missing words while requiring substantial overlap

### Accuracy Score Interpretation

| Score | Meaning | Business Interpretation |
|-------|---------|------------------------|
| **1.0** | Perfect match | Field extracted exactly correctly |
| **0.9** | Substring match (text fields) | Minor OCR artifact, semantically correct |
| **0.8** | Partial component match (dates, phones) | Mostly correct, usable with validation |
| **0.5-0.7** | Significant partial match | May require human review |
| **0.0** | Complete mismatch or wrong/missing | Field extraction failed |

### When Custom Logic Matters Most

**Scenario 1: Production Readiness Assessment**
- Binary metrics: 65% accuracy (many partial matches scored as failures)
- Custom metrics: 82% accuracy (partial credit for near-matches)
- **Impact**: Custom logic reveals model is closer to production-ready than binary metrics suggest

**Scenario 2: Model Comparison**
- Model A: 50 perfect extractions, 50 failures (binary: 50%)
- Model B: 40 perfect, 60 near-matches (binary: 40%, custom: 64%)
- **Impact**: Custom logic correctly identifies Model B as better for real-world use

**Scenario 3: Field-Specific Optimization**
- Binary: "Phone extraction: 30% accurate"
- Custom: "Phone extraction: 30% perfect, 40% at 0.8 (80% digits correct)"
- **Impact**: Reveals OCR quality issue rather than model capability issue

---

## Evaluation Metrics

### Field-Level Accuracy

Calculated per field across all documents using custom scoring:

```python
field_accuracy = sum(partial_scores) / total_fields_evaluated
# partial_scores are floats from 0.0 to 1.0, not binary 0/1
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

### Important Note: "sklearn Classification Reports"

The system generates reports **formatted like sklearn's classification_report()**, but these are **cosmetic formatting only**. The underlying metrics come from the custom comparison logic described above, not from sklearn's binary classification.

**What the "classification report" actually contains**:
- Precision/Recall/F1-Score values are **derived from custom accuracy scores**, not sklearn
- These metrics are converted from your field-level accuracy percentages
- The report format mimics sklearn for familiarity, but the calculation method is completely different

**Why this matters**:
- You cannot directly compare these reports to sklearn classification reports from other systems
- The "support" values and averaging methods are approximations for visualization
- The actual evaluation logic is the custom fuzzy matching described in the previous section

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
- **List separators**: Use ` | ` (space-pipe-space) for lists, but NOT for text fields
- **Pipes in addresses**: For text fields (ADDRESS, NAME), pipes are normalized to spaces automatically (v1.1+)

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

### Understanding Custom Logic Decisions

**Question**: Why did my field get score X instead of Y?

**Debug workflow**:

1. **Enable debug mode** to see step-by-step comparison:
```python
from common.evaluation_metrics import calculate_field_accuracy

score = calculate_field_accuracy(
    extracted_value="Your Extracted Value",
    ground_truth_value="Ground Truth Value",
    field_name="FIELD_NAME",
    debug=True  # Shows detailed comparison steps
)
```

2. **Review field type assignment**:
```python
from common.config import get_all_field_types
field_types = get_all_field_types()
print(f"FIELD_NAME is type: {field_types.get('FIELD_NAME')}")
```

3. **Check normalization**:
   - Both values converted to lowercase
   - Formatting chars removed: `,`, `$`, `%`, `(`, `)`, spaces
   - Then compared

4. **Verify thresholds**:
   - Text fields: Need 80%+ word overlap for partial credit
   - Phone fields: Need 80%+ digit match for 0.8 score
   - Monetary: Need within 1% tolerance
   - Dates: Need 2+ matching components

**Common surprises**:
- `"Acme Corp"` vs `"Acme Corporation"` → 50% overlap → 0.0 (below 80% threshold)
- `"$100.00"` vs `"$101.50"` → 1.5% difference → 0.0 (exceeds 1% tolerance)
- `"04/05/2025"` vs `"05/04/2025"` → Same components, different order → 1.0 (components match)
- `"06-082-698-025"` vs `"06 082 698 025"` → Same digits → 1.0 (ABN format ignored)
- `"123 Main St | Sydney"` vs `"123 Main St Sydney"` → Pipes normalized → 1.0 (text fields only)

### Adjusting Custom Logic Thresholds

**To modify tolerances**, edit `common/evaluation_metrics.py`:

**Monetary tolerance** (line ~215):
```python
# Default: 1% tolerance
tolerance = abs(ground_truth_num * 0.01) if ground_truth_num != 0 else 0.01

# For stricter (0.1%):
tolerance = abs(ground_truth_num * 0.001) if ground_truth_num != 0 else 0.001
```

**Text overlap threshold** (line ~362):
```python
# Default: 80% word overlap required
if overlap >= 0.8:
    return overlap

# For more lenient (70%):
if overlap >= 0.7:
    return overlap
```

**Phone digit matching** (line ~243):
```python
# Default: 80% digits match → score 0.8
score = 0.8 if match_ratio >= 0.8 else (0.5 if match_ratio >= 0.6 else 0.0)

# For stricter (90% required):
score = 0.8 if match_ratio >= 0.9 else 0.0
```

**When to adjust thresholds**:
- Your document set has consistently higher/lower OCR quality than expected
- Business requirements demand stricter accuracy for specific fields
- Empirical testing shows current thresholds are too lenient/strict for your use case

**⚠️ Important**: Document any threshold changes and re-run all evaluations to ensure consistency.

---

## Best Practices

### 1. Consistent Ground Truth Format

✅ **Do**:
```csv
INVOICE_DATE,TOTAL_AMOUNT,LINE_ITEM_DESCRIPTIONS,BUSINESS_ADDRESS
15/03/2025,$1234.56,Item 1 | Item 2 | Item 3,123 Main St | Sydney NSW 2000
```

**Notes**:
- Use pipes `|` for list fields (LINE_ITEM_DESCRIPTIONS)
- Use pipes `|` for multi-line text fields (BUSINESS_ADDRESS) - normalized automatically (v1.1+)
- Date format: DD/MM/YYYY
- Money format: Include `$` and decimals

❌ **Don't**:
```csv
INVOICE_DATE,TOTAL_AMOUNT,LINE_ITEM_DESCRIPTIONS,BUSINESS_ADDRESS
15-Mar-2025,1234.56,Item 1, Item 2, Item 3,"123 Main St
Sydney NSW 2000"
```

**Why**: Inconsistent separators and date formats break comparison logic

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

## Summary: Key Takeaways

### What Makes This Evaluation System Unique

1. **Custom fuzzy matching logic** instead of binary sklearn classification
2. **Partial credit scoring** (0.0-1.0) reflects real-world OCR quality
3. **Field-type-specific tolerances** encode business requirements
4. **NOT_FOUND handling** correctly evaluates absent fields
5. **Structured data comparison** for lists and transactions
6. **Pipe normalization** (v1.1+) handles multi-line text fields automatically
7. **Word-based matching** with 80% threshold for text similarity
8. **Format-agnostic numeric IDs** - only digits matter for ABN/Tax IDs

### When to Use This Approach

✅ **Use custom logic when**:
- Evaluating OCR/vision extraction quality
- Partial matches have business value
- Different fields require different tolerances
- You need to quantify "how close" near-matches are

❌ **Consider binary classification when**:
- Fields are truly binary (present/absent only)
- No tolerance for variation is acceptable
- You need strict sklearn-compatible metrics
- Integration with existing sklearn-based systems required

### Quick Reference: Evaluation Workflow

```bash
# 1. Prepare ground truth CSV
evaluation_data/ground_truth.csv

# 2. Run extraction evaluation (remote GPU)
python llama_keyvalue.py
python internvl3_keyvalue.py

# 3. Review reports
output/reports/{model}_comprehensive_evaluation_report_{timestamp}.md
output/reports/{model}_classification_report_{timestamp}.md

# 4. Debug specific fields
python -c "from common.evaluation_metrics import calculate_field_accuracy; \
  score = calculate_field_accuracy('extracted', 'ground_truth', 'FIELD', debug=True)"
```

### Implementation File Reference

- **Custom comparison logic**: `common/evaluation_metrics.py:99-370` (`calculate_field_accuracy()`)
- **Field type definitions**: `common/config.py` (field classification functions)
- **Evaluation orchestration**: `common/evaluation_metrics.py:491-708` (`evaluate_extraction_results()`)
- **Report generation**: `common/reporting.py:229-408` (`generate_classification_report()`)

---

## Related Documentation

- **[PROMPT_TESTING_GUIDE.md](docs/PROMPT_TESTING_GUIDE.md)**: How to test and refine extraction prompts
- **[PIPELINE_FLOW.md](docs/PIPELINE_FLOW.md)**: Complete extraction pipeline architecture
- **[CLAUDE.md](CLAUDE.md)**: Project setup and environment configuration
- **[common/evaluation_metrics.py](common/evaluation_metrics.py)**: Source code for custom comparison logic

---

**Questions or Issues?**

- See [Custom Comparison Logic: Design Rationale](#custom-comparison-logic-design-rationale) for justification of design choices
- See [Troubleshooting](#troubleshooting) for debugging evaluation results
- See [Adjusting Custom Logic Thresholds](#adjusting-custom-logic-thresholds) to modify tolerances
- Check existing evaluation runs in `output/reports/` for examples
