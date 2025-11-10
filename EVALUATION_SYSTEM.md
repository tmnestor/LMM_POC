# Evaluation System Documentation

## Overview

The evaluation system compares extracted data from vision-language models against ground truth data to assess extraction accuracy. It uses **position-aware F1 scoring** with **type-specific comparison logic** to provide fair and comprehensive evaluation.

## Evaluation Method

**Current Method**: `order_aware_f1` (Position-Aware F1 Scoring)

Set in notebooks:
```python
os.environ['EVALUATION_METHOD'] = 'order_aware_f1'
```

## Step-by-Step Process

### 1. Extract Data from Image

```python
# Model generates response in field-by-field format
response = processor.decode(output[0], skip_special_tokens=True)

# Parse field-by-field format (NOT JSON)
extracted_data = parse_extraction_response(response)

# Result: {'DOCUMENT_TYPE': 'INVOICE', 'BUSINESS_ABN': '12 345 678 901', ...}
```

### 2. Load Document-Specific Field Mappings

```python
# Load field mappings from YAML
field_defs_path = Path('config/field_definitions.yaml')
with open(field_defs_path, 'r') as f:
    field_defs = yaml.safe_load(f)

DOC_TYPE_FIELDS = {
    'invoice': field_defs['document_fields']['invoice']['fields'],
    'receipt': field_defs['document_fields']['receipt']['fields'],
    'bank_statement': field_defs['document_fields']['bank_statement']['fields'],
    'statement': field_defs['document_fields']['bank_statement']['fields'],  # Alias
}
```

### 3. Manual Field-by-Field Evaluation

```python
# Get relevant fields for this document type
relevant_fields = DOC_TYPE_FIELDS.get(doc_type, [])

# Evaluate each field
field_scores = {}
total_f1 = 0.0
fields_evaluated = 0
fields_matched = 0

for field in relevant_fields:
    extracted_value = extracted_data.get(field, "NOT_FOUND")
    gt_value = ground_truth.get(field, "NOT_FOUND")

    # Skip if both are NOT_FOUND
    if extracted_value == "NOT_FOUND" and gt_value == "NOT_FOUND":
        continue

    fields_evaluated += 1

    # Calculate field accuracy using type-specific logic
    metrics = calculate_field_accuracy_with_method(
        extracted_value, gt_value, field,
        method='order_aware_f1'
    )

    field_scores[field] = metrics
    total_f1 += metrics['f1_score']

    if metrics['f1_score'] > 0.9:
        fields_matched += 1

# Calculate overall accuracy
overall_accuracy = total_f1 / fields_evaluated if fields_evaluated > 0 else 0.0

# Store evaluation results
evaluation = {
    'overall_accuracy': overall_accuracy,
    'fields_evaluated': fields_evaluated,
    'fields_matched': fields_matched,
    'total_fields': len(relevant_fields),
    'field_scores': field_scores
}
```

For each field, the system uses type-specific comparison logic:

## Field Type Comparison Rules

### Monetary Fields
**Fields**: `GST_AMOUNT`, `TOTAL_AMOUNT`, `LINE_ITEM_PRICES`, `LINE_ITEM_TOTAL_PRICES`

**Logic**:
- Remove `$` and formatting characters
- Parse to float
- Compare with **1% tolerance** for rounding errors
- **Exact match or 0.0** (no partial credit for amounts)

**Example**:
```python
extracted: "$95.50"
ground_truth: "$95.50"
→ Parse: 95.50 == 95.50 (within 1% tolerance)
→ Score: 1.0
```

### Boolean Fields
**Fields**: `IS_GST_INCLUDED`

**Logic**:
- Parse string to boolean
- Accepts: `"True"`, `"true"`, `"1"` → True
- Accepts: `"False"`, `"false"`, `"0"` → False
- **Exact boolean match required**

**Example**:
```python
extracted: "True"
ground_truth: "True"
→ Parse: True == True
→ Score: 1.0
```

### Text Fields
**Fields**: `SUPPLIER_NAME`, `BUSINESS_ADDRESS`, `PAYER_NAME`, `PAYER_ADDRESS`, `LINE_ITEM_DESCRIPTIONS`

**Logic**:
1. **Case normalization**: Convert to lowercase
2. **Formatting removal**: Remove `,`, `$`, `%`, `(`, `)`, spaces
3. **Fuzzy matching**: Levenshtein distance for similarity
4. **ANLS threshold**: 0.5 (50% similarity minimum)
5. **Partial credit**: Score = similarity if >= 0.5, else 0.0

**Example**:
```python
extracted: "Acme Corp"
ground_truth: "ACME CORPORATION"

# Step 1: Normalize case
→ "acme corp" vs "acme corporation"

# Step 2: Remove formatting
→ "acmecorp" vs "acmecorporation"

# Step 3: Calculate similarity
→ Edit distance: 5 (add "ation")
→ Max length: 15
→ Similarity: 1.0 - (5/15) = 0.67

# Step 4: Apply threshold
→ 0.67 >= 0.5 → Score: 0.67
```

### ID Fields
**Fields**: `BUSINESS_ABN`, `INVOICE_NUMBER`, `BSB`, `ACCOUNT_NUMBER`

**Logic**:
1. Remove label prefixes (ABN:, BSB:, etc.)
2. Remove ALL spaces, dashes, formatting
3. Case-insensitive comparison
4. **Exact match required** (no fuzzy matching)

**Example**:
```python
extracted: "12 345 678 901"
ground_truth: "12345678901"
→ Strip formatting: "12345678901" == "12345678901"
→ Score: 1.0
```

### Date Fields
**Fields**: `INVOICE_DATE`, `TRANSACTION_DATES`, `STATEMENT_DATE_RANGE`

**Logic**:
- Normalize different date formats (DD/MM/YYYY, DD-MM-YYYY, DD-MMM-YY)
- Handle month names (Jan, January, etc.)
- Compare date components (day, month, year)
- **Exact date match required**

**Example**:
```python
extracted: "16/07/2025"
ground_truth: "16-Jul-25"
→ Normalize: (16, 07, 2025) == (16, 07, 2025)
→ Score: 1.0
```

### List Fields (Position-Aware F1)
**Fields**: `LINE_ITEM_DESCRIPTIONS`, `LINE_ITEM_QUANTITIES`, `LINE_ITEM_PRICES`, `TRANSACTION_DATES`, `TRANSACTION_AMOUNTS_PAID`

**Logic**:
- Split by `" | "` separator
- **Position-aware matching**: Items must match at the SAME index
- Calculate F1 score: `F1 = 2 * (Precision * Recall) / (Precision + Recall)`
- True Positives (TP): Items matching at correct position
- False Positives (FP): Extra items extracted
- False Negatives (FN): Missing items

**Example 1 - Perfect Match**:
```python
Extracted:    ["apple", "banana", "cherry"]
Ground Truth: ["apple", "banana", "cherry"]

Position 0: "apple" == "apple" → TP
Position 1: "banana" == "banana" → TP
Position 2: "cherry" == "cherry" → TP

TP=3, FP=0, FN=0
Precision = 3/3 = 1.0
Recall = 3/3 = 1.0
F1 = 2 * (1.0 * 1.0) / (1.0 + 1.0) = 1.0
```

**Example 2 - Wrong Order**:
```python
Extracted:    ["banana", "apple", "cherry"]
Ground Truth: ["apple", "banana", "cherry"]

Position 0: "banana" != "apple" → FN (substitution error)
Position 1: "apple" != "banana" → FN (substitution error)
Position 2: "cherry" == "cherry" → TP

TP=1, FP=0, FN=2
Precision = 1/3 = 0.33
Recall = 1/3 = 0.33
F1 = 2 * (0.33 * 0.33) / (0.33 + 0.33) = 0.33
```

**Example 3 - Missing Items**:
```python
Extracted:    ["apple", "banana"]
Ground Truth: ["apple", "banana", "cherry"]

Position 0: "apple" == "apple" → TP
Position 1: "banana" == "banana" → TP
Position 2: Missing → FN

TP=2, FP=0, FN=1
Precision = 2/2 = 1.0
Recall = 2/3 = 0.67
F1 = 2 * (1.0 * 0.67) / (1.0 + 0.67) = 0.80
```

**Example 4 - Extra Items**:
```python
Extracted:    ["apple", "banana", "cherry", "date"]
Ground Truth: ["apple", "banana", "cherry"]

Position 0: "apple" == "apple" → TP
Position 1: "banana" == "banana" → TP
Position 2: "cherry" == "cherry" → TP
Position 3: Extra item → FP

TP=3, FP=1, FN=0
Precision = 3/4 = 0.75
Recall = 3/3 = 1.0
F1 = 2 * (0.75 * 1.0) / (0.75 + 1.0) = 0.86
```

#### Type-Aware List Matching

**Monetary Lists** (LINE_ITEM_PRICES, LINE_ITEM_TOTAL_PRICES):
- Parse each item as float
- Compare with 1% tolerance
- Position-aware matching

**Quantity Lists** (LINE_ITEM_QUANTITIES):
- Convert floats to integers (2.0 → 2)
- Exact integer match required
- Position-aware matching

**Text Lists** (LINE_ITEM_DESCRIPTIONS):
- Fuzzy text matching (75% threshold)
- Substring matching allowed
- Position-aware matching

### Quantity Fields
**Fields**: `LINE_ITEM_QUANTITIES`

**Logic**:
- Convert float strings to integers (2.0 → 2, 1.0 → 1)
- **Exact integer match required**

**Example**:
```python
extracted: "2.0 | 1.0 | 5.0"
ground_truth: "2 | 1 | 5"
→ Convert: [2, 1, 5] == [2, 1, 5]
→ Score: 1.0
```

## NOT_FOUND Handling

The system properly handles missing fields:

```python
# Both NOT_FOUND = Correct (field doesn't exist)
extracted: "NOT_FOUND"
ground_truth: "NOT_FOUND"
→ Score: 1.0

# Should have extracted = False negative
extracted: "NOT_FOUND"
ground_truth: "some value"
→ Score: 0.0

# False positive = Hallucination
extracted: "some value"
ground_truth: "NOT_FOUND"
→ Score: 0.0
```

## Document-Type Awareness

Evaluation only compares fields relevant to the document type:

| Document Type | Fields Evaluated | Count |
|--------------|------------------|-------|
| **Invoice** | DOCUMENT_TYPE, BUSINESS_ABN, SUPPLIER_NAME, BUSINESS_ADDRESS, PAYER_NAME, PAYER_ADDRESS, INVOICE_DATE, LINE_ITEM_DESCRIPTIONS, LINE_ITEM_QUANTITIES, LINE_ITEM_PRICES, LINE_ITEM_TOTAL_PRICES, GST_AMOUNT, IS_GST_INCLUDED, TOTAL_AMOUNT | 14 |
| **Receipt** | Same as Invoice | 14 |
| **Bank Statement** | DOCUMENT_TYPE, TRANSACTION_DATES, LINE_ITEM_DESCRIPTIONS, TRANSACTION_AMOUNTS_PAID, STATEMENT_DATE_RANGE | 5 |

## Evaluation Output

The evaluation returns a dictionary:

```python
{
    'overall_accuracy': 0.85,        # Average F1 score across all fields
    'fields_matched': 12,            # Fields with score > 0.5
    'total_fields': 14,              # Fields evaluated (doc-type specific)
    'fields_extracted': 13,          # Fields != "NOT_FOUND"
    'field_scores': {                # Individual field F1 scores
        'DOCUMENT_TYPE': 1.0,
        'BUSINESS_ABN': 1.0,
        'SUPPLIER_NAME': 0.9,
        'TOTAL_AMOUNT': 1.0,
        'LINE_ITEM_DESCRIPTIONS': 0.67,
        ...
    }
}
```

## Key Principles

### 1. Fairness
- **Partial credit** for close matches (text fields)
- **Type-aware** comparisons (monetary vs text vs dates)
- **Tolerance** for minor variations (1% for amounts, format differences for dates)

### 2. Strictness
- **Position-aware** for lists (order matters)
- **Exact match** for IDs and amounts
- **No credit** below similarity thresholds

### 3. Transparency
- **Field-level scores** available for analysis
- **Detailed breakdown** of TP/FP/FN for debugging
- **Consistent scoring** across all models

## Alternative Evaluation Methods

The system supports multiple evaluation methods (configured via `EVALUATION_METHOD` environment variable):

1. **order_aware_f1** (Default): Position-aware F1 - strict, order matters
2. **f1** (Position-agnostic): Set-based F1 - lenient, only values matter
3. **kieval**: Correction cost metric - application-centric
4. **correlation**: Cross-list correlation validation

## Configuration

**Location**: `common/evaluation_metrics.py`

**Key Functions**:
- `load_ground_truth()`: Load GT from CSV with `dtype=str`
- `calculate_field_accuracy_with_method()`: Field comparison router (main evaluation function)
- `calculate_field_accuracy()`: Legacy accuracy calculation (0.0 to 1.0)
- `calculate_field_accuracy_f1()`: Position-aware F1 implementation
- `calculate_field_accuracy_f1_position_agnostic()`: Position-agnostic F1
- `calculate_field_accuracy_kieval()`: KIEval correction cost metric
- `_parse_boolean_value()`: Boolean string parsing
- `_compare_dates_fuzzy()`: Date normalization and comparison
- `_transaction_item_matches()`: Transaction field matching
- `_fuzzy_text_match()`: Text similarity with Levenshtein distance

## Ground Truth Requirements

**CSV Format**:
- `dtype=str` when loading (prevents bool auto-conversion)
- `keep_default_na=False` to preserve "NOT_FOUND" strings
- Image names WITHOUT file extensions (e.g., `invoice_001` not `invoice_001.jpeg`)

**Example**:
```python
gt_df = pd.read_csv('ground_truth.csv', dtype=str, keep_default_na=False)
```

## Usage in Notebooks

Both `llama_batch_universal.ipynb` and `llama_batch_oracle.ipynb` use:

```python
# Set evaluation method
os.environ['EVALUATION_METHOD'] = 'order_aware_f1'

# Load ground truth (with dtype=str to prevent bool conversion)
ground_truth = load_ground_truth(ground_truth_path, dtype=str)

# Load document-specific field mappings
field_defs_path = Path('config/field_definitions.yaml')
with open(field_defs_path, 'r') as f:
    field_defs = yaml.safe_load(f)

DOC_TYPE_FIELDS = {
    'invoice': field_defs['document_fields']['invoice']['fields'],
    'receipt': field_defs['document_fields']['receipt']['fields'],
    'bank_statement': field_defs['document_fields']['bank_statement']['fields'],
}

# Get relevant fields for this document type
relevant_fields = DOC_TYPE_FIELDS.get(doc_type, [])

# Manual field-by-field evaluation
field_scores = {}
total_f1 = 0.0
fields_evaluated = 0
fields_matched = 0

for field in relevant_fields:
    extracted_value = extracted_data.get(field, "NOT_FOUND")
    gt_value = gt_data.get(field, "NOT_FOUND")

    if extracted_value == "NOT_FOUND" and gt_value == "NOT_FOUND":
        continue

    fields_evaluated += 1

    metrics = calculate_field_accuracy_with_method(
        extracted_value, gt_value, field,
        method=os.environ.get('EVALUATION_METHOD', 'order_aware_f1')
    )

    field_scores[field] = metrics
    total_f1 += metrics['f1_score']

    if metrics['f1_score'] > 0.9:
        fields_matched += 1

# Calculate overall accuracy
overall_accuracy = total_f1 / fields_evaluated if fields_evaluated > 0 else 0.0

# Create evaluation results dictionary
evaluation = {
    'overall_accuracy': overall_accuracy,
    'fields_evaluated': fields_evaluated,
    'fields_matched': fields_matched,
    'total_fields': len(relevant_fields),
    'field_scores': field_scores
}

# Access results
accuracy = evaluation['overall_accuracy'] * 100
fields_matched = evaluation['fields_matched']
total_fields = evaluation['total_fields']
```

## Performance Metrics

**Output Metrics**:
- **Overall Accuracy**: Average F1 across all fields (0-100%)
- **Field Coverage**: Percentage of fields extracted (not NOT_FOUND)
- **Fields Matched**: Count of fields with score > 0.5
- **Processing Time**: Time to extract per image

**CSV Export**:
- Compatible with `model_comparison.ipynb`
- Includes all extracted fields + evaluation metrics
- Enables cross-model comparison
