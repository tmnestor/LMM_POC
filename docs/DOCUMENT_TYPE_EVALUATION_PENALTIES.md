# Document Type Classification Penalty Analysis

**Report Date:** 2025-11-10
**System:** Vision-Language Model Evaluation Framework
**Scope:** How incorrect DOCUMENT_TYPE classifications affect evaluation accuracy

---

## Executive Summary

The evaluation system implements a **two-tier penalty mechanism** for incorrect document type classifications:

1. **Direct Penalty**: Binary scoring (0.0 or 1.0, no partial credit)
2. **Cascading Penalty**: Wrong field set evaluated, causing systematic accuracy degradation

**Key Finding:** The cascading penalty is the most severe - misclassifying a bank statement as an invoice can result in evaluating 6+ non-existent fields, each scoring 0.0 accuracy, dramatically reducing overall performance metrics.

---

## 1. Direct Penalty: Binary Scoring

### Mechanism

Unlike other extraction fields that support fuzzy matching, DOCUMENT_TYPE uses **all-or-nothing comparison**:

```python
# From common/evaluation_metrics.py:178-209
if field_name == "DOCUMENT_TYPE":
    # Canonical type mapping (normalizes variations)
    type_mapping = {
        "invoice": "invoice",
        "tax invoice": "invoice",
        "estimate": "invoice",
        "quote": "invoice",
        "receipt": "receipt",
        "bank statement": "bank_statement",
        "account statement": "bank_statement",
    }

    extracted_canonical = type_mapping.get(extracted_lower, extracted_lower)
    ground_truth_canonical = type_mapping.get(ground_truth_lower, ground_truth_lower)

    if extracted_canonical == ground_truth_canonical:
        return 1.0  # Perfect match
    else:
        return 0.0  # Complete mismatch - NO partial credit
```

### Impact

- **No partial credit** for near-misses
- Variations are normalized ("Tax Invoice" → "invoice")
- Every wrong classification = guaranteed 0.0 score for DOCUMENT_TYPE field

### Comparison to Other Fields

| Field Type | Scoring Method | Example Partial Scores |
|-----------|---------------|----------------------|
| **DOCUMENT_TYPE** | Binary (0.0 or 1.0) | None - all or nothing |
| Text fields | Fuzzy matching | Substring match: 0.9, Word overlap 80%: 0.8+ |
| Numeric fields | Percentage difference | Within 5%: 0.95, Within 10%: 0.90 |
| Date fields | Component matching | Same month/year: 0.7+ |

---

## 2. Cascading Penalty: Wrong Field Set Evaluation

### Mechanism

**This is the critical penalty** that causes severe accuracy degradation.

The evaluation system uses the **extracted** DOCUMENT_TYPE (not ground truth) to determine which fields to evaluate:

```python
# From common/evaluation_metrics.py:599-620
# Get document type from EXTRACTED data (not ground truth!)
doc_type_raw = extracted_data.get("DOCUMENT_TYPE", "invoice").lower()

# Map detected type to schema type
type_mapping = {
    "invoice": "invoice",
    "tax invoice": "invoice",
    "receipt": "receipt",
    "bank statement": "bank_statement",
}
doc_type = type_mapping.get(doc_type_raw, "invoice")

# Get document-specific fields based on WRONG type
from common.config import get_document_type_fields
fields_to_evaluate = get_document_type_fields(doc_type)

# Evaluate these fields (many won't exist on actual document)
for field in fields_to_evaluate:
    extracted_value = extracted_data.get(field, "NOT_FOUND")
    ground_truth_value = gt_data.get(field, "NOT_FOUND")
    accuracy_score = calculate_field_accuracy(...)
```

### Why This Is Severe

**Wrong denominator in accuracy calculation:**

```
Accuracy = correct_fields / fields_evaluated
```

When DOCUMENT_TYPE is wrong:
- `fields_to_evaluate` = **wrong field set** for actual document type
- Many fields will be "NOT_FOUND" on the actual document
- Each "NOT_FOUND" → 0.0 accuracy
- Overall accuracy score plummets

---

## 3. Real-World Impact Examples

### Example 1: Bank Statement Misclassified as Invoice

**Scenario:**
- **True Document Type:** Bank Statement
- **Extracted Type:** Invoice (WRONG)
- **Ground Truth Fields:** 5 bank-specific fields
- **Evaluated Fields:** 11 invoice-specific fields

| Category | Fields | Evaluation Result |
|----------|--------|-------------------|
| **Should evaluate** | ACCOUNT_NUMBER, ACCOUNT_NAME, BSB, OPENING_BALANCE, CLOSING_BALANCE | ❌ Not evaluated |
| **Actually evaluates** | INVOICE_NUMBER, ABN, SUPPLIER_NAME, TOTAL, SUBTOTAL, TAX_AMOUNT, DUE_DATE, INVOICE_DATE, LINE_ITEMS, PAYMENT_TERMS, GST_STATUS | ✅ Evaluated but **don't exist** |

**Impact:**
- 6+ invoice fields don't exist on bank statement → "NOT_FOUND" → 0.0 accuracy each
- Actual bank statement fields are ignored
- **Overall accuracy tanks** from wrong denominator

### Example 2: Invoice Misclassified as Bank Statement

**Scenario:**
- **True Document Type:** Invoice
- **Extracted Type:** Bank Statement (WRONG)
- **Ground Truth Fields:** 11 invoice fields
- **Evaluated Fields:** 5 bank statement fields

**Impact:**
- Only 5 fields evaluated instead of 11
- Misses 6+ invoice-specific fields (LINE_ITEMS, TAX_AMOUNT, etc.)
- Lower field coverage
- **Incomplete evaluation** - many relevant fields ignored

### Example 3: Field Set Sizes by Document Type

| Document Type | Field Count | Key Fields |
|--------------|-------------|------------|
| **Invoice** | 11 | INVOICE_NUMBER, ABN, TOTAL, SUBTOTAL, TAX_AMOUNT, LINE_ITEMS, DUE_DATE, SUPPLIER_NAME, PAYMENT_TERMS, INVOICE_DATE, GST_STATUS |
| **Receipt** | 11 | Similar to invoice (receipt-specific fields) |
| **Bank Statement** | 5 | ACCOUNT_NUMBER, ACCOUNT_NAME, BSB, OPENING_BALANCE, CLOSING_BALANCE |

**Mismatch Impact:**
- Invoice → Bank Statement: **6 field deficit** (evaluates 5 instead of 11)
- Bank Statement → Invoice: **6 field surplus** (evaluates 11 instead of 5, 6+ are "NOT_FOUND")

---

## 4. Technical Implementation Details

### File Locations

| File | Function | Lines | Purpose |
|------|----------|-------|---------|
| `common/evaluation_metrics.py` | `calculate_field_accuracy()` | 101-432 | Binary scoring for DOCUMENT_TYPE |
| `common/evaluation_metrics.py` | `evaluate_extraction_results()` | 553-770 | Field set selection using extracted type |
| `common/config.py` | `get_document_type_fields()` | 731-788 | Document-specific field definitions |
| `common/batch_processor.py` | Document routing | Various | Uses type for processing decisions |

### Field Set Selection Logic

```python
# From common/config.py:731-788
def get_document_type_fields(doc_type: str) -> list[str]:
    """Get fields to extract based on document type."""

    invoice_fields = [
        "INVOICE_NUMBER", "ABN", "SUPPLIER_NAME", "TOTAL",
        "SUBTOTAL", "TAX_AMOUNT", "DUE_DATE", "INVOICE_DATE",
        "LINE_ITEMS", "PAYMENT_TERMS", "GST_STATUS"
    ]

    bank_statement_fields = [
        "ACCOUNT_NUMBER", "ACCOUNT_NAME", "BSB",
        "OPENING_BALANCE", "CLOSING_BALANCE"
    ]

    receipt_fields = invoice_fields  # Similar schema

    if doc_type == "bank_statement":
        return bank_statement_fields
    elif doc_type == "receipt":
        return receipt_fields
    else:  # Default to invoice
        return invoice_fields
```

---

## 5. No Explicit Penalty Weighting

### What Was NOT Found

After searching the codebase for:
- "penalty"
- "weight"
- "cascade"
- "document type error"
- Explicit multipliers

**Result:** No explicit penalty weighting system exists.

### How Penalty Actually Works

The penalty is **structural and implicit**:

1. **No penalty multiplier** applied to wrong DOCUMENT_TYPE scores
2. **No special weighting** that gives DOCUMENT_TYPE higher importance than other fields
3. **No explicit "wrong doc type" error flag** in evaluation results
4. Damage occurs through **field set mismatch**, not scoring adjustment

**Analogy:** Like being given the wrong answer key for an exam - you're evaluated on questions that don't match the test you took.

---

## 6. Comparison: DOCUMENT_TYPE vs Other Fields

### Special Treatment

| Aspect | DOCUMENT_TYPE | Other Fields |
|--------|---------------|--------------|
| **Scoring** | Binary (0.0 or 1.0) | Fuzzy matching with partial credit |
| **Normalization** | Canonical type mapping | Case-insensitive, whitespace normalized |
| **Partial Credit** | ❌ None | ✅ Substring, word overlap, numeric tolerance |
| **Cascading Effects** | ✅ Wrong field set evaluated | ❌ None |
| **Impact on Other Fields** | Severe - changes evaluation denominator | None - isolated to single field |

### Why DOCUMENT_TYPE Is Critical

1. **Determines evaluation schema** - wrong type = wrong benchmark
2. **Affects ALL field scores** - not isolated to single field
3. **No recovery mechanism** - once wrong, entire evaluation compromised
4. **Binary failure mode** - no partial credit to soften impact

---

## 7. Implications for Model Evaluation

### For Model Performance Assessment

**When reviewing accuracy scores, consider:**

1. **Low overall accuracy** may indicate wrong document type classification
2. **Many "NOT_FOUND" fields** suggests field set mismatch
3. **DOCUMENT_TYPE = 0.0** is a red flag for cascading penalties
4. Need to **separate document type errors** from field extraction errors

### Recommended Evaluation Approach

```python
# Check if document type is correct FIRST
if result["DOCUMENT_TYPE"]["accuracy"] == 0.0:
    print("⚠️ WARNING: Wrong document type detected")
    print("   All field accuracy scores may be invalid")
    print("   Using wrong evaluation schema")

# Report document type accuracy separately
doc_type_accuracy = correct_doc_types / total_documents
field_extraction_accuracy = field_scores / fields_evaluated

# Only trust field accuracy when doc type is correct
valid_results = [r for r in results if r["DOCUMENT_TYPE"]["accuracy"] == 1.0]
```

### For Model Comparison

**When comparing Llama vs InternVL3:**

1. **Track document type accuracy separately** from field extraction accuracy
2. **Filter results** by correct/incorrect doc type classification
3. **Calculate conditional metrics**:
   - Field accuracy GIVEN correct document type
   - Field accuracy GIVEN incorrect document type
4. **Identify systematic biases** (e.g., one model consistently misclassifies receipts)

---

## 8. Recommendations

### For Current System

1. **Add explicit document type error tracking**
   - Flag evaluations where doc type was wrong
   - Report "invalid evaluation due to wrong doc type"

2. **Separate metrics**
   ```
   Document Type Accuracy: 95%
   Field Extraction Accuracy (correct type): 87%
   Field Extraction Accuracy (wrong type): 23%  # Shows cascade impact
   ```

3. **Add warnings in evaluation output**
   - Highlight when wrong field set was evaluated
   - Show which fields were evaluated vs should have been evaluated

### For Future Improvements

1. **Consider ground truth-based field selection**
   - Use ground truth doc type for field set (breaks current design)
   - Avoids cascading penalty but may hide doc type errors

2. **Add document type confidence scores**
   - Allow models to express uncertainty
   - Could trigger multi-type evaluation for borderline cases

3. **Implement type-specific accuracy tracking**
   - Invoice classification accuracy
   - Bank statement classification accuracy
   - Identify which types cause most errors

---

## Appendix: Code References

### Key Functions

```python
# Binary scoring for DOCUMENT_TYPE
# common/evaluation_metrics.py:178-209
def calculate_field_accuracy(field_name, extracted, ground_truth):
    if field_name == "DOCUMENT_TYPE":
        if extracted_canonical == ground_truth_canonical:
            return 1.0
        else:
            return 0.0
```

```python
# Field set selection using extracted type
# common/evaluation_metrics.py:599-620
def evaluate_extraction_results(image_path, extracted_data, ground_truth):
    doc_type = extracted_data.get("DOCUMENT_TYPE", "invoice").lower()
    fields_to_evaluate = get_document_type_fields(doc_type)

    for field in fields_to_evaluate:
        accuracy = calculate_field_accuracy(...)
```

### Related Files

- `common/evaluation_metrics.py` - Main evaluation logic
- `common/config.py` - Document type field definitions
- `common/document_type_metrics.py` - Type-specific validation rules
- `common/batch_processor.py` - Document type routing

---

## Conclusion

The evaluation system's treatment of DOCUMENT_TYPE creates a **double penalty**:

1. **Direct:** Binary scoring with no partial credit (0.0 for any wrong classification)
2. **Cascading:** Wrong field set evaluated, causing systematic accuracy degradation across all fields

The cascading penalty is particularly severe because it affects the evaluation denominator, making it appear as though the model failed on many fields when in reality it was evaluated against the wrong schema.

**Key Takeaway:** Document type classification is not just another field - it's the **foundation of the entire evaluation**. A wrong classification invalidates the entire evaluation result for that document.
