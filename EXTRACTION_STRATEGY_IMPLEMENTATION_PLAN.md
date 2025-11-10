# Document Classification Penalty Elimination - Implementation Plan

**Date:** 2025-11-10
**Purpose:** Quantify the impact of document classification errors and test alternative extraction strategies
**Models:** Llama-3.2-Vision-11B, InternVL3-2B

---

## Executive Summary

Current two-stage extraction creates a **critical dependency**: wrong document classification → wrong extraction schema → wrong fields extracted → catastrophic evaluation failure.

This plan implements and tests two alternative strategies that eliminate document classification errors entirely, allowing us to **quantify the true cost** of classification mistakes.

---

## Problem Statement

### Current Two-Stage Process

```
Stage 1: Classify Document
   ↓ (POTENTIAL ERROR HERE)
Stage 2: Select Schema → Extract Fields
   ↓
Evaluation: Compare against ground truth
```

**Failure Mode:**
1. Stage 1 misclassifies bank statement as invoice
2. Stage 2 uses invoice schema (14 fields instead of 5)
3. Model extracts wrong fields (looks for INVOICE_NUMBER on bank statement)
4. Evaluation compares 14 invoice fields vs 5 bank statement ground truth
5. Result: ~0% accuracy due to complete schema mismatch

### Cascading Penalties

1. **Direct:** DOCUMENT_TYPE field = 0.0 accuracy (binary scoring)
2. **Extraction:** Model uses wrong prompt, looks for non-existent fields
3. **Evaluation:** Wrong field set evaluated (14 vs 5 or vice versa)
4. **Systemic:** Multiple related fields (DATES, DESCRIPTIONS, AMOUNTS) penalized for same error

**Impact:** Single classification error can reduce overall accuracy from ~80% to ~20%

---

## Proposed Solutions

### Method 1: Universal Extraction

**Concept:** Ask for all 17 fields regardless of document type

```
Single Prompt → "Extract all fields from this document"
                ↓
Model returns: {field: value or NOT_FOUND}
                ↓
Evaluation: Compare only fields that should exist
```

**Advantages:**
- ✅ No classification errors possible
- ✅ Single prompt for all documents
- ✅ Simpler production pipeline
- ✅ Model decides which fields exist

**Challenges:**
- ❓ Can model handle 17 fields without confusion?
- ❓ Will model hallucinate irrelevant fields?
- ❓ How to evaluate fairly vs two-stage?

### Method 2: Oracle Classification

**Concept:** Use ground truth document type to select correct prompt

```
Read GT doc type → Select schema (5 fields for bank_statement, 14 for invoice/receipt)
                   ↓
Model extracts: Only relevant fields for that document type
                ↓
Evaluation: Always uses correct field set
```

**Advantages:**
- ✅ Perfect classification (oracle knowledge)
- ✅ Isolates extraction quality from classification errors
- ✅ Upper bound on performance
- ✅ Shows what's possible with perfect classification

**Challenges:**
- ⚠️ Not production-viable (requires ground truth)
- ℹ️ Baseline for comparison only

### Method 0: Current Two-Stage (Baseline)

**For comparison:** Keep current approach as baseline

---

## Universal Field Set Definition

**Source:** `config/field_definitions.yaml`
**Total Universal Fields:** 17 (YAML says count: 19, but 17 fields listed - using actual list)
**Invoice/Receipt Fields:** 14 (identical schema for both)
**Bank Statement Fields:** 5

### Field Breakdown (From YAML)

#### Common Fields (2) - Appear in ALL document types
1. `DOCUMENT_TYPE` - Document classification
2. `LINE_ITEM_DESCRIPTIONS` - Itemized entries (transactions for bank, line items for invoice)

#### Invoice/Receipt ONLY Fields (12) - Not in bank statements
3. `BUSINESS_ABN` - Australian Business Number
4. `SUPPLIER_NAME` - Vendor/merchant name
5. `BUSINESS_ADDRESS` - Supplier/merchant address
6. `PAYER_NAME` - Customer/payer name
7. `PAYER_ADDRESS` - Customer/payer address
8. `INVOICE_DATE` - Invoice/receipt date
9. `LINE_ITEM_QUANTITIES` - Quantities for line items
10. `LINE_ITEM_PRICES` - Unit prices
11. `LINE_ITEM_TOTAL_PRICES` - Total price per line item
12. `IS_GST_INCLUDED` - GST inclusion flag (extract from document)
13. `GST_AMOUNT` - GST/tax amount
14. `TOTAL_AMOUNT` - Total amount

#### Bank Statement ONLY Fields (3) - Not in invoices/receipts
15. `STATEMENT_DATE_RANGE` - Statement period (extract from document)
16. `TRANSACTION_DATES` - Transaction dates list
17. `TRANSACTION_AMOUNTS_PAID` - Transaction amounts list

### Document Type Field Counts

| Document Type | Total Fields | Structure |
|--------------|-------------|-----------|
| **Invoice** | 14 | 2 common + 12 invoice-only |
| **Receipt** | 14 | 2 common + 12 invoice-only (same as invoice) |
| **Bank Statement** | 5 | 2 common + 3 bank-only |
| **Universal** | 17 | 2 common + 12 invoice-only + 3 bank-only |

---

## Evaluation Strategy: Option C (Hybrid)

### Primary Metric: Relevant Fields Only

```python
# Only evaluate fields that SHOULD exist for this document type
doc_type = ground_truth["DOCUMENT_TYPE"]
relevant_fields = get_document_type_fields(doc_type)

# Calculate accuracy on relevant fields only
accuracy = evaluate(extracted[relevant_fields], ground_truth[relevant_fields])
```

**This ensures fair comparison with two-stage method.**

### Secondary Metric: False Positive Tracking

```python
# Track if model extracts fields that SHOULDN'T exist
irrelevant_fields = all_17_fields - relevant_fields

false_positives = []
for field in irrelevant_fields:
    if extracted[field] != "NOT_FOUND":
        false_positives.append(field)

false_positive_rate = len(false_positives) / len(irrelevant_fields)
```

**This measures model confusion from universal prompt.**

### Metrics to Report

| Metric | Description | Purpose |
|--------|-------------|---------|
| **Primary Accuracy** | Accuracy on relevant fields only | Fair comparison with two-stage |
| **False Positive Rate** | % of irrelevant fields extracted | Measures hallucination/confusion |
| **False Positive Count** | Number of irrelevant fields extracted | Absolute confusion metric |
| **Field Coverage** | % of relevant fields found | Completeness metric |
| **Processing Time** | Time per image | Efficiency comparison |

---

## Implementation Plan

### Phase 1: Llama-3.2-Vision-11B

#### Notebooks to Create

1. **`llama_batch_universal.ipynb`**
   - Universal extraction (all 17 fields)
   - Single prompt for all documents
   - Option C evaluation
   - Outputs: `llama_universal_batch_results_*.csv`

2. **`llama_batch_oracle.ipynb`**
   - Oracle classification (uses GT doc type)
   - Correct prompt selection (5 or 14 fields)
   - Standard evaluation
   - Outputs: `llama_oracle_batch_results_*.csv`

3. **Keep existing: `llama_batch.ipynb`**
   - Current two-stage approach
   - Baseline for comparison
   - Outputs: `llama_batch_results_*.csv`

#### Prompt Design

**Universal Prompt (Method 1):**
```
Extract the following fields from this document. If a field is not present or not applicable, respond with "NOT_FOUND".

Fields to extract (17 total):
1. DOCUMENT_TYPE
2. BUSINESS_ABN
3. SUPPLIER_NAME
4. BUSINESS_ADDRESS
5. PAYER_NAME
6. PAYER_ADDRESS
7. INVOICE_DATE
8. STATEMENT_DATE_RANGE
9. LINE_ITEM_DESCRIPTIONS
10. LINE_ITEM_QUANTITIES
11. LINE_ITEM_PRICES
12. LINE_ITEM_TOTAL_PRICES
13. IS_GST_INCLUDED
14. GST_AMOUNT
15. TOTAL_AMOUNT
16. TRANSACTION_DATES
17. TRANSACTION_AMOUNTS_PAID

Respond in JSON format: {"FIELD_NAME": "value or NOT_FOUND"}
```

**Oracle Prompt (Method 2):**
```python
# Read ground truth doc type first
doc_type = ground_truth["DOCUMENT_TYPE"]

if doc_type == "bank_statement":
    prompt = get_bank_statement_prompt()  # 5 fields
elif doc_type == "invoice":
    prompt = get_invoice_prompt()  # 14 fields
elif doc_type == "receipt":
    prompt = get_receipt_prompt()  # 14 fields (same schema as invoice)
```

### Phase 2: InternVL3-2B

After Llama results are analyzed, create:

1. **`ivl3_2b_batch_universal.ipynb`**
   - Same structure as Llama universal
   - InternVL3-specific prompt formatting

2. **`ivl3_2b_batch_oracle.ipynb`**
   - Same structure as Llama oracle
   - InternVL3-specific prompt formatting

3. **Keep existing: `ivl3_2b_batch_non_quantized.ipynb`**
   - Current two-stage baseline

---

## Comparison Strategy

### Three-Way Comparison

| Method | Classification | Extraction Schema | Error Source |
|--------|---------------|------------------|--------------|
| **Two-Stage (Baseline)** | Model predicts | Based on model prediction | Classification + Extraction |
| **Universal (Method 1)** | None (all fields) | All 17 fields | Extraction only |
| **Oracle (Method 2)** | Ground truth | Based on GT | Extraction only |

### Key Comparisons

#### 1. Quantify Classification Penalty
```
Classification_Penalty = Oracle_Accuracy - TwoStage_Accuracy
```
Shows accuracy loss purely from classification errors.

#### 2. Test Universal Extraction Viability
```
Universal_vs_Oracle = Universal_Accuracy - Oracle_Accuracy
```
Shows if universal prompt causes confusion (should be ≈0 if viable).

#### 3. Identify Model Limitations
```
If Universal_Accuracy ≈ Oracle_Accuracy:
    → Universal extraction is viable
    → Classification stage is unnecessary overhead

If Universal_Accuracy << Oracle_Accuracy:
    → Universal prompt confuses model
    → Need classification stage
```

### Success Metrics

**Goal:** Quantify classification penalty and identify best production approach

| Scenario | Universal Accuracy | Interpretation | Recommendation |
|----------|-------------------|----------------|----------------|
| Universal ≈ Oracle | ~80% | Model handles universal well | **Use Method 1 (Universal)** |
| Universal < Oracle by 5-10% | ~70-75% | Minor confusion from universal prompt | Test in production, monitor |
| Universal << Oracle by >15% | <65% | Significant confusion | **Keep two-stage OR improve classification** |

---

## Evaluation Code Changes

### New Evaluation Function

```python
def evaluate_with_false_positive_tracking(
    extracted_data: Dict[str, str],
    ground_truth: Dict[str, str],
    image_path: str,
    method: str = "universal"
) -> Dict:
    """
    Evaluate extraction with Option C strategy.

    Args:
        extracted_data: All extracted fields (17 for universal)
        ground_truth: Ground truth data
        image_path: Image identifier
        method: "universal", "oracle", or "two_stage"

    Returns:
        {
            'primary_accuracy': float,  # Relevant fields only
            'false_positive_rate': float,  # Irrelevant fields extracted
            'false_positive_count': int,
            'false_positive_fields': List[str],
            'relevant_field_count': int,
            'field_scores': Dict[str, float]
        }
    """
    # Get document type from ground truth
    doc_type = ground_truth.get("DOCUMENT_TYPE", "invoice").lower()

    # Get relevant fields for this document type
    relevant_fields = get_document_type_fields(doc_type)
    all_fields = get_all_universal_fields()  # All 17 fields
    irrelevant_fields = set(all_fields) - set(relevant_fields)

    # PRIMARY METRIC: Evaluate relevant fields only
    primary_scores = {}
    for field in relevant_fields:
        extracted_value = extracted_data.get(field, "NOT_FOUND")
        ground_truth_value = ground_truth.get(field, "NOT_FOUND")

        accuracy = calculate_field_accuracy_with_method(
            extracted_value, ground_truth_value, field
        )
        primary_scores[field] = accuracy

    primary_accuracy = sum(s['f1_score'] for s in primary_scores.values()) / len(relevant_fields)

    # SECONDARY METRIC: Track false positives
    false_positives = []
    for field in irrelevant_fields:
        if extracted_data.get(field, "NOT_FOUND") != "NOT_FOUND":
            false_positives.append(field)

    return {
        'primary_accuracy': primary_accuracy * 100,  # 0-100%
        'false_positive_rate': len(false_positives) / len(irrelevant_fields) if irrelevant_fields else 0,
        'false_positive_count': len(false_positives),
        'false_positive_fields': false_positives,
        'relevant_field_count': len(relevant_fields),
        'field_scores': primary_scores
    }
```

### CSV Output Format

**Additional columns for universal/oracle methods:**

```csv
image_name, document_type, primary_accuracy, false_positive_rate, false_positive_count,
false_positive_fields, relevant_field_count, processing_time, method,
[... all 17 field columns ...]
```

---

## Workflow

### Step-by-Step Execution

#### Step 1: Llama Baseline (Current Two-Stage)
```bash
# Run on remote H200 machine
jupyter notebook llama_batch.ipynb

# Output: llama_batch_results_TIMESTAMP.csv
```

#### Step 2: Llama Universal Extraction
```bash
jupyter notebook llama_batch_universal.ipynb

# Output: llama_universal_batch_results_TIMESTAMP.csv
```

#### Step 3: Llama Oracle Classification
```bash
jupyter notebook llama_batch_oracle.ipynb

# Output: llama_oracle_batch_results_TIMESTAMP.csv
```

#### Step 4: Llama Comparison Analysis
```bash
# Use post_processing_evaluation.ipynb or create new comparison notebook
# Compare: Two-stage vs Universal vs Oracle
```

#### Step 5: Repeat for InternVL3-2B
```bash
jupyter notebook ivl3_2b_batch_non_quantized.ipynb  # Baseline
jupyter notebook ivl3_2b_batch_universal.ipynb      # Universal
jupyter notebook ivl3_2b_batch_oracle.ipynb         # Oracle
```

#### Step 6: Final Comparison
```bash
# Use model_comparison.ipynb
# Compare all 6 runs:
# - llama_two_stage, llama_universal, llama_oracle
# - ivl3_two_stage, ivl3_universal, ivl3_oracle
```

---

## Expected Outcomes

### Hypothesis

**If classification penalty is significant (>10% accuracy loss):**
- Oracle accuracy >> Two-stage accuracy
- Universal accuracy ≈ Oracle accuracy (if model handles universal well)
- **Conclusion:** Classification is the bottleneck

**If extraction is the bottleneck:**
- Oracle accuracy ≈ Two-stage accuracy
- Universal accuracy may be lower (confusion from 17 fields)
- **Conclusion:** Need better extraction prompts, not better classification

### Deliverables

1. **Quantitative Metrics**
   - Classification penalty: `Oracle_Accuracy - TwoStage_Accuracy`
   - Universal viability: `Universal_Accuracy - Oracle_Accuracy`
   - Per-document-type breakdown
   - False positive rates for universal method

2. **Comparison Report**
   - Side-by-side accuracy tables
   - Document type performance breakdown
   - Processing time comparison
   - Recommendations for production

3. **Updated Documentation**
   - `DOCUMENT_TYPE_EVALUATION_PENALTIES.md` - Add Method 1 & 2 results
   - Executive summary with recommendations

---

## Implementation Order

### Priority 1: Create Llama Notebooks (THIS SPRINT) ✅
- [x] `llama_batch_universal.ipynb` - **COMPLETE** (24 cells)
- [x] `llama_batch_oracle.ipynb` - **COMPLETE** (13 cells)
- [ ] Test on small batch (5-10 images) - **PENDING: Remote execution on H200**
- [ ] Verify CSV outputs are correct
- [ ] Run full evaluation

### Priority 2: Analyze Llama Results
- [ ] Compare three methods
- [ ] Calculate classification penalty
- [ ] Assess universal extraction viability
- [ ] Document findings

### Priority 3: Create InternVL3 Notebooks (NEXT SPRINT)
- [ ] `ivl3_2b_batch_universal.ipynb`
- [ ] `ivl3_2b_batch_oracle.ipynb`
- [ ] Run full evaluation
- [ ] Compare with Llama results

### Priority 4: Final Analysis & Recommendations
- [ ] Cross-model comparison
- [ ] Production strategy recommendation
- [ ] Update penalty documentation

---

## Field Dependencies and Relationships

### Correlated Fields

**Transaction Fields (Bank Statement):**
- `TRANSACTION_DATES`, `LINE_ITEM_DESCRIPTIONS`, and `TRANSACTION_AMOUNTS_PAID` are related
- Same list position represents the same transaction
- `STATEMENT_DATE_RANGE` is extracted separately from the document (typically found at top of statement)
- **Evaluation consideration:** Position-aware or correlation-aware F1 needed for transaction lists

**Line Item Fields (Invoice/Receipt):**
- `LINE_ITEM_DESCRIPTIONS`, `LINE_ITEM_QUANTITIES`, `LINE_ITEM_PRICES`, `LINE_ITEM_TOTAL_PRICES` are related
- Same list position represents the same line item
- **Evaluation consideration:** Position-aware or correlation-aware F1 needed

### Implications for Universal Extraction

When models extract all 17 fields, watch for:
1. **Document type confusion:** Invoice fields on bank statements (false positives)
2. **List alignment:** Related lists should have same length

---

## Technical Considerations

### Prompt Engineering

**Universal Prompt Design:**
- Clear field definitions
- Explicit NOT_FOUND instructions
- JSON format enforcement
- Field order (common fields first?)

**Oracle Prompt Design:**
- Reuse existing two-stage Stage 2 prompts
- Just skip Stage 1 classification
- Use GT doc type for schema selection

### Memory Management

- Universal extraction may use more tokens (17 fields vs 5/14)
- Monitor GPU memory usage
- Batch size may need adjustment

### Ground Truth CSV

Ensure `evaluation_data/ground_truth.csv` has:
- `DOCUMENT_TYPE` column (for oracle method)
- All 17 field columns (some will be NOT_FOUND)
- Consistent field naming with universal schema

---

## Success Criteria

### Must Achieve
1. ✅ All 6 notebooks run without errors
2. ✅ CSV outputs compatible with `model_comparison.ipynb`
3. ✅ Quantitative classification penalty measured
4. ✅ Universal extraction viability determined

### Should Achieve
1. ✅ Oracle accuracy > Two-stage accuracy (proves classification penalty exists)
2. ✅ Universal accuracy within 5% of Oracle (proves universal is viable)
3. ✅ False positive rate < 10% for universal method
4. ✅ Processing time comparable across methods

### Nice to Have
1. ✅ Per-document-type penalty breakdown
2. ✅ Field-level error analysis
3. ✅ Automated comparison report generation
4. ✅ Production deployment recommendation

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Universal prompt confuses models | Test on small batch first, iterate prompt |
| Ground truth missing doc types | Validate GT CSV before running |
| Memory issues with 17 fields | Reduce batch size, monitor GPU |
| Inconsistent field naming | Create field mapping from `field_definitions.yaml` |
| Long processing times | Run overnight, use progress tracking |

---

## Next Steps

1. **Review this plan** with senior data scientists
2. **Confirm field definitions** from `config/field_definitions.yaml`
3. **Validate ground truth CSV** has all required columns
4. **Create first notebook**: `llama_batch_universal.ipynb`
5. **Test on small batch** before full run

---

## Appendix: Field Mapping Reference

```yaml
# From config/field_definitions.yaml (ACTUAL DEFINITIONS)

COMMON_FIELDS (2):
  - DOCUMENT_TYPE
  - LINE_ITEM_DESCRIPTIONS

BANK_STATEMENT_ONLY_FIELDS (3):
  - STATEMENT_DATE_RANGE
  - TRANSACTION_DATES
  - TRANSACTION_AMOUNTS_PAID

BANK_STATEMENT_TOTAL_FIELDS (5):
  - DOCUMENT_TYPE
  - STATEMENT_DATE_RANGE
  - LINE_ITEM_DESCRIPTIONS
  - TRANSACTION_DATES
  - TRANSACTION_AMOUNTS_PAID

INVOICE_RECEIPT_ONLY_FIELDS (12):
  - BUSINESS_ABN
  - SUPPLIER_NAME
  - BUSINESS_ADDRESS
  - PAYER_NAME
  - PAYER_ADDRESS
  - INVOICE_DATE
  - LINE_ITEM_QUANTITIES
  - LINE_ITEM_PRICES
  - LINE_ITEM_TOTAL_PRICES
  - IS_GST_INCLUDED
  - GST_AMOUNT
  - TOTAL_AMOUNT

INVOICE_RECEIPT_TOTAL_FIELDS (14):
  - DOCUMENT_TYPE
  - BUSINESS_ABN
  - SUPPLIER_NAME
  - BUSINESS_ADDRESS
  - PAYER_NAME
  - PAYER_ADDRESS
  - INVOICE_DATE
  - LINE_ITEM_DESCRIPTIONS
  - LINE_ITEM_QUANTITIES
  - LINE_ITEM_PRICES
  - LINE_ITEM_TOTAL_PRICES
  - IS_GST_INCLUDED
  - GST_AMOUNT
  - TOTAL_AMOUNT

UNIVERSAL_FIELDS (17):
  - DOCUMENT_TYPE
  - BUSINESS_ABN
  - SUPPLIER_NAME
  - BUSINESS_ADDRESS
  - PAYER_NAME
  - PAYER_ADDRESS
  - INVOICE_DATE
  - STATEMENT_DATE_RANGE
  - LINE_ITEM_DESCRIPTIONS
  - LINE_ITEM_QUANTITIES
  - LINE_ITEM_PRICES
  - LINE_ITEM_TOTAL_PRICES
  - IS_GST_INCLUDED
  - GST_AMOUNT
  - TOTAL_AMOUNT
  - TRANSACTION_DATES
  - TRANSACTION_AMOUNTS_PAID
```

---

**Document Version:** 2.1
**Last Updated:** 2025-11-10
**Status:** Phase 1 Implementation Complete - Ready for Testing
**Changes:**
- v1.1: Updated field definitions to match actual YAML (17 universal fields, correct field names)
- v1.2: Explicitly mentioned receipt schema (14 fields, same as invoice) in Oracle method
- v1.3-1.5: REMOVED - incorrect derivation documentation
- v2.0: **CRITICAL CORRECTION** - ALL fields must be extracted from documents, NO derived fields
  - IS_GST_INCLUDED is extracted (not derived from GST_AMOUNT)
  - STATEMENT_DATE_RANGE is extracted (not derived from TRANSACTION_DATES)
  - Models must extract all field values directly from the document
- v2.1: **IMPLEMENTATION COMPLETE** - Both Llama notebooks created and validated
  - `llama_batch_universal.ipynb` - 24 cells, loads prompt from `prompts/universal.yaml`
  - `llama_batch_oracle.ipynb` - 13 cells, uses ground truth doc type for schema selection
  - Ready for remote testing on H200 machine
