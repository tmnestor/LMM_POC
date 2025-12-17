# Model Comparison Summary

**InternVL3.5-8B vs LayoutLM** across 15 comparable document fields.

## Key Results

InternVL3.5-8B achieves **72.5% mean F1** compared to LayoutLM's 54.6%—a **+17.9% improvement**.

| Metric | InternVL3.5-8B | LayoutLM |
|--------|----------------|----------|
| Mean F1 | 72.5% | 54.6% |
| Critical Fields | 82.6% | 51.7% |

### Areas for Improvement

InternVL3.5-8B underperforms LayoutLM on two fields:

| Field | InternVL3.5-8B | LayoutLM | Gap |
|-------|----------------|----------|-----|
| LINE_ITEM_TOTAL_PRICES | 29% | 51% | -22% |
| LINE_ITEM_DESCRIPTIONS | 44% | 56% | -12% |

## Statistical Significance

The improvement is statistically significant (p=0.0105) with a **large effect size** (Cohen's d=0.83). The 95% confidence interval (+6% to +29%) excludes zero.

## Additional Capabilities

InternVL3.5-8B exclusively supports DOCUMENT_TYPE (72.8%) and STATEMENT_DATE_RANGE (92.8%)—fields unavailable in LayoutLM—with 82.8% mean accuracy.

## F1 Calculation Methodology

F1 scores for InternVL3.5-8B were computed using `calculate_field_accuracy_f1()` with **position-aware matching**:

### Single-Value Fields
- **Text fields** (SUPPLIER_NAME, BUSINESS_ADDRESS): Levenshtein distance with 0.5 ANLS threshold
- **ID fields** (BUSINESS_ABN): Exact match after normalization
- **Monetary fields** (TOTAL_AMOUNT, GST_AMOUNT): Numeric comparison with 1% tolerance
- **Date fields** (INVOICE_DATE): Semantic date parsing with format flexibility

### List Fields (Pipe-Separated Values)
Uses **position-aware (order-aware)** matching where items must match both in value AND position:

- **TP** (True Positive): Items matching at same position
- **FP** (False Positive): Extra extracted items or mismatches
- **FN** (False Negative): Missing or mismatched ground truth items

**Formulas:**
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 × (Precision × Recall) / (Precision + Recall)

This strict position-aware approach penalizes ordering errors, ensuring extracted line items align correctly with ground truth.
