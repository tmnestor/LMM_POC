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

## Schema Fields

The evaluation covers 17 fields extracted from business documents (invoices, receipts, bank statements):

| Field | Description |
|-------|-------------|
| DOCUMENT_TYPE | Classification: Invoice, Receipt, or Bank Statement |
| BUSINESS_ABN | Australian Business Number (11-digit identifier) |
| SUPPLIER_NAME | Vendor or merchant name |
| BUSINESS_ADDRESS | Supplier's address |
| PAYER_NAME | Customer or payer name |
| PAYER_ADDRESS | Customer's address |
| INVOICE_DATE | Date of invoice or receipt |
| LINE_ITEM_DESCRIPTIONS | Product/service descriptions (list) |
| LINE_ITEM_QUANTITIES | Quantities per line item (list) |
| LINE_ITEM_PRICES | Unit prices per line item (list) |
| LINE_ITEM_TOTAL_PRICES | Extended prices per line item (list) |
| IS_GST_INCLUDED | Whether GST is included in totals |
| GST_AMOUNT | Goods and Services Tax amount |
| TOTAL_AMOUNT | Invoice/receipt total |
| STATEMENT_DATE_RANGE | Bank statement period |
| TRANSACTION_DATES | Dates of bank transactions (list) |
| TRANSACTION_AMOUNTS_PAID | Transaction amounts (list) |

### Critical Fields

Four fields are designated as **critical** due to their importance for financial reconciliation:

- **BUSINESS_ABN**: Required for tax compliance and vendor identification
- **SUPPLIER_NAME**: Essential for vendor matching
- **GST_AMOUNT**: Required for GST/BAS reporting
- **TOTAL_AMOUNT**: Core financial data for reconciliation

### Field Categories (Panel B)

Fields are grouped into five categories for analysis:

| Category | Fields |
|----------|--------|
| **Identity** | DOCUMENT_TYPE*, BUSINESS_ABN, SUPPLIER_NAME |
| **Address** | BUSINESS_ADDRESS, PAYER_NAME, PAYER_ADDRESS |
| **Dates** | INVOICE_DATE, STATEMENT_DATE_RANGE*, TRANSACTION_DATES |
| **Line Items** | LINE_ITEM_DESCRIPTIONS, LINE_ITEM_QUANTITIES, LINE_ITEM_PRICES, LINE_ITEM_TOTAL_PRICES |
| **Financial** | IS_GST_INCLUDED, GST_AMOUNT, TOTAL_AMOUNT, TRANSACTION_AMOUNTS_PAID |

*Exclusive to InternVL3.5-8B

## Accuracy vs F1: Understanding the Metrics

This report uses two distinct evaluation metrics that serve different purposes:

### Document-Level Accuracy (`overall_accuracy`)

Used in: **Executive Dashboard**, **Summary Statistics**

- **Definition**: Percentage of fields correctly extracted per document
- **Formula**: `(fields_matched / total_fields) × 100`
- **Scope**: Evaluates complete document extraction quality
- **Scale**: 0-100%
- **Use case**: Quick comparison of overall model performance

### Per-Field F1 Score

Used in: **Detailed Field Analysis**, **Statistical Comparisons**

- **Definition**: Harmonic mean of Precision and Recall computed per field type
- **Formula**: `F1 = 2 × (Precision × Recall) / (Precision + Recall)`
- **Scope**: Evaluates extraction quality for each specific field (e.g., SUPPLIER_NAME, TOTAL_AMOUNT)
- **Scale**: 0-100%
- **Use case**: Identifying which fields perform well vs need improvement

### Why These Metrics Differ

The same model may show different values for Accuracy vs F1 because:

1. **Aggregation level**: Accuracy averages across documents; F1 averages across field types
2. **Field weighting**: Accuracy weights all fields equally per document; F1 treats each field type independently
3. **List fields impact**: Position-aware F1 for list fields (LINE_ITEM_*) can be stricter than document accuracy

**Example**: A model might achieve 85% document accuracy but 72% mean F1 if it struggles with specific field types like LINE_ITEM_TOTAL_PRICES.

---

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

### Position-Aware Matching Example

```
Ground Truth: ["apple", "banana", "cherry"]
Extracted:    ["banana", "apple", "cherry"]

Position 0: "banana" ≠ "apple"  → FP + FN
Position 1: "apple" ≠ "banana" → FP + FN
Position 2: "cherry" = "cherry" → TP

Result: TP=1, FP=2, FN=2
Precision = 1/3 = 33.3%
Recall = 1/3 = 33.3%
F1 = 33.3%
```

This strict position-aware approach penalizes ordering errors, ensuring extracted line items align correctly with ground truth.

### Alternative: Position-Agnostic Matching

A separate function `calculate_field_accuracy_f1_position_agnostic()` is available for set-based matching where order doesn't matter:

```
Ground Truth: ["apple", "banana", "cherry"]
Extracted:    ["banana", "apple", "cherry"]

Set comparison: All 3 items present
Result: TP=3, FP=0, FN=0 → F1 = 100%
```

The notebook uses **position-aware** matching by default for stricter evaluation.
