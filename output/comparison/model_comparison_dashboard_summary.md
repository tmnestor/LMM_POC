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
