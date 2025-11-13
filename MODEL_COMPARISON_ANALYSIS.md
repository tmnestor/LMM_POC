# Model Comparison Analysis Report

**Auto-Generated from Notebook**: 2025-11-13 22:58:29
**Source**: `model_comparison.ipynb`
**Dataset**: 195 documents (30 bank statements, 61 invoices, 104 receipts)
**Evaluation Fields**: 17 business document fields

---

## Executive Summary

### Overall Performance Metrics

| Model | F1 Score | Precision | Recall | Accuracy | Median Speed | Throughput |
|-------|----------|-----------|--------|----------|--------------|------------|
| **Llama-3.2-Vision-11B** | 0.0000 | 0.0000 | 0.0000 | 0.00% | 0.0s | 0.0 docs/min |
| **InternVL3-Quantized-8B** | 0.0000 | 0.0000 | 0.0000 | 0.00% | 0.0s | 0.0 docs/min |
| **InternVL3-NonQuantized-2B** | 0.0000 | 0.0000 | 0.0000 | 0.00% | 0.0s | 0.0 docs/min |

### Key Findings

**Winner (F1 Score)**: InternVL3-NonQuantized-2B

**Highest Precision**: InternVL3-NonQuantized-2B (0.0000)

**Highest Recall**: InternVL3-NonQuantized-2B (0.0000)

**Fastest**: InternVL3-NonQuantized-2B (0.0s)

---

## Visualizations

All visualizations are generated in `output/visualizations/`:

### 1. Executive Performance Dashboard
![Executive Dashboard](output/visualizations/executive_comparison.png)

**6-panel comprehensive view:**
- Overall accuracy distribution (box plots)
- Processing speed comparison
- Accuracy by document type
- Processing time by document type
- Efficiency analysis (accuracy vs speed)
- Performance summary table

### 2. Document Type Classification
![Document Type Confusion](output/visualizations/doctype_confusion_matrix.png)

**3-model confusion matrices** showing classification performance for:
- Bank Statements (30 docs, 15.4%)
- Invoices (61 docs, 31.3%)
- Receipts (104 docs, 53.3%)

### 3. Field Extraction Status
![Field Confusion Heatmap](output/visualizations/field_confusion_heatmap.png)

**Breakdown of extraction status:**
- Correct extractions (matches ground truth)
- Incorrect extractions (wrong value)
- Not Found (field not extracted)

### 4. Per-Field Metrics
![Per-Field Metrics](output/visualizations/per_field_metrics.png)

**4-panel analysis:**
- F1 Score by field
- Precision by field
- Recall by field
- Accuracy by field

### 5. Hallucination Analysis
![Hallucination Analysis](output/visualizations/hallucination_analysis.png)

**9-panel breakdown:**
- Overall hallucination rates
- Hallucinations vs correct NOT_FOUND
- Hallucination-recall tradeoff
- Per-field hallucination (3 models)
- Document-level distribution (3 models)

---

## Per-Field Performance Summary

### Field-Level Accuracy by Model

| Field | Llama-11B | InternVL3-8B | InternVL3-2B | Best Model | Best Score |
|-------|-----------|--------------|--------------|------------|------------|
| DOCUMENT_TYPE | 0.0% | 0.0% | 0.0% | InternVL3-NonQuantized-2B | 1.0% |
| TOTAL_AMOUNT | 0.0% | 0.0% | 0.0% | InternVL3-NonQuantized-2B | 1.0% |
| GST_AMOUNT | 0.0% | 0.0% | 0.0% | InternVL3-NonQuantized-2B | 1.0% |
| PAYER_ADDRESS | 0.0% | 0.0% | 0.0% | InternVL3-NonQuantized-2B | 1.0% |
| STATEMENT_DATE_RANGE | 0.0% | 0.0% | 0.0% | InternVL3-Quantized-8B | 1.0% |
| LINE_ITEM_QUANTITIES | 0.0% | 0.0% | 0.0% | Llama-3.2-Vision | 1.0% |
| BUSINESS_ABN | 0.0% | 0.0% | 0.0% | Llama-3.2-Vision | 1.0% |
| LINE_ITEM_TOTAL_PRICES | 0.0% | 0.0% | 0.0% | Llama-3.2-Vision | 1.0% |
| INVOICE_DATE | 0.0% | 0.0% | 0.0% | InternVL3-NonQuantized-2B | 0.8% |
| BUSINESS_ADDRESS | 0.0% | 0.0% | 0.0% | InternVL3-NonQuantized-2B | 0.8% |
| PAYER_NAME | 0.0% | 0.0% | 0.0% | InternVL3-NonQuantized-2B | 0.7% |
| SUPPLIER_NAME | 0.0% | 0.0% | 0.0% | InternVL3-NonQuantized-2B | 0.7% |
| LINE_ITEM_DESCRIPTIONS | 0.0% | 0.0% | 0.0% | Llama-3.2-Vision | 0.9% |
| TRANSACTION_DATES | 0.0% | 0.0% | 0.0% | Llama-3.2-Vision | 0.7% |
| LINE_ITEM_PRICES | 0.0% | 0.0% | 0.0% | Llama-3.2-Vision | 1.0% |
| IS_GST_INCLUDED | 0.0% | 0.0% | 0.0% | Llama-3.2-Vision | 1.0% |
| TRANSACTION_AMOUNTS_PAID | 0.0% | 0.0% | 0.0% | InternVL3-NonQuantized-2B | 0.0% |

---

## Model Specialization

### Fields Where Each Model Performs Best

| Model | Best-Performing Fields | Percentage | Count |
|-------|----------------------|------------|-------|
| **Llama-3.2-Vision-11B** | 0.0% | 0/17 | SECONDARY |
| **InternVL3-Quantized-8B** | 0.0% | 0/17 | SECONDARY |
| **InternVL3-NonQuantized-2B** | 0.0% | 0/17 | NO SPECIALIZATION |

---

## Deployment Recommendations

Based on the analysis above:

1. **Document Classification (PRIMARY)**: Use the model with highest document type classification accuracy
2. **Field Extraction (SECONDARY)**: Consider ensemble approach leveraging each model's field specialization
3. **High-Volume Processing**: Balance speed vs quality based on throughput requirements
4. **Hallucination Sensitivity**: Choose model based on tolerance for false positives vs false negatives

---

## Related Documentation

- [three_model_field_comparison.md](three_model_field_comparison.md) - Detailed field-by-field analysis
- [ACCURACY_PARADOX_EXPLAINED.md](ACCURACY_PARADOX_EXPLAINED.md) - Why F1 > Accuracy for extraction
- [HALLUCINATION_ANALYSIS_ADDED.md](HALLUCINATION_ANALYSIS_ADDED.md) - Hallucination analysis methodology

---

**Report Auto-Generated**: 2025-11-13 22:58:29
**Source Notebook**: `model_comparison.ipynb`
**Visualizations**: `output/visualizations/`
**Next Update**: Re-run notebook to refresh all metrics and visualizations
