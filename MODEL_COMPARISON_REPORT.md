# Model Comparison Analysis Report

**Auto-Generated from Notebook**: 2025-12-09 13:43:53
**Source**: `model_comparison_reporter.ipynb`
**Dataset**: 9 documents (3 bank statements, 3 invoices, 3 receipts)
**Evaluation Fields**: 17 business document fields

---

## Executive Summary

### Overall Performance Metrics

| Model | F1 Score | Precision | Recall | Accuracy | Median Speed | Throughput |
|-------|----------|-----------|--------|----------|--------------|------------|
| **Llama-3.2-Vision-11B** | 0.9032 | 0.9032 | 0.9032 | 97.64% | 142.0s | 0.4 docs/min |
| **InternVL3-8B** | 0.6741 | 0.7664 | 0.6016 | 67.06% | 66.3s | 0.9 docs/min |
| **InternVL3.5-8B** | 0.8858 | 0.8957 | 0.8762 | 90.48% | 55.4s | 1.1 docs/min |

### Key Findings

**Winner (F1 Score)**: Llama-3.2-Vision-11B

**Highest Precision**: Llama-3.2-Vision-11B (0.9032)

**Highest Recall**: Llama-3.2-Vision-11B (0.9032)

**Fastest**: InternVL3.5-8B (55.4s)

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
- Bank Statements (3 docs, 33.3%)
- Invoices (3 docs, 33.3%)
- Receipts (3 docs, 33.3%)

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


### 5. Field-Level Accuracy Analysis
![Field-Level Accuracy](output/visualizations/field_level_accuracy.png)

**3-panel comprehensive view:**
- Field accuracy comparison (horizontal bar chart across all models)
- Field accuracy heatmap (color-coded performance matrix)
- Model specialization distribution (fields where each model performs best)

### 6. Hallucination Analysis
![Hallucination Analysis](output/visualizations/hallucination_analysis.png)

**9-panel breakdown:**
- Overall hallucination rates
- Hallucinations vs correct NOT_FOUND
- Hallucination-recall tradeoff
- Per-field hallucination (3 models)
- Document-level distribution (3 models)

### Hallucination Rates

| Model | Hallucination Rate | Correct NOT_FOUND Rate | Total Hallucinations |
|-------|-------------------|------------------------|----------------------|
| **Llama-11B** | 0.0% | 100.0% | 0 |
| **InternVL3-8B** | 0.0% | 100.0% | 0 |
| **InternVL3-2B** | 0.0% | 100.0% | 0 |

**Interpretation:**
- **Hallucination Rate**: % of NOT_FOUND fields where model invented a value
- **Correct NOT_FOUND Rate**: % of NOT_FOUND fields correctly identified as absent

---

## Per-Field Performance Summary

### Field-Level Accuracy by Model

| Field | Llama-11B | InternVL3-8B | InternVL3-2B | Best Model | Best Score |
|-------|-----------|--------------|--------------|------------|------------|
| DOCUMENT_TYPE | 100.0% | 100.0% | 100.0% | InternVL3-8B | 100.0% |
| GST_AMOUNT | 100.0% | 100.0% | 100.0% | InternVL3-8B | 100.0% |
| PAYER_NAME | 100.0% | 98.3% | 100.0% | InternVL3.5-8B | 100.0% |
| PAYER_ADDRESS | 100.0% | 94.8% | 100.0% | InternVL3.5-8B | 100.0% |
| BUSINESS_ABN | 100.0% | 83.3% | 100.0% | InternVL3.5-8B | 100.0% |
| TOTAL_AMOUNT | 100.0% | 83.3% | 100.0% | InternVL3.5-8B | 100.0% |
| LINE_ITEM_DESCRIPTIONS | 99.3% | 69.0% | 99.1% | Llama-3.2-Vision-11B | 99.3% |
| LINE_ITEM_QUANTITIES | 100.0% | 83.3% | 83.3% | Llama-3.2-Vision-11B | 100.0% |
| STATEMENT_DATE_RANGE | 100.0% | 66.7% | 100.0% | InternVL3.5-8B | 100.0% |
| SUPPLIER_NAME | 85.7% | 85.7% | 85.7% | InternVL3-8B | 85.7% |
| INVOICE_DATE | 83.3% | 83.3% | 83.3% | InternVL3-8B | 83.3% |
| BUSINESS_ADDRESS | 83.3% | 78.9% | 83.3% | InternVL3.5-8B | 83.3% |
| LINE_ITEM_PRICES | 100.0% | 33.3% | 100.0% | InternVL3.5-8B | 100.0% |
| TRANSACTION_DATES | 97.8% | 33.3% | 100.0% | InternVL3.5-8B | 100.0% |
| LINE_ITEM_TOTAL_PRICES | 100.0% | 33.3% | 83.3% | Llama-3.2-Vision-11B | 100.0% |
| TRANSACTION_AMOUNTS_PAID | 66.7% | 0.0% | 100.0% | InternVL3.5-8B | 100.0% |
| IS_GST_INCLUDED | 100.0% | 0.0% | 0.0% | Llama-3.2-Vision-11B | 100.0% |

---

## Model Specialization

### Fields Where Each Model Performs Best

| Model | Best-Performing Fields | Percentage | Count |
|-------|----------------------|------------|-------|
| **Llama-3.2-Vision-11B** | 23.5% | 4/17 | SECONDARY |
| **InternVL3-8B** | 23.5% | 4/17 | SECONDARY |
| **InternVL3.5-8B** | 52.9% | 9/17 | PRIMARY |

---

## Deployment Recommendations

Based on the analysis above:

### 1. Document Classification (PRIMARY)
Use the model with highest document type classification accuracy for initial routing and categorization.

### 2. Field Extraction Strategy (SECONDARY)
Consider an ensemble approach leveraging each model's field specialization:
- Use model-specific strengths for particular fields
- Implement confidence-based routing
- Fall back to best overall performer for general fields

### 3. High-Volume Processing
Balance speed vs quality based on throughput requirements:
- **Fastest processing**: InternVL3.5-8B (~55.4s/doc)
- **Best accuracy**: Llama-3.2-Vision-11B (97.64% overall)
- **Best balance**: Consider throughput constraints and acceptable accuracy threshold

### 4. Hallucination Sensitivity: Critical Business Decision

#### Understanding Hallucination in Document Extraction

**Hallucination** = Model extracts a value when ground truth is `NOT_FOUND`

**Example:**
- Ground Truth: `BUSINESS_ABN = NOT_FOUND` (field doesn't exist in document)
- Model Output: `BUSINESS_ABN = "12345678901"` ← **HALLUCINATION** (invented data)

#### The Tradeoff: Precision vs Recall

**High Precision (Low Hallucination)**
- Model only extracts when very confident
- **Few false positives** (hallucinations)
- **Many false negatives** (missed fields)
- Conservative approach: "Only extract what you're sure about"

**High Recall (Risk of Hallucination)**
- Model extracts aggressively to catch all fields
- **Few false negatives** (catches most fields)
- **More false positives** (risk of hallucinations)
- Aggressive approach: "Extract everything, review later"

#### Relationship to Metrics

```
Precision = Correct Extractions / All Extractions
  → High precision = Low hallucination rate
  → Model is cautious, only extracts when confident

Recall = Correct Extractions / All Fields That Should Be Extracted
  → High recall = Catches more fields
  → Risk: May hallucinate to achieve higher coverage

Hallucination Rate = Hallucinations / NOT_FOUND Opportunities
  → Direct measure of false positive risk
  → Critical for production reliability
```

#### Model Selection Guide Based on Use Case

**Choose HIGH PRECISION Model (Llama-3.2-Vision-11B: 90.32%) if:**
- ✅ Processing financial/regulatory data (invoices, tax documents)
- ✅ Automated processing with no human review
- ✅ **False data is worse than missing data**
- ✅ You can afford to manually review `NOT_FOUND` fields
- ✅ Compliance and audit requirements
- ✅ Low tolerance for hallucinations

**Example**: Bank reconciliation where a hallucinated amount could cause financial errors.

**Choose HIGH RECALL Model (Llama-3.2-Vision-11B: 90.32%) if:**
- ✅ Comprehensive data capture is critical
- ✅ Human review pipeline can catch errors
- ✅ **Missing data is worse than wrong data**
- ✅ Initial screening/discovery use case
- ✅ Maximizing field coverage is priority
- ✅ Can tolerate some false positives

**Example**: Legal document discovery where missing a field could have serious consequences.

**Choose BALANCED Model (for high-volume processing) if:**
- ✅ High-volume processing requirements
- ✅ Need reasonable precision and recall
- ✅ Speed is a critical factor
- ✅ Standard business document processing

**Example**: Receipt processing for expense management with human spot-checking.

#### Your Model Performance Profile

Based on the analysis:

| Model | Precision | Recall | F1 | Best For |
|-------|-----------|--------|----|----|
| **Llama-3.2-Vision-11B** | 90.32% | 90.32% | 0.9032 | 🏆 Best Precision🏆 Best Recall🏆 Best F1 |
| **InternVL3-8B** | 76.64% | 60.16% | 0.6741 |  |
| **InternVL3.5-8B** | 89.57% | 87.62% | 0.8858 |  |

**Key Insights:**
- **Precision Leader**: Llama-3.2-Vision-11B (90.32%)
- **Recall Leader**: Llama-3.2-Vision-11B (90.32%)
- **F1 Leader**: Llama-3.2-Vision-11B (0.9032)
- **Speed vs Accuracy Tradeoff**: Consider throughput requirements against quality needs

#### Efficiency Analysis

**Performance Efficiency Score** = Accuracy × Throughput (docs/min)

| Model | Avg Accuracy | Avg Speed | Throughput | Efficiency Score |
|-------|--------------|-----------|------------|------------------|
| **Llama-3.2-Vision-11B** | 97.64% | 142.0s | 0.4 docs/min | 41.3 |
| **InternVL3-8B** | 67.06% | 66.3s | 0.9 docs/min | 60.7 |
| **InternVL3.5-8B** | 90.48% | 55.4s | 1.1 docs/min | 98.0 |

**Highest Efficiency**: InternVL3.5-8B





#### Document-Type Specific Recommendations

**Best Model by Document Type:**

- **Bank Statement**: Llama-3.2-Vision-11B (95.31% accuracy)
- **Invoice**: Llama-3.2-Vision-11B (97.62% accuracy)
- **Receipt**: Llama-3.2-Vision-11B (100.00% accuracy)

#### Field Performance Insights

**Fields with Significant Model Performance Differences (>20% spread):**

- **TRANSACTION_AMOUNTS_PAID**: Use InternVL3.5-8B (100% vs 0%, +100% advantage)
- **IS_GST_INCLUDED**: Use Llama-3.2-Vision-11B (100% vs 0%, +100% advantage)
- **LINE_ITEM_PRICES**: Use InternVL3.5-8B (100% vs 33%, +67% advantage)
- **TRANSACTION_DATES**: Use InternVL3.5-8B (100% vs 33%, +67% advantage)
- **LINE_ITEM_TOTAL_PRICES**: Use Llama-3.2-Vision-11B (100% vs 33%, +67% advantage)

**⚠️ Problematic Fields Requiring Attention (<50% avg accuracy):**

- **IS_GST_INCLUDED**: 33% average accuracy - Consider prompt optimization or additional fine tuning


#### Production Deployment Strategy

**Phase 1: Initial Deployment**
1. Choose model based on your primary business constraint:
   - **Financial accuracy** → Highest precision model
   - **Data completeness** → Highest recall model
   - **High volume** → Fastest processing model

**Phase 2: Monitoring**
2. Track in production:
   - Hallucination rate on `NOT_FOUND` fields
   - Manual review costs (false negatives)
   - Error correction costs (false positives)

**Phase 3: Optimization**
3. Adjust strategy based on actual costs:
   - If missing fields cost more → Switch to higher recall model
   - If hallucinations cost more → Switch to higher precision model
   - If volume is issue → Consider faster model with review pipeline

**Phase 4: Advanced Optimization**
4. Consider ensemble approaches:
   - Use high-precision model for critical fields (amounts, dates)
   - Use high-recall model for descriptive fields (line items)
   - Route by document confidence scores

---

## Related Documentation

- [FIELD_COMPARISON.md](FIELD_COMPARISON.md) - Detailed field-by-field analysis
- [ACCURACY_PARADOX_EXPLAINED.md](ACCURACY_PARADOX_EXPLAINED.md) - Why Accuracy > F1 for extraction
- [HALLUCINATION_ANALYSIS.md](HALLUCINATION_ANALYSIS.md) - Hallucination analysis methodology

---

**Report Auto-Generated**: {timestamp}
**Source Notebook**: `model_comparison_reporter.ipynb`
**Visualizations**: `output/visualizations/`
**Next Update**: Re-run notebook to refresh all metrics and visualizations
    