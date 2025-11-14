# Model Comparison Analysis Report

**Auto-Generated from Notebook**: 2025-11-14 16:54:42
**Source**: `model_comparison_reporter.ipynb`
**Dataset**: 195 documents (25 bank statements, 102 invoices, 68 receipts)
**Evaluation Fields**: 17 business document fields

---

## Executive Summary

### Overall Performance Metrics

| Model | F1 Score | Precision | Recall | Accuracy | Median Speed | Throughput |
|-------|----------|-----------|--------|----------|--------------|------------|
| **Llama-3.2-Vision-11B** | 0.5363 | 0.5811 | 0.4978 | 55.83% | 29.1s | 2.1 docs/min |
| **InternVL3-Quantized-8B** | 0.6032 | 0.7771 | 0.4929 | 56.27% | 69.5s | 0.9 docs/min |
| **InternVL3-NonQuantized-2B** | 0.0000 | 0.0000 | 0.0000 | 34.64% | 20.6s | 2.9 docs/min |

### Key Findings

**Winner (F1 Score)**: InternVL3-Quantized-8B

**Highest Precision**: InternVL3-Quantized-8B (0.7771)

**Highest Recall**: Llama-3.2-Vision-11B (0.4978)

**Fastest**: InternVL3-NonQuantized-2B (20.6s)

---

## Visualizations

All visualizations are generated in `output/visualizations/`:

### 1. Executive Performance Dashboard
\![Executive Dashboard](output/visualizations/executive_comparison.png)

**6-panel comprehensive view:**
- Overall accuracy distribution (box plots)
- Processing speed comparison
- Accuracy by document type
- Processing time by document type
- Efficiency analysis (accuracy vs speed)
- Performance summary table

### 2. Document Type Classification
\![Document Type Confusion](output/visualizations/doctype_confusion_matrix.png)

**3-model confusion matrices** showing classification performance for:
- Bank Statements (25 docs, 12.8%)
- Invoices (102 docs, 52.3%)
- Receipts (68 docs, 34.9%)

### 3. Field Extraction Status
\![Field Confusion Heatmap](output/visualizations/field_confusion_heatmap.png)

**Breakdown of extraction status:**
- Correct extractions (matches ground truth)
- Incorrect extractions (wrong value)
- Not Found (field not extracted)

### 4. Per-Field Metrics
\![Per-Field Metrics](output/visualizations/per_field_metrics.png)

**4-panel analysis:**
- F1 Score by field
- Precision by field
- Recall by field
- Accuracy by field

### 5. Field-Level Accuracy Analysis
\![Field-Level Accuracy](output/visualizations/field_level_accuracy.png)

**3-panel comprehensive view:**
- Field accuracy comparison (horizontal bar chart across all models)
- Field accuracy heatmap (color-coded performance matrix)
- Model specialization distribution (fields where each model performs best)

### 6. Hallucination Analysis
\![Hallucination Analysis](output/visualizations/hallucination_analysis.png)

**9-panel breakdown:**
- Overall hallucination rates
- Hallucinations vs correct NOT_FOUND
- Hallucination-recall tradeoff
- Per-field hallucination (3 models)
- Document-level distribution (3 models)

### Hallucination Rates

| Model | Hallucination Rate | Correct NOT_FOUND Rate | Total Hallucinations |
|-------|-------------------|------------------------|----------------------|
| **Llama-11B** | 30.1% | 69.9% | 442 |
| **InternVL3-8B** | 11.4% | 88.6% | 167 |
| **InternVL3-2B** | 13.9% | 86.1% | 204 |

**Interpretation:**
- **Hallucination Rate**: % of NOT_FOUND fields where model invented a value
- **Correct NOT_FOUND Rate**: % of NOT_FOUND fields correctly identified as absent

---

## Per-Field Performance Summary

### Field-Level Accuracy by Model

| Field | Llama-11B | InternVL3-8B | InternVL3-2B | Best Model | Best Score |
|-------|-----------|--------------|--------------|------------|------------|
| TOTAL_AMOUNT | 82.4% | 81.0% | 78.5% | Llama-3.2-Vision | 82.4% |
| INVOICE_DATE | 79.3% | 78.8% | 66.7% | Llama-3.2-Vision | 79.3% |
| SUPPLIER_NAME | 76.0% | 76.8% | 56.6% | InternVL3-Quantized-8B | 76.8% |
| PAYER_NAME | 75.9% | 74.6% | 57.1% | Llama-3.2-Vision | 75.9% |
| DOCUMENT_TYPE | 70.8% | 84.6% | 78.2% | Llama-3.2-Vision | 70.8% |
| GST_AMOUNT | 61.9% | 80.5% | 54.7% | InternVL3-Quantized-8B | 80.5% |
| PAYER_ADDRESS | 60.6% | 61.0% | 43.5% | InternVL3-Quantized-8B | 61.0% |
| TOTAL_ITEM_QUANTITIES | 35.1% | 74.2% | 44.9% | InternVL3-Quantized-8B | 74.2% |
| BUSINESS_ABN | 65.0% | 43.1% | 43.9% | Llama-3.2-Vision | 65.0% |
| BUSINESS_ADDRESS | 49.6% | 56.6% | 41.3% | InternVL3-Quantized-8B | 56.6% |
| LINE_ITEM_DESCRIPTIONS | 25.6% | 68.6% | 26.3% | InternVL3-Quantized-8B | 55.8% |
| LINE_ITEM_TOTAL_PRICES | 22.5% | 34.9% | 28.2% | InternVL3-Quantized-8B | 34.9% |
| TRANSACTION_DATES | 33.8% | 13.8% | 23.6% | Llama-3.2-Vision | 33.8% |
| IS_GST_INCLUDED | 62.4% | 0.0% | 0.0% | Llama-3.2-Vision | 62.4% |
| STATEMENT_DATE_RANGE | 32.9% | 10.9% | 10.6% | Llama-3.2-Vision | 32.9% |
| LINE_ITEM_PRICES | 13.6% | 22.7% | 17.4% | InternVL3-Quantized-8B | 22.7% |
| TRANSACTION_AMOUNTS_PAID | 20.7% | 20.7% | 3.4% | Llama-3.2-Vision | 20.7% |

---

## Model Specialization

### Fields Where Each Model Performs Best

| Model | Best-Performing Fields | Percentage | Count |
|-------|------------------------|------------|-------|
| **Llama-3.2-Vision-11B** | 58.8% | 10/17 | PRIMARY |
| **InternVL3-Quantized-8B** | 41.2% | 7/17 | SECONDARY |
| **InternVL3-NonQuantized-2B** | 0.0% | 0/17 | NO SPECIALIZATION |

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
- **Fastest processing**: InternVL3-NonQuantized-2B (~20.6s/doc)
- **Best accuracy**: InternVL3-Quantized-8B (56.27% overall)
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

Recall = Correct Extractions / All Fields That Should Be Extracted
→ High recall = Catches more fields
→ Risk: May hallucinate to achieve higher coverage

Hallucination Rate = Hallucinations / NOT_FOUND Opportunities
→ Direct measure of false positive risk
→ Critical for production reliability
```

#### Model Selection Guide Based on Use Case

**Choose HIGH PRECISION Model (InternVL3-Quantized-8B: 77.71%) if:**
- ☑️ Processing financial/regulatory data (invoices, tax documents)
- ☑️ Automated processing with no human review
- ☑️ **False data is worse than missing data**
- ☑️ You can afford to manually review `NOT_FOUND` fields
- ☑️ Compliance and audit requirements
- ☑️ Low tolerance for hallucinations

**Example**: Bank reconciliation where a hallucinated amount could cause financial errors.

**Choose HIGH RECALL Model (Llama-3.2-Vision-11B: 49.78%) if:**
- ☑️ Comprehensive data capture is critical
- ☑️ Human review pipeline can catch errors
- ☑️ **Missing data is worse than wrong data**
- ☑️ Initial screening/discovery use case
- ☑️ Maximizing field coverage is priority
- ☑️ Can tolerate some false positives

**Example**: Legal document discovery where missing a field could have serious consequences.

**Choose BALANCED Model (for high-volume processing) if:**
- ☑️ High-volume processing requirements
- ☑️ Need reasonable precision and recall
- ☑️ Speed is a critical factor
- ☑️ Standard business document processing

**Example**: Receipt processing for expense management with human spot-checking.

#### Your Model Performance Profile

Based on the analysis:

| Model | Precision | Recall | F1 | Best For |
|-------|-----------|--------|-----|----------|
| **Llama-3.2-Vision-11B** | 58.11% | 49.78% | 0.5363 | 🏆 Best Recall |
| **InternVL3-Quantized-8B** | 77.71% | 49.29% | 0.6032 | 🏆 Best Precision 🏆 Best F1 |
| **InternVL3-NonQuantized-2B** | 0.00% | 0.00% | 0.0000 | ❌ |

**Key Insights:**
- **Precision Leader**: InternVL3-Quantized-8B (77.71%)
- **Recall Leader**: Llama-3.2-Vision-11B (49.78%)
- **F1 Leader**: InternVL3-Quantized-8B (0.6032)
- **Speed vs Accuracy Tradeoff**: Consider throughput requirements against quality needs

---

## Efficiency Analysis

**Performance Efficiency Score** = Accuracy × Throughput (docs/min)

| Model | Avg Accuracy | Avg Speed | Throughput | Efficiency Score |
|-------|--------------|-----------|------------|------------------|
| **Llama-3.2-Vision-11B** | 55.14% | 29.1s | 2.1 docs/min | 115.2 |
| **InternVL3-Quantized-8B** | 56.27% | 69.5s | 0.9 docs/min | 48.6 |
| **InternVL3-NonQuantized-2B** | 34.64% | 20.6s | 2.9 docs/min | 101.1 |

**Highest Efficiency**: Llama-3.2-Vision-11B

#### Document-Type Specific Recommendations

**Best Model by Document Type:**

- **Bank Statement**: Llama-3.2-Vision-11B (43.07% accuracy)
- **Invoice**: InternVL3-Quantized-8B (57.78% accuracy)
- **Receipt**: InternVL3-Quantized-8B (67.23% accuracy)

#### Field Performance Insights

**Fields with Significant Model Performance Differences (>20% spread):**

- **IS_GST_INCLUDED**: Use Llama-3.2-Vision (62% vs 0%, +62% advantage)
- **LINE_ITEM_QUANTITIES**: Use InternVL3-Quantized-8B (74% vs 35%, +39% advantage)
- **LINE_ITEM_DESCRIPTIONS**: Use Llama-3.2-Vision (56% vs 26%, +30% advantage)
- **GST_AMOUNT**: Use InternVL3-Quantized-8B (81% vs 55%, +26% advantage)
- **STATEMENT_DATE_RANGE**: Use Llama-3.2-Vision (33% vs 11%, +22% advantage)

**⚠️ Problematic Fields Requiring Attention (<50% avg accuracy):**

- **BUSINESS_ADDRESS**: 49% average accuracy - Consider prompt optimization or additional fine tuning
- **LINE_ITEM_DESCRIPTIONS**: 40% average accuracy - Consider prompt optimization or additional fine tuning
- **LINE_ITEM_TOTAL_PRICES**: 29% average accuracy - Consider prompt optimization or additional fine tuning
- **TRANSACTION_DATES**: 24% average accuracy - Consider prompt optimization or additional fine tuning
- **IS_GST_INCLUDED**: 21% average accuracy - Consider prompt optimization or additional fine tuning

---

## Production Deployment Strategy

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
