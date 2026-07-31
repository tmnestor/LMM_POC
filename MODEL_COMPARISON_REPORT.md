# Model Comparison Analysis Report

**Auto-Generated from Notebook**: 2026-07-28 15:48:43
**Source**: `model_comparison_reporter.ipynb`
**Dataset**: 165 documents (55 bank statements, 55 invoices, 55 receipts)
**Evaluation Fields**: 18 business document fields

---

## Executive Summary

### Overall Performance Metrics

| Model | Mean F1 | Median F1 | Precision | Recall | Accuracy | Median Speed | Throughput |
|-------|---------|-----------|-----------|--------|----------|--------------|------------|
| **MODEL1** | 0.6409 | 0.7586 | 0.8035 | 0.5330 | 80.35% | 44.5s | 1.3 docs/min |
| **MODEL2** | 0.5521 | 0.6452 | 0.6867 | 0.4617 | 68.67% | 24.9s | 2.4 docs/min |
| **MODEL3** | 0.4493 | 0.5161 | 0.5525 | 0.3786 | 55.25% | 10.1s | 5.9 docs/min |

### Key Findings

**Winner (F1 Score)**: MODEL1

**Highest Precision**: MODEL1 (0.8035)

**Highest Recall**: MODEL1 (0.5330)

**Fastest**: MODEL3 (10.1s)

---

## Visualizations

All visualizations are generated in `output/visualizations/`:

### 1. Executive Performance Dashboard

**6-panel comprehensive view:**
- Overall accuracy distribution (box plots)
- Processing speed comparison
- Accuracy by document type
- Processing time by document type
- Efficiency analysis (accuracy vs speed)
- Performance summary table

### 2. Document Type Classification

**3-model confusion matrices** showing classification performance for:
- Bank Statements (55 docs, 33.3%)
- Invoices (55 docs, 33.3%)
- Receipts (55 docs, 33.3%)

### 3. Field Extraction Status

**Breakdown of extraction status:**
- Correct extractions (matches ground truth)
- Incorrect extractions (wrong value)
- Not Found (field not extracted)

### 4. Per-Field Metrics

**4-panel analysis:**
- F1 Score by field
- Precision by field
- Recall by field
- Accuracy by field


### 5. Field-Level F1 Analysis

**3-panel comprehensive view:**
- Field accuracy comparison (horizontal bar chart across all models)
- Field accuracy heatmap (color-coded performance matrix)
- Model specialization distribution (fields where each model performs best)

### 6. Hallucination Analysis

**9-panel breakdown:**
- Overall hallucination rates
- Hallucinations vs correct NOT_FOUND
- Hallucination-recall tradeoff
- Per-field hallucination (3 models)
- Document-level distribution (3 models)

### Hallucination Rates

| Model | Hallucination Rate | Correct NOT_FOUND Rate | Total Hallucinations |
|-------|-------------------|------------------------|----------------------|
| **MODEL1** | 0.0% | 100.0% | 0 |
| **MODEL2** | 0.0% | 100.0% | 0 |
| **MODEL3** | 0.0% | 100.0% | 0 |

**Interpretation:**
- **Hallucination Rate**: % of NOT_FOUND fields where model invented a value
- **Correct NOT_FOUND Rate**: % of NOT_FOUND fields correctly identified as absent

---

## Per-Field Performance Summary

### Field-Level F1 by Model

| Field | {MODEL1_NAME} | {MODEL2_NAME} | {MODEL3_NAME} | Best Model | Best Score |
|-------|-----------|--------------|--------------|------------|------------|
| PAYER_ADDRESS | 97.0% | 94.5% | 90.3% | MODEL1 | 97.0% |
| STATEMENT_DATE_RANGE | 96.4% | 93.6% | 90.0% | MODEL1 | 96.4% |
| LINE_ITEM_TOTAL_PRICES | 96.0% | 92.3% | 89.7% | MODEL1 | 96.0% |
| LINE_ITEM_PRICES | 98.1% | 93.0% | 85.9% | MODEL1 | 98.1% |
| LINE_ITEM_QUANTITIES | 97.0% | 94.4% | 85.2% | MODEL1 | 97.0% |
| LINE_ITEM_DESCRIPTIONS | 96.3% | 92.5% | 86.6% | MODEL1 | 96.3% |
| TRANSACTION_DATES | 95.1% | 91.0% | 85.5% | MODEL1 | 95.1% |
| BUSINESS_ABN | 95.8% | 90.3% | 84.2% | MODEL1 | 95.8% |
| GST_AMOUNT | 93.9% | 90.9% | 84.2% | MODEL1 | 93.9% |
| INVOICE_DATE | 95.2% | 89.1% | 83.0% | MODEL1 | 95.2% |
| TOTAL_AMOUNT | 95.2% | 91.5% | 77.6% | MODEL1 | 95.2% |
| BUSINESS_ADDRESS | 95.8% | 85.5% | 82.4% | MODEL1 | 95.8% |
| IS_GST_INCLUDED | 91.5% | 92.1% | 78.2% | MODEL2 | 92.1% |
| TRANSACTION_AMOUNTS_PAID | 92.1% | 87.3% | 79.9% | MODEL1 | 92.1% |
| PAYER_NAME | 94.5% | 83.0% | 70.3% | MODEL1 | 94.5% |
| DOCUMENT_TYPE | 92.7% | 77.6% | 64.8% | MODEL1 | 92.7% |
| SUPPLIER_NAME | 91.5% | 73.9% | 64.8% | MODEL1 | 91.5% |
| data_provenance | 0.0% | 0.0% | 0.0% | MODEL1 | 0.0% |

---

## Model Specialization

### Fields Where Each Model Performs Best

| Model | Best-Performing Fields | Percentage | Count |
|-------|----------------------|------------|-------|
| **MODEL1** | 94.4% | 17/18 | PRIMARY |
| **MODEL2** | 5.6% | 1/18 | SECONDARY |
| **MODEL3** | 0.0% | 0/18 | NO SPECIALIZATION |

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
- **Fastest processing**: MODEL3 (~10.1s/doc)
- **Best accuracy**: MODEL1 (80.35% overall)
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

**Choose HIGH PRECISION Model (MODEL1: 80.35%) if:**
- ✅ Processing financial/regulatory data (invoices, tax documents)
- ✅ Automated processing with no human review
- ✅ **False data is worse than missing data**
- ✅ You can afford to manually review `NOT_FOUND` fields
- ✅ Compliance and audit requirements
- ✅ Low tolerance for hallucinations

**Example**: Bank reconciliation where a hallucinated amount could cause financial errors.

**Choose HIGH RECALL Model (MODEL1: 53.30%) if:**
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
| **MODEL1** | 80.35% | 53.30% | 0.6409 | 🏆 Best Precision🏆 Best Recall🏆 Best F1 |
| **MODEL2** | 68.67% | 46.17% | 0.5521 |  |
| **MODEL3** | 55.25% | 37.86% | 0.4493 |  |

**Key Insights:**
- **Precision Leader**: MODEL1 (80.35%)
- **Recall Leader**: MODEL1 (53.30%)
- **F1 Leader**: MODEL1 (0.6409)
- **Speed vs Accuracy Tradeoff**: Consider throughput requirements against quality needs

#### Efficiency Analysis

**Performance Efficiency Score** = Accuracy × Throughput (docs/min)

| Model | Avg Accuracy | Avg Speed | Throughput | Efficiency Score |
|-------|--------------|-----------|------------|------------------|
| **MODEL1** | 80.35% | 44.5s | 1.3 docs/min | 108.2 |
| **MODEL2** | 68.67% | 24.9s | 2.4 docs/min | 165.2 |
| **MODEL3** | 55.25% | 10.1s | 5.9 docs/min | 327.7 |

**Highest Efficiency**: MODEL3





#### Document-Type Specific Recommendations

**Best Model by Document Type:**

- **Bank Statement**: MODEL1 (59.09% accuracy)
- **Invoice**: MODEL1 (90.00% accuracy)
- **Receipt**: MODEL1 (91.97% accuracy)

#### Field Performance Insights

**Fields with Significant Model Performance Differences (>20% spread):**

- **DOCUMENT_TYPE**: Use MODEL1 (93% vs 65%, +28% advantage)
- **SUPPLIER_NAME**: Use MODEL1 (92% vs 65%, +27% advantage)
- **PAYER_NAME**: Use MODEL1 (95% vs 70%, +24% advantage)

**⚠️ Problematic Fields Requiring Attention (<50% avg accuracy):**

- **data_provenance**: 0% average accuracy - Consider prompt optimization or additional fine tuning


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
    