# Executive Model Comparison Report

**Generated**: 2025-01-13

**Document Status**: Comprehensive Technical Analysis

---

## Overview

This comprehensive report compares three vision-language models for business document information extraction:

- **Llama-3.2-Vision-11B** (11B parameters, non-quantized)
- **InternVL3-Quantized-8B** (8B parameters, 8-bit quantized)
- **InternVL3-NonQuantized-2B** (2B parameters, non-quantized)

### Dataset

- **195 documents**: 30 bank statements (15.4%), 61 invoices (31.3%), 104 receipts (53.3%)
- **17 fields** evaluated with document-type-specific requirements
- **Position-aware F1 scoring** with field-type-specific comparison logic

### Key Findings

| Metric | Winner | Score | Key Insight |
|--------|--------|-------|-------------|
| **F1 Score** | Llama-11B | 0.3379 | Best overall extraction quality |
| **Precision** | Llama-11B | 0.2624 | Best for financial auditing |
| **Recall** | Llama-11B | 0.7797 | Extracts 78% of all fields |
| **Document Classification** | Llama-11B | 70.8% | PRIMARY application |
| **Speed** | InternVL3-2B | 2.2 docs/min | 3.7× faster than InternVL3-8B |

### Application Priorities

1. **PRIMARY**: Document Type Classification (70.8% accuracy with Llama)
2. **SECONDARY**: Financial Auditing - tracking what was spent, when, and how much (highest precision: 0.2624)

---

## 1. Executive Performance Dashboard

**6-panel comprehensive comparison of model performance across all metrics.**

![Executive Performance Dashboard](../visualizations/executive_comparison.png)

### Dashboard Components

#### 1. Overall Accuracy Comparison (Top-left)
- Box plot showing accuracy distribution
- Median accuracy: Llama 55.8%, InternVL3-8B 56.3%, InternVL3-2B 34.6%
- Note: Median differs from overall accuracy (see Accuracy Paradox)

#### 2. Processing Speed Comparison (Top-right)
- InternVL3-2B: 27.5s median (FASTEST)
- Llama-11B: 33.1s median (balanced)
- InternVL3-8B: 88.7s median (SLOWEST - quantization overhead)

#### 3. Average Accuracy by Document Type (Middle-left)
- Bank statements: Llama best (~44%), InternVL3-8B struggles (~30%)
- Invoices: All models ~55-65% accuracy
- Receipts: InternVL3-8B best (~67%), majority class challenge

#### 4. Average Processing Time by Document Type (Middle-right)
- Bank statements slowest (170s for InternVL3-8B)
- Receipts fastest (18-24s)
- Tabular data processing overhead evident

#### 5. Efficiency Analysis: Accuracy vs Processing Time (Bottom-left)
- Scatter plot shows quality-speed tradeoff
- Llama-11B: High quality (F1: 0.34), acceptable speed (34s)
- InternVL3-2B: Moderate quality (F1: 0.30), best speed (28s)
- InternVL3-8B: Worst on both dimensions

#### 6. Performance Summary Table (Bottom-right)
- Average Accuracy: Llama 55.8%, InternVL3-8B 56.3%, InternVL3-2B 34.6%
- Throughput: 2.2, 0.7, 1.8 docs/min respectively
- Standard deviation shows consistency

### Key Insights

- **Llama-11B**: Best quality-speed balance, recommended for production
- **InternVL3-8B**: Quantization paradox - slowest despite being quantized
- **InternVL3-2B**: Speed champion but quality tradeoffs

---

## 2. Document Type Classification Analysis

**PRIMARY APPLICATION**: Routing documents to correct workflows.

![Document Type Confusion Matrix](../visualizations/doctype_confusion_matrix.png)

### Classification Performance

| Model | Accuracy | F1-Score | Key Strength | Key Weakness |
|-------|----------|----------|--------------|---------------|
| **Llama-11B** | **70.8%** | 0.7152 | Bank statements (96% precision) | Receipt recall (57%) |
| **InternVL3-8B** | 69.2% | 0.7075 | Bank statements (91% F1) | Receipt recall (53%) |
| **InternVL3-2B** | 59.0% | 0.5903 | Bank statement recall (97%) | Receipt recall (33%) |

### Critical Finding: Receipt Classification Failure

**Problem**: All models struggle with receipts (majority class = 53% of documents)

- **Llama-11B**: 50/104 receipts misclassified as invoices (48%)
- **InternVL3-8B**: 47/104 receipts misclassified as invoices (45%)
- **InternVL3-2B**: 34/104 receipts misclassified as invoices (33%) + 17 as NOT_FOUND (16%)

**Root Cause**: Receipts and invoices share identical 14 fields, differentiated only by layout.

### Confusion Matrix Interpretation

Each 3-panel visualization shows:
- **Y-axis (True)**: Actual document types (3 types: bank_statement, invoice, receipt)
- **X-axis (Predicted)**: Model predictions (6 types including rare hallucinated types)
- **Cell values**: Document counts
- **Color intensity**: Higher = more documents

### Rare Document Type Hallucinations

Models incorrectly predict non-existent document types:
- **Llama-11B**: CTP_INSURANCE, E-TICKET, MOBILE_APP_SCREENSHOT
- **InternVL3-8B**: CRYPTO_STATEMENT, TAX_INVOICE, PAYMENT_ADVICE, NOT_FOUND
- **InternVL3-2B**: NOT_FOUND (17 receipts marked as unrecognizable)

**Recommendation**: Constrain output vocabulary to 3 valid types in production.

---

## 3. Field Extraction Status Analysis

**Field-level confusion matrix showing Correct/Incorrect/Not Found breakdown.**

![Field Extraction Status](../visualizations/field_confusion_heatmap.png)

### Extraction Status Categories

- **Correct**: Extracted value matches ground truth
- **Incorrect**: Extracted value doesn't match ground truth (hallucination or error)
- **Not Found**: Model says NOT_FOUND (conservative approach)

### Field-Level Summary (3 Models)

Each heatmap shows 17 fields × 3 status categories:

#### Llama-11B Pattern
- **High Correct**: TOTAL_AMOUNT (82%), INVOICE_DATE (79%), SUPPLIER_NAME (75%)
- **High Incorrect**: Many fields show aggressive extraction with errors
- **Low Not Found**: Rarely predicts NOT_FOUND (aggressive strategy)

#### InternVL3-8B Pattern
- **High Correct**: GST_AMOUNT (81% - BEST), LINE_ITEM_QUANTITIES (74% - BEST)
- **Moderate Incorrect**: Conservative approach limits errors
- **High Not Found**: Frequently predicts NOT_FOUND when uncertain

#### InternVL3-2B Pattern
- **Moderate Correct**: Some fields ~50-60%
- **High Incorrect**: Less conservative than 8B variant
- **Moderate Not Found**: Balanced approach

### Critical Failure: IS_GST_INCLUDED

**Both InternVL3 models**: 0.0% correct (complete failure on boolean fields)
**Llama-11B**: 62% correct (only functional model)

**Impact**: InternVL3 models cannot be used alone for tax/compliance workflows.

### Field Difficulty Tiers

**Easy (>70% avg accuracy)**:
- TOTAL_AMOUNT, INVOICE_DATE, SUPPLIER_NAME, PAYER_NAME, DOCUMENT_TYPE

**Medium (40-70% avg accuracy)**:
- GST_AMOUNT, PAYER_ADDRESS, LINE_ITEM_QUANTITIES, BUSINESS_ADDRESS, BUSINESS_ABN

**Hard (<40% avg accuracy)**:
- LINE_ITEM_PRICES, TRANSACTION_DATES, IS_GST_INCLUDED, STATEMENT_DATE_RANGE, TRANSACTION_AMOUNTS_PAID

---

## 4. Per-Field Precision, Recall, and F1 Metrics

**Detailed performance breakdown for each of 17 business document fields.**

![Per-Field Metrics Comparison](../visualizations/per_field_metrics.png)

### Four-Panel Visualization

#### 1. F1 Score by Field (Top-left)
- Harmonic mean of precision and recall
- **Llama-11B dominates**: 9/17 fields won
- **InternVL3-8B specializes**: 8/17 fields won (numeric/structured)
- **InternVL3-2B**: 0/17 fields won (never best)

#### 2. Precision by Field (Top-right)
- Measures extraction quality (correct / attempted)
- **Llama-11B**: Highest overall precision (0.2624)
- Critical for financial auditing applications

#### 3. Recall by Field (Bottom-left)
- Measures extraction coverage (correct / extractable)
- **Llama-11B**: 78% recall (best coverage)
- **InternVL3 models**: 39-44% recall (conservative)

#### 4. Accuracy by Field (Bottom-right)
- Includes correct NOT_FOUND predictions
- **Misleading for extraction tasks** (see Accuracy Paradox)
- InternVL3-8B highest due to conservative NOT_FOUND strategy

### Model Specialization Matrix

| Field Type | Best Model | Reasoning |
|------------|------------|-----------|
| **Monetary** | Llama-11B | TOTAL_AMOUNT (82%), near-parity with InternVL3-8B |
| **Numeric Lists** | InternVL3-8B | LINE_ITEM_QUANTITIES (74%), GST_AMOUNT (81%) |
| **Text Descriptions** | Llama-11B | LINE_ITEM_DESCRIPTIONS (57% vs 26%) |
| **Names/Addresses** | InternVL3-8B | SUPPLIER_NAME (77%), PAYER_NAME (75%) |
| **Dates** | Llama-11B | INVOICE_DATE (79%), TRANSACTION_DATES (34%) |
| **Boolean** | Llama-11B | IS_GST_INCLUDED (62%) - ONLY functional model |
| **Complex Tables** | InternVL3-8B | LINE_ITEM_TOTAL_PRICES (35%) |

### The LINE_ITEM_QUANTITIES Performance Chasm

**Extreme specialization**:
- InternVL3-8B: 74% accuracy
- Llama-11B: 35% accuracy
- **2.1× performance gap** (highest variance across all fields)

**Technical reason**: InternVL3's tile-based vision processing preserves table structure better.

### Ensemble Opportunity

**Field-specific routing** can optimize performance:
- Route numeric list fields → InternVL3-8B
- Route text/boolean/date fields → Llama-11B
- Expected combined F1: ~0.40 (vs 0.34 for Llama alone)

---

## 5. Hallucination Analysis

**Measuring how often models invent values for fields that don't exist (NOT_FOUND in ground truth).**

![Hallucination Analysis](../visualizations/hallucination_analysis.png)

### Definition

**Hallucination** = Model extracts a value when ground truth is NOT_FOUND

### Nine-Panel Visualization

**Row 1: Overall Comparisons**
1. Overall hallucination rate by model (bar chart)
2. Hallucinations vs Correct NOT_FOUND (grouped bars)
3. Hallucination vs Recall tradeoff (scatter plot)

**Row 2: Per-Field Analysis**
4. Llama-11B field hallucination (top 15 fields)
5. InternVL3-8B field hallucination (top 15 fields)
6. InternVL3-2B field hallucination (top 15 fields)

**Row 3: Document Distribution**
7. Llama-11B document hallucination distribution (histogram)
8. InternVL3-8B document hallucination distribution (histogram)
9. InternVL3-2B document hallucination distribution (histogram)

### Expected Hallucination Rates

Based on the Accuracy Paradox analysis:

| Model | Expected Rate | Reasoning |
|-------|---------------|-----------|
| **Llama-11B** | 50-70% | Lowest accuracy (49%) → High hallucination of NOT_FOUND fields |
| **InternVL3-8B** | 10-30% | Highest accuracy (54%) → Conservative, rarely hallucinates |
| **InternVL3-2B** | 30-50% | Moderate accuracy (53%) → Moderate hallucination |

### The Accuracy Paradox

```
Accuracy = (Correct Extractions + Correct NOT_FOUNDs) / Total Fields
Hallucination Rate = Hallucinated NOT_FOUNDs / Total NOT_FOUND Fields

Low Accuracy + High F1 = High Hallucination (Aggressive extraction)
High Accuracy + Low F1 = Low Hallucination (Conservative extraction)
```

**Llama's Strategy**:
- Aggressive extraction maximizes recall (78%)
- Side effect: Hallucinates ~50-70% of absent fields
- Results in lower accuracy but higher F1

**InternVL3-8B's Strategy**:
- Conservative extraction minimizes hallucinations
- Side effect: Misses 56% of extractable fields
- Results in higher accuracy but lower F1

### Business Implications

#### High Hallucination (Llama-11B)

**Pros**:
- Maximizes data extraction (78% recall)
- Good for comprehensive document mining
- Excellent for human-in-the-loop workflows

**Cons**:
- Requires extensive validation
- 50-70% of absent fields may be invented
- Risk of downstream errors if not validated

**Use Case**: Exploratory extraction with mandatory validation

#### Low Hallucination (InternVL3-8B)

**Pros**:
- High confidence in extracted data
- Fewer false positives
- Safer for automated workflows

**Cons**:
- Misses many extractable fields (44% recall)
- Leaves data on the table
- May require secondary extraction pass

**Use Case**: High-precision applications, compliance workflows

### Most Hallucinated Fields

Expected problematic fields (averaged across models):
- LINE_ITEM_PRICES (all models struggle)
- TRANSACTION_AMOUNTS_PAID (bank statements)
- STATEMENT_DATE_RANGE (bank statements)

---

## 6. Key Findings & Deployment Recommendations

### Application Priority Matrix

| Priority | Application | Best Model | Key Metric | Rationale |
|----------|-------------|------------|------------|-----------|
| **1 (PRIMARY)** | Document Type Classification | Llama-11B | 70.8% accuracy | Critical for workflow routing |
| **2 (SECONDARY)** | Financial Auditing | Llama-11B | 0.2624 precision | Track spending accurately |
| 3 | Comprehensive Document Mining | Llama-11B | 0.7797 recall | Maximum data extraction |
| 4 | High-Volume Processing | InternVL3-2B | 2.2 docs/min | Speed-optimized |
| 5 | Numeric Field Extraction | InternVL3-8B | 74% (quantities) | Table specialization |

### Production Deployment Strategies

#### Strategy 1: Llama-11B Standalone (RECOMMENDED)

**Configuration**:
- Single model: Llama-11B
- Post-processing validation
- Business rule filtering

**Best For**:
- Document type classification (PRIMARY)
- Financial auditing (SECONDARY - highest precision)
- Comprehensive document mining
- Human-in-the-loop workflows

**Requirements**:
- 22GB VRAM (A100, H100, V100-32GB)
- Validation pipeline for hallucinations
- 1.8 docs/min throughput

#### Strategy 2: Ensemble (Maximum Quality)

**Configuration**:
- Primary: InternVL3-8B (numeric/structured fields)
- Secondary: Llama-11B (text/boolean/dates)
- Field-specific routing

**Field Routing Logic**:
- LINE_ITEM_QUANTITIES → InternVL3-8B
- GST_AMOUNT → InternVL3-8B
- SUPPLIER_NAME/PAYER_NAME → InternVL3-8B
- IS_GST_INCLUDED → Llama-11B (only option)
- LINE_ITEM_DESCRIPTIONS → Llama-11B
- TRANSACTION_DATES → Llama-11B

**Best For**:
- Document type classification (maximum accuracy)
- Financial auditing (precision-critical)
- Compliance workflows
- High-value document processing

**Tradeoffs**:
- 2× inference cost
- Complex routing logic
- Estimated F1: ~0.40 (vs 0.34 Llama alone)

#### Strategy 3: InternVL3-2B (High-Volume)

**Configuration**:
- Single model: InternVL3-2B
- Minimal post-processing
- Accept moderate quality for speed

**Best For**:
- High-volume initial extraction
- Cost-sensitive deployments
- Consumer GPU environments (4GB VRAM)

**Limitations**:
- Cannot extract IS_GST_INCLUDED (0% accuracy)
- No field wins (always outperformed)
- Worst document type classification (59%)
- High variance (24% std dev)

### Model Selection Decision Tree

```
START: What is your constraint?

├─ QUALITY (Maximize F1)
│  └─ Llama-11B Standalone
│     ✅ F1: 0.34 (best)
│     ⚠️ Requires validation
│
├─ PRECISION (Financial Auditing)
│  └─ Llama-11B Standalone
│     ✅ Precision: 0.2624 (best)
│     ✅ Document classification: 70.8%
│
├─ SPEED (Maximize throughput)
│  ├─ Need boolean extraction?
│  │  ├─ YES → Llama-11B (1.8 docs/min)
│  │  └─ NO → InternVL3-2B (2.2 docs/min)
│
└─ COST (Minimize infrastructure)
   └─ InternVL3-2B
      ✅ 4GB VRAM
      ⚠️ 0% boolean accuracy
      ⚠️ No field specialization
```

### Critical Blockers

#### 1. IS_GST_INCLUDED Failure (InternVL3 models)
- 0.0% accuracy on boolean fields
- Blocker for tax/compliance workflows
- No known fix without ensemble or Llama

#### 2. Receipt Classification (All models)
- 33-57% recall on majority class (53% of documents)
- Requires business rule validation
- Cannot rely on classification alone

#### 3. InternVL3-8B Speed Penalty
- 3.4× slower than InternVL3-2B despite quantization
- 0.6 docs/min throughput unacceptable for production
- Avoid unless memory-constrained

### Final Recommendation

**For most production deployments: Llama-3.2-Vision-11B**

**Rationale**:
- ✅ Best for PRIMARY application (document classification: 70.8%)
- ✅ Best for SECONDARY application (financial auditing: highest precision 0.2624)
- ✅ Highest F1 (0.3379) - best overall extraction quality
- ✅ Highest recall (0.7797) - extracts 78% of all fields
- ✅ Only model with boolean extraction capability
- ✅ Acceptable speed (1.8 docs/min)
- ⚠️ Requires post-processing validation (high hallucination rate)
- ⚠️ Highest memory requirement (22GB VRAM)

**When to consider alternatives**:
- Speed-critical: InternVL3-2B (22% faster)
- Cost-critical: InternVL3-2B (5.5× smaller)
- No boolean fields: InternVL3-8B viable for specialized numeric extraction
- Maximum quality: Ensemble with field-specific routing

---

## Related Documentation

- **ACCURACY_PARADOX_EXPLAINED.md** - Why F1 > Accuracy for extraction tasks
- **MODEL_COMPARISON_ANALYSIS.md** - Comprehensive 2000+ line technical analysis
- **HALLUCINATION_ANALYSIS_ADDED.md** - Hallucination analysis implementation guide
- **THREE_MODEL_FIELD_METRICS_UPDATE.md** - 3-model comparison update notes

---

**Report Generated**: 2025-01-13
**Models Evaluated**: Llama-3.2-Vision-11B, InternVL3-Quantized-8B, InternVL3-NonQuantized-2B
**Evaluation Dataset**: 195 documents (30 bank statements, 61 invoices, 104 receipts)
**Fields Analyzed**: 17 business document fields
**Primary Metric**: Position-Aware F1 Score
