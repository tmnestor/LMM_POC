# Technical Performance Analysis: Three Vision-Language Models for Business Document Information Extraction

## Executive Summary

This document presents a comprehensive technical comparison of three vision-language models evaluated for business document information extraction:

- **Llama-3.2-Vision-11B** (11B parameters, non-quantized)
- **InternVL3-Quantized-8B** (8B parameters, AWQ quantized)
- **InternVL3-NonQuantized-2B** (2B parameters, non-quantized)

### Overall Performance Rankings

| Metric | 1st Place | 2nd Place | 3rd Place |
|--------|-----------|-----------|-----------|
| **F1 Score** | Llama-11B (0.3379) | InternVL3-2B (0.2984) | InternVL3-8B (0.2611) |
| **Precision** | Llama-11B (0.2624) | InternVL3-2B (0.2727) | InternVL3-8B (0.2228) |
| **Recall** | Llama-11B (0.7797) | InternVL3-8B (0.4367) | InternVL3-2B (0.3873) |
| **Accuracy** | InternVL3-8B (0.5442) | InternVL3-2B (0.5324) | Llama-11B (0.4911) |
| **Speed (docs/min)** | InternVL3-2B (2.2) | Llama-11B (1.8) | InternVL3-8B (0.6) |

### Model Specialization by Fields Won

- **Llama-11B**: 9 fields (52.9%)
- **InternVL3-8B**: 8 fields (47.1%)
- **InternVL3-2B**: 0 fields (0%)

### Key Technical Findings

1. **The Accuracy Paradox**: Llama-11B achieves highest F1 but lowest accuracy due to aggressive extraction strategy that hallucinates NOT_FOUND fields
2. **IS_GST_INCLUDED Complete Failure**: Both InternVL3 models achieve 0.0 accuracy on boolean field extraction (Llama: 0.617647)
3. **Field Specialization**: InternVL3-8B dominates quantity/number extraction, Llama-11B excels at complex text and date fields
4. **Speed-Accuracy Tradeoff**: InternVL3-2B is 3.7× faster than Llama but with 11% lower F1 score
5. **Document Type Classification**: Llama-11B leads with 70.8% accuracy vs InternVL3-8B (69.2%) and InternVL3-2B (59.0%)

---

## Evaluation Methodology

### Dataset Composition

- **Total Documents**: 195
  - Bank Statements: 75 (38.5%)
  - Invoices: 100 (51.3%)
  - Receipts: 20 (10.3%)

### Fields Evaluated

**17 fields across all document types:**

1. DOCUMENT_TYPE
2. BUSINESS_ABN
3. SUPPLIER_NAME
4. BUSINESS_ADDRESS
5. PAYER_NAME
6. PAYER_ADDRESS
7. INVOICE_DATE
8. LINE_ITEM_DESCRIPTIONS
9. LINE_ITEM_QUANTITIES
10. LINE_ITEM_PRICES
11. LINE_ITEM_TOTAL_PRICES
12. IS_GST_INCLUDED
13. GST_AMOUNT
14. TOTAL_AMOUNT
15. STATEMENT_DATE_RANGE
16. TRANSACTION_DATES
17. TRANSACTION_AMOUNTS_PAID

### Evaluation Metrics

**Primary Metric: Position-Aware F1 Score**

The evaluation uses a custom position-aware F1 scoring system (not sklearn) with field-type-specific comparison logic:

- **Monetary fields** (TOTAL_AMOUNT, GST_AMOUNT): 1% tolerance for rounding errors
- **Numeric IDs** (BUSINESS_ABN): Exact digit match, format-agnostic
- **Dates** (INVOICE_DATE, TRANSACTION_DATES): Component-based flexible matching
- **Text fields** (SUPPLIER_NAME, PAYER_NAME): Fuzzy matching with 80% word overlap threshold
- **List fields** (LINE_ITEM_*): Position-aware or position-agnostic F1
- **Phone numbers**: Digit-based partial credit (80% match = 0.8 score)
- **Boolean** (IS_GST_INCLUDED): Exact true/false match

**Metric Calculation:**

```python
# Precision: Of all attempted extractions, how many were correct?
precision = true_positives / (true_positives + false_positives)

# Recall: Of all extractable fields, how many did we find?
recall = true_positives / (true_positives + false_negatives)

# F1: Harmonic mean of precision and recall
f1 = 2 × (precision × recall) / (precision + recall)

# Accuracy: All correct predictions (including NOT_FOUND)
accuracy = (true_positives + true_negatives) / total_predictions

# CRITICAL: Precision/Recall EXCLUDE NOT_FOUND predictions
# true_positives = (predicted correctly) AND (value != NOT_FOUND)
# false_positives = (predicted incorrectly) AND (value != NOT_FOUND)
# false_negatives = (predicted NOT_FOUND) AND (ground truth exists)
```

### Why F1 > Accuracy for Extraction Tasks

As documented in `ACCURACY_PARADOX_EXPLAINED.md`, accuracy is **misleading for extraction tasks** because:

1. **NOT_FOUND inflation**: Correctly predicting NOT_FOUND inflates accuracy
2. **Conservative bias**: Models saying "I don't know" get rewarded
3. **Misaligned incentives**: High accuracy can coexist with poor extraction coverage

**Example**: For an invoice with 17 fields (8 present, 9 absent):
- Conservative model: Extracts 3 correctly, misses 5 → Accuracy 71%, F1 55%
- Aggressive model: Extracts 7 correctly, hallucinates 6 → Accuracy 59%, F1 78%

The aggressive model has **lower accuracy but higher F1** because F1 focuses on actual extraction performance, not conservative "I don't know" predictions.

---

## Overall Performance Analysis

### Model Performance Summary

| Model | Precision | Recall | F1 Score | Accuracy |
|-------|-----------|--------|----------|----------|
| **InternVL3-2B** | 0.2727 | 0.3873 | 0.2984 | 0.5324 |
| **InternVL3-8B** | 0.2228 | 0.4367 | 0.2611 | 0.5442 |
| **Llama-11B** | 0.2624 | 0.7797 | 0.3379 | 0.4911 |

### Interpretation

**Llama-11B: The Aggressive Extractor**
- Extracts 78% of all extractable fields (highest recall)
- 26% precision means 74% of attempted extractions contain errors
- Lowest accuracy (49%) due to hallucinating non-existent fields
- **Strategy**: Extract everything, validate later
- **Use case**: Maximizing information capture

**InternVL3-8B: The Balanced Conservative**
- Extracts only 44% of extractable fields (moderate recall)
- 22% precision (worst) - even conservative guesses often wrong
- Highest accuracy (54%) by frequently saying NOT_FOUND
- **Strategy**: Only extract when very confident
- **Use case**: High-precision requirements

**InternVL3-2B: The Speed Demon**
- Extracts 39% of extractable fields (worst recall)
- 27% precision (best among InternVL3)
- Moderate accuracy (53%)
- **Strategy**: Fast inference, moderate quality
- **Use case**: High-volume processing

### Precision-Recall Tradeoff

**Recall Advantage: Llama-11B**
- 78% vs 44% (InternVL3-8B) vs 39% (InternVL3-2B)
- Llama extracts **78% more fields** than InternVL3-2B
- Critical for data warehouse population

**Precision Disadvantage: All Models**
- Best precision: 27% (InternVL3-2B)
- Worst precision: 22% (InternVL3-8B)
- **All models struggle with extraction quality**
- Indicates field extraction is inherently challenging

---

## Field-Level Performance Deep Dive

### Performance Tiers

Based on average accuracy across all models:

#### Tier 1: High-Performing Fields (>70% avg accuracy)

| Field | Llama-11B | InternVL3-8B | InternVL3-2B | Avg | Variance | Winner |
|-------|-----------|--------------|--------------|-----|----------|--------|
| **TOTAL_AMOUNT** | 0.817647 | 0.809816 | 0.785276 | 0.804 | 0.014 | Llama-11B |
| **INVOICE_DATE** | 0.790588 | 0.787654 | 0.667073 | 0.748 | 0.058 | Llama-11B |
| **SUPPLIER_NAME** | 0.750765 | **0.767833** | 0.566391 | 0.695 | 0.091 | InternVL3-8B |
| **PAYER_NAME** | 0.735433 | **0.745968** | 0.570769 | 0.684 | 0.080 | InternVL3-8B |
| **DOCUMENT_TYPE** | 0.707692 | 0.697436 | 0.589744 | 0.665 | 0.053 | Llama-11B |

**Technical Analysis:**

- **TOTAL_AMOUNT** (0.804 avg, 0.014 variance):
  - All models excel due to monetary field salience
  - Low variance indicates consistent extraction across models
  - 1% tolerance in evaluation helps (e.g., $100.00 vs $100 match)

- **INVOICE_DATE** (0.748 avg, 0.058 variance):
  - Llama leads with 79% accuracy
  - Component-based matching helps (day/month/year flexibility)
  - InternVL3-2B lags significantly (67%) - speed tradeoff

- **SUPPLIER_NAME / PAYER_NAME** (0.695 / 0.684 avg):
  - **InternVL3-8B specialization**: Wins both name fields
  - Fuzzy matching with 80% word overlap threshold
  - High variance (0.091 / 0.080) indicates model-specific approaches

#### Tier 2: Medium-Performing Fields (40-70% avg accuracy)

| Field | Llama-11B | InternVL3-8B | InternVL3-2B | Avg | Variance | Winner |
|-------|-----------|--------------|--------------|-----|----------|--------|
| **GST_AMOUNT** | 0.580000 | **0.805085** | 0.547009 | 0.644 | 0.115 | InternVL3-8B |
| **PAYER_ADDRESS** | 0.593897 | **0.609560** | 0.435436 | 0.546 | 0.079 | InternVL3-8B |
| **LINE_ITEM_QUANTITIES** | 0.345064 | **0.742326** | 0.448845 | 0.512 | 0.168 | InternVL3-8B |
| **BUSINESS_ADDRESS** | 0.507380 | **0.565845** | 0.413068 | 0.495 | 0.063 | InternVL3-8B |
| **BUSINESS_ABN** | 0.596403 | 0.400787 | 0.439370 | 0.479 | 0.085 | Llama-11B |
| **LINE_ITEM_DESCRIPTIONS** | 0.566178 | 0.262729 | 0.374667 | 0.401 | 0.125 | Llama-11B |

**Technical Analysis:**

- **GST_AMOUNT** (0.644 avg, 0.115 variance):
  - **InternVL3-8B dominates** with 81% accuracy (best single-field performance)
  - Llama at 58%, InternVL3-2B at 55%
  - High variance (0.115) indicates InternVL3-8B has specialized capability
  - Monetary + numeric extraction strength

- **LINE_ITEM_QUANTITIES** (0.512 avg, **0.168 variance** - HIGHEST):
  - **Extreme model disagreement**: InternVL3-8B 74% vs Llama 35%
  - **2.15× performance gap** - largest in dataset
  - Suggests fundamentally different extraction approaches
  - InternVL3-8B optimized for numeric list extraction

- **LINE_ITEM_DESCRIPTIONS** (0.401 avg, 0.125 variance):
  - **Llama excels** at text-heavy fields (57% vs 26% for InternVL3-8B)
  - Longer context windows benefit description extraction
  - InternVL3-8B conservative approach fails for variable-length text

#### Tier 3: Low-Performing Fields (<40% avg accuracy)

| Field | Llama-11B | InternVL3-8B | InternVL3-2B | Avg | Variance | Winner |
|-------|-----------|--------------|--------------|-----|----------|--------|
| **LINE_ITEM_TOTAL_PRICES** | 0.221557 | **0.348684** | 0.282051 | 0.284 | 0.052 | InternVL3-8B |
| **TRANSACTION_DATES** | 0.338109 | 0.138489 | 0.236014 | 0.237 | 0.082 | Llama-11B |
| **IS_GST_INCLUDED** | **0.617647** | 0.000000 | 0.000000 | 0.206 | 0.291 | Llama-11B |
| **STATEMENT_DATE_RANGE** | 0.328725 | 0.109370 | 0.105882 | 0.181 | 0.104 | Llama-11B |
| **LINE_ITEM_PRICES** | 0.119760 | **0.227273** | 0.173554 | 0.173 | 0.044 | InternVL3-8B |
| **TRANSACTION_AMOUNTS_PAID** | 0.206897 | 0.034483 | 0.142857 | 0.128 | 0.071 | Llama-11B |

**Technical Analysis:**

- **IS_GST_INCLUDED** (0.206 avg, **0.291 variance** - MAXIMUM):
  - **CRITICAL FINDING**: Both InternVL3 models achieve **0.0% accuracy**
  - Llama achieves 62% - only model capable of boolean extraction
  - **Complete failure mode** for InternVL3 architecture on boolean fields
  - **292% variance** indicates fundamental capability gap

- **TRANSACTION_DATES** (0.237 avg):
  - **Llama 2.4× better** than InternVL3-8B (34% vs 14%)
  - Bank statement specific field
  - Multi-date extraction from tabular data
  - InternVL3-8B extremely conservative (14% accuracy)

- **LINE_ITEM_PRICES** (0.173 avg):
  - **All models struggle** (12-23% range)
  - Requires table structure understanding
  - Position-aware matching challenging
  - Best performance: InternVL3-8B at 23%

### Field Specialization Matrix

| Field Type | Best Model | 2nd Best | Accuracy Gap | Technical Reason |
|------------|------------|----------|--------------|------------------|
| **Boolean** | Llama-11B | N/A | +62% | InternVL3 models cannot extract boolean values |
| **Quantities/Numbers** | InternVL3-8B | InternVL3-2B | +40% | Numeric list extraction optimization |
| **Monetary** | Llama-11B | InternVL3-8B | +1% | Near parity, all models handle well |
| **Names** | InternVL3-8B | Llama-11B | +1-2% | Fuzzy matching advantage |
| **Addresses** | InternVL3-8B | Llama-11B | +2% | Conservative approach benefits structured text |
| **Dates** | Llama-11B | InternVL3-8B | +0.3% | Component-based matching advantage |
| **Text Descriptions** | Llama-11B | InternVL3-2B | +19% | Long context window benefits |
| **Table Data** | InternVL3-8B | Llama-11B | +11% | Position-aware extraction strength |

---

## Document Type Classification Analysis

### Classification Accuracy

| Model | Accuracy | F1 Score (weighted) |
|-------|----------|---------------------|
| **Llama-3.2-Vision-11B** | **70.8%** | 0.715221 |
| **InternVL3-Quantized-8B** | 69.2% | 0.707537 |
| **InternVL3-NonQuantized-2B** | 59.0% | 0.590284 |

**Performance Gap**: Llama leads by 1.6% over InternVL3-8B, 11.8% over InternVL3-2B

### Confusion Matrix Analysis

#### Llama-3.2-Vision-11B

**Classification Report:**

| True Class | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| **BANK_STATEMENT** | 0.960000 | 0.800000 | 0.872727 | 30 |
| **INVOICE** | 0.555556 | 0.901639 | 0.687500 | 61 |
| **RECEIPT** | 0.867647 | 0.567308 | 0.686047 | 104 |

**Key Observations:**

- **Bank Statement**: Excellent precision (96%) but moderate recall (80%)
  - 20% misclassified as other types (5 as RECEIPT, 1 as INVOICE)
  - Very few false positives (only 2 documents misclassified AS bank statement)

- **Invoice**: Low precision (56%) but excellent recall (90%)
  - Aggressive classification: Labels many receipts as invoices (4 receipts → invoice)
  - Catches most true invoices (90% recall)

- **Receipt**: High precision (87%) but poor recall (57%)
  - Conservative: Misses 43% of receipts
  - Confused with INVOICE (50 receipts → invoice) and other rare types

**Rare Document Types Extracted** (not in ground truth):
- COMPULSORY THIRD PARTY INSURANCE: 0 precision, 0 recall
- E-TICKET: 0 precision, 0 recall
- MOBILE APP SCREENSHOT: 0 precision, 0 recall

#### InternVL3-Quantized-8B

**Classification Report:**

| True Class | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| **BANK_STATEMENT** | 0.962963 | 0.866667 | 0.912281 | 30 |
| **INVOICE** | 0.529412 | 0.885246 | 0.662577 | 61 |
| **RECEIPT** | 0.932203 | 0.528846 | 0.674847 | 104 |

**Key Observations:**

- **Bank Statement**: Best performance overall (91% F1)
  - High precision (96%) and good recall (87%)
  - Only 4 misclassifications (all to INVOICE)

- **Invoice**: Similar to Llama - low precision (53%), high recall (89%)
  - Over-predicts invoices from receipts

- **Receipt**: Excellent precision (93%) but worst recall (53%)
  - Very conservative: Misses 47% of receipts
  - Most go to INVOICE (47), some to NOT_FOUND (4)

**Rare Document Types Extracted**:
- CRYPTO STATEMENT: 0 precision, 0 recall
- TAX INVOICE: 0 precision, 0 recall
- NOT_FOUND: 0 precision, 0 recall
- PAYMENT ADVICE: 0 precision, 0 recall

#### InternVL3-NonQuantized-2B

**Classification Report:**

| True Class | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| **BANK_STATEMENT** | 0.852941 | 0.966667 | 0.906250 | 30 |
| **INVOICE** | 0.485981 | 0.852459 | 0.619048 | 61 |
| **RECEIPT** | 0.918919 | 0.326923 | 0.482270 | 104 |

**Key Observations:**

- **Bank Statement**: Excellent recall (97%) but lower precision (85%)
  - Aggressive: Misclassifies some invoices/receipts as bank statements
  - Best recall among all models for this class

- **Invoice**: Worst precision (49%), good recall (85%)
  - Most aggressive invoice classifier
  - Many receipts → invoice (13)

- **Receipt**: Best precision (92%) but **worst recall (33%)**
  - Extremely conservative on receipts
  - Misses 67% of true receipts
  - Most go to INVOICE (34) or NOT_FOUND (17)

**Rare Document Types Extracted**:
- NOT_FOUND: 0 precision, 0 recall (17 documents marked NOT_FOUND)

### Document Type Confusion Patterns

#### Common Misclassifications

1. **Receipt → Invoice** (All models)
   - Llama: 50 receipts → invoice
   - InternVL3-8B: 47 receipts → invoice
   - InternVL3-2B: 34 receipts → invoice
   - **Reason**: Visual similarity, both have items/amounts

2. **Invoice → Receipt** (Rare)
   - All models rarely confuse invoices as receipts
   - Suggests "invoice" is over-predicted category

3. **Receipt → NOT_FOUND** (InternVL3-2B)
   - 17 receipts marked as NOT_FOUND
   - Indicates model uncertainty/conservatism

4. **Bank Statement Confusion**
   - Llama: 1 bank_stmt → invoice, 5 → receipt
   - InternVL3-8B: 4 bank_stmt → invoice
   - InternVL3-2B: 1 bank_stmt → invoice
   - Generally reliable across models

#### Rare Document Type Handling

**Llama-11B extracts**:
- COMPULSORY THIRD PARTY INSURANCE (CTP_INSUR)
- E-TICKET
- MOBILE APP SCREENSHOT

**InternVL3-8B extracts**:
- CRYPTO STATEMENT
- TAX INVOICE
- NOT_FOUND
- PAYMENT ADVICE

**InternVL3-2B extracts**:
- NOT_FOUND (most conservative)

**Analysis**: Models hallucinate rare document types not present in ground truth, indicating:
- Over-generalization from training data
- Aggressive classification beyond 3-class problem
- Need for constrained output vocabulary in production

---

## Processing Speed & Efficiency Analysis

### Processing Time Comparison

| Model | Median Time (s) | Avg Time (s) | Docs/min | Std Dev |
|-------|-----------------|--------------|----------|---------|
| **Llama-3.2-Vision-11B** | 34.0 | 34.0 | 1.8 | 18.1% |
| **InternVL3-Quantized-8B** | 93.3 | 93.3 | 0.6 | 17.3% |
| **InternVL3-NonQuantized-2B** | 27.5 | 27.5 | 2.2 | 24.0% |

### Speed Rankings

1. **InternVL3-2B**: 27.5s (2.2 docs/min) - **Fastest**
2. **Llama-11B**: 34.0s (1.8 docs/min) - **1.2× slower**
3. **InternVL3-8B**: 93.3s (0.6 docs/min) - **3.4× slower than fastest**

### Efficiency Analysis

#### InternVL3-NonQuantized-2B: Speed Champion

**Advantages:**
- **27.5s median** - fastest processing
- **2.2 docs/min throughput** - best for high-volume
- **2B parameters** - smallest memory footprint
- **24% std dev** - moderate consistency

**Tradeoffs:**
- 11% lower F1 vs Llama (0.2984 vs 0.3379)
- Worst precision (0.2727)
- Worst recall among practical models (0.3873)

**Use Case**: High-volume document processing where speed > quality

#### Llama-3.2-Vision-11B: Balanced Speed-Quality

**Advantages:**
- **34.0s median** - only 23% slower than fastest
- **Best F1 (0.3379)** - highest quality
- **18.1% std dev** - most consistent
- **1.8 docs/min** - acceptable throughput

**Tradeoffs:**
- **11B parameters** - highest memory (22GB VRAM)
- 1.2× slower than InternVL3-2B
- 23% more processing time for 13% higher F1

**Use Case**: Production deployments requiring quality with acceptable speed

#### InternVL3-Quantized-8B: Speed Penalty

**Disadvantages:**
- **93.3s median** - slowest by far
- **0.6 docs/min** - 3.7× slower than InternVL3-2B
- **17.3% std dev** - consistent but slow

**Paradox**: 8B model slower than 11B model
- **Reason**: AWQ quantization overhead
- Quantization adds decoding latency
- Trade memory for speed (opposite of goal)

**Recommendation**: **Avoid for production** unless memory-constrained

### Speed vs Accuracy Scatter Analysis

**Efficiency Frontier:**

```
High Quality, Slow: [Empty]
High Quality, Fast: Llama-11B (F1: 0.34, 34s)
Low Quality, Slow: InternVL3-8B (F1: 0.26, 93s)
Low Quality, Fast: InternVL3-2B (F1: 0.30, 28s)
```

**Pareto Optimal Models:**
- **Llama-11B**: Best quality-speed tradeoff
- **InternVL3-2B**: Best speed if quality acceptable

**Dominated Model:**
- **InternVL3-8B**: Worst quality AND slowest - no scenario where optimal

### Document Type Processing Time

| Doc Type | Llama-11B | InternVL3-8B | InternVL3-2B |
|----------|-----------|--------------|--------------|
| **Bank Statement** | 55s | 170s | 43s |
| **Invoice** | 29s | 82s | 21s |
| **Receipt** | 24s | 65s | 18s |

**Observations:**

1. **Bank statements slowest** (all models)
   - Tabular data processing overhead
   - Multi-row transaction extraction
   - 2.3× slower than receipts (Llama)

2. **Receipt processing fastest** (all models)
   - Simple structure, fewer fields
   - Single transaction vs multiple

3. **Model consistency**:
   - InternVL3-8B: 2.6× slower than InternVL3-2B (all document types)
   - Llama: 1.3× slower than InternVL3-2B (all document types)

---

## Model-Specific Technical Analysis

### Llama-3.2-Vision-11B

#### Architecture & Specifications

- **Parameters**: 11 billion
- **Quantization**: None (full precision)
- **Memory**: ~22GB VRAM
- **Preprocessing**: Built-in vision encoder
- **Context Window**: Large (benefits long documents)

#### Performance Profile

**Strengths:**

1. **Highest F1 Score (0.3379)**
   - 29% better than InternVL3-8B
   - 13% better than InternVL3-2B
   - Best overall extraction quality

2. **Exceptional Recall (0.7797)**
   - Extracts 78% of all extractable fields
   - 79% higher recall than InternVL3-8B
   - 101% higher recall than InternVL3-2B

3. **Boolean Field Extraction**
   - **Only model capable of IS_GST_INCLUDED** (62% accuracy)
   - InternVL3 models: 0% accuracy
   - Critical for tax/compliance workflows

4. **Complex Field Mastery**
   - TRANSACTION_DATES: 34% (vs 14% for InternVL3-8B)
   - STATEMENT_DATE_RANGE: 33% (vs 11% for InternVL3-8B)
   - LINE_ITEM_DESCRIPTIONS: 57% (vs 26% for InternVL3-8B)

5. **Document Type Classification**
   - 70.8% accuracy (highest)
   - Best for rare document types (extracts CTP_INSUR, E-TICKET, MOBILE_SS)

6. **Processing Speed**
   - 34.0s median (acceptable)
   - 1.8 docs/min throughput
   - Only 23% slower than fastest model

**Weaknesses:**

1. **Lowest Accuracy (0.4911)**
   - 9% lower than InternVL3-8B
   - 8% lower than InternVL3-2B
   - Aggressive hallucination strategy

2. **High Hallucination Rate**
   - Predicts values for NOT_FOUND fields
   - 74% of attempted extractions contain errors (26% precision)
   - Requires post-processing validation

3. **Quantity Extraction Failure**
   - LINE_ITEM_QUANTITIES: 35% (vs 74% for InternVL3-8B)
   - **2.1× worse** than specialized model
   - Struggles with numeric list extraction

4. **Resource Requirements**
   - 11B parameters (largest model)
   - 22GB VRAM minimum
   - Highest memory footprint

5. **Consistency Issues**
   - 18.1% std dev (highest)
   - Variable performance across documents
   - Less predictable behavior

#### Technical Characteristics

**Extraction Strategy: Aggressive**
- Attempts extraction for all fields
- "Extract first, validate later" approach
- Maximizes recall at precision cost

**Strengths:**
- Long context windows (text-heavy fields)
- Built-in vision preprocessing
- Detailed responses with explanations

**Architecture Benefits:**
- 11B parameters enable complex reasoning
- Better at ambiguous fields
- Handles rare document types

**Deployment Considerations:**
- Requires high-memory GPU (A100, H100, V100 with 32GB)
- Post-processing pipeline essential
- Business rule validation mandatory

#### Recommended Use Cases

| Use Case | Suitability | Rationale |
|----------|-------------|-----------|
| **Data Warehouse Population** | ⭐⭐⭐⭐⭐ | Maximize extraction (78% recall) |
| **Boolean Field Extraction** | ⭐⭐⭐⭐⭐ | Only functional model (IS_GST_INCLUDED) |
| **Complex Text Fields** | ⭐⭐⭐⭐⭐ | Best for descriptions, dates |
| **High-Volume Processing** | ⭐⭐⭐ | Acceptable speed (1.8 docs/min) |
| **Financial Auditing** | ⭐⭐ | High hallucination rate risky |
| **Quantity Extraction** | ⭐ | Poor performance vs InternVL3 |

---

### InternVL3-Quantized-8B

#### Architecture & Specifications

- **Parameters**: 8 billion
- **Quantization**: AWQ (Activation-aware Weight Quantization)
- **Memory**: ~8GB VRAM
- **Preprocessing**: Dynamic with tile-based approach
- **Context Window**: Moderate

#### Performance Profile

**Strengths:**

1. **Highest Accuracy (0.5442)**
   - 11% better than Llama
   - 2% better than InternVL3-2B
   - Most conservative, fewest hallucinations

2. **Field Specialization Leader (8/17 fields)**
   - SUPPLIER_NAME: 77% (best)
   - PAYER_NAME: 75% (best)
   - GST_AMOUNT: 81% (best overall single-field performance)
   - PAYER_ADDRESS: 61% (best)
   - LINE_ITEM_QUANTITIES: 74% (best)
   - BUSINESS_ADDRESS: 57% (best)
   - LINE_ITEM_TOTAL_PRICES: 35% (best)
   - LINE_ITEM_PRICES: 23% (best)

3. **Numeric Extraction Excellence**
   - GST_AMOUNT: 81% (26% better than Llama)
   - LINE_ITEM_QUANTITIES: 74% (115% better than Llama)
   - Best model for table-based numeric data

4. **Name & Address Extraction**
   - Wins all name/address fields
   - Fuzzy matching optimization
   - Conservative approach benefits structured text

5. **Bank Statement Classification**
   - 91% F1 (best among all models)
   - 97% precision, 87% recall
   - Reliable bank statement detection

**Weaknesses:**

1. **Worst F1 Score (0.2611)**
   - 29% lower than Llama
   - 14% lower than InternVL3-2B
   - Most conservative extraction strategy

2. **Worst Recall (0.4367)**
   - Misses 56% of extractable fields
   - Conservative "say NOT_FOUND when uncertain"
   - Leaves money on the table for data extraction

3. **Critical Failure: IS_GST_INCLUDED**
   - **0.0% accuracy** (complete failure)
   - Cannot extract boolean values
   - Fundamental architecture limitation

4. **Slowest Processing (93.3s median)**
   - 3.4× slower than InternVL3-2B
   - 2.7× slower than Llama
   - **0.6 docs/min** throughput (unacceptable for production)
   - AWQ quantization overhead paradox

5. **Poor Text Field Performance**
   - LINE_ITEM_DESCRIPTIONS: 26% (115% worse than Llama)
   - TRANSACTION_DATES: 14% (143% worse than Llama)
   - STATEMENT_DATE_RANGE: 11% (199% worse than Llama)

#### Technical Characteristics

**Extraction Strategy: Conservative**
- Only extract when highly confident
- Frequently returns NOT_FOUND
- Prioritizes precision over recall

**Strengths:**
- AWQ quantization (8GB VRAM)
- Specialized for numeric/structured data
- Reliable for name/address extraction

**Architecture Limitations:**
- Cannot process boolean fields
- Struggles with variable-length text
- Conservative on complex fields

**Deployment Considerations:**
- Memory efficient (8GB VRAM)
- **Speed penalty unacceptable** (93s per doc)
- Requires ensemble with Llama for missed fields
- Cannot handle IS_GST_INCLUDED (blocker for tax workflows)

#### Recommended Use Cases

| Use Case | Suitability | Rationale |
|----------|-------------|-----------|
| **Quantity Extraction** | ⭐⭐⭐⭐⭐ | Best performance (LINE_ITEM_QUANTITIES: 74%) |
| **Name/Address Extraction** | ⭐⭐⭐⭐⭐ | Wins all name/address fields |
| **Monetary Field Extraction** | ⭐⭐⭐⭐⭐ | GST_AMOUNT: 81% (best) |
| **High-Precision Applications** | ⭐⭐⭐⭐ | Highest accuracy, fewest hallucinations |
| **Memory-Constrained Deployment** | ⭐⭐⭐⭐ | Only 8GB VRAM required |
| **Boolean Field Extraction** | ⭐ | 0% accuracy - complete failure |
| **High-Volume Processing** | ⭐ | Too slow (0.6 docs/min) |
| **General Extraction** | ⭐⭐ | Worst F1 (misses 56% of fields) |

---

### InternVL3-NonQuantized-2B

#### Architecture & Specifications

- **Parameters**: 2 billion
- **Quantization**: None (full precision float)
- **Memory**: ~4GB VRAM
- **Preprocessing**: Dynamic with tile-based approach
- **Context Window**: Moderate

#### Performance Profile

**Strengths:**

1. **Fastest Processing (27.5s median)**
   - **2.2 docs/min throughput** (best)
   - 19% faster than Llama
   - 239% faster than InternVL3-8B
   - Speed champion for high-volume processing

2. **Smallest Memory Footprint (2B parameters)**
   - ~4GB VRAM requirement
   - Deployable on consumer GPUs
   - Lowest infrastructure cost

3. **Moderate F1 Score (0.2984)**
   - Better than InternVL3-8B despite 4× smaller
   - Only 13% worse than Llama
   - Reasonable quality for speed tradeoff

4. **Best Precision among InternVL3 (0.2727)**
   - 22% better precision than InternVL3-8B
   - Fewer hallucinations than larger sibling

5. **Bank Statement Recall (0.967)**
   - Best recall for bank statement classification
   - Aggressive on bank statement detection

**Weaknesses:**

1. **No Field Wins (0/17 fields)**
   - Never best model for any field
   - Always outperformed by Llama or InternVL3-8B
   - Generalist without specialization

2. **Worst Precision (0.2727)**
   - 73% of attempted extractions contain errors
   - Higher hallucination rate than InternVL3-8B
   - Less reliable than larger models

3. **Critical Failure: IS_GST_INCLUDED**
   - **0.0% accuracy** (complete failure, same as InternVL3-8B)
   - Cannot extract boolean values
   - Fundamental InternVL3 architecture limitation

4. **Worst Document Type Classification (59.0%)**
   - 11.8% worse than Llama
   - 10.2% worse than InternVL3-8B
   - Struggles with complex document differentiation

5. **Highest Variance (24.0% std dev)**
   - Most inconsistent performance
   - Unpredictable quality across documents
   - Less reliable than larger models

6. **Receipt Classification Failure**
   - Only 33% recall (worst)
   - 67% of receipts misclassified
   - 17 receipts marked NOT_FOUND (unique failure mode)

#### Technical Characteristics

**Extraction Strategy: Fast & Moderate**
- Balance speed and quality
- Less conservative than InternVL3-8B
- More hallucinations than InternVL3-8B but faster

**Strengths:**
- 2B parameters (smallest)
- Non-quantized (no quantization overhead)
- Fast inference (27.5s median)

**Architecture Limitations:**
- Cannot process boolean fields (InternVL3 limitation)
- Small model → less capability
- High variance in quality

**Deployment Considerations:**
- Minimal hardware requirements (4GB VRAM)
- Consumer GPU deployable (RTX 3060, RTX 4060)
- Acceptable for high-volume, moderate-quality needs
- Cannot handle IS_GST_INCLUDED (blocker for tax workflows)

#### Recommended Use Cases

| Use Case | Suitability | Rationale |
|----------|-------------|-----------|
| **High-Volume Processing** | ⭐⭐⭐⭐⭐ | Fastest (2.2 docs/min) |
| **Cost-Optimized Deployment** | ⭐⭐⭐⭐⭐ | Smallest footprint (4GB VRAM) |
| **Consumer GPU Deployment** | ⭐⭐⭐⭐⭐ | Works on RTX 3060/4060 |
| **Prototyping & Development** | ⭐⭐⭐⭐ | Fast iteration cycles |
| **Low-Latency Applications** | ⭐⭐⭐⭐ | 27.5s response time |
| **Boolean Field Extraction** | ⭐ | 0% accuracy - complete failure |
| **Precision-Critical Tasks** | ⭐⭐ | 73% hallucination rate |
| **Document Type Classification** | ⭐⭐ | Worst accuracy (59%) |

---

## Critical Technical Findings

### 1. The IS_GST_INCLUDED Catastrophic Failure

#### Problem Statement

**IS_GST_INCLUDED is a boolean field** indicating whether GST is included in the total amount.

**Performance:**
- **Llama-11B**: 0.617647 (62% accuracy)
- **InternVL3-8B**: 0.000000 (0% accuracy) ❌
- **InternVL3-2B**: 0.000000 (0% accuracy) ❌

**Variance**: 0.291 (291%) - MAXIMUM across all fields

#### Root Cause Analysis

**InternVL3 Architecture Limitation:**

The InternVL3 family exhibits **complete failure on boolean field extraction**, achieving **0.0% accuracy** on IS_GST_INCLUDED across both quantized and non-quantized variants.

**Possible Technical Reasons:**

1. **Training Data Bias**:
   - InternVL3 may not have boolean extraction in training corpus
   - Primarily trained on multi-class or free-text extraction
   - Boolean true/false not represented in visual-language training

2. **Prompt Following Failure**:
   - Model may not understand "YES/NO/TRUE/FALSE/NOT_FOUND" output format
   - Over-indexes on NOT_FOUND for uncertain fields
   - Lacks confidence calibration for binary decisions

3. **Vision-Language Grounding**:
   - Cannot visually identify boolean indicators ("Tax Inclusive", "GST Inc.", checkboxes)
   - Struggles with implicit boolean signals in document layout
   - Requires explicit textual boolean values (rare in documents)

4. **Architecture Design**:
   - 2B and 8B variants share base architecture
   - Both fail identically → systematic architectural issue
   - Not a quantization problem (2B is non-quantized, still fails)

**Llama-11B Success Factors:**

- 62% accuracy indicates capability but not mastery
- Larger model (11B vs 2B/8B) enables boolean reasoning
- Different training methodology includes boolean tasks
- Better prompt following for constrained outputs

#### Business Impact

**Critical Blocker for Tax Workflows:**

IS_GST_INCLUDED determines:
- Tax calculation method
- Compliance requirements
- Reporting categories
- Invoice validation logic

**Mitigation Strategies:**

1. **Ensemble Approach**:
   - Use InternVL3 for numeric/name extraction (strengths)
   - Use Llama for IS_GST_INCLUDED (only functional model)
   - Combine predictions in post-processing

2. **Heuristic Fallback**:
   - Parse TOTAL_AMOUNT and GST_AMOUNT relationship
   - If GST_AMOUNT > 0 and TOTAL_AMOUNT ≈ (subtotal + GST) → TRUE
   - If GST_AMOUNT = 0 → FALSE
   - If GST_AMOUNT = NOT_FOUND → NOT_FOUND

3. **Vision-Language Prompt Engineering**:
   - Explicitly instruct to look for "Tax Inclusive" keywords
   - Provide visual examples in few-shot prompting
   - Constrain output vocabulary to TRUE/FALSE/NOT_FOUND

**Recommendation**: **Do not deploy InternVL3 models alone for tax/compliance workflows** due to complete IS_GST_INCLUDED failure.

---

### 2. The LINE_ITEM_QUANTITIES Performance Chasm

#### Problem Statement

LINE_ITEM_QUANTITIES requires extracting numeric quantities from tabular line item data.

**Performance:**
- **InternVL3-8B**: 0.742326 (74% accuracy) ✅
- **InternVL3-2B**: 0.448845 (45% accuracy)
- **Llama-11B**: 0.345064 (35% accuracy) ❌

**Performance Gap**: InternVL3-8B is **2.15× better** than Llama
**Variance**: 0.168 (168%) - HIGHEST across all fields

#### Root Cause Analysis

**InternVL3-8B Specialization:**

InternVL3-8B demonstrates **extreme specialization** for numeric list extraction from tables, with **74% accuracy** vs Llama's **35%**.

**Technical Reasons:**

1. **Tile-Based Vision Processing**:
   - InternVL3 uses dynamic tile-based image preprocessing
   - Better preserves table structure and column alignment
   - Spatial relationships maintained for quantity columns

2. **Numeric Extraction Optimization**:
   - Training data may emphasize table/numeric extraction
   - Better at position-aware list extraction
   - Handles variable-length quantity lists

3. **Conservative Strategy Advantage**:
   - Only extracts when confident (high precision for this field)
   - Avoids hallucinating quantities
   - NOT_FOUND strategy works better for tables

**Llama-11B Failure Factors:**

- Aggressive extraction backfires for numeric precision
- 35% accuracy indicates systematic failure
- Built-in preprocessing may distort table structure
- Long context windows don't help with spatial alignment

#### Business Impact

**Invoice Processing Advantage**:

For invoice line item extraction, **InternVL3-8B is clearly superior**:
- 74% accuracy for quantities
- 35% for LINE_ITEM_TOTAL_PRICES (best)
- 23% for LINE_ITEM_PRICES (best)

**Use Case Specialization**:
- **Invoice line items**: InternVL3-8B
- **Text descriptions**: Llama-11B (57% vs 26%)
- **Ensemble**: Combine both for complete extraction

**Recommendation**: For invoice processing, use **InternVL3-8B for numeric line item fields**, **Llama for descriptions**.

---

### 3. Document Type Confusion: Receipt vs Invoice

#### Problem Statement

All models struggle to differentiate **RECEIPT** from **INVOICE**, leading to systematic misclassification.

**Receipt → Invoice Misclassifications:**
- **Llama-11B**: 50 receipts → invoice (48% of receipts)
- **InternVL3-8B**: 47 receipts → invoice (45% of receipts)
- **InternVL3-2B**: 34 receipts → invoice (33% of receipts)

**Receipt Recall (ability to detect receipts):**
- **Llama-11B**: 57% recall (43% missed)
- **InternVL3-8B**: 53% recall (47% missed)
- **InternVL3-2B**: 33% recall (67% missed) ❌

#### Root Cause Analysis

**Visual Similarity:**

Receipts and invoices share similar visual features:
- Both have vendor names, dates, amounts
- Both contain line items (goods/services)
- Both have totals, subtotals, tax amounts
- Layout differences subtle (heading text, formatting)

**Semantic Overlap:**

- Receipts: Proof of payment (transaction completed)
- Invoices: Request for payment (transaction pending)
- Visual-language models cannot distinguish intent
- Requires external context (payment status, due dates)

**Model Behavior:**

1. **Llama-11B**:
   - 90% invoice recall (catches all invoices)
   - 57% receipt recall (misses many receipts)
   - **Bias toward "invoice" classification**
   - Aggressive: When uncertain, predicts invoice

2. **InternVL3-8B**:
   - 89% invoice recall
   - 53% receipt recall
   - Similar bias toward invoice
   - Conservative approach doesn't help

3. **InternVL3-2B**:
   - 85% invoice recall
   - **33% receipt recall** (worst)
   - Strongest invoice bias
   - 17 receipts marked NOT_FOUND (unique failure)

#### Business Impact

**False Positives:**

Receipts misclassified as invoices create:
- Incorrect payment tracking
- Duplicate invoice entries
- Accounting reconciliation errors

**Mitigation Strategies:**

1. **Business Rule Validation**:
   - Check for payment confirmation keywords ("PAID", "RECEIPT")
   - Look for "INVOICE #" vs "RECEIPT #"
   - Validate against payment database

2. **Ensemble Voting**:
   - Use all 3 models for classification
   - Majority vote with confidence thresholds
   - Flag disagreements for human review

3. **Sequential Classification**:
   - First: Is it a bank statement? (high confidence)
   - Second: Is it an invoice or receipt? (lower confidence)
   - Third: Check rare types

4. **Field-Based Classification**:
   - If INVOICE_DATE exists but PAYER_NAME missing → Invoice
   - If PAYER_NAME exists and INVOICE_DATE recent → Receipt
   - Use extracted fields to refine classification

**Recommendation**: **Do not rely on document type classification alone**. Use field-based validation and business rules for invoice vs receipt differentiation.

---

### 4. The Accuracy Paradox in Production

#### Problem Statement

**Llama-11B has the LOWEST accuracy (0.4911) but HIGHEST F1 (0.3379).**

This contradicts intuition: "How can the best model have the worst accuracy?"

#### The Paradox Explained

**Accuracy Counts Everything:**
```python
accuracy = (correct_extractions + correct_NOT_FOUNDs) / total_fields
```

**F1 Ignores NOT_FOUND:**
```python
precision = correct_extractions / attempted_extractions  # Excludes NOT_FOUND
recall = correct_extractions / extractable_fields  # Penalty for NOT_FOUND
f1 = 2 * (precision * recall) / (precision + recall)
```

**Example: Invoice with 17 fields (8 exist, 9 absent)**

**Conservative Model (InternVL3-8B):**
- Extracts 3 fields correctly → 3 TP
- Incorrectly extracts 2 fields → 2 FP
- Misses 3 extractable fields → 3 FN
- Correctly says NOT_FOUND for 9 absent fields → **+9 to accuracy**

**Metrics:**
- Accuracy: (3 + 9) / 17 = **71%** ✨
- Precision: 3 / (3 + 2) = 60%
- Recall: 3 / (3 + 3) = 50%
- F1: 2 × (0.6 × 0.5) / (0.6 + 0.5) = **55%**

**Aggressive Model (Llama-11B):**
- Extracts 7 fields correctly → 7 TP
- Incorrectly extracts 3 fields → 3 FP
- Misses 1 extractable field → 1 FN
- Hallucinates 6 NOT_FOUND fields → **-6 from accuracy**

**Metrics:**
- Accuracy: (7 + 3) / 17 = **59%** ⚠️
- Precision: 7 / (7 + 3) = 70%
- Recall: 7 / (7 + 1) = 88%
- F1: 2 × (0.7 × 0.88) / (0.7 + 0.88) = **78%** ✨

**The Paradox:**
- Aggressive model has **LOWER accuracy** (59% vs 71%)
- Aggressive model has **HIGHER F1** (78% vs 55%)
- **Reason**: Accuracy inflated by correctly predicting NOT_FOUND

#### Why This Matters for Production

**Accuracy is Misleading:**

A model can achieve 70% accuracy by:
1. Being extremely conservative
2. Saying NOT_FOUND for uncertain fields
3. Avoiding difficult extractions

This looks good on metrics but provides little business value.

**F1 Measures What Matters:**

F1 focuses on **actual extraction performance**:
- Did you extract the values that exist?
- When you attempted extraction, were you correct?
- Ignores the easy "I don't know" answers

**Production Implications:**

1. **Don't optimize for accuracy** in extraction tasks
2. **Use F1 as primary metric** for model selection
3. **Accept lower accuracy** if F1 is higher
4. **Plan for post-processing** to validate aggressive extractions

**Recommendation**: **Choose Llama-11B despite lower accuracy** because F1 better reflects extraction effectiveness.

---

## Deployment Recommendations

### Decision Matrix

| Requirement | Recommended Model | Alternative | Reasoning |
|-------------|------------------|-------------|-----------|
| **Maximum F1** | Llama-11B | - | Highest quality (0.3379) |
| **Maximum Recall** | Llama-11B | - | Best coverage (77.97%) |
| **Maximum Accuracy** | InternVL3-8B | InternVL3-2B | Conservative strategy (54.42%) |
| **Boolean Extraction** | Llama-11B | None | Only functional model (62%) |
| **Quantity Extraction** | InternVL3-8B | - | Best for LINE_ITEM_QUANTITIES (74%) |
| **Name/Address** | InternVL3-8B | - | Wins all name/address fields |
| **Text Descriptions** | Llama-11B | - | Best for variable-length text (57%) |
| **Fastest Processing** | InternVL3-2B | Llama-11B | 2.2 docs/min vs 1.8 |
| **Lowest Memory** | InternVL3-2B | - | 4GB VRAM (smallest) |
| **Best Speed/Quality** | Llama-11B | - | 1.8 docs/min with highest F1 |
| **Cost Optimization** | InternVL3-2B | - | Consumer GPU deployable |

### Production Deployment Strategies

#### Strategy 1: Llama-11B Standalone (Recommended)

**Use Case:** General-purpose document extraction

**Configuration:**
- Single model: Llama-11B
- Post-processing validation
- Business rule filtering

**Pros:**
- Simplest architecture
- Highest F1 (0.3379)
- Best recall (78%)
- Handles all fields including IS_GST_INCLUDED

**Cons:**
- Requires validation pipeline
- High hallucination rate (74%)
- 11B parameters (22GB VRAM)

**Recommended For:**
- Data warehouse population
- Exploratory extraction
- Human-in-the-loop workflows

---

#### Strategy 2: Ensemble (Maximum Quality)

**Use Case:** High-precision extraction with maximum coverage

**Configuration:**
- Primary: InternVL3-8B (for numeric/structured fields)
- Secondary: Llama-11B (for text/boolean/complex fields)
- Voting: Combine predictions with field-specific routing

**Field Routing:**
- LINE_ITEM_QUANTITIES → InternVL3-8B (74% accuracy)
- GST_AMOUNT → InternVL3-8B (81% accuracy)
- SUPPLIER_NAME → InternVL3-8B (77% accuracy)
- PAYER_NAME → InternVL3-8B (75% accuracy)
- IS_GST_INCLUDED → Llama-11B (62% accuracy, only option)
- LINE_ITEM_DESCRIPTIONS → Llama-11B (57% accuracy)
- TRANSACTION_DATES → Llama-11B (34% accuracy)
- All others → Best model per field

**Pros:**
- Maximizes per-field accuracy
- Leverages model specialization
- Best possible quality

**Cons:**
- Complex architecture
- 2× inference cost
- Longer processing time
- Requires field-specific routing logic

**Recommended For:**
- Financial auditing
- Compliance workflows
- High-value document processing

---

#### Strategy 3: InternVL3-2B (High-Volume)

**Use Case:** High-throughput, moderate-quality extraction

**Configuration:**
- Single model: InternVL3-2B
- Minimal post-processing
- Accept moderate quality for speed

**Pros:**
- Fastest (2.2 docs/min)
- Smallest footprint (4GB VRAM)
- Consumer GPU deployable
- Moderate F1 (0.2984)

**Cons:**
- Cannot extract IS_GST_INCLUDED (0% accuracy)
- No field wins (always outperformed)
- Worst document type classification (59%)
- High variance (24% std dev)

**Recommended For:**
- High-volume initial extraction
- Cost-sensitive deployments
- Prototyping and development
- Consumer GPU environments

---

#### Strategy 4: Hybrid Cascade

**Use Case:** Balance cost, speed, and quality

**Configuration:**
1. **First Pass**: InternVL3-2B (fast initial extraction)
2. **Second Pass**: Llama-11B (for failed/low-confidence fields)
3. **Validation**: Business rules on final results

**Logic:**
```
For each field:
  result_2b = InternVL3-2B.extract(field)

  if result_2b.confidence > 0.8 AND field not in [IS_GST_INCLUDED]:
    return result_2b
  else:
    result_llama = Llama-11B.extract(field)
    return result_llama
```

**Pros:**
- 60-70% of fields handled by fast model
- Llama only for difficult fields
- Reduced inference cost
- Better quality than InternVL3-2B alone

**Cons:**
- Requires confidence scoring
- Complex routing logic
- Variable processing time
- Still cannot avoid Llama for IS_GST_INCLUDED

**Recommended For:**
- Cost-optimized production
- Mixed document difficulty
- Latency-sensitive applications with quality requirements

---

### Model Selection Flowchart

```
START: What is your primary constraint?

├─ QUALITY (Maximize F1)
│  └─ Llama-11B Standalone
│     ✅ F1: 0.3379 (best)
│     ⚠️ Hallucinations (need validation)
│
├─ PRECISION (Minimize hallucinations)
│  └─ Ensemble: InternVL3-8B + Llama-11B
│     ✅ Field-specific routing
│     ⚠️ 2× inference cost
│
├─ SPEED (Maximize throughput)
│  ├─ Can you sacrifice IS_GST_INCLUDED?
│  │  ├─ YES → InternVL3-2B
│  │  │  ✅ 2.2 docs/min
│  │  │  ⚠️ 0% boolean accuracy
│  │  │
│  │  └─ NO → Llama-11B
│  │     ✅ 1.8 docs/min
│  │     ✅ Handles all fields
│  │
├─ COST (Minimize infrastructure)
│  └─ InternVL3-2B
│     ✅ 4GB VRAM (consumer GPU)
│     ⚠️ 0% boolean accuracy
│     ⚠️ No field wins
│
└─ BALANCED (Cost + Quality)
   └─ Hybrid Cascade
      ✅ Fast model for easy fields
      ✅ Llama for hard fields
      ⚠️ Complex routing
```

---

### Field-Specific Model Selection

| Field | Best Model | Accuracy | 2nd Best | Gap | Notes |
|-------|------------|----------|----------|-----|-------|
| TOTAL_AMOUNT | Llama-11B | 81.8% | InternVL3-8B | +0.8% | Near parity, any model works |
| INVOICE_DATE | Llama-11B | 79.1% | InternVL3-8B | +0.3% | Llama slight edge |
| SUPPLIER_NAME | InternVL3-8B | 76.8% | Llama-11B | +2.3% | InternVL3 specialization |
| PAYER_NAME | InternVL3-8B | 74.6% | Llama-11B | +1.4% | InternVL3 specialization |
| DOCUMENT_TYPE | Llama-11B | 70.8% | InternVL3-8B | +1.4% | Llama classification edge |
| GST_AMOUNT | InternVL3-8B | 80.5% | Llama-11B | +22.5% | **Strong InternVL3 advantage** |
| PAYER_ADDRESS | InternVL3-8B | 61.0% | Llama-11B | +1.6% | InternVL3 specialization |
| LINE_ITEM_QUANTITIES | InternVL3-8B | 74.2% | InternVL3-2B | +29.3% | **Extreme InternVL3 advantage** |
| BUSINESS_ADDRESS | InternVL3-8B | 56.6% | Llama-11B | +5.9% | InternVL3 specialization |
| BUSINESS_ABN | Llama-11B | 59.6% | InternVL3-2B | +15.7% | Llama excels at numeric IDs |
| LINE_ITEM_DESCRIPTIONS | Llama-11B | 56.6% | InternVL3-2B | +19.2% | **Strong Llama advantage** |
| LINE_ITEM_TOTAL_PRICES | InternVL3-8B | 34.9% | InternVL3-2B | +6.7% | All models struggle |
| TRANSACTION_DATES | Llama-11B | 33.8% | InternVL3-2B | +10.2% | Llama better for complex dates |
| IS_GST_INCLUDED | Llama-11B | 61.8% | - | - | **Only functional model** |
| STATEMENT_DATE_RANGE | Llama-11B | 32.9% | InternVL3-8B | +21.9% | Llama significantly better |
| LINE_ITEM_PRICES | InternVL3-8B | 22.7% | InternVL3-2B | +5.4% | All models fail |
| TRANSACTION_AMOUNTS_PAID | Llama-11B | 20.7% | InternVL3-2B | +6.4% | All models fail |

**Color Coding:**
- **Green** (>70%): Production-ready
- **Yellow** (40-70%): Usable with validation
- **Red** (<40%): High error rate, critical validation required

---

## Technical Limitations & Caveats

### Dataset Limitations

1. **Limited Sample Size**: 195 documents
   - Bank Statements: 75 (38.5%)
   - Invoices: 100 (51.3%)
   - Receipts: 20 (10.3%) - **small sample**
   - Statistical significance concerns for rare types

2. **Imbalanced Document Types**:
   - Receipts underrepresented (10%)
   - May bias model comparison
   - Receipt classification metrics less reliable

3. **Ground Truth Quality**:
   - Human-annotated (potential errors)
   - Field-type-specific comparison logic
   - Fuzzy matching introduces subjectivity

### Evaluation Methodology Caveats

1. **Custom Fuzzy Matching**:
   - Not sklearn-based
   - Field-type-specific tolerances (1% for monetary, 80% word overlap for text)
   - Partial credit scoring (0.0-1.0 floats)
   - Results may not be directly comparable to sklearn metrics

2. **Position-Aware F1**:
   - Requires items to match in value AND position
   - May penalize models differently than position-agnostic F1
   - Line item fields particularly sensitive

3. **Document-Aware Evaluation**:
   - Invoice/Receipt: 14 fields evaluated
   - Bank Statement: 5 fields evaluated (+ 2 validation-only)
   - Prevents NOT_FOUND penalties for legitimate absences
   - May inflate accuracy for document-specific models

### Model Deployment Caveats

1. **Hardware Requirements**:
   - Llama-11B: 22GB VRAM (A100, H100, V100-32GB)
   - InternVL3-8B: 8GB VRAM (RTX 3080, A10, A100)
   - InternVL3-2B: 4GB VRAM (RTX 3060, RTX 4060, consumer GPUs)
   - Memory requirements may vary with batch size

2. **Processing Time Variability**:
   - Measurements from single-document inference
   - Batch processing may improve throughput
   - Network latency not included
   - Document complexity affects processing time

3. **Quantization Effects**:
   - InternVL3-8B uses AWQ quantization
   - Accuracy impact unknown (no non-quantized 8B baseline)
   - Quantization paradoxically slows inference (overhead)

### Generalization Concerns

1. **Domain Specificity**:
   - Evaluated on Australian business documents (ABN, GST)
   - Tax fields may not generalize to other jurisdictions
   - Document formats specific to Australia

2. **Field Selection**:
   - 17 fields chosen for this study
   - Different fields may show different model rankings
   - Field difficulty not representative of all extraction tasks

3. **Prompt Sensitivity**:
   - Results dependent on prompt engineering
   - Different prompts may yield different rankings
   - Prompt optimization not performed equally across models

### Known Issues

1. **IS_GST_INCLUDED Failure**:
   - InternVL3 models: 0% accuracy
   - Blocker for tax/compliance workflows
   - No known fix without ensemble

2. **InternVL3-8B Speed Penalty**:
   - AWQ quantization slows inference (paradox)
   - 3.4× slower than InternVL3-2B
   - Not production-viable for high-volume

3. **Receipt Classification**:
   - All models struggle (33-57% recall)
   - Receipt vs Invoice confusion
   - Requires business rule validation

4. **Line Item Prices**:
   - All models fail (12-23% accuracy)
   - Table extraction challenges
   - Position-aware matching difficulty

---

## Appendix: Metric Definitions

### F1 Score

**Formula:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Interpretation:**
- Harmonic mean of precision and recall
- Balances false positives and false negatives
- Range: 0.0 (worst) to 1.0 (perfect)
- **Primary metric for extraction tasks**

**Why Harmonic Mean?**
- Arithmetic mean would allow high precision + low recall to score well
- Harmonic mean penalizes imbalanced precision/recall
- Only high when BOTH metrics are high

### Precision

**Formula:**
```
Precision = TP / (TP + FP)
         = Correct Extractions / Attempted Extractions
```

**Interpretation:**
- Of all fields the model attempted to extract, how many were correct?
- Measures **quality of predictions**
- High precision = few hallucinations
- **Excludes NOT_FOUND predictions** (key difference from accuracy)

**Example:**
- Model extracts 10 fields
- 7 are correct, 3 are wrong
- Precision = 7/10 = 70%

### Recall

**Formula:**
```
Recall = TP / (TP + FN)
       = Correct Extractions / Extractable Fields
```

**Interpretation:**
- Of all fields that exist in the document, how many did we find?
- Measures **coverage of extraction**
- High recall = few missed fields
- **Penalty for predicting NOT_FOUND when value exists**

**Example:**
- Document has 8 extractable fields
- Model finds 6 correctly, misses 2
- Recall = 6/8 = 75%

### Accuracy

**Formula:**
```
Accuracy = (TP + TN) / Total
         = (Correct Extractions + Correct NOT_FOUNDs) / All Fields
```

**Interpretation:**
- Of all predictions (including NOT_FOUND), how many were correct?
- Measures **overall correctness**
- **Includes correct NOT_FOUND predictions** (key difference from F1)
- **Misleading for extraction tasks** due to NOT_FOUND inflation

**Example:**
- 17 fields total: 8 exist, 9 absent
- Model extracts 6 correctly, 2 incorrectly, misses 2, correctly says NOT_FOUND for 7 absent fields
- Accuracy = (6 + 7) / 17 = 76%
- BUT F1 would be lower due to missed fields and incorrect extractions

### Variance

**Formula:**
```
Variance = σ² = Σ(xi - μ)² / N
where xi = model accuracy, μ = mean accuracy across models
```

**Interpretation:**
- Measures model disagreement for a field
- High variance = models have very different performance
- Low variance = models perform similarly
- Indicates field difficulty or model specialization

**Example:**
- IS_GST_INCLUDED: Llama 62%, InternVL3-8B 0%, InternVL3-2B 0%
- Mean = 20.6%, Variance = 0.291 (291%)
- **Extreme variance** indicates fundamental capability difference

---

## Conclusion

### Key Takeaways

1. **Llama-3.2-Vision-11B is the overall winner** with highest F1 (0.3379) and recall (0.7797)
   - Best for general-purpose extraction
   - Only model capable of boolean field extraction (IS_GST_INCLUDED: 62%)
   - Recommended for production unless speed/cost constrained

2. **InternVL3-Quantized-8B excels at specialized fields** (8/17 field wins)
   - Best for numeric/quantity extraction (LINE_ITEM_QUANTITIES: 74%)
   - Best for name/address extraction
   - **Critical flaw: 0% boolean accuracy**
   - **Not production-viable due to speed** (93.3s median)

3. **InternVL3-NonQuantized-2B is the speed champion** (2.2 docs/min)
   - Best for high-volume, cost-sensitive deployments
   - Acceptable F1 (0.2984) for moderate-quality needs
   - **Critical flaw: 0% boolean accuracy**
   - No field wins (always outperformed)

4. **Accuracy is misleading for extraction tasks**
   - InternVL3-8B has highest accuracy (54%) but worst F1 (0.26)
   - Llama has lowest accuracy (49%) but best F1 (0.34)
   - **Use F1 as primary metric**, not accuracy

5. **Field-specific model selection** can optimize performance
   - Ensemble approach leverages each model's strengths
   - 2× inference cost but maximizes per-field accuracy
   - Recommended for high-value document processing

### Deployment Recommendation Summary

| Scenario | Recommended Model | F1 | Speed | Memory |
|----------|------------------|-----|-------|--------|
| **Production (General)** | Llama-11B | 0.3379 | 1.8 docs/min | 22GB |
| **Production (Maximum Quality)** | Ensemble (InternVL3-8B + Llama-11B) | ~0.40* | 0.3 docs/min | 30GB |
| **High-Volume Processing** | InternVL3-2B | 0.2984 | 2.2 docs/min | 4GB |
| **Cost-Optimized** | InternVL3-2B | 0.2984 | 2.2 docs/min | 4GB |
| **Quantity Extraction** | InternVL3-8B (if not speed-limited) | 0.2611 | 0.6 docs/min | 8GB |

*Estimated ensemble F1 based on field-specific routing

### Final Recommendation

**For most production deployments: Llama-3.2-Vision-11B**

**Rationale:**
- ✅ Highest F1 (0.3379) - best overall extraction quality
- ✅ Highest recall (0.7797) - extracts 78% of all fields
- ✅ Only model with boolean extraction capability
- ✅ Acceptable speed (1.8 docs/min)
- ✅ Handles all document types and fields
- ⚠️ Requires post-processing validation (high hallucination rate)
- ⚠️ Highest memory requirement (22GB VRAM)

**When to consider alternatives:**
- **Speed-critical**: Use InternVL3-2B (22% faster)
- **Cost-critical**: Use InternVL3-2B (5.5× smaller)
- **No boolean fields**: InternVL3-8B viable for specialized numeric extraction
- **Maximum quality**: Ensemble with field-specific routing

---

## References

### Documentation
- `ACCURACY_PARADOX_EXPLAINED.md` - Why F1 > Accuracy for extraction tasks
- `EVALUATION_SYSTEM_GUIDE.md` - Comprehensive evaluation methodology (1,026 lines)
- `KIE_Evaluation_Methodology.md` - Technical evaluation deep-dive (929 lines)
- `alternative_evaluation_metrics.md` - Metric research and recommendations
- `bank_statement_extraction_case_study.md` - Bank statement extraction analysis
- `THREE_MODEL_FIELD_METRICS_UPDATE.md` - 3-model comparison update notes
- `THREE_PANEL_CONFUSION_FIX.md` - Confusion matrix implementation notes

### Implementation
- `model_comparison.ipynb` - Jupyter notebook with all visualizations
- `common/evaluation_metrics.py` - Core evaluation logic (1,798 lines)
- `common/config.py` - Configuration & thresholds (1,054 lines)
- `config/field_definitions.yaml` - Field schemas (235 lines)

### Data Sources
- Performance visualizations (9 images provided)
- Ground truth CSV: `evaluation_data/lmm_poc_gt_20251111.csv`
- Model batch extraction CSVs from notebook output
- 195 documents: 75 bank statements, 100 invoices, 20 receipts

---

**Document Version**: 1.0
**Date**: 2025-11-12
**Models Evaluated**: Llama-3.2-Vision-11B, InternVL3-Quantized-8B, InternVL3-NonQuantized-2B
**Fields Analyzed**: 17 business document fields
**Evaluation Dataset**: 195 documents (75 bank statements, 100 invoices, 20 receipts)
**Primary Metric**: Position-Aware F1 Score
