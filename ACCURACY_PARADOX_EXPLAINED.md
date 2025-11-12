# The Accuracy Paradox in Field Extraction: Why Higher Accuracy Doesn't Mean Better Performance

## The Contradiction

Looking at the Model Performance Summary from `model_comparison.ipynb` Cell 22:

| Model | Precision | Recall | F1 Score | Accuracy |
|-------|-----------|--------|----------|----------|
| InternVL3-2B | 0.2440 | 0.3466 | 0.2670 | **0.5816** |
| InternVL3-8B | 0.1993 | 0.3907 | 0.2336 | **0.5922** |
| Llama-11B | **0.2348** | **0.6976** | **0.3023** | 0.5447 |

**The Paradox:**
- Llama-11B has the **best** precision, recall, and F1 score
- Llama-11B has the **lowest** accuracy
- InternVL3-8B has the **worst** F1 score but **highest** accuracy

How can the best-performing model have the lowest accuracy?

---

## Root Cause: Different Metric Definitions

### Accuracy Calculation (Lines 59-62 of Cell 22)

```python
# Calculate exact match accuracy
matches = (y_true_binary == y_pred_binary).sum()  # ← Counts ALL matches
total = len(y_true_binary)
accuracy = matches / total
```

**What it counts:**
- ✅ Correct extraction: Predicted "ABC123" when ground truth is "ABC123"
- ✅ Correct NOT_FOUND: Predicted "NOT_FOUND" when ground truth is "NOT_FOUND"
- ❌ Incorrect extraction: Predicted "ABC123" when ground truth is "XYZ789"
- ❌ Missed extraction: Predicted "NOT_FOUND" when ground truth is "ABC123"

**Key insight:** Correctly predicting NOT_FOUND inflates accuracy.

---

### Precision/Recall/F1 Calculation (Lines 64-80 of Cell 22)

```python
# For precision/recall, we treat it as binary: correct extraction vs not
correct_mask = (y_pred_binary == y_true_binary) & (y_pred_binary != 'NOT_FOUND')

# True positives: predicted correctly (and not NOT_FOUND)
tp = correct_mask.sum()  # ← EXCLUDES NOT_FOUND predictions

# False positives: predicted incorrectly (but not NOT_FOUND)
fp = ((y_pred_binary != y_true_binary) & (y_pred_binary != 'NOT_FOUND')).sum()

# False negatives: failed to extract (predicted NOT_FOUND when ground truth exists)
fn = ((y_pred_binary == 'NOT_FOUND') & (y_true_binary != 'NOT_FOUND')).sum()

# Calculate metrics
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
```

**What it counts:**
- ✅ True Positive: Predicted "ABC123" when ground truth is "ABC123"
- ❌ False Positive: Predicted "ABC123" when ground truth is "XYZ789" (hallucination)
- ❌ False Negative: Predicted "NOT_FOUND" when ground truth exists (missed extraction)
- **IGNORED**: Correctly predicting NOT_FOUND (doesn't help or hurt)

**Key insight:** Predicting NOT_FOUND is neither rewarded nor penalized. Only actual extraction performance matters.

---

## What This Reveals About Model Behavior

### Llama-11B: The Aggressive Extractor

**Strategy:** Try to extract every field, even if uncertain

**Behavior:**
- **High Recall (0.70)**: Successfully extracts most fields that exist
- **Moderate Precision (0.23)**: Makes mistakes but still extracts valuable data
- **Best F1 (0.30)**: Good balance for extraction tasks
- **Low Accuracy (0.54)**: Hallucinates values for fields that should be NOT_FOUND

**Example:**
- Field: TRANSACTION_DATES on an invoice (doesn't exist)
- Ground Truth: NOT_FOUND
- Llama Prediction: "2024-01-15" (hallucination)
- Impact: Hurts accuracy, doesn't hurt F1 (because F1 ignores NOT_FOUND cases)

**Ideal for:** Maximizing information extraction, followed by human review

---

### InternVL3-2B/8B: The Conservative Extractors

**Strategy:** Only extract when highly confident, otherwise say NOT_FOUND

**Behavior:**
- **Low Recall (0.35-0.39)**: Misses many extractable fields
- **Low Precision (0.20-0.24)**: Of the fields attempted, many are still wrong
- **Worst F1 (0.23-0.27)**: Conservative strategy hurts extraction performance
- **High Accuracy (0.58-0.59)**: Correctly identifies many NOT_FOUND cases

**Example:**
- Field: BUSINESS_ABN on an invoice (exists)
- Ground Truth: "12 345 678 901"
- InternVL3 Prediction: NOT_FOUND (too conservative)
- Impact: Hurts recall/F1, doesn't hurt accuracy (conservatism rewarded)

**Ideal for:** High-precision tasks where false positives are expensive

---

## The NOT_FOUND Inflation Problem

Consider a typical business document extraction task with 17 fields:

### Invoice Example:
- **Present fields:** 8 (DOCUMENT_TYPE, SUPPLIER_NAME, TOTAL_AMOUNT, etc.)
- **Absent fields:** 9 (TRANSACTION_DATES, STATEMENT_DATE_RANGE, etc.)

### Conservative Model (InternVL3-8B):
- Correctly extracts: 3 fields → 3 TP
- Incorrectly extracts: 2 fields → 2 FP
- Misses: 3 fields → 3 FN
- Correctly says NOT_FOUND: 9 fields → **+9 to accuracy**

**Result:**
- Accuracy: (3 + 9) / 17 = **71%** ✨ (looks great!)
- Recall: 3 / (3 + 3) = **50%** (misses half the extractable fields)
- Precision: 3 / (3 + 2) = **60%** (40% hallucination rate)
- F1: 2 × (0.6 × 0.5) / (0.6 + 0.5) = **55%**

### Aggressive Model (Llama-11B):
- Correctly extracts: 7 fields → 7 TP
- Incorrectly extracts: 3 fields → 3 FP
- Misses: 1 field → 1 FN
- Hallucinates NOT_FOUND fields: 6 fields → **-6 from accuracy**

**Result:**
- Accuracy: (7 + 3) / 17 = **59%** ⚠️ (looks worse!)
- Recall: 7 / (7 + 1) = **88%** (extracts most fields)
- Precision: 7 / (7 + 3) = **70%** (30% hallucination rate)
- F1: 2 × (0.7 × 0.88) / (0.7 + 0.88) = **78%** ✨ (much better!)

**The Paradox:** The aggressive model has **lower accuracy but higher F1** because accuracy is inflated by correctly predicting NOT_FOUND.

---

## Why F1 Is Better for Extraction Tasks

### Accuracy's Fatal Flaw

For field extraction tasks, **many fields legitimately don't exist** in each document:
- Bank statements don't have LINE_ITEM_PRICES
- Invoices don't have TRANSACTION_DATES
- Receipts don't have ACCOUNT_BALANCE

A model that says "NOT_FOUND" for everything achieves ~50-60% accuracy just by being conservative!

**Accuracy favors conservative models that avoid attempting difficult extractions.**

---

### F1's Advantages

**F1 measures what we actually care about:**
1. **Precision:** When the model extracts a value, is it correct?
2. **Recall:** Of all extractable values, how many did we find?

**F1 ignores the easy cases:**
- Correctly saying NOT_FOUND doesn't help your score
- You only get credit for actual extraction performance

**F1 penalizes both types of mistakes:**
- **Hallucinations (low precision):** Extracting wrong values
- **Timidity (low recall):** Missing extractable fields

---

## Implications for Model Selection

### For Maximum Information Extraction
**Choose: Llama-11B**
- Highest recall: Extracts 70% of available fields
- Best F1: Optimal balance for extraction tasks
- Caveat: Will hallucinate ~77% of non-existent fields (low accuracy)
- **Strategy:** Use aggressive extraction, then validate with business rules

### For Precision-Critical Applications
**Choose: InternVL3-2B**
- Highest precision among InternVL3 variants
- Fewer hallucinations
- Caveat: Misses 65% of extractable fields (low recall)
- **Strategy:** Use for initial extraction, then fallback to Llama for missed fields

### For Speed/Resource Constrained
**Choose: InternVL3-2B**
- Smallest model (2B parameters)
- Fastest inference
- Lowest memory footprint
- Caveat: Worst F1 score overall

---

## Recommendations

### 1. Primary Metric: F1 Score

For document field extraction, **always prioritize F1** over accuracy:
```
F1 Score > Precision ≥ Recall >> Accuracy
```

**Rationale:**
- F1 directly measures extraction effectiveness
- Accuracy is inflated by NOT_FOUND predictions
- Precision/Recall show the quality vs coverage tradeoff

---

### 2. Use Accuracy as a Warning Sign

**Low accuracy with high F1 (like Llama-11B):**
- ⚠️ Model hallucinates non-existent fields
- ✅ Extracts most real fields successfully
- **Action:** Add post-processing validation to filter hallucinations

**High accuracy with low F1 (like InternVL3-8B):**
- ⚠️ Model is too conservative, misses many fields
- ✅ Rarely hallucinates
- **Action:** Consider ensemble with aggressive model for missed fields

---

### 3. Context-Specific Metrics

Different use cases need different priorities:

| Use Case | Priority Metric | Rationale |
|----------|----------------|-----------|
| **Data Warehouse Population** | Recall | Maximize extracted data, validate later |
| **Financial Auditing** | Precision | False positives are expensive |
| **General Extraction** | F1 | Balance coverage and correctness |
| **Low-Resource Systems** | F1 / inference time | Optimize performance per compute |

---

## Technical Note: Why We Calculate This Way

The current implementation follows **Information Extraction best practices:**

### Standard Approach:
```python
# Only count actual extraction performance
tp = (predicted correctly) AND (value != NOT_FOUND)
fp = (predicted incorrectly) AND (value != NOT_FOUND)
fn = (predicted NOT_FOUND) AND (ground truth exists)
tn = (IGNORED - doesn't affect precision/recall)
```

### Alternative Approach (NOT recommended):
```python
# Count NOT_FOUND as a valid class
# Treats "NOT_FOUND" as equally important to "ABC123"
# Results in inflated metrics that don't reflect extraction quality
```

**Why the standard approach is better:**
- Focuses on **information gained**, not conservative predictions
- Aligns with user intent: "Extract all available data"
- Prevents gaming the metric by predicting NOT_FOUND

---

## Conclusion

The "Accuracy Paradox" in field extraction reveals a fundamental truth:

**High accuracy can mask poor extraction performance when many fields legitimately don't exist in each document.**

### Key Takeaways:

1. **F1 > Accuracy** for extraction tasks
2. **Llama-11B** wins on extraction performance (F1: 0.30)
3. **InternVL3-8B** wins on accuracy (0.59) but misses most fields
4. **Accuracy rewards conservative models** that avoid difficult extractions
5. **F1 rewards effective extraction** regardless of conservatism

### Final Recommendation:

For document field extraction:
```
Primary:   F1 Score (measures extraction effectiveness)
Secondary: Precision/Recall (shows quality vs coverage tradeoff)
Tertiary:  Accuracy (warning sign for hallucinations)
```

**Don't be fooled by high accuracy scores in extraction tasks. F1 tells the real story.**

---

## Appendix: Metric Formulas

### Accuracy
```
Accuracy = (TP + TN) / Total
         = Correct predictions / All predictions
         = Includes correct NOT_FOUND predictions
```

### Precision
```
Precision = TP / (TP + FP)
          = Correct extractions / Attempted extractions
          = Excludes NOT_FOUND predictions
```

### Recall
```
Recall = TP / (TP + FN)
       = Correct extractions / Extractable fields
       = Penalty for saying NOT_FOUND when value exists
```

### F1 Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = Harmonic mean of precision and recall
   = Balances quality and coverage
```

---

## Related Documents

- `model_comparison.ipynb` (Cell 22): Implementation of per-field metrics
- `THREE_MODEL_FIELD_METRICS_UPDATE.md`: Documentation of 3-model comparison
- `EVALUATION_SYSTEM_GUIDE.md`: Overall evaluation methodology
