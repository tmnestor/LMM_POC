# Alternative Evaluation Metrics for List Extraction

## Executive Summary

The current **positional matching with ground truth denominator** (matches/gt_length) is indeed too generous because it:
- Does not penalize **false positives** (extra items extracted)
- Does not penalize **ordering errors** (correct items in wrong positions)
- Rewards partial matches without considering extraction quality

This document presents **5 alternative metrics** from recent research (2024-2025) that address these limitations.

---

## Problem with Current Metric

### Current Implementation
```python
# From evaluation_metrics.py lines 963-980
score = matches / len(ground_truth_items) if ground_truth_items else 0.0
```

### Issues
1. **No False Positive Penalty**: Extracting 3 correct + 10 wrong items scores same as 3 correct + 0 wrong (both = 60% if gt=5)
2. **No Order Awareness**: [A, B, C] scores same as [C, A, B] (both = 100%)
3. **Overly Generous**: User observation that scores don't reflect extraction quality

### Example Showing Problem
```
Ground Truth: ["EFTPOS $50", "ATM $100", "Purchase $75"]  (3 items)
Extracted:    ["EFTPOS $50", "ATM $100", "SALARY $2000", "INTEREST $10", "Purchase $75"]  (5 items)

Current Score: 3/3 = 100% ✓ All ground truth found!
Reality:       2 false positives (SALARY, INTEREST) should reduce score
```

---

## Recommended Metrics (Ranked by Suitability)

## 1. **F1 Score with Precision & Recall** ⭐ RECOMMENDED

### Why Best for Your Use Case
- **Balances completeness (recall) with accuracy (precision)**
- Penalizes both false positives AND false negatives
- Standard metric in information extraction (2024 research)
- Easy to interpret and explain in presentations

### Formula
```
Precision = TP / (TP + FP)  # Accuracy of extractions
Recall    = TP / (TP + FN)  # Completeness of extraction
F1 Score  = 2 * (Precision * Recall) / (Precision + Recall)
```

Where:
- **TP (True Positives)**: Correctly extracted items
- **FP (False Positives)**: Incorrectly extracted items (e.g., credits extracted as debits)
- **FN (False Negatives)**: Missing items from ground truth

### Example Calculation
```
Ground Truth: ["EFTPOS $50", "ATM $100", "Purchase $75"]  (3 items)
Extracted:    ["EFTPOS $50", "ATM $100", "SALARY $2000", "INTEREST $10", "Purchase $75"]  (5 items)

TP = 3 (all ground truth items found)
FP = 2 (SALARY, INTEREST should not be extracted)
FN = 0 (no missing items)

Precision = 3/(3+2) = 3/5 = 60%
Recall    = 3/(3+0) = 3/3 = 100%
F1        = 2*(0.6*1.0)/(0.6+1.0) = 1.2/1.6 = 75%

Current Metric: 100% (too generous!)
F1 Score:       75%  (properly penalizes false positives)
```

### Implementation
```python
def calculate_f1_score(extracted_items, ground_truth_items, field_name):
    """
    Calculate F1 score for list extraction.

    Args:
        extracted_items: List of extracted values
        ground_truth_items: List of ground truth values
        field_name: Field name for context-aware matching

    Returns:
        dict with precision, recall, f1_score
    """
    # True Positives: Items in both extracted and ground truth
    tp = 0
    for ext_item in extracted_items:
        if any(_transaction_item_matches(ext_item, gt_item, field_name)
               for gt_item in ground_truth_items):
            tp += 1

    # False Positives: Items in extracted but not in ground truth
    fp = len(extracted_items) - tp

    # False Negatives: Items in ground truth but not in extracted
    fn = 0
    for gt_item in ground_truth_items:
        if not any(_transaction_item_matches(ext_item, gt_item, field_name)
                   for ext_item in extracted_items):
            fn += 1

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }
```

### Advantages
✅ Penalizes false positives (over-extraction)
✅ Penalizes false negatives (under-extraction)
✅ Standard metric in NER and information extraction
✅ Well-understood in ML community
✅ Easy to explain: "Balance between accuracy and completeness"

### Disadvantages
❌ Does not consider ordering (position-agnostic)
❌ All errors weighted equally (no cost differentiation)

---

## 2. **ANLS (Average Normalized Levenshtein Similarity)** ⭐ GOOD FOR FUZZY MATCHING

### Description
- Developed for Document Visual Question Answering (DocVQA)
- Uses edit distance to measure similarity between strings
- Standard practice in document extraction tasks (2024)
- **Threshold at 0.5**: Scores below 50% similarity = 0.0

### Formula
```
For each item pair (extracted, ground_truth):
    edit_distance = levenshtein(extracted, ground_truth)
    max_length = max(len(extracted), len(ground_truth))
    similarity = 1.0 - (edit_distance / max_length)

    if similarity >= 0.5:
        score = similarity
    else:
        score = 0.0

ANLS = average(scores across all ground_truth items)
```

### Example
```
Ground Truth: "EFTPOS WITHDRAWAL WOOLWORTHS $50.00"
Extracted:    "EFTPOS WITHDRAWAL WOOLWORTH $50.00"  # Missing 'S'

Edit distance: 1 character
Max length:    38
Similarity:    1.0 - (1/38) = 0.974 > 0.5 ✓
Score:         97.4%
```

### Why Useful
- Handles OCR errors and minor typos gracefully
- Used by LayoutLM, Donut, and other document AI models
- Prevents over-penalization for spelling variations

### When NOT to Use
- When exact matches are required (e.g., dollar amounts)
- When order matters
- For list-level evaluation (designed for single strings)

### Implementation
```python
from Levenshtein import distance as levenshtein_distance

def calculate_anls(extracted_items, ground_truth_items, threshold=0.5):
    """Calculate Average Normalized Levenshtein Similarity."""
    if not ground_truth_items:
        return 0.0

    total_score = 0.0
    for gt_item in ground_truth_items:
        # Find best match in extracted items
        best_score = 0.0
        for ext_item in extracted_items:
            edit_dist = levenshtein_distance(str(ext_item), str(gt_item))
            max_len = max(len(str(ext_item)), len(str(gt_item)))
            similarity = 1.0 - (edit_dist / max_len) if max_len > 0 else 0.0

            # Apply threshold
            if similarity >= threshold:
                best_score = max(best_score, similarity)

        total_score += best_score

    return total_score / len(ground_truth_items)
```

**Note**: Requires `python-Levenshtein` package: `pip install python-Levenshtein`

---

## 3. **KIEval (Key Information Extraction Evaluation)** ⭐ INDUSTRY-FOCUSED

### Description
- **Latest research (2025)** from arXiv
- Designed for **production RPA applications**
- Focuses on **correction costs** rather than accuracy
- Entity-level AND group-level evaluation

### Conceptual Difference
| Traditional | KIEval |
|-------------|--------|
| "How accurate is extraction?" | "How much effort to fix extraction?" |
| Precision/Recall | Substitution/Addition/Deletion costs |
| Model-centric | Application-centric |

### Formula
```
Substitution = min(FP, FN)  # Items requiring value edits
Addition     = FN - Substitution  # Missing items to add
Deletion     = FP - Substitution  # Extra items to remove

Total_Error  = Substitution + Addition + Deletion
Score        = 1.0 - (Total_Error / Total_Items)
```

### Example
```
Ground Truth: ["EFTPOS $50", "ATM $100", "Purchase $75"]  (3 items)
Extracted:    ["EFTPOS $50", "ATM $100", "SALARY $2000", "Purchase $75"]  (4 items)

TP = 3, FP = 1, FN = 0

Substitution = min(1, 0) = 0
Addition     = 0 - 0 = 0
Deletion     = 1 - 0 = 1  # Remove SALARY

Total_Error  = 0 + 0 + 1 = 1 correction needed
Total_Items  = max(3, 4) = 4
Score        = 1.0 - (1/4) = 75%
```

### Why Useful for Your Case
- Aligns with **business value**: "How much manual correction is needed?"
- Differentiates error types (missing vs extra vs wrong)
- **Exact match required** (no partial credit for "close enough")
- Designed for structured document extraction

### Limitations
- Requires exact matches (no fuzzy matching)
- More complex to explain than F1
- Does not consider ordering

### Implementation
```python
def calculate_kieval(extracted_items, ground_truth_items, field_name):
    """
    Calculate KIEval score based on correction costs.

    Returns:
        dict with substitution, addition, deletion, total_error, score
    """
    # Count matches
    tp = sum(1 for ext in extracted_items
             if any(_transaction_item_matches(ext, gt, field_name)
                    for gt in ground_truth_items))

    fp = len(extracted_items) - tp  # Extra items
    fn = len(ground_truth_items) - tp  # Missing items

    # Calculate correction operations
    substitution = min(fp, fn)  # Items to edit
    addition = fn - substitution  # Items to add
    deletion = fp - substitution  # Items to delete

    total_error = substitution + addition + deletion
    total_items = max(len(extracted_items), len(ground_truth_items))

    score = 1.0 - (total_error / total_items) if total_items > 0 else 0.0

    return {
        "score": score,
        "substitution": substitution,
        "addition": addition,
        "deletion": deletion,
        "total_error": total_error,
        "total_items": total_items
    }
```

---

## 4. **Order-Aware F1 (Positional Matching)**

### Description
Extension of F1 that considers item position in lists. Useful for transaction lists where **chronological order matters**.

### Formula
```
For each position i:
    if extracted[i] matches ground_truth[i]:
        positional_match[i] = 1
    else:
        positional_match[i] = 0

Positional_Precision = sum(positional_match) / len(extracted)
Positional_Recall    = sum(positional_match) / len(ground_truth)
Positional_F1        = 2 * (P * R) / (P + R)
```

### Example
```
Ground Truth: ["EFTPOS $50", "ATM $100", "Purchase $75"]
Extracted:    ["ATM $100", "EFTPOS $50", "Purchase $75"]  # First two swapped

Position 0: "ATM $100" ≠ "EFTPOS $50" → 0
Position 1: "EFTPOS $50" ≠ "ATM $100" → 0
Position 2: "Purchase $75" = "Purchase $75" → 1

Positional matches: 1/3 = 33.3%
Standard F1:        100% (all items present)
Order-Aware F1:     50% (penalizes ordering errors)
```

### When to Use
- Transaction lists where date order matters
- Sequential data (invoice line items top-to-bottom)
- When chronological accuracy is critical

### When NOT to Use
- Unordered sets (e.g., list of product categories)
- When order is arbitrary

### Implementation
```python
def calculate_order_aware_f1(extracted_items, ground_truth_items, field_name):
    """
    Calculate F1 score with positional matching.
    """
    if not ground_truth_items:
        return {"f1_score": 0.0, "positional_matches": 0}

    # Positional matches
    min_len = min(len(extracted_items), len(ground_truth_items))
    positional_matches = 0

    for i in range(min_len):
        if _transaction_item_matches(extracted_items[i], ground_truth_items[i], field_name):
            positional_matches += 1

    # Precision: correct positions / extracted count
    precision = positional_matches / len(extracted_items) if extracted_items else 0.0

    # Recall: correct positions / ground truth count
    recall = positional_matches / len(ground_truth_items) if ground_truth_items else 0.0

    # F1
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "positional_matches": positional_matches,
        "total_positions": len(ground_truth_items)
    }
```

---

## 5. **Weighted Error Penalty Score**

### Description
Custom metric that assigns **different costs** to different error types based on business impact.

### Concept
```
Not all errors are equal:
- Missing a $10,000 transaction is worse than missing a $10 fee
- False positive (credit as debit) is worse than false negative (missed debit)
- Early-position errors more visible than late-position errors
```

### Formula
```
For each ground truth item:
    if found in extracted:
        score += 1.0
    else:
        score += 0.0  # Missing item penalty

For each extracted item not in ground truth:
    score -= false_positive_penalty  # Default 0.5

For each position mismatch:
    score -= position_penalty * (1 / (position + 1))  # Earlier = worse

Final_Score = max(0, score / len(ground_truth))
```

### Example with Custom Weights
```python
def calculate_weighted_score(extracted_items, ground_truth_items, field_name,
                             fp_penalty=0.5, position_penalty=0.1):
    """
    Calculate score with weighted error penalties.

    Args:
        fp_penalty: Penalty for each false positive (0-1)
        position_penalty: Penalty for position errors (0-1)
    """
    if not ground_truth_items:
        return 0.0

    score = 0.0

    # Check each ground truth item
    for i, gt_item in enumerate(ground_truth_items):
        found = False
        correct_position = False

        for j, ext_item in enumerate(extracted_items):
            if _transaction_item_matches(ext_item, gt_item, field_name):
                found = True
                if i == j:
                    correct_position = True
                    score += 1.0  # Full credit
                else:
                    # Positional penalty (earlier positions more important)
                    pos_weight = 1.0 / (i + 1)
                    score += 1.0 - (position_penalty * pos_weight)
                break

        if not found:
            score += 0.0  # Missing item = 0 points

    # Penalize false positives
    fp_count = 0
    for ext_item in extracted_items:
        if not any(_transaction_item_matches(ext_item, gt_item, field_name)
                   for gt_item in ground_truth_items):
            fp_count += 1

    score -= (fp_count * fp_penalty)

    # Normalize
    final_score = max(0.0, score / len(ground_truth_items))

    return final_score
```

### Advantages
✅ Highly customizable to business needs
✅ Can model real-world correction costs
✅ Captures both completeness and accuracy
✅ Can weight different error types differently

### Disadvantages
❌ Requires tuning penalty parameters
❌ Less standardized (harder to compare across papers)
❌ More complex to explain

---

## Metric Comparison Table

| Metric | Penalizes FP? | Penalizes FN? | Order-Aware? | Fuzzy Match? | Complexity | Best For |
|--------|---------------|---------------|--------------|--------------|------------|----------|
| **Current (matches/gt_len)** | ❌ No | ✅ Yes | ❌ No | ❌ No | Low | Simple recall |
| **F1 Score** | ✅ Yes | ✅ Yes | ❌ No | ❌ No | Low | **General use** ⭐ |
| **ANLS** | ❌ No | ✅ Yes | ❌ No | ✅ Yes | Medium | OCR/typo tolerance |
| **KIEval** | ✅ Yes | ✅ Yes | ❌ No | ❌ No | Medium | Production RPA |
| **Order-Aware F1** | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No | Medium | Sequential data |
| **Weighted Penalty** | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No | High | Custom business needs |

---

## Recommendation for Your Bank Statement Project

### Primary Metric: **F1 Score** ⭐

**Why:**
1. ✅ **Addresses your concern**: Properly penalizes false positives (credits extracted as debits)
2. ✅ **Standard in field**: All 2024 information extraction papers use it
3. ✅ **Easy to explain**: "Balance between accuracy and completeness"
4. ✅ **Interpretable**: Precision shows extraction accuracy, Recall shows completeness
5. ✅ **Well-tested**: Proven in NER, entity extraction, document processing

**Example:**
```
Current (too generous):
- 3 correct + 2 wrong = 3/3 = 100%

F1 Score (proper):
- Precision = 3/(3+2) = 60%
- Recall = 3/3 = 100%
- F1 = 75%
```

### Secondary Metric: **ANLS** (for robustness analysis)

**Why:**
- Handles minor OCR/extraction variations
- Tests if score improvements are due to better understanding vs better character recognition
- Standard in DocVQA evaluation

### Implementation Plan

1. **Replace current scoring** in `evaluation_metrics.py` lines 963-980
2. **Add F1 calculation** using implementation above
3. **Update display** in `batch_processor.py` to show:
   - Precision (extraction accuracy)
   - Recall (completeness)
   - F1 (overall score)
   - TP/FP/FN counts for transparency
4. **Re-run evaluations** on same test set to compare:
   - Current metric scores
   - New F1 scores
   - Expect F1 to be 15-30% lower (more realistic)

### Expected Impact

```
Image 008 (current example):
TRANSACTION_DATES:
- Ground Truth: 6 items
- Extracted: 8 items (includes 2 credits)
- Current Score: 100% (6/6 found)
- Expected F1: ~75% (penalizes 2 false positives)

Overall Accuracy:
- Current: 40-60% (too generous)
- Expected F1: 30-50% (more realistic)
- After prompt fixes: 70-90% (realistic target)
```

---

## Code Integration Example

### New Function in `evaluation_metrics.py`

```python
def calculate_field_accuracy_f1(extracted_value, ground_truth_value, field_name, debug=False):
    """
    Calculate F1-based accuracy for a field with better false positive handling.

    Returns:
        dict with f1_score, precision, recall, tp, fp, fn
    """
    # Handle NOT_FOUND cases
    if ground_truth_value == "NOT_FOUND":
        return {
            "f1_score": 1.0 if extracted_value == "NOT_FOUND" else 0.0,
            "precision": 1.0 if extracted_value == "NOT_FOUND" else 0.0,
            "recall": 1.0,
            "tp": 0, "fp": 0, "fn": 0
        }

    # Handle single values
    if '|' not in str(extracted_value) and '|' not in str(ground_truth_value):
        match = _fuzzy_match(str(extracted_value), str(ground_truth_value), field_name)
        return {
            "f1_score": 1.0 if match else 0.0,
            "precision": 1.0 if match else 0.0,
            "recall": 1.0 if match else 0.0,
            "tp": 1 if match else 0,
            "fp": 0 if match else 1,
            "fn": 0 if match else 1
        }

    # Handle list values (transaction fields)
    extracted_items = [i.strip() for i in str(extracted_value).split('|') if i.strip()]
    ground_truth_items = [i.strip() for i in str(ground_truth_value).split('|') if i.strip()]

    # Calculate TP, FP, FN
    tp = 0
    matched_gt_indices = set()

    for ext_item in extracted_items:
        for i, gt_item in enumerate(ground_truth_items):
            if i not in matched_gt_indices and _transaction_item_matches(ext_item, gt_item, field_name):
                tp += 1
                matched_gt_indices.add(i)
                break

    fp = len(extracted_items) - tp
    fn = len(ground_truth_items) - tp

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    if debug:
        print(f"  F1 Metrics for {field_name}:")
        print(f"    TP={tp}, FP={fp}, FN={fn}")
        print(f"    Precision={precision:.2%}, Recall={recall:.2%}, F1={f1_score:.2%}")

    return {
        "f1_score": f1_score,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }
```

### Updated Display in `batch_processor.py`

```python
# Replace lines 696-732 with F1-aware display
if verbose:
    for field in filtered_ground_truth.keys():
        extracted_val = extracted_data.get(field, "NOT_FOUND")
        ground_val = filtered_ground_truth.get(field, "NOT_FOUND")

        # Get F1 metrics
        metrics = calculate_field_accuracy_f1(extracted_val, ground_val, field, debug=False)
        f1 = metrics["f1_score"]
        precision = metrics["precision"]
        recall = metrics["recall"]

        # Determine status
        if f1 == 1.0:
            status = "✅"
            color = "green"
        elif f1 >= 0.5:
            status = "≈"
            color = "yellow"
        else:
            status = "❌"
            color = "red"

        # Display with F1 breakdown
        rprint(f"[{color}]{status} {field}: F1={f1:.1%} (P={precision:.1%}, R={recall:.1%})[/{color}]")

        if status != "✅" and verbose:
            rprint(f"[yellow]     Extracted: {extracted_val}[/yellow]")
            rprint(f"[yellow]     Ground Truth: {ground_val}[/yellow]")
            rprint(f"[cyan]     TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}[/cyan]")
```

---

## References

1. **KIEval** (2025): "KIEval: Evaluation Metric for Document Key Information Extraction"
   https://arxiv.org/html/2503.05488
   - Application-centric metric for RPA
   - Correction cost framework

2. **ANLS*** (2024): "A Universal Document Processing Metric for Generative Large Language Models"
   https://arxiv.org/html/2402.03848v2
   - Extended ANLS for nested structures
   - 0.5 threshold for partial matching

3. **Information Extraction Evaluation** (2024): "Assessing the quality of information extraction"
   https://arxiv.org/html/2404.04068v1
   - Precision/Recall/F1 for entity extraction
   - Industry best practices

4. **NER Metrics** (Stack Overflow): "Computing precision and recall in Named Entity Recognition"
   https://stackoverflow.com/questions/1783653/computing-precision-and-recall-in-named-entity-recognition
   - Practical implementation guidance

5. **Order-Aware Metrics** (2024): "Reading Order Independent Metrics for Information Extraction"
   https://arxiv.org/abs/2404.18664
   - Order-aware vs order-agnostic evaluation

---

## Next Steps

1. **Implement F1 scoring** in `evaluation_metrics.py`
2. **Update batch_processor.py** display with P/R/F1 breakdown
3. **Re-run current test set** to establish new baseline
4. **Compare scores**:
   - Current (too generous): ~60-100%
   - F1 (realistic): ~30-70%
   - After prompt improvements: ~70-90%
5. **Update documentation** with F1 methodology
6. **Test on H200** with updated metrics

The F1 score will provide a **more honest assessment** of extraction quality while remaining **easy to explain and interpret** for presentations.
