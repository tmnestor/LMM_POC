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

## 5. **Cross-List Correlation Validation** ⭐ FOR TRANSACTION ALIGNMENT

### Description
Validates that **related lists maintain semantic alignment** across fields. Critical for transaction data where dates, descriptions, and amounts must correspond to the same transaction at the same index position.

### The Problem
Standard F1 scoring evaluates each list independently, missing semantic misalignment:

```python
# Ground Truth - semantically aligned
TRANSACTION_DATES:        ["01/01/2025", "02/01/2025", "03/01/2025"]
LINE_ITEM_DESCRIPTIONS:   ["Coffee",     "Lunch",      "Dinner"]
TRANSACTION_AMOUNTS_PAID: ["$5",         "$15",        "$25"]
# Transaction 1: Coffee on 01/01 cost $5 ✓

# Extracted - individually correct but semantically WRONG
TRANSACTION_DATES:        ["01/01/2025", "02/01/2025", "03/01/2025"]  # F1=100%
LINE_ITEM_DESCRIPTIONS:   ["Dinner",     "Coffee",     "Lunch"]       # F1=100% (all items present!)
TRANSACTION_AMOUNTS_PAID: ["$5",         "$15",        "$25"]         # F1=100%
# Transaction 1: Dinner on 01/01 cost $5 ✗ WRONG!

Overall F1: 100% but semantically incorrect!
```

### Solution: Correlation-Aware F1

Combines **within-list F1** with **cross-list alignment penalty**.

#### Formula
```python
For each transaction row i:
    dates_match = TRANSACTION_DATES[i] matches ground_truth_dates[i]
    desc_match = LINE_ITEM_DESCRIPTIONS[i] matches ground_truth_descriptions[i]
    amt_match = TRANSACTION_AMOUNTS_PAID[i] matches ground_truth_amounts[i]

    # Row is correct only if ALL fields align
    row_correct = dates_match AND desc_match AND amt_match

Alignment_Score = correct_rows / total_rows
Combined_F1 = (standard_F1 + Alignment_Score) / 2
```

### Implementation Approaches

#### Option A: Strict Alignment (Recommended for Invoices/Statements)
All related fields must align at the same index.

```python
def calculate_correlation_aware_f1(extracted_data, ground_truth_data,
                                    related_field_groups, debug=False):
    """
    Calculate F1 with cross-list correlation validation.

    Args:
        extracted_data: Dict with extracted fields
        ground_truth_data: Dict with ground truth fields
        related_field_groups: List of field tuples that must align
            Example: [("TRANSACTION_DATES", "LINE_ITEM_DESCRIPTIONS", "TRANSACTION_AMOUNTS_PAID")]

    Returns:
        dict with standard_f1, alignment_score, combined_f1
    """
    results = {}

    # Calculate standard F1 for each field
    field_f1_scores = {}
    for field in extracted_data.keys():
        f1_metrics = calculate_field_accuracy_f1(
            extracted_data[field],
            ground_truth_data[field],
            field
        )
        field_f1_scores[field] = f1_metrics["f1_score"]

    # Calculate alignment scores for related field groups
    alignment_scores = []

    for field_group in related_field_groups:
        # Parse all fields in the group into lists
        extracted_lists = {}
        ground_truth_lists = {}

        for field in field_group:
            extracted_lists[field] = [
                i.strip() for i in str(extracted_data[field]).split('|') if i.strip()
            ]
            ground_truth_lists[field] = [
                i.strip() for i in str(ground_truth_data[field]).split('|') if i.strip()
            ]

        # Check alignment row-by-row
        min_len = min(len(lst) for lst in ground_truth_lists.values())
        aligned_rows = 0

        for i in range(min_len):
            # Check if all fields match at position i
            row_aligned = True
            for field in field_group:
                if i < len(extracted_lists[field]):
                    # Use field-specific matching
                    if not _transaction_item_matches(
                        extracted_lists[field][i],
                        ground_truth_lists[field][i],
                        field
                    ):
                        row_aligned = False
                        break
                else:
                    row_aligned = False
                    break

            if row_aligned:
                aligned_rows += 1

        # Alignment score for this group
        alignment_score = aligned_rows / min_len if min_len > 0 else 0.0
        alignment_scores.append(alignment_score)

        if debug:
            print(f"  Field Group {field_group}:")
            print(f"    Aligned rows: {aligned_rows}/{min_len}")
            print(f"    Alignment score: {alignment_score:.1%}")

    # Overall alignment score
    overall_alignment = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.0

    # Overall standard F1
    overall_f1 = sum(field_f1_scores.values()) / len(field_f1_scores) if field_f1_scores else 0.0

    # Combined score (weighted average)
    combined_f1 = (overall_f1 + overall_alignment) / 2

    return {
        "standard_f1": overall_f1,
        "alignment_score": overall_alignment,
        "combined_f1": combined_f1,
        "field_f1_scores": field_f1_scores,
        "alignment_details": alignment_scores
    }
```

#### Option B: Partial Credit Alignment
Gives partial credit for partially aligned rows.

```python
def calculate_partial_alignment_score(extracted_lists, ground_truth_lists, field_group):
    """
    Calculate alignment with partial credit.
    Row gets score = (matching_fields / total_fields) for that row.
    """
    min_len = min(len(lst) for lst in ground_truth_lists.values())
    total_alignment = 0.0

    for i in range(min_len):
        matching_fields = 0

        for field in field_group:
            if i < len(extracted_lists[field]):
                if _transaction_item_matches(
                    extracted_lists[field][i],
                    ground_truth_lists[field][i],
                    field
                ):
                    matching_fields += 1

        # Partial credit for this row
        row_score = matching_fields / len(field_group)
        total_alignment += row_score

    return total_alignment / min_len if min_len > 0 else 0.0
```

### Example Calculation

```python
# Ground Truth
TRANSACTION_DATES:        ["01/01/2025", "02/01/2025", "03/01/2025"]
LINE_ITEM_DESCRIPTIONS:   ["Coffee",     "Lunch",      "Dinner"]
TRANSACTION_AMOUNTS_PAID: ["$5",         "$15",        "$25"]

# Extracted (descriptions reversed)
TRANSACTION_DATES:        ["01/01/2025", "02/01/2025", "03/01/2025"]
LINE_ITEM_DESCRIPTIONS:   ["Dinner",     "Coffee",     "Lunch"]
TRANSACTION_AMOUNTS_PAID: ["$5",         "$15",        "$25"]

# Standard F1 for each field
TRANSACTION_DATES:        F1 = 100% (all dates correct)
LINE_ITEM_DESCRIPTIONS:   F1 = 100% (all items present)
TRANSACTION_AMOUNTS_PAID: F1 = 100% (all amounts present)
Overall Standard F1:      100%

# Alignment Check (row-by-row)
Row 0: Date✓ + Desc✗ (Dinner≠Coffee) + Amt✓ → 2/3 fields aligned → 0% (strict) or 67% (partial)
Row 1: Date✓ + Desc✗ (Coffee≠Lunch) + Amt✓ → 2/3 fields aligned → 0% (strict) or 67% (partial)
Row 2: Date✓ + Desc✗ (Lunch≠Dinner) + Amt✓ → 2/3 fields aligned → 0% (strict) or 67% (partial)

Alignment Score (strict):  0/3 = 0%
Alignment Score (partial): (67%+67%+67%)/3 = 67%

# Combined Scores
Strict Combined F1:  (100% + 0%) / 2 = 50%  ← Properly penalized!
Partial Combined F1: (100% + 67%) / 2 = 83%
```

### When to Use

✅ **Use for:**
- Transaction lists (dates, descriptions, amounts must align)
- Invoice line items (description, quantity, price, total)
- Sequential data where order and alignment matter
- Multi-field entities (name + address + phone must correspond)

❌ **Don't use for:**
- Independent field lists (no semantic relationship)
- Unordered sets
- Single-value fields

### Advantages
✅ Detects semantic misalignment that standard F1 misses
✅ Ensures extracted data is **usable** (not just present)
✅ Critical for downstream applications (accounting, RPA)
✅ Reflects real-world data quality requirements

### Disadvantages
❌ More complex to implement
❌ Requires knowing which fields are related
❌ May be overly strict (strict mode)
❌ Can double-penalize errors (field F1 + alignment penalty)

### Integration with Batch Processor

```python
# In batch_processor.py - add correlation validation
def evaluate_with_correlation(extracted_data, ground_truth_data):
    """Evaluate with cross-list correlation checking."""

    # Define related field groups for each document type
    doc_type = extracted_data.get("DOCUMENT_TYPE", "").lower()

    if "bank" in doc_type or "statement" in doc_type:
        related_groups = [
            ("TRANSACTION_DATES", "LINE_ITEM_DESCRIPTIONS", "TRANSACTION_AMOUNTS_PAID")
        ]
    elif "invoice" in doc_type or "receipt" in doc_type:
        related_groups = [
            ("LINE_ITEM_DESCRIPTIONS", "LINE_ITEM_QUANTITIES",
             "LINE_ITEM_PRICES", "LINE_ITEM_TOTAL_PRICES")
        ]
    else:
        related_groups = []

    # Calculate correlation-aware F1
    if related_groups:
        metrics = calculate_correlation_aware_f1(
            extracted_data,
            ground_truth_data,
            related_groups,
            debug=True
        )

        return metrics
    else:
        # Fall back to standard F1 if no related groups
        return calculate_standard_f1(extracted_data, ground_truth_data)
```

### Display Output

```python
# Enhanced display with alignment metrics
rprint("[bold cyan]Correlation Analysis:[/bold cyan]")
rprint(f"  Standard F1:      {metrics['standard_f1']:.1%}")
rprint(f"  Alignment Score:  {metrics['alignment_score']:.1%}")
rprint(f"  [bold]Combined F1:      {metrics['combined_f1']:.1%}[/bold]")

if metrics['alignment_score'] < 0.8:
    rprint("[yellow]  ⚠️ Warning: Low alignment - check field ordering![/yellow]")
```

---

## 6. **Weighted Error Penalty Score**

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

| Metric | Penalizes FP? | Penalizes FN? | Order-Aware? | Cross-List Correlation? | Fuzzy Match? | Complexity | Best For |
|--------|---------------|---------------|--------------|------------------------|--------------|------------|----------|
| **Current (matches/gt_len)** | ❌ No | ✅ Yes | ❌ No | ❌ No | ❌ No | Low | Simple recall |
| **F1 Score** | ✅ Yes | ✅ Yes | ❌ No | ❌ No | ❌ No | Low | **General use** ⭐ |
| **ANLS** | ❌ No | ✅ Yes | ❌ No | ❌ No | ✅ Yes | Medium | OCR/typo tolerance |
| **KIEval** | ✅ Yes | ✅ Yes | ❌ No | ❌ No | ❌ No | Medium | Production RPA |
| **Order-Aware F1** | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No | ❌ No | Medium | Sequential data |
| **Cross-List Correlation** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No | High | **Transaction alignment** ⭐ |
| **Weighted Penalty** | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No | ❌ No | High | Custom business needs |

---

## Recommendation for Your Bank Statement Project

### Primary Metric: **F1 Score** ⭐

**Why:**
1. ✅ **Addresses your concern**: Properly penalizes false positives (credits extracted as debits)
2. ✅ **Standard in field**: All 2024 information extraction papers use it
3. ✅ **Easy to explain**: "Balance between accuracy and completeness"
4. ✅ **Interpretable**: Precision shows extraction accuracy, Recall shows completeness
5. ✅ **Well-tested**: Proven in NER, entity extraction, document processing

### Advanced Metric: **Cross-List Correlation F1** ⭐⭐ (For Production)

**Why Add This:**
1. ✅ **Semantic correctness**: Ensures dates/descriptions/amounts align properly
2. ✅ **Catches order errors**: Detects when lists are individually correct but misaligned
3. ✅ **Production-critical**: Prevents unusable extractions (Coffee costing $25)
4. ✅ **Downstream validation**: Ensures data can be used in accounting/RPA systems
5. ✅ **Real-world quality**: Reflects actual data usability, not just presence

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
