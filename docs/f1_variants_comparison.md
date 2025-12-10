# F1 Score Variants Comparison

This document explains the five F1 score variants used for evaluating field extraction accuracy.

## Overview

| Variant | Column | Description |
|---------|--------|-------------|
| **Smart** | `F1_Smart` | Automatically selects best method per field type (RECOMMENDED) |
| **Binary** | `F1_Binary` | Entire field value must match exactly (strictest) |
| **Position-Aware** | `F1_Aware` | Items must match at the same position, partial credit |
| **Position-Agnostic** | `F1_Agnostic` | Set-based matching, exact items, order doesn't matter |
| **Fuzzy Agnostic** | `F1_Fuzzy` | Set-based + token overlap for partial credit (most lenient) |

## F1_Smart: Field-Type-Specific Selection (RECOMMENDED)

F1_Smart automatically selects the most appropriate F1 method based on the field's data type:

| Use Case | Method | Fields |
|----------|--------|--------|
| Strict validation (e.g., ABN, amounts) | Binary | BUSINESS_ABN, GST_AMOUNT, TOTAL_AMOUNT, IS_GST_INCLUDED, DOCUMENT_TYPE |
| Ordered lists (e.g., line items matching rows) | Position-Aware | LINE_ITEM_QUANTITIES, LINE_ITEM_PRICES, LINE_ITEM_TOTAL_PRICES, TRANSACTION_AMOUNTS_PAID |
| Unordered extraction (e.g., transaction descriptions) | Position-Agnostic | LINE_ITEM_DESCRIPTIONS, TRANSACTION_DATES |
| Fuzzy text comparison (e.g., names, addresses) | Fuzzy | SUPPLIER_NAME, BUSINESS_ADDRESS, PAYER_NAME, PAYER_ADDRESS, INVOICE_DATE, STATEMENT_DATE_RANGE |

## Example Scenarios

### Scenario 1: Same items, different order

**Extracted:** `[A, B, C]` vs **Ground Truth:** `[A, C, B]`

| F1_Binary | F1_Aware | F1_Agnostic | F1_Fuzzy |
|-----------|----------|-------------|----------|
| 0.0 | 0.33 | 1.0 | 1.0 |

**Explanation:**
- **Binary:** Entire string differs → 0.0
- **Aware:** Only A matches at position 0 → 1/3 precision, 1/3 recall → F1 = 0.33
- **Agnostic:** All 3 items match (set-based) → 1.0
- **Fuzzy:** All 3 items match → 1.0

### Scenario 2: Partial extraction (missing items)

**Extracted:** `[A, B, C]` vs **Ground Truth:** `[A, B, C, D, E]`

| F1_Binary | F1_Aware | F1_Agnostic | F1_Fuzzy |
|-----------|----------|-------------|----------|
| 0.0 | 0.6 | 0.6 | 0.6 |

**Explanation:**
- **Binary:** String differs → 0.0
- **Aware:** 3/3 precision, 3/5 recall → F1 = 0.6
- **Agnostic:** 3/3 precision, 3/5 recall → F1 = 0.6
- **Fuzzy:** Same as agnostic → 0.6

### Scenario 3: Fuzzy matching advantage

**Extracted:** `[PIZZA HUT SUBURB]` vs **Ground Truth:** `[Pizza Hut]`

| F1_Binary | F1_Aware | F1_Agnostic | F1_Fuzzy |
|-----------|----------|-------------|----------|
| 0.0 | 0.0 | 0.0 | ~0.67 |

**Explanation:**
- **Binary/Aware/Agnostic:** Normalized strings don't match exactly → 0.0
- **Fuzzy:** Token overlap ("pizza", "hut") gives partial credit → ~0.67

## Formula Reference

For all variants:
```
F1 = 2 * precision * recall / (precision + recall)
```

Where:
- **Precision** = True Positives / Extracted Count
- **Recall** = True Positives / Ground Truth Count
