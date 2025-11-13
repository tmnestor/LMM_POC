# Model Comparison Analysis Fix Summary

**Date**: 2025-11-14
**Fixed By**: Claude Code
**Notebook**: `model_comparison.ipynb`

## Problem Identified

The generated `MODEL_COMPARISON_ANALYSIS.md` report had multiple critical errors:

1. **All metrics showing 0.0000**: F1 Score, Precision, Recall all displayed as 0.0000
2. **Contradictory data**: Field-level columns showed 0.0% but "Best Score" showed different values (0.7-1.0%)
3. **Zero hallucination rates**: Showed 0% hallucination which was incorrect
4. **Wrong field count**: Should analyze only 17 fields (excluding TRANSACTION_AMOUNTS_RECEIVED and ACCOUNT_BALANCE)

## Root Cause Analysis

### Cell 31: Report Generation Function

**Problem**: The code attempted to access columns that don't exist in the batch results CSV files:

```python
# WRONG - These columns don't exist in the CSV files
llama_overall_f1 = llama_df['f1_score'].mean()
llama_overall_precision = llama_df['precision'].mean()
llama_overall_recall = llama_df['recall'].mean()
```

**What Actually Exists in CSV Files**:
```
image_file, document_type, processing_time, overall_accuracy,
fields_extracted, fields_matched, total_fields, ...
```

The batch CSVs contain:
- `overall_accuracy` ✅ (exists)
- `processing_time` ✅ (exists)
- `fields_extracted`, `fields_matched`, `total_fields` ✅ (exist)

But NOT:
- `f1_score` ❌ (doesn't exist)
- `precision` ❌ (doesn't exist)
- `recall` ❌ (doesn't exist)

### Cell 26: Hallucination Analysis

**Status**: ✅ Already correct - no changes needed

Cell 26 already had:
1. Correct variable names (llama_df, internvl3_quantized_df, internvl3_non_quantized_df)
2. Field exclusion logic (skips TRANSACTION_AMOUNTS_RECEIVED and ACCOUNT_BALANCE)

## Solution Implemented

### Cell 31: Complete Rewrite

**Key Changes**:

1. **New calculate_metrics() function** that properly computes F1/Precision/Recall from existing columns:

```python
def calculate_metrics(df, model_name):
    """Calculate precision, recall, F1 from batch results DataFrame."""

    # Use columns that actually exist
    accuracy = df['overall_accuracy'].mean()
    speed = df['processing_time'].median()

    # Calculate metrics from field matching data
    valid_rows = df[df['fields_extracted'] > 0].copy()
    valid_rows['precision'] = valid_rows['fields_matched'] / valid_rows['fields_extracted']
    valid_rows['recall'] = valid_rows['fields_matched'] / valid_rows['total_fields']

    precision = valid_rows['precision'].mean()
    recall = valid_rows['recall'].mean()

    # F1 = 2 * precision * recall / (precision + recall)
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return f1, precision, recall, accuracy, speed
```

2. **Fixed field_performance access** to use correct model column names:

```python
# OLD - Wrong column names
llama_acc = field_performance.loc[field, 'Llama-11B']  # Column doesn't exist
internvl_8b_acc = field_performance.loc[field, 'InternVL3-8B']  # Column doesn't exist

# NEW - Correct column names
llama_acc = fp.loc[field, 'Llama-3.2-Vision']  # Correct
internvl_8b_acc = fp.loc[field, 'InternVL3-Quantized-8B']  # Correct
internvl_2b_acc = fp.loc[field, 'InternVL3-NonQuantized-2B']  # Correct
```

3. **Fixed percentage conversion** - field_performance stores values as decimals (0-1), not percentages:

```python
# Convert to percentages
report += f"| {field} | {llama_acc*100:.1f}% | {internvl_8b_acc*100:.1f}% | ..."
```

4. **Fixed model specialization** to use correct column names from field_performance:

```python
llama_fields = (fp['best_model'] == 'Llama-3.2-Vision').sum()
internvl_8b_fields = (fp['best_model'] == 'InternVL3-Quantized-8B').sum()
internvl_2b_fields = (fp['best_model'] == 'InternVL3-NonQuantized-2B').sum()
```

5. **Added comprehensive debug output** to help diagnose future issues:

```python
rprint(f"[green]✅ {model_name} metrics calculated successfully[/green]")
rprint(f"[cyan]  F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}, Acc={accuracy:.2f}%, Speed={speed:.1f}s[/cyan]")
```

## Expected Results After Fix

After running the updated notebook on the remote server, the report should show:

1. **Non-zero metrics**: F1, Precision, Recall calculated from actual field matching data
2. **Consistent field performance**: Per-field accuracies matching best_model and best_score
3. **Correct field count**: Only 17 fields analyzed (excluding TRANSACTION_AMOUNTS_RECEIVED, ACCOUNT_BALANCE)
4. **Accurate hallucination rates**: Based on field-level comparison with ground truth
5. **Correct model specialization**: Showing which model performs best on which fields

## Metrics Calculation Formula

**From Batch Results CSVs**:

```
Precision = fields_matched / fields_extracted  (per document)
Recall = fields_matched / total_fields  (per document)
F1 = 2 * precision * recall / (precision + recall)
Accuracy = overall_accuracy column (already computed per document)
Speed = median(processing_time)
```

**Aggregation**:
- Precision/Recall/F1: Mean across all documents
- Speed: Median across all documents
- Accuracy: Mean of overall_accuracy column

## Verification Steps

After syncing to remote server and running the notebook:

1. Check that Cell 31 prints debug output showing calculated metrics
2. Verify MODEL_COMPARISON_ANALYSIS.md has non-zero F1/Precision/Recall values
3. Verify field-level accuracy table shows consistent percentages
4. Verify model specialization shows correct field counts (not all 0/17)
5. Verify hallucination analysis shows realistic rates (not 0%)

## Files Modified

- `model_comparison.ipynb` - Cell 31 (Cell index 30) completely rewritten

## Files Unchanged

- Cell 26 (hallucination analysis) - Already correct, no changes needed

## Related Issues Fixed

1. ❌ **Wrong column names**: Changed from 'Llama-11B' to 'Llama-3.2-Vision', etc.
2. ❌ **Missing percentage conversion**: Added multiplication by 100 for field_performance display
3. ❌ **Accessing non-existent columns**: Now calculates metrics instead of reading them
4. ❌ **Silent exception handling**: Added debug output to show what's actually happening

## Technical Details

### Why Original Code Failed

The original code assumed the batch results CSVs would contain pre-calculated `f1_score`, `precision`, and `recall` columns. However, the actual batch processors (llama_batch.ipynb, ivl3_8b_batch_quantized.ipynb, etc.) only output:

- Raw extraction results per document
- Per-field values (DOCUMENT_TYPE, TOTAL_AMOUNT, etc.)
- `fields_extracted`, `fields_matched`, `total_fields` counts
- `overall_accuracy` (percentage of fields matched)
- `processing_time`

The try/except blocks caught the KeyError exceptions when trying to access non-existent columns and silently defaulted all metrics to 0.

### Why New Code Works

The new code:

1. Only accesses columns that actually exist in the CSV files
2. Calculates F1/Precision/Recall from the field matching counts
3. Properly accesses field_performance DataFrame with correct column names
4. Converts field_performance decimal values (0-1) to percentages (0-100)
5. Provides debug output to verify calculations

## Next Steps

1. User will sync notebook to remote server
2. User will restart kernel
3. User will run all cells from top to bottom
4. Claude Code will verify the generated MODEL_COMPARISON_ANALYSIS.md has correct values

## Success Criteria

✅ All models show non-zero F1, Precision, Recall values
✅ Field-level accuracy table shows realistic percentages (not all 0.0%)
✅ Model specialization shows correct distribution of best-performing fields
✅ Hallucination rates show realistic percentages (not all 0%)
✅ Per-field metrics match between summary table and specialization section
