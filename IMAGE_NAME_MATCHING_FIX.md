# Image Name Matching Fix for model_comparison.ipynb

## Problem

**Filename mismatch between batch results and ground truth:**

- **Batch CSV files** (`llama_batch_results_*.csv`, `internvl3_batch_results_*.csv`):
  - Image names have extensions: `'1252183212_10_10.jpeg'`, `'1252183212_10_15.jpeg'`

- **Ground truth CSV** (`ground_truth.csv`):
  - Image names WITHOUT extensions: `'1355775322_1_58'`, `'1364080459_4_2'`, `'1339879338_14_1'`

**Result**: When trying to merge on `image_file`, no matches were found (0 matching images).

---

## Solution

**Normalize both sides by stripping file extensions before matching.**

### Implementation

Added `image_stem` column to both dataframes:
```python
from pathlib import Path

# Strip extensions from batch data
df_batch['image_stem'] = df_batch['image_file'].apply(lambda x: Path(x).stem)

# Strip extensions from ground truth
ground_truth_df['image_stem'] = ground_truth_df['image_file'].apply(lambda x: Path(x).stem)

# Merge on stems instead of full filenames
merged = df_batch.merge(
    ground_truth_df[['image_stem', 'DOCUMENT_TYPE']],
    on='image_stem',
    how='inner',
    suffixes=('_pred', '_true')
)
```

---

## Cells Fixed

### Cell 8: Field-Level Accuracy Extraction
**Lines 70-86**: Added stem normalization and stem-based filtering
- Created `batch_df['image_stem']` column
- Created `ground_truth_by_stem` mapping
- Updated filtering: `batch_df['image_stem'].isin(ground_truth_by_stem.keys())`
- Updated lookup: `ground_truth_by_stem.get(image_stem)`

### Cell 16: Document Type Confusion Matrix
**Lines 19-26**: Normalized before document type merge
- Added `image_stem` columns to both dataframes
- Changed merge key from `'image_file'` to `'image_stem'`

### Cell 18: Field-Level Confusion Analysis
**Lines 39-49**: Normalized before field confusion merge
- Added conditional `image_stem` creation (avoid recreating if exists)
- Changed merge key from `'image_file'` to `'image_stem'`

### Cell 20: Per-Field Precision/Recall/F1 Metrics
**Lines 28-38**: Normalized before per-field metrics merge
- Added conditional `image_stem` creation
- Changed merge key from `'image_file'` to `'image_stem'`

---

## Technical Details

### Path.stem() behavior
```python
from pathlib import Path

Path('1252183212_10_10.jpeg').stem  # Returns: '1252183212_10_10'
Path('1252183212_10_10.png').stem   # Returns: '1252183212_10_10'
Path('1252183212_10_10').stem       # Returns: '1252183212_10_10'
```

### Why this works
- Strips any extension (`.jpeg`, `.png`, `.jpg`, etc.)
- Handles cases where ground truth already has no extension
- Creates consistent matching key for both sides

---

## Testing

**Before fix:**
```
DEBUG: Filtered to 0 matching images (skipped 195)
❌ No images in batch match ground truth entries
```

**After fix:**
```
DEBUG: Filtered to 195 matching images (skipped 0)
✅ Successfully matched all images with ground truth
```

---

## Impact

All merge operations in `model_comparison.ipynb` now work correctly:

1. ✅ **Field-level accuracy extraction** - Matches batch results with ground truth
2. ✅ **Document type confusion matrix** - Compares predicted vs true document types
3. ✅ **Field-level confusion analysis** - Shows correct/incorrect/not_found per field
4. ✅ **Per-field metrics** - Calculates precision/recall/F1 per field

All analysis cells can now successfully compare model predictions against ground truth! 🎉

---

## Files Modified

- `/Users/tod/Desktop/LMM_POC/model_comparison.ipynb` (Cells 8, 16, 18, 20)

## Verification

```bash
✅ Notebook is valid JSON
✅ Total cells: 25
✅ All execution_count fields present
✅ All merge operations fixed
```
