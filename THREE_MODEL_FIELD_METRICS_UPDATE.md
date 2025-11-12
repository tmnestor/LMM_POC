# Three-Model Field-Level Metrics Update

## Summary

Updated field-level confusion matrices and per-field metrics visualizations to disambiguate InternVL3 into 8B (quantized) and 2B (non-quantized) versions.

## Changes Made

### Cell 20: Field-Level Confusion Heatmap (160 lines)

**Added 3rd model support:**

```python
# OLD: Only 2 models
llama_confusion = create_field_confusion_data(llama_batch_df, ground_truth, 'Llama')
internvl_confusion = create_field_confusion_data(internvl_batch_df, ground_truth, 'InternVL3')
all_confusion = pd.concat([llama_confusion, internvl_confusion], ignore_index=True)

# NEW: All 3 models with clear naming
llama_confusion = create_field_confusion_data(llama_batch_df, ground_truth, 'Llama-11B')
internvl_confusion = create_field_confusion_data(internvl_batch_df, ground_truth, 'InternVL3-8B')
internvl_nq_confusion = create_field_confusion_data(internvl_nq_batch_df, ground_truth, 'InternVL3-2B')
all_confusion = pd.concat([llama_confusion, internvl_confusion, internvl_nq_confusion], ignore_index=True)
```

**Changed to 3-panel layout:**

```python
# OLD: 2 panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 12))
plot_confusion_heatmap(all_confusion, 'Llama', ax1)
plot_confusion_heatmap(all_confusion, 'InternVL3', ax2)

# NEW: 3 panels
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 12))
plot_confusion_heatmap(all_confusion, 'Llama-11B', ax1)
plot_confusion_heatmap(all_confusion, 'InternVL3-8B', ax2)
plot_confusion_heatmap(all_confusion, 'InternVL3-2B', ax3)
```

**Updated summary statistics loop:**

```python
# OLD: 2 models
for model in ['Llama', 'InternVL3']:
    ...

# NEW: 3 models
for model in ['Llama-11B', 'InternVL3-8B', 'InternVL3-2B']:
    ...
```

---

### Cell 22: Per-Field Metrics (174 lines)

**Added 3rd model metrics:**

```python
# OLD: Only 2 models
llama_metrics = compute_per_field_metrics(llama_batch_df, ground_truth, 'Llama')
internvl_metrics = compute_per_field_metrics(internvl_batch_df, ground_truth, 'InternVL3')
all_metrics = pd.concat([llama_metrics, internvl_metrics], ignore_index=True)

# NEW: All 3 models
llama_metrics = compute_per_field_metrics(llama_batch_df, ground_truth, 'Llama-11B')
internvl_metrics = compute_per_field_metrics(internvl_batch_df, ground_truth, 'InternVL3-8B')
internvl_nq_metrics = compute_per_field_metrics(internvl_nq_batch_df, ground_truth, 'InternVL3-2B')
all_metrics = pd.concat([llama_metrics, internvl_metrics, internvl_nq_metrics], ignore_index=True)
```

**Added 3rd color for charts:**

```python
# OLD: 2 colors
pivot_f1.plot(kind='barh', ax=ax1, color=['#3498db', '#e74c3c'])

# NEW: 3 colors (blue, red, green)
model_colors = ['#3498db', '#e74c3c', '#2ecc71']
pivot_f1.plot(kind='barh', ax=ax1, color=model_colors)
```

All 4 bar charts (F1, Precision, Recall, Accuracy) now use the 3-color scheme.

---

## Expected Output

### Cell 20: Field Extraction Status

**Console Output:**
```
Creating field-level confusion matrices...
✅ Field-level confusion heatmaps created (3 models)

Field Confusion Summary:

Llama-11B:
  Correct: X/Y (Z%)
  Incorrect: X/Y (Z%)
  Not Found: X/Y (Z%)

InternVL3-8B:
  Correct: X/Y (Z%)
  Incorrect: X/Y (Z%)
  Not Found: X/Y (Z%)

InternVL3-2B:
  Correct: X/Y (Z%)
  Incorrect: X/Y (Z%)
  Not Found: X/Y (Z%)
```

**Visual Output:**
- 3 heatmaps side-by-side (24" wide figure)
- Panel 1: "Llama-11B - Field Extraction Status"
- Panel 2: "InternVL3-8B - Field Extraction Status"
- Panel 3: "InternVL3-2B - Field Extraction Status"
- Each shows fields × (correct, incorrect, not_found) status

---

### Cell 22: Per-Field Metrics

**Console Output:**
```
Computing per-field precision/recall/F1 metrics...

Per-Field Metrics Comparison:
[DataFrame showing all fields × all 3 models]

✅ Metrics saved to: output/csv/per_field_metrics.csv
✅ Per-field metrics visualizations created (3 models)

Model Performance Summary:
                precision    recall  f1_score  accuracy
model
InternVL3-2B      X.XXXX    X.XXXX    X.XXXX    X.XXXX
InternVL3-8B      X.XXXX    X.XXXX    X.XXXX    X.XXXX
Llama-11B         X.XXXX    X.XXXX    X.XXXX    X.XXXX
```

**Visual Output:**
- 2×2 grid of bar charts
- All charts now show 3 bars per field (3 models)
- Color scheme:
  - Blue: Llama-11B
  - Red: InternVL3-8B
  - Green: InternVL3-2B

---

## Model Naming Convention

| Old Name | New Name | Description |
|----------|----------|-------------|
| Llama | Llama-11B | Llama-3.2-Vision-11B |
| InternVL3 | InternVL3-8B | InternVL3-8B (8-bit quantized) |
| (missing) | InternVL3-2B | InternVL3-2B (non-quantized) |

---

## Files Modified

- `model_comparison.ipynb`
  - Cell 20: 157 → 160 lines (added 3rd model)
  - Cell 22: 169 → 174 lines (added 3rd model + 3-color scheme)

---

## Verification

Run this verification script on the remote server:

```bash
python verify_notebook_version.py
```

Expected checks to pass:
- ✅ Cell 20: Has `internvl_nq_confusion`
- ✅ Cell 20: Has `subplots(1, 3` for 3-panel layout
- ✅ Cell 22: Has `internvl_nq_metrics`
- ✅ Cell 22: Has `model_colors` list
- ✅ All cells reference all 3 model names

---

## Benefits

1. **Clear Differentiation**: Users can now see which InternVL3 variant performed better
2. **Complete Comparison**: All 3 deployed models shown in every visualization
3. **Consistent Naming**: Model names match document type confusion matrices (Cell 18)
4. **Better Insights**: Can compare 8B quantized vs 2B non-quantized InternVL3 performance
