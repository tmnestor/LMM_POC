# New Analysis Sections Added to model_comparison.ipynb

## Summary

Successfully integrated **6 new cells** into `model_comparison.ipynb` (now 25 cells, was 19).

All implementations use:
- ✅ **pandas** for data manipulation
- ✅ **seaborn** for visualizations
- ✅ **sklearn** for classification metrics

No raw matplotlib - all visualizations use seaborn's high-level API.

---

## Section 5.4: Document Type Confusion Matrix

**Location**: Cells 15-16

**Purpose**: Show document type classification accuracy with classic confusion matrix

**Features**:
- Side-by-side confusion matrices (Llama vs InternVL3)
- sklearn `confusion_matrix` for calculation
- seaborn heatmaps for visualization
- Classification report showing precision/recall/F1 per document type
- Per-document-type metrics comparison (bar charts)

**Outputs**:
- `doctype_confusion_matrix.png` - Side-by-side heatmaps
- `doctype_metrics_comparison.png` - Precision/recall/F1 bar charts
- `llama_doctype_classification_report.csv`
- `internvl_doctype_classification_report.csv`

**Metrics Calculated**:
- Precision per document type
- Recall per document type
- F1-score per document type
- Support (sample count) per document type
- Overall classification accuracy

---

## Section 5.5: Field-Level Confusion Analysis

**Location**: Cells 17-18

**Purpose**: Show field extraction status (correct/incorrect/not_found) for each field

**Features**:
- Side-by-side heatmaps showing extraction status per field
- Compares all 19 fields (DOCUMENT_TYPE, BUSINESS_ABN, etc.)
- Visual identification of problematic fields
- Summary statistics (% correct/incorrect/not_found)

**Outputs**:
- `field_confusion_heatmap.png` - Side-by-side heatmaps
- Console summary showing extraction success rates

**Analysis Reveals**:
- Which fields are most accurately extracted
- Which fields are frequently incorrect vs not found
- Model-specific strengths and weaknesses per field

---

## Section 5.6: Per-Field Precision, Recall, and F1 Metrics

**Location**: Cells 19-20

**Purpose**: Detailed per-field performance using sklearn classification metrics

**Features**:
- Binary classification approach (correct extraction vs not)
- Precision: Of all predicted values, what % were correct?
- Recall: Of all ground truth values, what % were extracted?
- F1 Score: Harmonic mean of precision and recall
- 4-panel visualization (F1/Precision/Recall/Accuracy)

**Outputs**:
- `per_field_metrics.png` - 4-panel horizontal bar charts
- `per_field_metrics.csv` - Full metrics table
- Summary statistics table showing model averages

**Metrics Calculated**:
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: 2 * (P * R) / (P + R)
- **Accuracy**: Correct extractions / Total
- **Support**: Total samples per field

**Key Insight**:
Treats field extraction as binary classification:
- TP: Predicted correctly (and not NOT_FOUND)
- FP: Predicted incorrectly (but not NOT_FOUND)
- FN: Failed to extract (predicted NOT_FOUND when ground truth exists)

---

## Integration Details

**Insertion Point**: After cell 14 (Detailed Performance Analysis)

**Cell Numbers**:
- Cell 15: Markdown - Section 5.4 heading
- Cell 16: Code - Document type confusion (186 lines)
- Cell 17: Markdown - Section 5.5 heading
- Cell 18: Code - Field confusion (151 lines)
- Cell 19: Markdown - Section 5.6 heading
- Cell 20: Code - Per-field metrics (163 lines)

**Total New Code**: 500 lines across 3 code cells

---

## Technical Implementation

### Document Type Confusion (Cell 16)
```python
from sklearn.metrics import confusion_matrix, classification_report
# Uses sklearn's confusion_matrix for document type classification
# Seaborn heatmap for visualization
# Pandas for data manipulation
```

### Field Confusion (Cell 18)
```python
# Custom confusion logic: correct/incorrect/not_found
# Seaborn heatmap with RdYlGn_r colormap
# Pandas pivot tables for matrix structure
```

### Per-Field Metrics (Cell 20)
```python
from sklearn.metrics import precision_recall_fscore_support
# Binary classification metrics per field
# Seaborn horizontal bar charts (no raw matplotlib)
# Pandas groupby for model comparison
```

---

## Data Dependencies

**Required DataFrames**:
- `llama_batch_df` - Llama batch results with extracted fields
- `internvl_batch_df` - InternVL3 batch results with extracted fields
- `ground_truth` - Ground truth DataFrame with true field values

**Required Columns**:
- `image_file` - Image filename (join key)
- `DOCUMENT_TYPE` - Document type (ground truth)
- `document_type` - Document type (predicted)
- All 19 FIELD_COLUMNS for field-level analysis

---

## Usage

Run cells 15-20 after running the data loading section (cells 4-8).

All visualizations are automatically saved to:
```
{CONFIG['output_dir']}/visualizations/
```

All CSVs are saved to:
```
{CONFIG['output_dir']}/
```

---

## Benefits

1. **Document Type Analysis**: Identifies systematic misclassifications
2. **Field-Level Insights**: Shows which fields are hard to extract
3. **Precision/Recall Trade-offs**: Reveals model behavior (conservative vs aggressive)
4. **Model Comparison**: Side-by-side visualizations for easy comparison
5. **Production Ready**: All metrics use sklearn's standard implementation

---

## Inspired By

LayoutLM evaluation script (`/Users/tod/Desktop/LayoutLM_evaluation/evaluate.py`)

Key adaptations:
- Adapted from token-level to field-level classification
- Added document type confusion matrix
- Maintained sklearn + seaborn approach (no raw matplotlib)
- Integrated into existing Jupyter notebook workflow
