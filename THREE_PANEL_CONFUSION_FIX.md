# Three-Panel Confusion Matrix with Abbreviations

## Changes Made

### **Cell 16: Data Preparation**

**Added preparation for InternVL3-NonQuantized model:**

```python
# OLD (only 2 models):
llama_batch_df = llama_df.copy() if not llama_df.empty else pd.DataFrame()
internvl_batch_df = internvl3_quantized_df.copy() if not internvl3_quantized_df.empty else pd.DataFrame()

# NEW (all 3 models):
llama_batch_df = llama_df.copy() if not llama_df.empty else pd.DataFrame()
internvl_batch_df = internvl3_quantized_df.copy() if not internvl3_quantized_df.empty else pd.DataFrame()
internvl_nq_batch_df = internvl3_non_quantized_df.copy() if not internvl3_non_quantized_df.empty else pd.DataFrame()  # ⭐ NEW
```

**Added verification for 3rd model:**
```python
if internvl_nq_batch_df.empty:
    rprint("[yellow]⚠️ Warning: InternVL3-NonQuantized batch data not loaded. Run cell 5 first.[/yellow]")
else:
    rprint(f"[cyan]InternVL3-NonQuantized batch data: {len(internvl_nq_batch_df)} rows[/cyan]")
```

---

### **Cell 18: Confusion Matrix Visualization**

#### **1. Added `abbreviate_doctype()` Function**

Shortens long document type names for readable axis labels:

```python
def abbreviate_doctype(name):
    """Shorten long document type names for visualization."""
    abbrev = {
        'COMPULSORY THIRD PARTY PERSONAL INJURY INSURANCE GREEN SLIP CERTIFICATE': 'CTP_INSUR',
        'E-TICKET ITINERARY, RECEIPT AND TAX INVOICE': 'E-TICKET',
        'MOBILE APP SCREENSHOT': 'MOBILE_SS',
        'PAYMENT ADVICE': 'PAYMENT',
        'CRYPTO STATEMENT': 'CRYPTO',
        'TAX INVOICE': 'TAX_INV',
        'INVOICE': 'INVOICE',
        'RECEIPT': 'RECEIPT',
        'BANK_STATEMENT': 'BANK_STMT',
        'NOT_FOUND': 'NOT_FOUND'
    }
    return abbrev.get(name, name[:12])  # Fallback: truncate to 12 chars
```

**Applied to labels:**
```python
labels = [abbreviate_doctype(label) for label in cm_df.columns.tolist()]
row_labels = [abbreviate_doctype(label) for label in cm_df.index.tolist()]
```

#### **2. Changed from 2-Panel to 3-Panel Layout**

**OLD:**
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
```

**NEW:**
```python
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
```

#### **3. Added Third Confusion Matrix**

Created confusion matrix for InternVL3-NonQuantized:

```python
internvl_nq_cm, internvl_nq_report, internvl_nq_labels, internvl_nq_row_labels, internvl_nq_y_true, internvl_nq_y_pred = create_doctype_confusion_matrix(
    internvl_nq_batch_df, ground_truth, 'InternVL3-NonQuantized'
)
```

**3rd subplot:**
```python
sns.heatmap(
    internvl_nq_cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=internvl_nq_labels,
    yticklabels=internvl_nq_row_labels,
    cbar_kws={'label': 'Count'},
    ax=ax3,
    linewidths=0.5,
    linecolor='gray'
)
ax3.set_title('InternVL3-NonQuantized-2B', fontsize=13, fontweight='bold')
```

#### **4. Improved Readability**

- Reduced title font size: 14 → 13
- Reduced axis label font size: 12 → 11
- Reduced tick label font size: 10 → 9
- Applied abbreviations to all labels

---

## Expected Output

### **Before:**

**2-Panel Plot:**
- Llama-3.2-Vision
- InternVL3 (only quantized)

**Long Labels:**
- "COMPULSORY THIRD PARTY PERSONAL INJURY INSURANCE GREEN SLIP CERTIFICATE"
- Unreadable, overlapping text

### **After:**

**3-Panel Plot:**
1. Llama-3.2-Vision-11B
2. InternVL3-Quantized-8B
3. InternVL3-NonQuantized-2B ⭐ NEW

**Abbreviated Labels:**
- "CTP_INSUR"
- "E-TICKET"
- "MOBILE_SS"
- Readable, fits nicely

---

## Visual Comparison

### **Panel 1: Llama-3.2-Vision-11B**
- Shows 3 rows (BANK_STMT, INVOICE, RECEIPT)
- Shows 6 columns (predicted types with abbreviations)

### **Panel 2: InternVL3-Quantized-8B**
- Shows 3 rows (BANK_STMT, INVOICE, RECEIPT)
- Shows 7 columns (predicted types including CRYPTO, NOT_FOUND)

### **Panel 3: InternVL3-NonQuantized-2B** ⭐ NEW
- Shows 3 rows (BANK_STMT, INVOICE, RECEIPT)
- Shows N columns (predicted types from 2B model)

---

## Abbreviation Mapping

| Original | Abbreviated |
|----------|-------------|
| COMPULSORY THIRD PARTY PERSONAL INJURY INSURANCE GREEN SLIP CERTIFICATE | CTP_INSUR |
| E-TICKET ITINERARY, RECEIPT AND TAX INVOICE | E-TICKET |
| MOBILE APP SCREENSHOT | MOBILE_SS |
| PAYMENT ADVICE | PAYMENT |
| CRYPTO STATEMENT | CRYPTO |
| TAX INVOICE | TAX_INV |
| INVOICE | INVOICE |
| RECEIPT | RECEIPT |
| BANK_STATEMENT | BANK_STMT |
| NOT_FOUND | NOT_FOUND |

---

## Files Modified

- `/Users/tod/Desktop/LMM_POC/model_comparison.ipynb`
  - Cell 16 (66 lines) - Added 3rd model preparation
  - Cell 18 (186 lines) - Added abbreviations and 3rd panel

---

## Verification

```bash
✅ Notebook is valid JSON
✅ Cell 16: 66 lines with proper formatting
✅ Cell 18: 186 lines with proper formatting
✅ All 3 models configured
✅ Abbreviation function implemented
✅ 3-panel layout implemented
✅ pd.crosstab still used (non-square matrices)
```

---

## Running the Updated Cells

1. **Run Cell 5**: Load all 3 model results
   - Creates: `llama_df`, `internvl3_quantized_df`, `internvl3_non_quantized_df`

2. **Run Cell 16**: Prepare data for confusion analysis
   - Creates: `llama_batch_df`, `internvl_batch_df`, `internvl_nq_batch_df`, `ground_truth`
   - Verifies all 3 models loaded

3. **Run Cell 18**: Generate 3-panel confusion matrices
   - Creates side-by-side comparison of all 3 models
   - Uses abbreviated labels for readability
   - Saves to `visualizations/doctype_confusion_matrix.png`

---

## Benefits

1. **Complete Model Comparison**: All 3 models shown side-by-side
2. **Readable Labels**: Abbreviations prevent overlapping text
3. **Consistent Format**: Same 3-row structure for all models
4. **Easy Comparison**: See how each model handles rare document types
5. **Production Ready**: Clean, professional visualization

---

## Next Steps

After running the updated cells, you'll be able to:
- Compare document type extraction across all 3 models
- Identify which rare types each model extracts
- See classification accuracy for each model
- Make informed decisions about model selection based on document type handling
