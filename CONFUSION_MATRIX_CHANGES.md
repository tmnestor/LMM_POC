# Confusion Matrix Changes - Quick Reference

## Before vs After

### BEFORE (Old Version - 2 Panels)
```
┌─────────────────────────────┬─────────────────────────────┐
│   Llama-3.2-Vision-11B      │   InternVL3-Quantized-8B    │
│                             │                             │
│ Y-axis: Long names like     │ Y-axis: Long names like     │
│ "COMPULSORY THIRD PARTY..." │ "COMPULSORY THIRD PARTY..." │
│                             │                             │
│ X-axis: Long names          │ X-axis: Long names          │
│                             │                             │
│ (Labels overlap/unreadable) │ (Labels overlap/unreadable) │
└─────────────────────────────┴─────────────────────────────┘
```

**Missing**: InternVL3-NonQuantized-2B model

### AFTER (New Version - 3 Panels)
```
┌──────────────────┬──────────────────┬──────────────────────┐
│ Llama-3.2-Vision │ InternVL3-Quant  │ InternVL3-NonQuant   │
│      -11B        │      -8B         │        -2B           │
├──────────────────┼──────────────────┼──────────────────────┤
│ Y: BANK_STMT     │ Y: BANK_STMT     │ Y: BANK_STMT         │
│    INVOICE       │    INVOICE       │    INVOICE           │
│    RECEIPT       │    RECEIPT       │    RECEIPT           │
│                  │                  │                      │
│ X: CTP_INSUR     │ X: CTP_INSUR     │ X: CTP_INSUR         │
│    E-TICKET      │    E-TICKET      │    E-TICKET          │
│    MOBILE_SS     │    MOBILE_SS     │    MOBILE_SS         │
│    PAYMENT       │    PAYMENT       │    PAYMENT           │
│    CRYPTO        │    CRYPTO        │    CRYPTO            │
│    TAX_INV       │    TAX_INV       │    TAX_INV           │
│    (etc.)        │    NOT_FOUND     │    (etc.)            │
│                  │    (etc.)        │                      │
└──────────────────┴──────────────────┴──────────────────────┘
```

**Features**: 3 panels, abbreviated labels, all models included

## Code Changes

### Cell 16: Data Preparation (66 lines)

**Key additions**:
```python
# NEW: 3rd model preparation
internvl_nq_batch_df = internvl3_non_quantized_df.copy() if not internvl3_non_quantized_df.empty else pd.DataFrame()

# NEW: Verification for 3rd model
if internvl_nq_batch_df.empty:
    rprint("[yellow]⚠️ Warning: InternVL3-NonQuantized batch data not loaded. Run cell 5 first.[/yellow]")
else:
    rprint(f"[cyan]InternVL3-NonQuantized batch data: {len(internvl_nq_batch_df)} rows[/cyan]")

# NEW: Column normalization for ground truth
if 'image_name' in ground_truth_full.columns and 'image_file' not in ground_truth_full.columns:
    ground_truth_full['image_file'] = ground_truth_full['image_name']

# NEW: Add image_stem for matching (strips extensions)
ground_truth_full['image_stem'] = ground_truth_full['image_file'].apply(lambda x: Path(x).stem)
```

### Cell 18: Confusion Matrix (186 lines)

**Key additions**:

#### 1. Abbreviation Function (NEW)
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
    return abbrev.get(name, name[:12])
```

#### 2. Three-Panel Layout (CHANGED)
```python
# OLD: 2 panels
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# NEW: 3 panels
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
```

#### 3. Third Confusion Matrix (NEW)
```python
# Create confusion matrix for InternVL3-NonQuantized
internvl_nq_cm, internvl_nq_report, internvl_nq_labels, internvl_nq_row_labels, internvl_nq_y_true, internvl_nq_y_pred = create_doctype_confusion_matrix(
    internvl_nq_batch_df, ground_truth, 'InternVL3-NonQuantized'
)

# Plot 3rd panel
sns.heatmap(
    internvl_nq_cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=internvl_nq_labels,
    yticklabels=internvl_nq_row_labels,
    cbar_kws={'label': 'Count'},
    ax=ax3,  # ← 3rd subplot
    linewidths=0.5,
    linecolor='gray'
)
ax3.set_title('InternVL3-NonQuantized-2B', fontsize=13, fontweight='bold')
ax3.set_xlabel('Predicted Document Type', fontsize=11)
ax3.set_ylabel('True Document Type', fontsize=11)
```

#### 4. Non-Square Matrix Support (CHANGED)
```python
# OLD: sklearn.confusion_matrix (forces square)
# labels = sorted(list(set(y_true) | set(y_pred)))
# cm = confusion_matrix(y_true, y_pred, labels=labels)

# NEW: pd.crosstab (allows 3×N non-square)
cm_df = pd.crosstab(y_true, y_pred, dropna=False)
cm = cm_df.values
labels = [abbreviate_doctype(label) for label in cm_df.columns.tolist()]
row_labels = [abbreviate_doctype(label) for label in cm_df.index.tolist()]
```

## How to Verify Changes Worked

### Console Output Check
After running Cell 18, you should see:
```
📊 Creating document type confusion matrices...

✅ Llama confusion matrix: (3, 6) - 3 true types vs 6 predicted types
✅ InternVL3-Quantized confusion matrix: (3, 7) - 3 true types vs 7 predicted types
✅ InternVL3-NonQuantized confusion matrix: (3, N) - 3 true types vs N predicted types

💾 Saved: visualizations/doctype_confusion_matrix.png
```

**Key indicators**:
- ✅ THREE confusion matrix lines (not two)
- ✅ Matrix shapes like (3, 6) not (6, 6) - non-square!
- ✅ Mentions "InternVL3-NonQuantized"

### Visual Output Check
The plot should show:
- ✅ **3 heatmaps** side-by-side (not 2)
- ✅ **Short labels** like "CTP_INSUR", "E-TICKET", "MOBILE_SS" (not long strings)
- ✅ **3 rows** on Y-axis (BANK_STMT, INVOICE, RECEIPT)
- ✅ **Variable columns** on X-axis (different for each model)
- ✅ **3rd panel** titled "InternVL3-NonQuantized-2B"

### File Size Check
The saved PNG should be **wider** than before (24 inches vs 18 inches) due to 3 panels.

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

## Matrix Dimensions

All confusion matrices should be **non-square**:
- **Y-axis (True)**: 3 rows (ground truth has only 3 document types)
- **X-axis (Predicted)**: Variable columns (models may extract 6-8+ different types)

**Example**: (3, 6) means 3 true types × 6 predicted types.

If you see (6, 6) or any square matrix, the changes haven't been applied!
