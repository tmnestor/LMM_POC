# Sync Troubleshooting Guide

## Problem
Local Mac file has all changes (3-panel confusion matrix with abbreviations), but remote server shows old version (2-panel with long labels).

## Verification Status

### ✅ Local Mac File (Verified)
- **Path**: `/Users/tod/Desktop/LMM_POC/model_comparison.ipynb`
- **Git Status**: Committed and pushed to origin/main
- **Commit**: 6b326a4 "feat: Enhance confusion matrix analysis by adding support for InternVL3-NonQuantized model"
- **Cell 16**: 66 lines, includes `internvl_nq_batch_df`
- **Cell 18**: 186 lines, includes `abbreviate_doctype()`, 3-panel subplot, all 3 models

## Troubleshooting Steps

### Step 1: Verify Remote Git Status

On the **remote server** (H200 or V100), run:

```bash
cd /home/jovyan/_LMM_POC
git status
git log --oneline -1 -- model_comparison.ipynb
```

**Expected output**:
```
commit 6b326a4 feat: Enhance confusion matrix analysis by adding support for InternVL3-NonQuantized model
```

**If you see an older commit**, you need to pull:
```bash
git pull origin main
```

### Step 2: Run Verification Script

On the **remote server**:

```bash
cd /home/jovyan/_LMM_POC
python verify_notebook_version.py
```

This will show exactly which features are present in the notebook file.

**Expected output**: All ✅ checks passed

**If checks fail**: The file hasn't been updated yet.

### Step 3: Force Jupyter to Reload

Even if the file is updated, Jupyter might have it cached. Try:

**Option A: Restart Kernel and Clear Output**
1. In Jupyter: `Kernel` → `Restart & Clear Output`
2. Close the notebook tab
3. Re-open `model_comparison.ipynb` from file browser
4. Run all cells fresh

**Option B: Restart Jupyter Server**
```bash
# Find Jupyter process
ps aux | grep jupyter

# Kill and restart (if you have access)
jupyter notebook stop 8888  # Or your port number
jupyter notebook
```

**Option C: Hard Reload in Browser**
- Mac Chrome: `Cmd + Shift + R`
- Mac Safari: `Cmd + Option + R`
- After reload, close and re-open the notebook

### Step 4: Check File Permissions

On the **remote server**:

```bash
ls -la /home/jovyan/_LMM_POC/model_comparison.ipynb
```

Ensure:
- File is readable
- File is owned by the correct user
- No file locks preventing updates

### Step 5: Manual File Verification

On the **remote server**, check cell line counts:

```bash
cd /home/jovyan/_LMM_POC
python3 -c "
import json
with open('model_comparison.ipynb', 'r') as f:
    nb = json.load(f)
print(f'Cell 16 lines: {len(nb[\"cells\"][16][\"source\"])}')
print(f'Cell 18 lines: {len(nb[\"cells\"][18][\"source\"])}')
print(f'Has ax3: {any(\"ax3\" in line for line in nb[\"cells\"][18][\"source\"])}')
"
```

**Expected output**:
```
Cell 16 lines: 66
Cell 18 lines: 186
Has ax3: True
```

**If these don't match**: File hasn't synced correctly.

### Step 6: Force Overwrite (Last Resort)

If git pull doesn't work, force overwrite:

```bash
cd /home/jovyan/_LMM_POC
git fetch origin
git reset --hard origin/main
```

⚠️ **WARNING**: This will discard any uncommitted local changes!

### Step 7: Check Disk Space

Sometimes sync fails due to disk space:

```bash
df -h /home/jovyan
```

Ensure sufficient space available.

## Expected Behavior After Fix

When you run **Cell 18** after successful sync, you should see:

### Console Output
```
📊 Creating document type confusion matrices...

✅ Llama confusion matrix: (3, 6) - 3 true types vs 6 predicted types
   Classification Report:
              precision    recall  f1-score   support
   [metrics for each document type]

✅ InternVL3-Quantized confusion matrix: (3, 7) - 3 true types vs 7 predicted types
   [similar report]

✅ InternVL3-NonQuantized confusion matrix: (3, N) - 3 true types vs N predicted types
   [similar report]

💾 Saved: visualizations/doctype_confusion_matrix.png
```

### Visual Output
- **3 confusion matrices** side-by-side in one figure
- **Abbreviated labels**: CTP_INSUR, E-TICKET, MOBILE_SS, PAYMENT, etc.
- **3 rows** (BANK_STMT, INVOICE, RECEIPT) for Y-axis (true labels)
- **Variable columns** for X-axis (predicted labels - different for each model)
- **Panel titles**:
  1. "Llama-3.2-Vision-11B"
  2. "InternVL3-Quantized-8B"
  3. "InternVL3-NonQuantized-2B"

## If Still Not Working

Run this diagnostic on **both local and remote**:

```bash
# Local Mac
cd /Users/tod/Desktop/LMM_POC
git log --oneline -5 -- model_comparison.ipynb
md5 model_comparison.ipynb  # Or shasum

# Remote Server
cd /home/jovyan/_LMM_POC
git log --oneline -5 -- model_comparison.ipynb
md5sum model_comparison.ipynb  # Compare with local
```

The **md5/sha checksums should match** if files are identical.

If checksums don't match, the git repositories are out of sync.
