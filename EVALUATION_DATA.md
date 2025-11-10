# Evaluation Data Directory Structure

## Overview
The `evaluation_data/` directory contains ground truth data and test images for evaluating vision-language model (VLM) extraction performance on business documents.

## Directory Structure

```
evaluation_data/
├── lmm_poc_gt_20251111.csv          # Master ground truth CSV (all document types)
├── bank/                             # Bank statement test set
│   ├── bank_gt.csv                   # Bank statement ground truth
│   └── *.jpeg                        # Bank statement images (75+ files)
├── invoice/                          # Invoice test set
│   ├── invoice_gt.csv                # Invoice ground truth
│   └── *.jpeg                        # Invoice images (100+ files)
├── receipt/                          # Receipt test set
│   ├── receipt_gt.csv                # Receipt ground truth
│   └── *.jpeg                        # Receipt images (20+ files)
└── synthetic/                        # Synthetic/generated test data
    ├── synthetic_gt.csv              # Synthetic data ground truth
    └── image_*.png                   # Synthetic document images
```

## File Organization

### Ground Truth Files

1. **Master GT**: `lmm_poc_gt_20251111.csv`
   - Contains ground truth for ALL document types (invoices, receipts, bank statements)
   - Image filename references WITHOUT file extensions (e.g., `invoice_001` not `invoice_001.jpeg`)
   - Use `dtype=str` when loading to preserve string types (prevents bool conversion)

2. **Type-Specific GTs**: `invoice_gt.csv`, `bank_gt.csv`, `receipt_gt.csv`
   - Subset ground truths for specific document types
   - Useful for document-type-specific evaluation
   - May contain type-specific fields only

### Image Files

**Naming Convention**: `{id}_{document_number}_{sequence}.{ext}`
- Bank statements: `{account_id}_{statement_id}_{page}.jpeg`
- Invoices: `{vendor_id}_{invoice_num}_{page}.jpeg`
- Receipts: `{vendor_id}_{receipt_num}_{page}.jpeg`
- Synthetic: `image_{number}.png`

## Usage Pattern

```python
import pandas as pd
from pathlib import Path

# Load master ground truth (all document types)
gt_df = pd.read_csv('evaluation_data/lmm_poc_gt_20251111.csv', dtype=str)

# Discover images by type
bank_images = sorted(Path('evaluation_data/bank').glob('*.jpeg'))
invoice_images = sorted(Path('evaluation_data/invoice').glob('*.jpeg'))
receipt_images = sorted(Path('evaluation_data/receipt').glob('*.jpeg'))

# Match images to ground truth using stem (no extension)
for img_path in bank_images:
    img_key = img_path.stem  # Remove .jpeg extension
    gt_row = gt_df[gt_df['image_file'] == img_key]
```

## Key Notes

1. **Image-GT Matching**: Ground truth keys use filenames WITHOUT extensions
   - Image: `1325183212_10_10.jpeg`
   - GT key: `1325183212_10_10`
   - Use `Path(img).stem` for lookups

2. **Data Loading**: Always use `dtype=str` to prevent pandas auto-conversion
   ```python
   gt_df = pd.read_csv(gt_path, dtype=str, keep_default_na=False)
   ```

3. **Document Type Distribution**:
   - Bank statements: ~75 images
   - Invoices: ~100 images
   - Receipts: ~20 images
   - Synthetic: ~9 images

4. **Field Coverage**: Not all fields apply to all document types
   - Invoices/Receipts: 14 fields (ABN, line items, GST, total, etc.)
   - Bank Statements: 5 fields (transactions, dates, amounts)
