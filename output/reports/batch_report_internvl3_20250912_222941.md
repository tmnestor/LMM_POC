# InternVL3 Batch Processing Report

**Generated:** 2025-09-12 22:30:42  
**Batch ID:** internvl3_20250912_222941  
**Model:** InternVL3-8B  

## Executive Summary

### Overall Performance
- **Total Images Processed:** 9
- **Successful Extractions:** 3 (33.3%)
- **Average Accuracy:** 75.71%
- **Deployment Status:** 🔴 **Needs Improvement**

### Processing Efficiency
- **Total Processing Time:** 34.03 seconds (0.6 minutes)
- **Average Time per Image:** 3.78 seconds
- **Throughput:** 15.9 images/minute

### Document Type Distribution
- **invoice:** 9 (100.0%)
- **unknown:** 6 (66.7%)

### Accuracy by Document Type
- **invoice:** 75.71%

### Top Performing Images
- image_006.png: 87.1% (invoice)
- image_005.png: 81.4% (invoice)
- image_007.png: 58.6% (invoice)

### Areas for Improvement
- image_007.png: 58.6% (invoice)
- image_005.png: 81.4% (invoice)
- image_006.png: 87.1% (invoice)

## Output Files Generated

All results have been saved to: `/home/jovyan/nfs_share/tod/LMM_POC/output`

- **CSV Files:** `csv/batch_internvl3_20250912_222941_*.csv`
- **Visualizations:** `visualizations/*_internvl3_20250912_222941.png`
- **Full Report:** `reports/batch_report_internvl3_20250912_222941.md`

## Technical Details

- **V100 Optimizations:** Enabled (ResilientGenerator, Memory Cleanup)
- **Quantization:** 8-bit with BitsAndBytesConfig
- **Max Tokens:** 4000
- **Device:** CUDA (auto-mapped)
