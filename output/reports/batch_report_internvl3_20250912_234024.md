# InternVL3 Batch Processing Report

**Generated:** 2025-09-12 23:41:18  
**Batch ID:** internvl3_20250912_234024  
**Model:** InternVL3-8B  

## Executive Summary

### Overall Performance
- **Total Images Processed:** 9
- **Successful Extractions:** 2 (22.2%)
- **Average Accuracy:** 72.86%
- **Deployment Status:** 🔴 **Needs Improvement**

### Processing Efficiency
- **Total Processing Time:** 25.11 seconds (0.4 minutes)
- **Average Time per Image:** 12.56 seconds
- **Throughput:** 4.8 images/minute

### Document Type Distribution
- **invoice:** 2 (22.2%)

### Accuracy by Document Type
- **invoice:** 72.86%

### Top Performing Images
- image_006.png: 87.1% (invoice)
- image_007.png: 58.6% (invoice)

### Areas for Improvement
- image_007.png: 58.6% (invoice)
- image_006.png: 87.1% (invoice)

## Output Files Generated

All results have been saved to: `/home/jovyan/nfs_share/tod/LMM_POC/output`

- **CSV Files:** `csv/batch_internvl3_20250912_234024_*.csv`
- **Visualizations:** `visualizations/*_internvl3_20250912_234024.png`
- **Full Report:** `reports/batch_report_internvl3_20250912_234024.md`

## Technical Details

- **V100 Optimizations:** Enabled (ResilientGenerator, Memory Cleanup)
- **Quantization:** 8-bit with BitsAndBytesConfig
- **Max Tokens:** 4000
- **Device:** CUDA (auto-mapped)
