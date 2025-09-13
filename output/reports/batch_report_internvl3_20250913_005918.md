# InternVL3 Batch Processing Report

**Generated:** 2025-09-13 01:00:35  
**Batch ID:** internvl3_20250913_005918  
**Model:** InternVL3-8B  

## Executive Summary

### Overall Performance
- **Total Images Processed:** 9
- **Successful Extractions:** 9 (100.0%)
- **Average Accuracy:** 73.81%
- **Deployment Status:** 🔴 **Needs Improvement**

### Processing Efficiency
- **Total Processing Time:** 74.09 seconds (1.2 minutes)
- **Average Time per Image:** 8.23 seconds
- **Throughput:** 7.3 images/minute

### Document Type Distribution
- **invoice:** 9 (100.0%)

### Accuracy by Document Type
- **invoice:** 73.81%

### Top Performing Images
- image_004.png: 90.0% (invoice)
- image_001.png: 87.1% (invoice)
- image_006.png: 87.1% (invoice)
- image_005.png: 81.4% (invoice)
- commbank_flat_complex.png: 68.6% (invoice)

### Areas for Improvement
- image_007.png: 58.6% (invoice)
- commbank_flat_simple.png: 61.4% (invoice)
- commbank_statement_001.png: 62.9% (invoice)
- image_002.png: 67.1% (invoice)
- commbank_flat_complex.png: 68.6% (invoice)

## Output Files Generated

All results have been saved to: `/home/jovyan/nfs_share/tod/LMM_POC/output`

- **CSV Files:** `csv/batch_internvl3_20250913_005918_*.csv`
- **Visualizations:** `visualizations/*_internvl3_20250913_005918.png`
- **Full Report:** `reports/batch_report_internvl3_20250913_005918.md`

## Technical Details

- **V100 Optimizations:** Enabled (ResilientGenerator, Memory Cleanup)
- **Quantization:** 8-bit with BitsAndBytesConfig
- **Max Tokens:** 4000
- **Device:** CUDA (auto-mapped)
