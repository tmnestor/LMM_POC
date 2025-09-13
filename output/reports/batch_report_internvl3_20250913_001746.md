# InternVL3 Batch Processing Report

**Generated:** 2025-09-13 00:18:21  
**Batch ID:** internvl3_20250913_001746  
**Model:** InternVL3-8B  

## Executive Summary

### Overall Performance
- **Total Images Processed:** 9
- **Successful Extractions:** 9 (100.0%)
- **Average Accuracy:** 26.19%
- **Deployment Status:** 🔴 **Needs Improvement**

### Processing Efficiency
- **Total Processing Time:** 33.06 seconds (0.6 minutes)
- **Average Time per Image:** 3.67 seconds
- **Throughput:** 16.3 images/minute

### Document Type Distribution
- **invoice:** 9 (100.0%)

### Accuracy by Document Type
- **invoice:** 26.19%

### Top Performing Images
- commbank_flat_simple.png: 85.7% (invoice)
- commbank_statement_001.png: 78.6% (invoice)
- commbank_flat_complex.png: 71.4% (invoice)
- image_001.png: 0.0% (invoice)
- image_002.png: 0.0% (invoice)

### Areas for Improvement
- image_001.png: 0.0% (invoice)
- image_002.png: 0.0% (invoice)
- image_004.png: 0.0% (invoice)
- image_005.png: 0.0% (invoice)
- image_006.png: 0.0% (invoice)

## Output Files Generated

All results have been saved to: `/home/jovyan/nfs_share/tod/LMM_POC/output`

- **CSV Files:** `csv/batch_internvl3_20250913_001746_*.csv`
- **Visualizations:** `visualizations/*_internvl3_20250913_001746.png`
- **Full Report:** `reports/batch_report_internvl3_20250913_001746.md`

## Technical Details

- **V100 Optimizations:** Enabled (ResilientGenerator, Memory Cleanup)
- **Quantization:** 8-bit with BitsAndBytesConfig
- **Max Tokens:** 4000
- **Device:** CUDA (auto-mapped)
