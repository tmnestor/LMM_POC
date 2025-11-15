# V100 vs H200 Hardware Comparison Report

This directory contains the comprehensive hardware comparison analysis and visualization tools.

## Files

### Main Report
- **v100_vs_h200_hardware_comparison.md** - Complete markdown report with analysis, tables, and embedded charts

### Data Files (CSV)
- `graph_data_processing_time.csv` - Processing time and throughput data
- `graph_data_accuracy.csv` - Accuracy metrics (avg, median, min, max)
- `graph_data_document_type_accuracy.csv` - Document type breakdown
- `graph_data_speed_vs_accuracy.csv` - Speed vs accuracy trade-off
- `graph_data_detailed_results.csv` - Complete dataset

### Visualization Script
- **create_comparison_charts.py** - Python script to generate all colored charts

### Generated Charts (PNG)
- `chart_executive_dashboard.png` - Comprehensive overview of all metrics
- `chart_processing_time.png` - Processing time horizontal bar chart
- `chart_accuracy_comparison.png` - Accuracy metrics grouped bar chart
- `chart_document_type_accuracy.png` - Document type performance (2 charts)
- `chart_speed_vs_accuracy.png` - Scatter plot showing trade-offs
- `chart_throughput_improvement.png` - Throughput % improvement over V100

## Regenerating Charts

To regenerate all charts with updated data:

```bash
cd /Users/tod/Desktop/LMM_POC/reports
python create_comparison_charts.py
```

**Requirements:**
- Python 3.7+
- pandas
- matplotlib

**Color Coding:**
- 🔴 Red: V100 Quantized
- 🔵 Blue: H200 Quantized
- 🟢 Green: H200 bfloat16
- 🟣 Purple: H200 InternVL3.5-8B

**Document Type Colors:**
- 🟢 Green: RECEIPT (easiest, highest accuracy)
- 🟠 Orange: INVOICE (moderate complexity)
- 🔴 Red: BANK_STATEMENT (most challenging, lowest accuracy)

## Key Findings Summary

### Speed Impact
- V100 is **7.8x slower** than H200 optimal configuration
- Flash Attention provides **3.0x speedup** on H200
- Throughput: 0.55 img/min (V100) → 4.29 img/min (H200)

### Accuracy Impact
- **Minimal accuracy degradation** on V100 (only 0.83% difference)
- V100 **outperforms** H200 Quantized for receipts (85.4% vs 83.57%)
- V100 **matches** H200 Quantized for invoices (80.2% vs 80.36%)

### Document Type Insights
- **RECEIPT:** V100 sufficient (85.4% accuracy)
- **INVOICE:** V100 sufficient (80.2% accuracy)
- **BANK_STATEMENT:** Model upgrade critical (InternVL3.5: +17% improvement)

### Recommendations
- **Receipt/Invoice workloads:** V100 Quantized is sufficient
- **Bank Statement workloads:** Upgrade to InternVL3.5-8B (model > hardware)
- **High-throughput workloads:** Upgrade to H200 for 7.8x speed improvement

## Chart Details

### 1. Executive Dashboard
Combines all key metrics in a single comprehensive view:
- Processing time comparison
- Average accuracy by configuration
- Document type accuracy breakdown
- Throughput comparison
- Speedup factors vs V100
- Summary statistics

### 2. Processing Time Chart
Horizontal bar chart showing processing time per image for each configuration.
- Includes time labels and throughput (images/min)
- Shows speedup factors vs V100
- Clearly visualizes the 7.8x speed difference

### 3. Accuracy Comparison Chart
Grouped bar chart showing all accuracy metrics:
- Average, median, minimum, maximum accuracy
- Color-coded by metric type
- Values labeled on each bar

### 4. Document Type Accuracy Chart
Dual view showing document type performance:
- **Left panel:** Grouped by configuration (shows how each config handles different doc types)
- **Right panel:** Grouped by document type (shows which configs work best for each type)

### 5. Speed vs Accuracy Scatter Plot
Shows the trade-off between throughput and accuracy:
- Each point represents a configuration
- Quadrants help identify ideal configurations (top-right)
- Annotations show configuration names

### 6. Throughput Improvement Chart
Horizontal bar chart showing percentage improvement over V100:
- V100 baseline at 0%
- Shows relative improvements for H200 configurations
- H200 bfloat16 shows 656% improvement

## Usage in Reports

The charts are embedded in the markdown report using relative paths:

```markdown
![Executive Dashboard](chart_executive_dashboard.png)
```

To view in Markdown viewers that support images, ensure the PNG files are in the same directory as the markdown file.

## Data Sources

All data was extracted from notebook execution outputs:
- **V100 Quantized:** User-provided screenshot (ivl3_8b_batch_quantized.ipynb on V100)
- **H200 Quantized:** ivl3_8b_batch_quantized.ipynb on H200
- **H200 bfloat16:** ivl3_8b_batch_h200.ipynb
- **H200 InternVL3.5:** ivl3_5_8b_batch.ipynb

All experiments used the same 9 synthetic business document images for fair comparison.
