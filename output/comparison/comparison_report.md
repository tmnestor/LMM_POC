# Model Accuracy Comparison Report

## Overview
- **Current Model**: Llama-3.2-11B
- **Competing Model**: LayoutLM
- **Comparable Fields**: 16
- **Fields with F1 Comparison**: 15
- **Additional Capabilities** (our model only): 1
- **Document Types**: Invoices, Receipts, Bank Statements (195 documents)

## Summary Statistics (Comparable Fields Only)

### Accuracy Metrics

| Metric | Llama-3.2-11B | LayoutLM |
|--------|-----------|----------|
| Mean Accuracy | 100.0% | 51.3% |
| Median Accuracy | 100.0% | 44.5% |
| Std Dev | 0.0% | 15.5% |
| Min | 100.0% | 31.6% |
| Max | 100.0% | 86.5% |

### F1 Score Metrics (Position-Agnostic)

| Metric | Llama-3.2-11B | LayoutLM |
|--------|-----------|----------|
| Mean F1 | 95.1% | 54.6% |
| Mean F1 Difference | +40.5% | - |
| Fields Llama-3.2-11B Better | 15 | - |
| Fields LayoutLM Better | 0 | - |

*Note: F1 scores are position-agnostic (set-based matching) for fair comparison between models.*

## Comparison Results

- **Mean Accuracy Difference**: +48.7% (positive = Llama-3.2-11B better)
- **Fields where Llama-3.2-11B better**: 16
- **Fields where LayoutLM better**: 0
- **Fields equal**: 0

## Critical Fields Analysis

Critical fields: BUSINESS_ABN, GST_AMOUNT, TOTAL_AMOUNT, SUPPLIER_NAME

- **Llama-3.2-11B Mean**: 100.0%
- **LayoutLM Mean**: 50.7%

## Statistical Significance

- **Paired t-test p-value**: 0.0000 (Significant at α=0.05)
- **Wilcoxon test p-value**: 0.0000 (Significant at α=0.05)
- **Cohen's d effect size**: 4.5976 (large)
- **95% CI for difference**: [+41.0%, +55.6%]

## Field-Level Details (Comparable Fields)

### Accuracy
| Field | Llama-3.2-11B | LayoutLM | Difference | Critical |
|-------|-----------|----------|------------|----------|
| BUSINESS_ABN | 100.0% | 31.6% | +68.4% | Yes |
| SUPPLIER_NAME | 100.0% | 42.7% | +57.3% | Yes |
| BUSINESS_ADDRESS | 100.0% | 40.8% | +59.2% |  |
| PAYER_NAME | 100.0% | 86.5% | +13.5% |  |
| PAYER_ADDRESS | 100.0% | 43.1% | +56.9% |  |
| INVOICE_DATE | 100.0% | 71.5% | +28.5% |  |
| LINE_ITEM_DESCRIPTIONS | 100.0% | 42.9% | +57.1% |  |
| LINE_ITEM_QUANTITIES | 100.0% | 40.7% | +59.3% |  |
| LINE_ITEM_PRICES | 100.0% | 46.0% | +54.0% |  |
| LINE_ITEM_TOTAL_PRICES | 100.0% | 34.7% | +65.3% |  |
| IS_GST_INCLUDED | 100.0% | 62.4% | +37.6% |  |
| GST_AMOUNT | 100.0% | 60.4% | +39.6% | Yes |
| TOTAL_AMOUNT | 100.0% | 68.0% | +32.0% | Yes |
| STATEMENT_DATE_RANGE | 100.0% | 40.0% | +60.0% |  |
| TRANSACTION_DATES | 100.0% | 45.9% | +54.1% |  |
| TRANSACTION_AMOUNTS_PAID | 100.0% | 64.1% | +35.9% |  |

### F1 Scores (Position-Agnostic)
| Field | Llama-3.2-11B F1 | LayoutLM F1 | Difference |
|-------|-----------|----------|------------|
| BUSINESS_ABN | 100.0% | 20.4% | +79.6% |
| SUPPLIER_NAME | 100.0% | 58.8% | +41.2% |
| BUSINESS_ADDRESS | 83.3% | 47.9% | +35.4% |
| PAYER_NAME | 100.0% | 89.5% | +10.5% |
| PAYER_ADDRESS | 100.0% | 46.3% | +53.7% |
| INVOICE_DATE | 83.3% | 83.3% | +0.1% |
| LINE_ITEM_DESCRIPTIONS | 65.3% | 55.8% | +9.5% |
| LINE_ITEM_QUANTITIES | 100.0% | 15.1% | +84.9% |
| LINE_ITEM_PRICES | 100.0% | 17.6% | +82.4% |
| LINE_ITEM_TOTAL_PRICES | 100.0% | 51.5% | +48.5% |
| IS_GST_INCLUDED | 100.0% | 76.9% | +23.1% |
| GST_AMOUNT | 100.0% | 50.6% | +49.4% |
| TOTAL_AMOUNT | 100.0% | 77.1% | +22.9% |
| TRANSACTION_DATES | 96.3% | 56.9% | +39.4% |
| TRANSACTION_AMOUNTS_PAID | 98.9% | 71.9% | +27.0% |

## Additional Capabilities (Llama-3.2-11B Only)

The following fields are **unique to our model** and not available in the competing model:

| Field | Llama-3.2-11B Accuracy | Category |
|-------|------------------|----------|
| DOCUMENT_TYPE | 100.0% | Identity |

**Mean accuracy of exclusive capabilities**: 100.0%

## Output Files

- `comparison_dashboard_a4.png` - A4 dashboard summary (300 dpi)
- `comparison_dashboard_a4.pdf` - A4 dashboard PDF
- `comparison_report.md` - This report
- `accuracy_comparison_bars.png` - Side-by-side bar chart
- `accuracy_difference_lollipop.png` - Difference visualization
- `accuracy_radar_chart.png` - Category radar chart
- `accuracy_heatmap.png` - Heatmap comparison
- `accuracy_boxplot.png` - Distribution comparison
- `accuracy_scatter.png` - Scatter plot with identity line
- `accuracy_by_category.png` - Category comparison chart
- `f1_comparison.png` - F1 score comparison (if F1 data available)
