# Model Accuracy Comparison Report

## Overview
- **Current Model**: InternVL3-8B
- **Competing Model**: Competing Model
- **Fields Compared**: 15
- **Document Types**: Invoices, Receipts, Bank Statements (195 documents)

## Summary Statistics

| Metric | InternVL3-8B | Competing Model |
|--------|-----------|----------|
| Mean Accuracy | 61.9% | 49.6% |
| Median Accuracy | 57.1% | 42.9% |
| Std Dev | 14.0% | 15.9% |
| Min | 42.1% | 31.6% |
| Max | 89.2% | 83.6% |

## Comparison Results

- **Mean Difference**: +12.3% (positive = InternVL3-8B better)
- **Fields where InternVL3-8B better**: 15
- **Fields where Competing Model better**: 0
- **Fields equal**: 0

## Critical Fields Analysis

Critical fields: BUSINESS_ABN, GST_AMOUNT, TOTAL_AMOUNT, SUPPLIER_NAME

- **InternVL3-8B Mean**: 61.3%
- **Competing Model Mean**: 48.1%

## Statistical Significance

- **Paired t-test p-value**: 0.0000 (Significant at α=0.05)
- **Wilcoxon test p-value**: 0.0001 (Significant at α=0.05)
- **Cohen's d effect size**: 0.8528 (large)
- **95% CI for difference**: [+10.6%, +14.2%]

## Field-Level Details

| Field | InternVL3-8B | Competing Model | Difference |
|-------|-----------|----------|------------|
| DOCUMENT_TYPE | N/A | N/A | N/A |
| BUSINESS_ABN ⚠️ | 42.1% | 31.6% | +10.5% |
| SUPPLIER_NAME ⚠️ | 51.8% | 32.5% | +19.3% |
| BUSINESS_ADDRESS | 56.2% | 41.4% | +14.8% |
| PAYER_NAME | 89.2% | 83.6% | +5.5% |
| PAYER_ADDRESS | 52.4% | 43.7% | +8.7% |
| INVOICE_DATE | 82.4% | 71.5% | +10.9% |
| LINE_ITEM_DESCRIPTIONS | 57.1% | 42.9% | +14.3% |
| LINE_ITEM_QUANTITIES | 53.4% | 40.7% | +12.7% |
| LINE_ITEM_PRICES | 57.3% | 46.0% | +11.3% |
| LINE_ITEM_TOTAL_PRICES | 46.8% | 34.7% | +12.2% |
| IS_GST_INCLUDED | 75.0% | N/A | N/A |
| GST_AMOUNT ⚠️ | 72.9% | 60.4% | +12.5% |
| TOTAL_AMOUNT ⚠️ | 78.6% | 68.0% | +10.6% |
| STATEMENT_DATE_RANGE | 60.0% | 40.0% | +20.0% |
| TRANSACTION_DATES | 54.3% | 42.9% | +11.4% |
| TRANSACTION_AMOUNTS_PAID | 74.3% | 64.1% | +10.2% |

## Output Files

- `field_comparison_results.csv` - Detailed field-level comparison
- `category_comparison_results.csv` - Category-level aggregation
- `accuracy_comparison_bars.png` - Side-by-side bar chart
- `accuracy_difference_lollipop.png` - Difference visualization
- `accuracy_radar_chart.png` - Category radar chart
- `accuracy_heatmap.png` - Heatmap comparison
- `accuracy_boxplot.png` - Distribution comparison
- `accuracy_scatter.png` - Scatter plot with identity line
