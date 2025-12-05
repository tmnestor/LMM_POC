# Unified Bank Statement Batch Evaluation Report

**Generated:** 2025-12-05 02:51:15
**Batch ID:** 20251205_022950
**Model:** InternVL3.5-8B (bfloat16, H200)
**Evaluation Method:** order_aware_f1

## Executive Summary

- **Total Images:** 12
- **Successful:** 12 (100.0%)
- **Failed:** 0

- **Average Accuracy:** 84.3%
- **Min Accuracy:** 40.0%
- **Max Accuracy:** 100.0%
- **Avg Processing Time:** 106.56s

## Per-Image Results

| Image | Accuracy | Date Format | Rows | Time |
|-------|----------|-------------|------|------|
| cba_date_grouped.png | 98.9% | balance_description_2turn | 10 | 76.2s |
| cba_date_grouped_cont.png | 47.5% | balance_description_2turn | 14 | 115.6s |
| transaction_summary.png | 100.0% | amount_description_2turn | 4 | 71.4s |
| cba_highligted.png | 100.0% | balance_description_2turn | 2 | 50.1s |
| low_contrast.png | 97.3% | debit_credit_description_2turn | 16 | 75.6s |
| cba_amount_balance.png | 88.0% | amount_description_2turn | 12 | 101.1s |
| cba_debit_credit.png | 97.8% | debit_credit_description_2turn | 15 | 62.3s |
| image_003.png | 100.0% | balance_description_2turn | 6 | 55.5s |
| image_009.png | 40.0% | balance_description_2turn | 13 | 100.1s |
| synthetic_chrono.png | 98.8% | balance_description_2turn | 27 | 244.6s |
| synthetic_reverse_chrono.png | 45.6% | balance_description_2turn | 21 | 235.3s |
| synthetic_multiline.png | 97.8% | balance_description_2turn | 10 | 90.7s |

## Field-Level Accuracy

| Field | Avg F1 |
|-------|--------|
| DOCUMENT_TYPE | 100.0% |
| STATEMENT_DATE_RANGE | 100.0% |
| TRANSACTION_DATES | 75.8% |
| LINE_ITEM_DESCRIPTIONS | 68.6% |
| TRANSACTION_AMOUNTS_PAID | 77.2% |