# Unified Bank Statement Batch Evaluation Report

**Generated:** 2025-12-04 07:29:40
**Batch ID:** 20251204_063513
**Model:** Llama-3.2-11B-Vision-Instruct
**Evaluation Method:** order_aware_f1

## Executive Summary

- **Total Images:** 12
- **Successful:** 12 (100.0%)
- **Failed:** 0

- **Average Accuracy:** 85.9%
- **Min Accuracy:** 40.0%
- **Max Accuracy:** 100.0%
- **Avg Processing Time:** 271.66s

## Per-Image Results

| Image | Accuracy | Date Format | Rows | Time |
|-------|----------|-------------|------|------|
| cba_date_grouped.png | 95.7% | balance_description_2turn | 9 | 228.9s |
| cba_date_grouped_cont.png | 90.2% | balance_description_2turn | 15 | 372.1s |
| transaction_summary.png | 100.0% | amount_description_2turn | 4 | 184.4s |
| cba_highligted.png | 100.0% | balance_description_2turn | 2 | 129.4s |
| low_contrast.png | 97.9% | debit_credit_description_2turn | 16 | 191.6s |
| cba_amount_balance.png | 88.0% | amount_description_2turn | 12 | 170.5s |
| cba_debit_credit.png | 99.3% | debit_credit_description_2turn | 15 | 208.9s |
| image_003.png | 100.0% | balance_description_2turn | 6 | 125.5s |
| image_009.png | 40.0% | balance_description_2turn | 11 | 251.0s |
| synthetic_chrono.png | 80.1% | balance_description_2turn | 26 | 571.3s |
| synthetic_reverse_chrono.png | 40.0% | balance_description_2turn | 20 | 574.5s |
| synthetic_multiline.png | 98.9% | balance_description_2turn | 10 | 251.8s |

## Field-Level Accuracy

| Field | Avg F1 |
|-------|--------|
| DOCUMENT_TYPE | 100.0% |
| STATEMENT_DATE_RANGE | 100.0% |
| TRANSACTION_DATES | 76.8% |
| LINE_ITEM_DESCRIPTIONS | 76.7% |
| TRANSACTION_AMOUNTS_PAID | 75.7% |