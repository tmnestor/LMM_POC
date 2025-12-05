# Unified Bank Statement Batch Evaluation Report

**Generated:** 2025-12-05 03:12:40
**Batch ID:** 20251205_025606
**Model:** InternVL3-8B (bfloat16, L40/A10G)
**Evaluation Method:** order_aware_f1

## Executive Summary

- **Total Images:** 12
- **Successful:** 12 (100.0%)
- **Failed:** 0

- **Average Accuracy:** 56.3%
- **Min Accuracy:** 20.0%
- **Max Accuracy:** 100.0%
- **Avg Processing Time:** 82.18s

## Per-Image Results

| Image | Accuracy | Date Format | Rows | Time |
|-------|----------|-------------|------|------|
| cba_date_grouped.png | 98.9% | balance_description_2turn | 10 | 65.6s |
| cba_date_grouped_cont.png | 20.0% | table_extraction_not_implemented | 0 | 2.1s |
| transaction_summary.png | 20.0% | amount_description_2turn | 0 | 118.5s |
| cba_highligted.png | 20.0% | table_extraction_not_implemented | 0 | 2.9s |
| low_contrast.png | 95.2% | debit_credit_description_2turn | 16 | 70.1s |
| cba_amount_balance.png | 82.2% | amount_description_2turn | 12 | 82.3s |
| cba_debit_credit.png | 40.0% | debit_credit_description_2turn | 0 | 50.0s |
| image_003.png | 100.0% | balance_description_2turn | 6 | 45.3s |
| image_009.png | 47.2% | balance_description_2turn | 4 | 81.4s |
| synthetic_chrono.png | 48.0% | balance_description_2turn | 23 | 200.0s |
| synthetic_reverse_chrono.png | 44.3% | balance_description_2turn | 20 | 192.2s |
| synthetic_multiline.png | 59.5% | balance_description_2turn | 8 | 75.7s |

## Field-Level Accuracy

| Field | Avg F1 |
|-------|--------|
| DOCUMENT_TYPE | 100.0% |
| STATEMENT_DATE_RANGE | 75.0% |
| TRANSACTION_DATES | 40.1% |
| LINE_ITEM_DESCRIPTIONS | 31.5% |
| TRANSACTION_AMOUNTS_PAID | 34.8% |