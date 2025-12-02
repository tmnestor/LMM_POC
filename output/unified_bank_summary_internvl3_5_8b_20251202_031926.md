# Unified Bank Statement Batch Evaluation Report

**Generated:** 2025-12-02 03:36:14
**Batch ID:** 20251202_031926
**Model:** InternVL3.5-8B (bfloat16)
**Evaluation Method:** order_aware_f1

## Executive Summary

- **Total Images:** 13
- **Successful:** 13 (100.0%)
- **Failed:** 0

- **Average Accuracy:** 59.6%
- **Min Accuracy:** 20.0%
- **Max Accuracy:** 100.0%
- **Avg Processing Time:** 77.04s

## Per-Image Results

| Image | Accuracy | Date Format | Rows | Time |
|-------|----------|-------------|------|------|
| cba_date_grouped.png | 70.0% | balance_description_2turn | 12 | 76.4s |
| cba_date_grouped_cont.png | 80.3% | balance_description_2turn | 19 | 115.9s |
| transaction_summary.png | 20.0% | table_extraction_not_implemented | 0 | 4.4s |
| cba_highligted.png | 30.0% | balance_description_2turn | 1 | 50.2s |
| low_contrast.png | 20.0% | table_extraction_not_implemented | 0 | 6.8s |
| nab_classic_highligted.png | 78.9% | balance_description_2turn | 17 | 192.0s |
| cba_amount_balance.png | 100.0% | balance_description_2turn | 12 | 81.6s |
| cba_debit_credit.png | 20.0% | table_extraction_not_implemented | 0 | 4.0s |
| westpac_debit_credit.png | 20.0% | table_extraction_not_implemented | 0 | 4.3s |
| cba_home_loan.png | 57.1% | balance_description_2turn | 0 | 50.6s |
| image_003.png | 100.0% | balance_description_2turn | 6 | 55.9s |
| image_009.png | 96.4% | balance_description_2turn | 15 | 100.5s |
| image_008.png | 82.3% | balance_description_2turn | 30 | 259.0s |

## Field-Level Accuracy

| Field | Avg F1 |
|-------|--------|
| DOCUMENT_TYPE | 100.0% |
| STATEMENT_DATE_RANGE | 53.8% |
| TRANSACTION_DATES | 45.8% |
| LINE_ITEM_DESCRIPTIONS | 41.5% |
| TRANSACTION_AMOUNTS_PAID | 56.9% |