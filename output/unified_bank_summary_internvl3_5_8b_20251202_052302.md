# Unified Bank Statement Batch Evaluation Report

**Generated:** 2025-12-02 05:45:53
**Batch ID:** 20251202_052302
**Model:** InternVL3.5-8B (bfloat16)
**Evaluation Method:** order_aware_f1

## Executive Summary

- **Total Images:** 13
- **Successful:** 13 (100.0%)
- **Failed:** 0

- **Average Accuracy:** 85.1%
- **Min Accuracy:** 48.7%
- **Max Accuracy:** 100.0%
- **Avg Processing Time:** 104.97s

## Per-Image Results

| Image | Accuracy | Date Format | Rows | Time |
|-------|----------|-------------|------|------|
| cba_date_grouped.png | 70.0% | balance_description_2turn | 12 | 76.9s |
| cba_date_grouped_cont.png | 80.3% | balance_description_2turn | 19 | 116.2s |
| transaction_summary.png | 100.0% | amount_description_2turn | 4 | 71.5s |
| cba_highligted.png | 60.0% | balance_description_2turn | 3 | 50.5s |
| low_contrast.png | 97.3% | debit_credit_description_2turn | 16 | 76.1s |
| nab_classic_highligted.png | 78.9% | balance_description_2turn | 17 | 192.6s |
| cba_amount_balance.png | 100.0% | balance_description_2turn | 12 | 81.7s |
| cba_debit_credit.png | 97.8% | debit_credit_description_2turn | 15 | 62.5s |
| westpac_debit_credit.png | 48.7% | debit_credit_description_2turn | 0 | 170.7s |
| cba_home_loan.png | 94.3% | balance_description_2turn | 4 | 50.7s |
| image_003.png | 100.0% | balance_description_2turn | 6 | 55.9s |
| image_009.png | 96.4% | balance_description_2turn | 15 | 100.3s |
| image_008.png | 82.3% | balance_description_2turn | 30 | 259.2s |

## Field-Level Accuracy

| Field | Avg F1 |
|-------|--------|
| DOCUMENT_TYPE | 100.0% |
| STATEMENT_DATE_RANGE | 92.3% |
| TRANSACTION_DATES | 78.6% |
| LINE_ITEM_DESCRIPTIONS | 67.2% |
| TRANSACTION_AMOUNTS_PAID | 87.3% |