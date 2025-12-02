# Unified Bank Statement Batch Evaluation Report

**Generated:** 2025-12-02 03:16:28
**Batch ID:** 20251202_030615
**Model:** InternVL3.5-8B (bfloat16)
**Evaluation Method:** order_aware_f1

## Executive Summary

- **Total Images:** 5
- **Successful:** 5 (100.0%)
- **Failed:** 0

- **Average Accuracy:** 85.8%
- **Min Accuracy:** 70.0%
- **Max Accuracy:** 100.0%
- **Avg Processing Time:** 121.48s

## Per-Image Results

| Image | Accuracy | Date Format | Rows | Time |
|-------|----------|-------------|------|------|
| cba_date_grouped.png | 70.0% | balance_description_2turn | 12 | 76.4s |
| cba_date_grouped_cont.png | 80.3% | balance_description_2turn | 19 | 115.9s |
| image_003.png | 100.0% | balance_description_2turn | 6 | 55.9s |
| image_009.png | 96.4% | balance_description_2turn | 15 | 100.1s |
| image_008.png | 82.3% | balance_description_2turn | 30 | 259.0s |

## Field-Level Accuracy

| Field | Avg F1 |
|-------|--------|
| DOCUMENT_TYPE | 100.0% |
| STATEMENT_DATE_RANGE | 100.0% |
| TRANSACTION_DATES | 79.7% |
| LINE_ITEM_DESCRIPTIONS | 70.8% |
| TRANSACTION_AMOUNTS_PAID | 78.4% |