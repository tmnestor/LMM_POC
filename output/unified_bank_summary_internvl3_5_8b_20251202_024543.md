# Unified Bank Statement Batch Evaluation Report

**Generated:** 2025-12-02 02:52:43
**Batch ID:** 20251202_024543
**Model:** InternVL3.5-8B (bfloat16)
**Evaluation Method:** order_aware_f1

## Executive Summary

- **Total Images:** 3
- **Successful:** 3 (100.0%)
- **Failed:** 0

- **Average Accuracy:** 92.9%
- **Min Accuracy:** 82.3%
- **Max Accuracy:** 100.0%
- **Avg Processing Time:** 138.30s

## Per-Image Results

| Image | Accuracy | Date Format | Rows | Time |
|-------|----------|-------------|------|------|
| image_003.png | 100.0% | balance_description_2turn | 6 | 55.4s |
| image_009.png | 96.4% | balance_description_2turn | 15 | 100.1s |
| image_008.png | 82.3% | balance_description_2turn | 30 | 259.4s |

## Field-Level Accuracy

| Field | Avg F1 |
|-------|--------|
| DOCUMENT_TYPE | 100.0% |
| STATEMENT_DATE_RANGE | 100.0% |
| TRANSACTION_DATES | 96.5% |
| LINE_ITEM_DESCRIPTIONS | 78.3% |
| TRANSACTION_AMOUNTS_PAID | 89.6% |