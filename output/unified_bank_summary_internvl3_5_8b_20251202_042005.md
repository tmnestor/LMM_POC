# Unified Bank Statement Batch Evaluation Report

**Generated:** 2025-12-02 04:22:54
**Batch ID:** 20251202_042005
**Model:** InternVL3.5-8B (bfloat16)
**Evaluation Method:** order_aware_f1

## Executive Summary

- **Total Images:** 2
- **Successful:** 2 (100.0%)
- **Failed:** 0

- **Average Accuracy:** 100.0%
- **Min Accuracy:** 100.0%
- **Max Accuracy:** 100.0%
- **Avg Processing Time:** 79.49s

## Per-Image Results

| Image | Accuracy | Date Format | Rows | Time |
|-------|----------|-------------|------|------|
| transaction_summary.png | 100.0% | amount_description_2turn | 4 | 73.8s |
| cba_amount_balance.png | 100.0% | balance_description_2turn | 12 | 85.2s |

## Field-Level Accuracy

| Field | Avg F1 |
|-------|--------|
| DOCUMENT_TYPE | 100.0% |
| STATEMENT_DATE_RANGE | 100.0% |
| TRANSACTION_DATES | 100.0% |
| LINE_ITEM_DESCRIPTIONS | 100.0% |
| TRANSACTION_AMOUNTS_PAID | 100.0% |