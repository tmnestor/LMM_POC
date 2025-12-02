# Unified Bank Statement Batch Evaluation Report

**Generated:** 2025-12-02 05:01:38
**Batch ID:** 20251202_050030
**Model:** InternVL3.5-8B (bfloat16)
**Evaluation Method:** order_aware_f1

## Executive Summary

- **Total Images:** 1
- **Successful:** 1 (100.0%)
- **Failed:** 0

- **Average Accuracy:** 57.1%
- **Min Accuracy:** 57.1%
- **Max Accuracy:** 57.1%
- **Avg Processing Time:** 62.05s

## Per-Image Results

| Image | Accuracy | Date Format | Rows | Time |
|-------|----------|-------------|------|------|
| cba_debit_credit.png | 57.1% | debit_credit_description_2turn | 0 | 62.0s |

## Field-Level Accuracy

| Field | Avg F1 |
|-------|--------|
| DOCUMENT_TYPE | 100.0% |
| STATEMENT_DATE_RANGE | 0.0% |
| TRANSACTION_DATES | 0.0% |
| LINE_ITEM_DESCRIPTIONS | 88.9% |
| TRANSACTION_AMOUNTS_PAID | 96.6% |