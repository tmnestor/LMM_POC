# Unified Bank Statement Batch Evaluation Report

**Generated:** 2025-12-02 05:07:05
**Batch ID:** 20251202_050557
**Model:** InternVL3.5-8B (bfloat16)
**Evaluation Method:** order_aware_f1

## Executive Summary

- **Total Images:** 1
- **Successful:** 1 (100.0%)
- **Failed:** 0

- **Average Accuracy:** 97.8%
- **Min Accuracy:** 97.8%
- **Max Accuracy:** 97.8%
- **Avg Processing Time:** 62.09s

## Per-Image Results

| Image | Accuracy | Date Format | Rows | Time |
|-------|----------|-------------|------|------|
| cba_debit_credit.png | 97.8% | debit_credit_description_2turn | 15 | 62.1s |

## Field-Level Accuracy

| Field | Avg F1 |
|-------|--------|
| DOCUMENT_TYPE | 100.0% |
| STATEMENT_DATE_RANGE | 100.0% |
| TRANSACTION_DATES | 100.0% |
| LINE_ITEM_DESCRIPTIONS | 88.9% |
| TRANSACTION_AMOUNTS_PAID | 100.0% |