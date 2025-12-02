# Unified Bank Statement Batch Evaluation Report

**Generated:** 2025-12-02 04:00:40
**Batch ID:** 20251202_035924
**Model:** InternVL3.5-8B (bfloat16)
**Evaluation Method:** order_aware_f1

## Executive Summary

- **Total Images:** 1
- **Successful:** 1 (100.0%)
- **Failed:** 0

- **Average Accuracy:** 40.0%
- **Min Accuracy:** 40.0%
- **Max Accuracy:** 40.0%
- **Avg Processing Time:** 70.78s

## Per-Image Results

| Image | Accuracy | Date Format | Rows | Time |
|-------|----------|-------------|------|------|
| transaction_summary.png | 40.0% | amount_description_2turn | 4 | 70.8s |

## Field-Level Accuracy

| Field | Avg F1 |
|-------|--------|
| DOCUMENT_TYPE | 100.0% |
| STATEMENT_DATE_RANGE | 100.0% |
| TRANSACTION_DATES | 0.0% |
| LINE_ITEM_DESCRIPTIONS | 0.0% |
| TRANSACTION_AMOUNTS_PAID | 0.0% |