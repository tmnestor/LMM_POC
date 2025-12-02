# Unified Bank Statement Batch Evaluation Report

**Generated:** 2025-12-02 04:33:26
**Batch ID:** 20251202_043008
**Model:** InternVL3.5-8B (bfloat16)
**Evaluation Method:** order_aware_f1

## Executive Summary

- **Total Images:** 1
- **Successful:** 1 (100.0%)
- **Failed:** 0

- **Average Accuracy:** 78.9%
- **Min Accuracy:** 78.9%
- **Max Accuracy:** 78.9%
- **Avg Processing Time:** 192.10s

## Per-Image Results

| Image | Accuracy | Date Format | Rows | Time |
|-------|----------|-------------|------|------|
| nab_classic_highligted.png | 78.9% | balance_description_2turn | 17 | 192.1s |

## Field-Level Accuracy

| Field | Avg F1 |
|-------|--------|
| DOCUMENT_TYPE | 100.0% |
| STATEMENT_DATE_RANGE | 100.0% |
| TRANSACTION_DATES | 97.1% |
| LINE_ITEM_DESCRIPTIONS | 0.0% |
| TRANSACTION_AMOUNTS_PAID | 97.1% |