# Executive Model Comparison Report

**Generated**: 2025-09-15 04:25:03

## Performance Dashboard

![Executive Performance Comparison](../visualizations/executive_comparison_20250915_042501.png)

## Executive Summary

### Llama-3.2-Vision
- **Average Accuracy**: 81.3%
- **Average Processing Time**: 53.4 seconds
- **Throughput**: 1.1 documents per minute
- **Documents Processed**: 9

### InternVL3-8B
- **Average Accuracy**: 70.8%
- **Average Processing Time**: 28.2 seconds
- **Throughput**: 2.1 documents per minute
- **Documents Processed**: 9

### InternVL3-2B
- **Average Accuracy**: 50.8%
- **Average Processing Time**: 10.3 seconds
- **Throughput**: 5.8 documents per minute
- **Documents Processed**: 9

## Document Type Performance

| document_type   |   InternVL3-2B |   InternVL3-8B |   Llama-3.2-Vision |
|:----------------|---------------:|---------------:|-------------------:|
| bank_statement  |        23.8095 |        44.7619 |            53.3333 |
| invoice         |        66.6667 |        80.9524 |            97.1429 |
| receipt         |        61.9048 |        86.6667 |            93.3333 |

## Key Findings

- **Accuracy Leader**: Llama-3.2-Vision
- **Speed Leader**: InternVL3-2B
- **Best for Invoices**: Llama-3.2-Vision
- **Best for Receipts**: Llama-3.2-Vision
- **Best for Bank Statements**: Llama-3.2-Vision


## Recommendations

Detailed recommendations and analysis available in the full comparison notebook.
