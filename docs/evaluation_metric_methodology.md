# Evaluation Metric Methodology

## How the Headline Accuracy Metric Is Calculated

The evaluation pipeline computes a single accuracy number per model through a three-layer aggregation. Each layer makes a deliberate design choice.

### Layer 1: Per-Field F1 Score

Each image is classified by document type (invoice, receipt, bank statement, travel expense), which determines **which fields** are evaluated. Field lists are loaded from `config/field_definitions.yaml` -- for example, an invoice has 14 fields, a travel expense has 9.

For each field, the extracted value is compared to ground truth using **position-aware F1**:

| Field Type | Comparison Method | Score Range |
|---|---|---|
| Single-value (e.g. `SUPPLIER_NAME`) | Normalised string match | Binary 0.0 or 1.0 |
| Boolean (e.g. `IS_GST_INCLUDED`) | Parsed boolean equality | Binary 0.0 or 1.0 |
| Date (e.g. `INVOICE_DATE`) | Semantic date comparison (handles format variation) | Binary 0.0 or 1.0 |
| Monetary (e.g. `TOTAL_AMOUNT`) | Numeric comparison after stripping currency symbols | Binary 0.0 or 1.0 |
| List (e.g. `LINE_ITEM_PRICES`) | Position-aware F1 over pipe-delimited items | Continuous 0.0 to 1.0 |

For list fields, items are delimited by `|` and matched **positionally** -- an item must match both value and position to count as a true positive:

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

Each field produces an `f1_score` in [0.0, 1.0].

### Layer 2: Per-Image Score

For an image with *N* evaluated fields:

```
image_score = (1 / N) * sum(field_f1_scores)
```

This is an **unweighted arithmetic mean** across the *N* fields for that image. Every field counts equally -- a list field with 20 transaction items has the same weight as a single-value field like `SUPPLIER_NAME`.

The number of fields *N* varies by document type, so an invoice (14 fields) and a travel expense (9 fields) are averaged over different denominators.

### Layer 3: Per-Model Score (the Headline Number)

```
model_score = (1 / M) * sum(image_scores)
```

where *M* is the number of images processed by that model. This is an **unweighted arithmetic mean across images**. Every image contributes equally regardless of how many fields it has or its document type.

### In Summary

```
Headline Score = mean across images of (mean across fields of (position-aware F1 per field))
```

A mean of means, with equal weight per image and equal weight per field within each image.

## Robustness Variants

The pipeline also computes three additional aggregations for comparison:

| Metric | Formula | Sensitivity |
|---|---|---|
| Mean of Means | `mean(per-image mean F1)` | Headline number; sensitive to outlier images |
| Median of Means | `median(per-image mean F1)` | Robust to outlier images |
| Mean of Medians | `mean(per-image median F1)` | Robust to outlier fields within each image |
| Median of Medians | `median(per-image median F1)` | Most robust; resistant to outliers at both levels |

When these four measures diverge significantly, it indicates skew caused by a small number of problematic images or fields.

## Why Per-Image Aggregation, Not Per-Field

An alternative approach would pool all field scores across all images into a single set and compute one grand mean. There are several reasons we chose per-image aggregation instead.

### 1. Operational Relevance

The unit of work in production is **a document**, not a field. Stakeholders ask "how reliably does the system process a document?" -- not "what is the average accuracy of the SUPPLIER_NAME field?" Per-image scoring directly answers the operational question: each document gets a pass/fail quality score, and the model score tells you what to expect for the next document.

### 2. Equal Representation Across Document Types

Document types have different field counts (e.g. invoices have 14 fields, travel expenses have 9). Per-field pooling would give invoices ~56% more influence on the headline number than travel expenses, purely because of schema size. Per-image aggregation gives every document equal weight, preventing the metric from being dominated by whichever document type has the most fields.

### 3. Avoiding Simpson's Paradox

Per-field pooling can mask systematic failures on specific document types. Consider a dataset with 100 invoices (14 fields each = 1,400 field scores) and 10 bank statements (5 fields each = 50 field scores). If the model scores 95% on invoices but 20% on bank statements, the pooled per-field average would be approximately 92.4% -- hiding a critical failure mode. Per-image aggregation would report 88.2%, which better reflects that a meaningful fraction of documents are poorly handled.

### 4. Consistent Comparison Across Models

Different models may handle different document types with varying field coverage. Per-image aggregation ensures that model comparison is not confounded by differences in which document types happen to have more fields. Each model is judged on how well it handles the average document in the evaluation set.

### 5. Alignment with Downstream Use

In a production pipeline, downstream consumers (human reviewers, ERP systems) process documents one at a time. A per-image score maps directly to the expected quality of each individual handoff. Per-field scores are useful for **diagnostic purposes** (identifying which fields need prompt tuning), but per-image scores are the right unit for **system-level evaluation**.

### When Per-Field Analysis Is Appropriate

Per-field aggregation is not discarded -- it serves a different purpose:

- **Prompt engineering**: Per-field scores identify which fields have low extraction quality, guiding targeted prompt improvements.
- **Model specialisation**: Per-field comparison across models reveals which model is strongest at each field, informing ensemble strategies.
- **Schema design**: Fields with consistently low F1 across all models may indicate ambiguous ground truth or poorly defined extraction targets.

These per-field breakdowns are available in the field-level analytics and comparison dashboards, complementing the per-image headline metric.
