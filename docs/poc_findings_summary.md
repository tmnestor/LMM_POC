# LMM Proof of Concept: Findings

## How to Read These Results

The charts use **F1 score** to measure extraction quality. F1 balances two things that matter in practice: **completeness** (did the model find all the fields?) and **trustworthiness** (is what it extracted actually correct?). A score of 100% means perfect extraction; 0% means complete failure. The scores represent the average across all fields extracted from each document -- dates, amounts, names, totals, and so on.

## Executive Summary

We evaluated three open-source vision-language models for automated document information extraction across bank statements, invoices, and receipts. **InternVL3.5-8B is the recommended model**, delivering the highest accuracy with the most consistent performance across all document types.

## Key Findings

### 1. InternVL3.5-8B delivers the best accuracy

The left box plot shows that InternVL3.5-8B achieves the highest median accuracy (~75%) with the **tightest distribution** -- meaning it performs reliably well across different documents rather than excelling on some and failing on others.

By contrast, Llama-11B -- despite being 40% larger -- shows the widest spread in accuracy, with some documents scoring below 20%. This inconsistency makes it a riskier choice for production use.

**Why Llama-11B scored poorly: hallucination.** Our evaluation penalises models that fabricate information. When a field genuinely does not appear on a document (e.g. no ABN printed on a receipt), the correct answer is "not found". InternVL3.5-8B reliably returns "not found" in these cases and earns the score. Llama-11B instead invents plausible-looking values -- an ABN that doesn't exist, a payer name that appears nowhere on the page -- and is penalised for every one. This also explains the wide variance in Llama's results: documents with more absent fields trigger more hallucinations, producing the low outliers visible in the box plot.

### 2. The accuracy advantage holds across all document types

The upper-right bar chart confirms InternVL3.5-8B (red) leads on every document category. Invoices and receipts show the clearest advantage. Bank statements are the most challenging type for all models due to complex multi-row transaction tables.

**Why bank statements are the hardest: row alignment.** A bank statement is a table where each row is a transaction. The extracted dates, descriptions, and amounts are separate lists -- but they must line up. The 5th date, the 5th description, and the 5th amount must all describe the same transaction. If the model drops one description but extracts all the dates and amounts, every row after the gap is misaligned -- the wrong amount gets attributed to the wrong transaction on the wrong date. The individual values might all be correct in isolation, but the reconstructed transaction ledger is wrong. Our evaluation measures this alignment, which is why bank statement scores are lower even when the model reads most of the text correctly.

### 3. Processing speed is comparable across models

The lower-right bar chart shows all three models process invoices and receipts in **25-30 seconds per page**. Bank statements take longer (65-80 seconds) due to the multi-turn extraction required for transaction tables.

InternVL3.5-8B processes documents at **1.6 pages/minute**, nearly matching the smaller InternVL3-2B (1.7 pages/min). The larger model size does not impose a meaningful speed penalty.

## Recommendation

**Deploy InternVL3.5-8B** as the production model. It offers:

- **Best accuracy** across all three document types
- **Most consistent results** (lowest variance between documents)
- **No speed trade-off** compared to the smaller model
- **Smaller than Llama-11B** -- lower GPU memory cost with better results

---

## InternVL3.5-8B vs LayoutLM: Head-to-Head Comparison

### Context

The previous section compared three candidate LMMs against each other. This section compares the winning model -- **InternVL3.5-8B** -- against the **existing Document Understanding (DU) pipeline** built on LayoutLM, a smaller specialised model (~381 million parameters) currently used in production.

### Headline Result

The left summary table shows that InternVL3.5-8B delivers an **18 percentage-point improvement** in overall extraction quality (72.5% vs 54.6%) and a **31 percentage-point improvement** on business-critical fields (82.6% vs 51.7%) -- at the cost of roughly half the throughput (1.6 vs 3.5 pages/min).

### Field-by-Field Breakdown

The right lollipop chart shows the F1 difference for every extracted field. Green bars (positive) mean InternVL3.5-8B is better; red bars (negative) mean LayoutLM is better. Larger dots indicate business-critical fields.

**InternVL3.5-8B outperforms LayoutLM on 13 of 16 fields.** The largest gains are on fields that require the model to understand spatial layout, read small print, or cross-reference multiple parts of a document -- areas where LayoutLM struggles most. These include business-critical fields like ABNs (+64.7%), GST amounts (+42.2%), and transaction dates (+35.6%).

**LayoutLM leads on only 3 fields** at the bottom of the chart (line item totals, descriptions, and payer name). These are candidates for improvement through prompt engineering rather than model changes.

### The Speed vs Accuracy Trade-off

InternVL3.5-8B is slower because it is **21x larger** than LayoutLM (8 billion vs 381 million parameters). This is expected and represents a deliberate trade-off:

- **Batch processing workloads** (overnight runs, bulk imports) are not time-sensitive -- the accuracy gain far outweighs the speed difference
- **Multi-GPU deployment** (as demonstrated in this POC with 4x NVIDIA L4 GPUs) scales throughput linearly, closing the gap for higher-volume scenarios
- **No retraining required** -- LayoutLM needs fine-tuning on labelled data for each new document format, while InternVL3.5-8B works out of the box with prompt-based configuration

### Bottom Line

InternVL3.5-8B extracts more complete and more trustworthy data from documents than the current DU pipeline, particularly for the fields that matter most to downstream business processes. The throughput trade-off is manageable for batch workloads and can be offset with additional GPU capacity.
