# Appendix A: Bank Statement-Specific Investigation

**Date**: December 2025
**Dataset**: 12 synthetic CBA (Commonwealth Bank) bank statement images
**Notebook**: `bank_statement/bank_model_comparison.ipynb`
**Test Environment**: Single L40 GPU (24GB VRAM)

---

## A.1 Overview

This appendix documents a focused investigation into bank statement extraction performance, using a **synthetic CBA-specific dataset** designed to test challenging extraction scenarios including:

- Date-grouped transactions
- Multi-line transaction descriptions
- Debit/credit column alignment
- Balance reconciliation
- Reverse chronological ordering

**Hardware Limitation**: The single L40 GPU (24GB VRAM) constrained InternVL3 models to **12 image patches maximum**, reducing their effective resolution. This compromises both InternVL3-8B and InternVL3.5-8B performance—results would likely improve with more VRAM allowing higher patch counts. Despite this limitation, InternVL3.5 still dramatically outperforms InternVL3, demonstrating the value of Cascade RL training.

---

## A.2 Models Tested

| Model | Parameters | Key Feature | Constraint |
|-------|------------|-------------|------------|
| **Llama-3.2-Vision** | 11B | Built-in preprocessing | Full resolution |
| **InternVL3-8B** | 8B | Native Multimodal Pre-training, V2PE | **12 patches max** ⚠️ |
| **InternVL3.5-8B** | 8B | **Cascade RL**, Visual Resolution Router | **12 patches max** ⚠️ |

### Key Difference: InternVL3 vs InternVL3.5

InternVL3.5 (August 2025) introduced **Cascade Reinforcement Learning**, a two-stage training approach:

1. **Stage 1 - Offline RL (MPO)**: Stable convergence with controlled sampling
2. **Stage 2 - Online RL (GSPO)**: Fine-grained alignment through online rollouts

This produces more consistent, reliable outputs—critical for structured document extraction where variance is as important as mean accuracy.

---

## A.3 Results Summary

| Model | Avg Accuracy | Avg Time | Std Dev | Docs/min |
|-------|-------------|----------|---------|----------|
| **Llama-3.2-Vision** | **85.9%** 🥇 | 271.7s | 22.3% | 0.22 |
| **InternVL3.5-8B** | **84.3%** 🥈 | 106.6s | 24.4% | 0.56 |
| **InternVL3-8B** | 56.3% | 82.2s | 30.8% | 0.73 |

### Critical Finding

**InternVL3.5-8B achieves 28 percentage points higher accuracy than InternVL3-8B** on bank statement extraction, despite:
- Near-identical DocVQA benchmark scores (~92%)
- Same model architecture (ViT-MLP-LLM)
- Same parameter count (8B)

This demonstrates that **standard benchmarks fail to capture real-world structured extraction difficulty**.

---

## A.4 Field-Level Performance

| Field | Llama-3.2 | InternVL3-8B | InternVL3.5-8B |
|-------|-----------|--------------|----------------|
| Document Type | 100% | 100% | 100% |
| Statement Date | 100% | 75% | 100% |
| Transaction Date | 77% | 40% | 76% |
| Line Item Descr | 77% | 32% | 69% |
| Transaction Amt | 76% | 35% | 77% |

**Analysis**: InternVL3-8B shows severe degradation on structured field extraction (Transaction Date: 40%, Line Item: 32%, Amount: 35%). InternVL3.5's Cascade RL recovers performance to near-Llama levels across all fields.

---

## A.5 Why InternVL3 Struggles

InternVL3 (April 2025) introduced architectural changes that may have hurt specialized extraction:

### Native Multimodal Pre-Training
- **Approach**: Joint language+vision training from scratch
- **Trade-off**: Optimizes for general multimodal understanding, may dilute document-specific capabilities

### Variable Visual Position Encoding (V2PE)
- **Approach**: Smaller position increments (δ < 1) for visual tokens
- **Risk**: Column alignment in bank statements depends on precise spatial positions
- **Impact**: Debit vs Credit column confusion increases

### Training Data Expansion
- InternVL3 expanded to ~200B tokens covering GUI, tools, 3D, video
- This breadth may have diluted document-specific capabilities

---

## A.6 Why InternVL3.5 Recovers

InternVL3.5's **Cascade RL** addresses InternVL3's regressions:

| Improvement | Mechanism | Impact on Bank Statements |
|-------------|-----------|---------------------------|
| **Lower variance** | Two-stage RL stabilizes outputs | Same transaction extracted consistently |
| **Better reasoning** | MMMU +10.7 points | Multi-step extraction pipeline works |
| **Token allocation** | Visual Resolution Router | More tokens to transaction tables |

### Benchmark Evidence

| Benchmark | InternVL3-8B | InternVL3.5-8B | Delta |
|-----------|--------------|----------------|-------|
| MMMU | 62.7% | 73.4% | **+10.7** |
| DocVQA | 92.7% | 92.3% | -0.4 |
| MathVista | ~58% | ~68% | **+10** |

The **MMMU improvement** (+10.7 points) is the key predictor—it tests multi-step reasoning, exactly what bank statement extraction requires.

---

## A.7 Efficiency Analysis

| Metric | Llama-3.2-Vision | InternVL3-8B | InternVL3.5-8B |
|--------|------------------|--------------|----------------|
| Avg Time | 271.7s | 82.2s | 106.6s |
| Accuracy | 85.9% | 56.3% | 84.3% |
| **Accuracy/Time** | 0.32%/s | 0.69%/s | **0.79%/s** 🏆 |

**InternVL3.5-8B offers the best accuracy-per-second ratio**—nearly matching Llama's accuracy at 2.5× the speed.

---

## A.8 Problem Documents

Documents where at least one model scored below 85%:

| Document | Issue | Best Model |
|----------|-------|------------|
| `cba_highlighted.png` | Highlighted regions confuse extraction | Llama (100%) |
| `transaction_summary.png` | Summary format vs transaction table | Llama (100%) |
| `cba_date_grouped_cont.png` | Continuation page, no headers | InternVL3.5 (50%) |
| `synthetic_reverse_chrono.png` | Reverse date ordering | All struggle (~43%) |
| `image_009.png` | Complex multi-section layout | All struggle (~42%) |

**Key Insight**: All models struggle with reverse chronological ordering and continuation pages—these may require prompt engineering or post-processing solutions.

---

## A.9 Recommendations

### For Bank Statement Production Workloads

1. **Best Accuracy**: Llama-3.2-Vision (85.9%)
   - Use when accuracy is paramount and latency is acceptable
   - 271.7s per document

2. **Best Value**: InternVL3.5-8B (84.3%)
   - Near-Llama accuracy at 2.5× speed
   - Best accuracy/time ratio (0.79%/s)
   - 106.6s per document

3. **Avoid**: InternVL3-8B
   - High variance (30.8% std dev) causes unreliable outputs
   - 28 points lower accuracy than InternVL3.5

### Future Testing

- **Qwen2.5-VL-7B**: Higher DocVQA (95.7%) suggests strong OCR—may outperform on bank statements
- **Prompt optimization**: Address reverse chronological and continuation page failures
- **Adapter-based post-processing**: Bank-specific adapters for CBA, ANZ, NAB, Westpac

---

## A.10 References

Wang, W., Gao, Z., Gu, L., Pu, H., Cui, L., Wei, X., Liu, Z., Jing, L., Ye, S., Shao, J., Wang, Z., Chen, Z., Zhang, H., Yang, G., Wang, H., Wei, Q., et al. (2025). InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency. *arXiv preprint arXiv:2508.18265*.

Zhu, J., Wang, W., Chen, Z., Liu, Z., Ye, S., Gu, L., Tian, H., Duang, Y., Su, W., Shao, J., Gao, Z., Cui, E., Wang, X., Cao, Y., Liu, Y., Wei, X., et al. (2025). InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models. *arXiv preprint arXiv:2504.10479*.

---

## A.11 Visualizations

Generated by `bank_statement/bank_model_comparison.ipynb`:

- `output/bank_model_dashboard.png` — Executive summary dashboard
- `output/bank_accuracy_comparison.png` — Accuracy distribution by model
- `output/bank_processing_time.png` — Processing time distribution
- `output/bank_field_f1_heatmap.png` — Field-level F1 scores
- `output/bank_per_document.png` — Per-document accuracy comparison
- `output/bank_accuracy_vs_time.png` — Efficiency scatter plot

---

*Appendix generated December 2025 from CBA-specific synthetic dataset on L40 GPU (24GB VRAM, 12 patch limit for InternVL models)*
