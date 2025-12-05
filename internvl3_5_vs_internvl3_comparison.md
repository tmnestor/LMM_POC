# InternVL3.5 vs InternVL3: Comparison for Document Processing

**Date:** December 2025
**Purpose:** Evaluate upgrade path from InternVL3 to InternVL3.5 for bank statement extraction

---

## Executive Summary

InternVL3.5 (released August 2025) offers **incremental improvements** over InternVL3 (April 2025) for OCR tasks, with more significant gains in document understanding and reasoning. For pure digit/number extraction, improvements are modest (~1 point on OCRBench). The main benefits are in complex table reasoning, document comprehension, and inference efficiency.

---

## Benchmark Comparison

### OCR Performance (OCRBench)

| Model | OCRBench Score | Change |
|-------|----------------|--------|
| InternVL3-2B | 835 | - |
| InternVL3.5-2B | 836 | +1 |
| InternVL3-78B | 906 | - |
| InternVL3.5-241B | 907 | +1 |

**Assessment:** Marginal improvement. InternVL3 already achieves excellent OCR scores.

### Document Understanding (DocVQA)

| Model | DocVQA Score | Change |
|-------|--------------|--------|
| InternVL3-2B | 88.3 | - |
| InternVL3.5-2B | 89.4 | +1.1 |
| InternVL3.5-38B | 94.0 | - |
| InternVL3.5-241B | 94.9 | - |

**Assessment:** More meaningful gains in document comprehension tasks.

### Chart/Table Understanding (ChartQA)

| Model | ChartQA Score |
|-------|---------------|
| InternVL3.5-2B | 80.7 |
| InternVL3.5-4B | 86.0 |

**Assessment:** Significant improvement (+5.3 points) with larger models for structured visual data.

---

## Key Architectural Improvements in InternVL3.5

### 1. Cascade Reinforcement Learning (Cascade RL)

Two-stage training process:
- **Stage 1 (Offline RL):** Stable convergence on reasoning tasks
- **Stage 2 (Online RL):** Refined alignment with human preferences

**Impact:** +16% gain in overall reasoning performance on benchmarks like MMMU and MathVista.

### 2. Visual Resolution Router (ViR)

Dynamically adjusts visual token resolution per-patch based on semantic richness:
- Reduces token count by **up to 50%**
- Maintains nearly full performance
- Patch-aware routing based on information density

**Impact:** Faster inference, lower memory usage, cost reduction for high-volume processing.

### 3. Decoupled Vision-Language Deployment (DvD)

Separates vision encoder and language model across different GPUs:
- More efficient resource utilization
- Better scaling for production deployments

---

## Relevance to Bank Statement Extraction

### What InternVL3.5 Improves

| Capability | Improvement Level | Notes |
|------------|-------------------|-------|
| Pure digit extraction | Minimal | InternVL3 already excellent |
| Table structure understanding | Moderate | Better reasoning about cell relationships |
| Document comprehension | Moderate | +1.1 points on DocVQA |
| Processing speed | Significant | 50% token reduction with ViR |
| Memory efficiency | Significant | DvD deployment strategy |

### When to Upgrade

**Recommended if:**
- Processing high volumes (cost/speed matters)
- Complex table layouts with merged cells
- Need reasoning about relationships between values
- Memory constraints on deployment hardware

**Not necessary if:**
- Current InternVL3 extraction accuracy is satisfactory
- Simple table structures
- Low processing volumes

---

## Model Availability

| Model | Parameters | HuggingFace |
|-------|------------|-------------|
| InternVL3.5-1B | 1B | OpenGVLab/InternVL3_5-1B |
| InternVL3.5-2B | 2B | OpenGVLab/InternVL3_5-2B |
| InternVL3.5-4B | 4B | OpenGVLab/InternVL3_5-4B |
| InternVL3.5-8B | 8B | OpenGVLab/InternVL3_5-8B |
| InternVL3.5-38B | 38B | OpenGVLab/InternVL3_5-38B |
| InternVL3.5-78B | 78B | OpenGVLab/InternVL3_5-78B |

---

## Recommendation

For the current bank statement extraction POC:

1. **Short-term:** Continue with InternVL3-8B - proven performance, stable codebase
2. **Evaluation:** Test InternVL3.5-8B on challenging documents (complex tables, poor scans)
3. **Production consideration:** InternVL3.5 with ViR for high-volume deployments (cost savings)

---

## Sources

- [InternVL3.5 Official Blog](https://internvl.github.io/blog/2025-08-26-InternVL-3.5/)
- [InternVL3.5 Paper (arXiv:2508.18265)](https://arxiv.org/abs/2508.18265)
- [InternVL3 Paper (arXiv:2504.10479)](https://arxiv.org/html/2504.10479v1)
- [InternVL3.5-8B on HuggingFace](https://huggingface.co/OpenGVLab/InternVL3_5-8B)
- [InternVL GitHub Repository](https://github.com/OpenGVLab/InternVL)
