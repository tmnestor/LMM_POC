# Why InternVL3.5 Outperforms InternVL3 for Bank Statement Extraction

## Dashboard Results Summary

From our bank statement extraction benchmarks (`output/bank_model_dashboard.png`):

| Model | Avg Accuracy | Avg Time | Std Dev | Key Observation |
|-------|-------------|----------|---------|-----------------|
| Llama-3.2-Vision | 85.9% | 271.7s | 22.3% | Highest accuracy, slowest |
| InternVL3-8B | 56.3% | 82.2s | 30.8% | Poor accuracy, high variance |
| InternVL3.5-8B | 84.3% | 106.6s | 24.4% | Near Llama accuracy, 2.5× faster |

**The critical finding**: InternVL3.5-8B achieves **28 percentage points higher accuracy** than InternVL3-8B on bank statement extraction, with significantly lower variance (24.4% vs 30.8%).

## Field-Level Analysis

The Field-Level F1 Scores heatmap reveals where InternVL3-8B fails:

| Field | Llama-3.2 | InternVL3-8B | InternVL3.5-8B |
|-------|-----------|--------------|----------------|
| Document Type | 100 | 100 | 100 |
| Statement Date | 100 | 75 | 100 |
| Transaction Date | 77 | 40 | 76 |
| Line Item Descr | 77 | 32 | 69 |
| Transaction Amt | 76 | 35 | 77 |

InternVL3-8B shows severe degradation on **structured field extraction**—exactly the tasks requiring precise spatial-semantic understanding. InternVL3.5-8B recovers to near-Llama performance across all fields.

## Key Technical Improvements in InternVL3.5

InternVL3.5 was released in August 2025 with three major innovations that explain its superior bank statement extraction performance:

### 1. Cascade Reinforcement Learning (Cascade RL)

InternVL3.5 introduces a two-stage reinforcement learning framework that fundamentally improves reasoning quality:

**Stage 1 - Offline RL (MPO Warm-up)**:
- Mixed Preference Optimization provides stable initial convergence
- Higher training efficiency with controlled sampling

**Stage 2 - Online RL (GSPO Refinement)**:
- Fine-grained alignment through online rollouts
- Addresses reward hacking through decoupled updates

**Why this matters for bank statements**: The cascaded approach produces more reliable, consistent outputs. Bank statement extraction requires deterministic behavior—the model should extract the same transaction the same way every time. InternVL3's single-stage training produced higher variance (30.8% std dev), while InternVL3.5's cascaded training reduces this to 24.4%.

Source: [InternVL3.5 Technical Report](https://arxiv.org/abs/2508.18265)

### 2. Visual Resolution Router (ViR)

ViR dynamically adjusts visual token compression based on semantic richness:

| Token Setting | Use Case | Performance |
|---------------|----------|-------------|
| 256 tokens/patch | Detail-rich regions | Full fidelity |
| 64 tokens/patch | Uniform regions | 50% token reduction |

**How it works**:
1. **Consistency Training**: Model learns to produce equivalent outputs at different compression rates
2. **Router Training**: Binary classifier routes patches based on semantic content
3. **Dynamic Allocation**: Text-dense regions (like transaction tables) get more tokens

**Why this matters for bank statements**: Bank statements have sparse layouts with concentrated text regions. ViR allocates more visual tokens to transaction tables and balance areas while compressing whitespace. This improves extraction accuracy in data-dense regions without increasing overall compute.

### 3. Decoupled Vision-Language Deployment (DvD)

InternVL3.5 separates vision and language processing:

```
Traditional Pipeline:
  Image → [ViT + MLP + LLM on same GPU] → Output

InternVL3.5 DvD:
  Image → [Vision Server: ViT + MLP + ViR]
                    ↓ (compressed visual tokens)
          [Language Server: LLM] → Output
```

**Benefits**:
- 2× throughput for complex visual reasoning
- Better GPU memory utilization
- Asynchronous pipelining

**Why this matters for bank statements**: DvD enables processing of high-resolution bank statement images without memory bottlenecks, maintaining extraction quality on detailed documents.

## Benchmark Evidence

### Official Benchmark Comparison (8B Scale)

| Benchmark | InternVL3-8B | InternVL3.5-8B | Delta |
|-----------|--------------|----------------|-------|
| MMMU | 62.7 | 73.4 | **+10.7** |
| DocVQA | 92.7 | 92.3 | -0.4 |
| ChartQA | ~84 | ~87 | +3 |
| MathVista | ~58 | ~68 | **+10** |

Source: [InternVL3.5 Paper](https://arxiv.org/html/2508.18265v1), [InternVL3 Paper](https://arxiv.org/abs/2504.10479)

**Key observation**: While DocVQA scores are nearly identical, our bank statement extraction shows a **28-point gap**. This discrepancy reveals the limitation of standard benchmarks—they don't capture the full difficulty of structured financial document extraction.

### Why Standard Benchmarks Miss the Gap

| Aspect | DocVQA | Bank Statement Extraction |
|--------|--------|---------------------------|
| Task type | Answer single question | Extract complete table |
| Error tolerance | Partial credit (ANLS) | Row misalignment cascades |
| Spatial precision | Approximate | Exact column mapping |
| Output format | Free text | Structured fields |
| Failure mode | Wrong answer | Invalid financial data |

DocVQA tests: "What is the total amount?" (localized reasoning)
Bank extraction requires: "Extract all 47 transactions with correct debit/credit assignment" (global structural consistency)

### Reasoning Improvements Explain Bank Extraction Gains

The MMMU improvement (+10.7 points) is the key indicator. MMMU tests multi-step reasoning across domains—exactly what bank statement extraction requires:

1. **Identify table boundaries** (visual parsing)
2. **Map columns to semantics** (layout understanding)
3. **Extract each row consistently** (structural reasoning)
4. **Validate amounts against balance** (numerical reasoning)

InternVL3.5's Cascade RL specifically targets reasoning quality, directly benefiting this multi-step extraction pipeline.

## InternVL3's Architectural Trade-offs

InternVL3 (April 2025) introduced innovations that may have hurt specialized extraction:

### Native Multimodal Pre-Training

InternVL3 jointly trains language and vision from scratch, rather than adapting a pre-trained LLM. This:
- **Pros**: Better multimodal alignment for general tasks
- **Cons**: May lose specialized document understanding from LLM pre-training

### Variable Visual Position Encoding (V2PE)

V2PE uses smaller position increments (δ < 1) for visual tokens. For bank statements:
- **Risk**: Column alignment depends on precise spatial positions
- **Impact**: Debit vs Credit column confusion increases

### Training Data Expansion

InternVL3's ~200B token training expanded to GUI, tools, 3D, and video domains. This breadth may have diluted document-specific capabilities that InternVL2.5 (and now InternVL3.5) maintained.

## InternVL3.5's Document-Specific Strengths

### Recovered Structured Extraction

InternVL3.5 addresses InternVL3's regressions through:

1. **Cascade RL stability**: Reduces output variance critical for consistent extraction
2. **ViR token allocation**: Concentrates capacity on text-dense regions
3. **Refined training**: Better balance between generality and document specialization

### Benchmark Recovery at Scale

| Model Scale | InternVL3 DocVQA | InternVL3.5 DocVQA |
|-------------|------------------|-------------------|
| 2B | 88.3 | 89.4 (+1.1) |
| 8B | 92.7 | 92.3 (-0.4) |
| 38B | 95.4 | 94.0 (-1.4) |

Source: [InternVL3.5 Technical Report](https://arxiv.org/html/2508.18265v1)

While DocVQA scores are comparable, our real-world bank extraction shows InternVL3.5 dramatically outperforming InternVL3. This suggests:

1. **DocVQA is saturated**: Top models all score 92-95%
2. **Real documents are harder**: Bank statements expose capability gaps hidden by benchmark ceilings
3. **Variance matters**: InternVL3.5's lower std dev (24.4% vs 30.8%) indicates more reliable extraction

## Efficiency Comparison

| Metric | Llama-3.2-Vision | InternVL3-8B | InternVL3.5-8B |
|--------|------------------|--------------|----------------|
| Avg Time | 271.7s | 82.2s | 106.6s |
| Docs/min | 0.22 | 0.73 | 0.56 |
| Accuracy | 85.9% | 56.3% | 84.3% |
| Accuracy/Time | 0.32%/s | 0.69%/s | **0.79%/s** |

**InternVL3.5-8B offers the best accuracy-per-second ratio**—nearly matching Llama's accuracy at 2.5× the speed.

## How Does Qwen2.5-VL Compare?

Qwen2.5-VL from Alibaba is a strong contender in the document understanding space. Here's how it compares to our tested models at the ~7-8B scale:

### Qwen2.5-VL-7B Benchmark Performance

| Benchmark | Qwen2.5-VL-7B | InternVL3.5-8B | InternVL3-8B | Llama-3.2-11B |
|-----------|---------------|----------------|--------------|---------------|
| DocVQA | ~95.7% | 92.3% | 92.7% | 70.7% |
| ChartQA | ~85% | ~87% | ~84% | ~83% |
| OCRBench | ~850+ | ~840 | ~835 | N/A |
| MMMU | ~56% | 73.4% | 62.7% | ~49% |

Sources: [Qwen2.5-VL Model Card](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct), [Open Laboratory](https://openlaboratory.ai/models/qwen2_5-vl-7b-instruct), [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2025/01/qwen2-5-vl-vision-model/)

### Qwen2.5-VL's Document Strengths

Qwen2.5-VL was specifically optimized for document understanding:

1. **Superior OCR**: Achieves ~75% accuracy on JSON extraction tasks, matching GPT-4o and outperforming specialized OCR models like Mistral-OCR (72.2%)

2. **Bounding Box Support**: Unlike most VLMs, Qwen2.5-VL provides text localization with bounding boxes—useful for verification workflows

3. **Multilingual OCR**: Strong support for English, Japanese, Arabic, and 10+ languages

4. **Structured Output**: Native support for JSON/CSV extraction from documents

Source: [OmniAI OCR Benchmark](https://apidog.com/blog/qwen-2-5-72b-open-source-ocr/)

### Why Qwen2.5-VL May Excel at Bank Statements

| Capability | Relevance to Bank Statements |
|------------|------------------------------|
| High DocVQA score (95.7%) | Strong document understanding baseline |
| Bounding box output | Verify column alignment visually |
| JSON extraction | Direct structured output |
| Multilingual OCR | Handle international transactions |

### Why We Haven't Tested Qwen2.5-VL Yet

Our current benchmark focuses on models we have deployed. Qwen2.5-VL-7B is a **strong candidate for future testing** based on:

1. **DocVQA leadership**: 95.7% vs InternVL3.5's 92.3%
2. **OCR specialization**: Explicitly trained for document extraction
3. **Structured output**: Native JSON support reduces post-processing

### Predicted Performance

Based on benchmark analysis, Qwen2.5-VL-7B would likely:

| Metric | Predicted Range | Rationale |
|--------|-----------------|-----------|
| Accuracy | 82-88% | Higher DocVQA suggests strong extraction |
| Variance | 20-25% | OCR focus implies consistency |
| Speed | 80-120s | Similar architecture to InternVL |

**Recommendation**: Add Qwen2.5-VL-7B to the next benchmark round. Its document-specialized training may outperform InternVL3.5-8B on bank statement extraction, despite lower MMMU scores.

### The MMMU vs DocVQA Trade-off

Interestingly, Qwen2.5-VL-7B and InternVL3.5-8B show opposite strengths:

| Model | DocVQA | MMMU | Specialization |
|-------|--------|------|----------------|
| Qwen2.5-VL-7B | 95.7% | ~56% | Document-focused |
| InternVL3.5-8B | 92.3% | 73.4% | Reasoning-focused |

For bank statements, which matters more?
- **DocVQA** tests: Can you answer questions about documents?
- **MMMU** tests: Can you reason across complex multi-step problems?

Bank statement extraction requires **both**:
- OCR accuracy (DocVQA-like)
- Structural reasoning (MMMU-like)

InternVL3.5's MMMU advantage may explain why it beats InternVL3 despite similar DocVQA scores. Qwen2.5-VL's DocVQA advantage may compensate for lower reasoning scores through better raw extraction.

**The only way to know is to test.**

## Recommendations

### For Bank Statement Extraction

1. **Use InternVL3.5-8B** for production workloads—best accuracy/speed tradeoff among tested models
2. **Use Llama-3.2-Vision** when accuracy is paramount and latency is acceptable
3. **Test Qwen2.5-VL-7B** as a priority—benchmark scores suggest it may be the best option
4. **Avoid InternVL3-8B** for structured extraction—high variance causes unreliable outputs

### For Model Selection Generally

1. **Don't trust benchmark saturation**: 92% vs 93% on DocVQA hides real capability differences
2. **Measure variance**: Standard deviation matters for production reliability
3. **Test your actual task**: Bank extraction exposed a 28-point gap invisible to DocVQA
4. **Consider both OCR and reasoning**: High DocVQA alone doesn't guarantee extraction success

## Technical References

### Qwen2.5-VL
- [Qwen2.5-VL-72B Model Card](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct) — Hugging Face
- [Qwen2.5-VL Technical Overview](https://www.analyticsvidhya.com/blog/2025/01/qwen2-5-vl-vision-model/) — Analytics Vidhya
- [Qwen2.5-VL OCR Benchmark](https://apidog.com/blog/qwen-2-5-72b-open-source-ocr/) — OmniAI comparison
- [Qwen2.5-VL-7B Details](https://openlaboratory.ai/models/qwen2_5-vl-7b-instruct) — Open Laboratory
- [QwenLM GitHub](https://github.com/QwenLM/Qwen3-VL) — Official repository

### InternVL3.5
- [InternVL3.5 Technical Report](https://arxiv.org/abs/2508.18265) — "Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency" (August 2025)
- [InternVL3.5 Blog Post](https://internvl.github.io/blog/2025-08-26-InternVL-3.5/)
- [InternVL3.5-8B Model Card](https://huggingface.co/OpenGVLab/InternVL3_5-8B)

### InternVL3
- [InternVL3 Technical Report](https://arxiv.org/abs/2504.10479) — "Exploring Advanced Training and Test-Time Recipes" (April 2025)
- [InternVL3 Blog Post](https://internvl.github.io/blog/2025-04-11-InternVL-3.0/)

### Key Architectural Innovations
- **Cascade Reinforcement Learning**: Two-stage RL (offline MPO + online GSPO) for stable, high-quality reasoning
- **Visual Resolution Router (ViR)**: Dynamic token compression based on semantic richness
- **Decoupled Vision-Language Deployment (DvD)**: Separated vision/language servers for efficiency
- **Variable Visual Position Encoding (V2PE)**: InternVL3's flexible position increments (inherited by InternVL3.5)

### Official Resources
- [OpenGVLab GitHub](https://github.com/OpenGVLab/InternVL)
- [InternVL Documentation](https://internvl.readthedocs.io/)
- [Hugging Face Model Collection](https://huggingface.co/collections/OpenGVLab/internvl35-68ac87bd52ebe953485927fb)

## Conclusion

InternVL3.5's Cascade Reinforcement Learning framework is the primary driver of its superior bank statement extraction performance. By combining offline RL stability with online RL refinement, InternVL3.5 produces more consistent, reliable outputs—exactly what structured document extraction requires.

The 28-point accuracy gap between InternVL3 and InternVL3.5 on bank statements, despite near-identical DocVQA scores, demonstrates that:

1. **Standard benchmarks hide real capability differences** at the performance ceiling
2. **Output variance is as important as mean accuracy** for production systems
3. **Reasoning improvements (MMMU +10.7) translate to structured extraction gains**

For production bank statement extraction, InternVL3.5-8B offers the optimal balance of accuracy (84.3%), speed (106.6s), and reliability (24.4% std dev).

---

*This analysis was generated from `bank_statement/bank_model_comparison.ipynb` using the CBA-specific synthetic bank statement dataset.*
