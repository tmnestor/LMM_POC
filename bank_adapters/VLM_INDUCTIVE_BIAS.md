# Why Vision Language Models Struggle with Bank Statements

## The Inductive Bias Problem

Vision Language Models (VLMs) like Llama-3.2-Vision and InternVL3 are powerful tools, but their architecture and training encode assumptions—called **inductive biases**—that fundamentally misalign with structured document extraction.

Understanding this mismatch explains why bank statement extraction is so challenging and motivates our adapter-based approach.

## What is Inductive Bias?

Inductive bias refers to the assumptions a learning algorithm makes to generalize from training data to unseen examples. Without these biases, learning would be impossible—you'd just memorize training data with no basis for prediction.

Every architectural choice encodes bias:
- **CNNs** assume translation invariance and local spatial hierarchies
- **Transformers** assume attention-based token relationships are sufficient
- **VLMs** assume dense cross-modal attention between visual and textual features

The No-Free-Lunch theorem tells us no algorithm dominates across all problems. Choosing a model means choosing biases—and hoping they match your domain's structure.

## VLM Training: Natural Images, Not Documents

Most VLMs are pre-trained on **natural image-caption pairs**: photographs of objects, scenes, people, and activities. This training creates biases optimized for:

| Natural Images | Bank Statements |
|----------------|-----------------|
| Continuous color gradients | High-contrast text on white |
| Objects with organic shapes | Rigid tabular structures |
| Semantic meaning from visual content | Semantic meaning from spatial position |
| Global scene understanding | Fine-grained character recognition |
| Captions describe what's visible | Extraction requires structural reasoning |

When a VLM sees a bank statement, it applies biases learned from millions of photos of dogs, landscapes, and street scenes. This is a fundamental mismatch.

## The Spatial-Semantic Gap

In natural images, **what** something is matters more than **where** it is. A dog is a dog whether it's in the left corner or center frame.

In bank statements, **position is meaning**:
- A number under "Debit" means money out
- The same number under "Credit" means money in
- "Balance" at the top is opening; at the bottom is closing
- Column alignment determines which transaction owns which amount

VLMs don't inherently understand that spatial position carries semantic weight in structured documents. Their attention mechanisms can learn this, but it fights against pre-training biases.

## Why Table Extraction Fails

Table extraction asks the VLM to do several things simultaneously:

1. **Detect table boundaries** — Where does the table start and end?
2. **Understand column semantics** — What does each column represent?
3. **Maintain row alignment** — Which cells belong to which row?
4. **Handle exceptions** — Merged cells, wrapped text, multi-line entries

Each step compounds error. A small misalignment in step 3 cascades into completely wrong data in the output.

**Our finding**: 3-turn table extraction has a high failure rate due to hallucinated structure and row misalignment. The VLM's biases work against the precise structural reasoning required.

## Why Balance Extraction Works Better

Balance extraction is a simpler task with clearer visual anchors:
- "Closing Balance" is a recognizable text pattern
- Monetary values near balance labels have consistent formatting
- The task doesn't require maintaining complex structural relationships

**Our finding**: 2-turn balance-based extraction significantly outperforms table extraction. The VLM can focus attention on recognizable patterns rather than reasoning about structure.

## The Big Four Challenge

Australian Big Four bank statements compound the problem with variation across three dimensions:

### Spatial Variation (where data appears)
| Bank | Layout Characteristics |
|------|----------------------|
| ANZ | Dense multi-column layouts |
| CommBank | Cleaner single-flow statements |
| NAB | Variable header/footer structures |
| Westpac | Multiple format versions in circulation |

### Syntactic Variation (how data is formatted)
| Element | Format Variations |
|---------|-------------------|
| Dates | DD/MM/YYYY, DD MMM YYYY, DD-MM-YY |
| Amounts | $1,234.56, 1234.56, (1234.56), 1234.56 CR |
| Debits/Credits | Separate columns, signed values, CR/DR suffix |
| Descriptions | Payee first, reference first, mixed |

### Semantic Variation (what data means)
| Concept | Terminology Variations |
|---------|------------------------|
| Money out | "Withdrawal", "Debit", "DR", "Payment" |
| Money in | "Credit", "Deposit", "CR", "Receipt" |
| Starting balance | "Opening Balance", "Balance B/F", "Previous Balance" |
| Ending balance | "Closing Balance", "Balance C/F", "Current Balance" |

A VLM trained on natural images has no prior knowledge of these conventions. Every statement is essentially a new puzzle.

## The Tradeoff We're Making

| Approach | VLM Burden | Code Burden | Failure Mode |
|----------|-----------|-------------|--------------|
| Pure VLM table extraction | High | Low | Unpredictable hallucinations |
| VLM + Bank-specific adapters | Lower | Higher | Predictable edge cases |

**We're trading unpredictable model failures for predictable code complexity.**

Deterministic bugs are easier to fix than stochastic ones. When a rule-based adapter fails, you can inspect the logic, add a test case, and fix it. When a VLM hallucinates table structure, you're fighting probabilities.

## Our Adapter-Based Solution

Rather than forcing the VLM to overcome its biases, we:

1. **Use VLM strengths** — OCR-like text extraction, visual anchor recognition
2. **Encode structure in code** — Bank-specific adapters know layout conventions
3. **Route by bank type** — Each adapter handles one bank's quirks
4. **Validate deterministically** — Schema-based checking catches errors reliably

```
VLM extraction (bank-agnostic, plays to strengths)
        ↓
Bank detector (rule-based or classifier)
        ↓
Bank-specific adapter
    ├── Spatial normalizer (this bank's layout → canonical positions)
    ├── Syntactic parser (this bank's formats → standard formats)
    └── Semantic mapper (this bank's terminology → canonical terms)
        ↓
Unified output schema
```

## The Deeper Lesson

This isn't a failure of VLMs—it's a recognition of what they're good at and what they're not.

**VLMs excel at:**
- Understanding visual content semantically
- Flexible interpretation of varied inputs
- Handling novel situations through generalization

**VLMs struggle with:**
- Precise structural reasoning
- Position-dependent semantics
- Consistent handling of rigid formats

The right architecture uses each tool for what it does best. VLMs provide flexible extraction; deterministic code provides structural reasoning. The combination outperforms either approach alone.

## Benchmark Evidence

Industry benchmarks confirm that document understanding remains challenging for general-purpose VLMs, even as they excel at natural image tasks.

### DocVQA Performance

[DocVQA](https://rrc.cvc.uab.es/?ch=17) contains 12,000+ document images with 50,000+ questions requiring OCR, layout understanding, and reasoning. The benchmark measures ANLS (Average Normalized Levenshtein Similarity) to handle OCR variations.

| Model | DocVQA Score | Notes |
|-------|-------------|-------|
| GPT-4o | 92.8% | Best proprietary |
| Qwen2.5-VL-7B | 94%+ | Recent open-source leader |
| Llama 3.2-90B Vision | 70.7% | Strong but lags document-specific models |
| InternVL2 | 91%+ | Competitive with proprietary |

Source: [LearnOpenCV VLM Benchmarks](https://learnopencv.com/vlm-evaluation-metrics/), [Nanonets VLM Comparison](https://nanonets.com/blog/vision-language-model-vlm-for-data-extraction/)

**Key insight**: Models trained primarily on natural images (Llama Vision) underperform document-specialized models by 20+ percentage points on DocVQA, despite similar parameter counts.

### ChartQA Performance

[ChartQA](https://arxiv.org/abs/2203.10244) tests chart reasoning with 10,000 charts and 36,000+ questions requiring visual parsing, data extraction, and mathematical reasoning.

| Model | ChartQA Score | Notes |
|-------|--------------|-------|
| GPT-4o | 90.8% | 6.7% below human performance |
| Claude 3.5 Sonnet | 90.8% | Strong chart reasoning |
| Gemini 1.5 Pro | 87.2% | Competitive |
| Open-source models | 70-85% | Significant gap remains |

Source: [Dextra Labs Top VLMs 2025](https://dextralabs.com/blog/top-10-vision-language-models/), [IDP Leaderboard](https://idp-leaderboard.org/details/)

**Key insight**: Chart understanding requires the same spatial-semantic reasoning as bank statements—position determines meaning. Even top models lag human performance significantly.

### Table Extraction Accuracy

The [IDP Leaderboard](https://idp-leaderboard.org/details/) evaluates table extraction across sparse, dense, structured, and unstructured formats.

Key findings from [Dot Square Lab benchmarking](https://dotsquarelab.com/resources/ai-document-intelligence-benchmark):
- Proprietary OCR services (Azure Document Intelligence, Mistral-OCR) showed **low accuracy at high cost** compared to VLM-based approaches
- VLM performance degrades significantly on tables without visible gridlines
- Row alignment errors compound across multi-page tables

**This directly parallels our experience**: Bank statement tables often lack gridlines, span multiple pages, and have inconsistent row boundaries—exactly the conditions where VLMs struggle most.

### The OCR-Document Gap

[DocVLM research (December 2024)](https://arxiv.org/abs/2412.08746) quantified how much OCR capability matters:

> "In limited-token regimes (448×448), DocVLM with 64 learned queries improves DocVQA results from 56.0% to 86.6% when integrated with InternVL2"

This 30+ percentage point improvement from adding explicit OCR encoding demonstrates that general VLMs lack sufficient text-extraction bias for document tasks.

### Cross-Lingual Performance Degradation

[Research on multilingual document understanding](https://arxiv.org/abs/2412.17787) found:
- DocVQA performance decreased **33.6%** for Chinese instructions
- ChartQA performance decreased **27.9%** for Chinese instructions

While our Australian bank statements are English, this highlights how sensitive VLMs are to distribution shift—any deviation from training data (including document formatting conventions) degrades performance.

## Why These Benchmarks Matter for Bank Statements

Bank statement extraction combines the hardest aspects of multiple benchmarks:

| Challenge | Benchmark | Bank Statement Equivalent |
|-----------|-----------|--------------------------|
| Fine-grained OCR | DocVQA | Account numbers, transaction references |
| Tabular structure | Table Extraction | Transaction rows, column alignment |
| Numerical reasoning | ChartQA | Balance calculations, debit/credit logic |
| Layout understanding | All | Bank-specific formatting conventions |

A model that scores 90%+ on DocVQA might still fail on bank statements because:
1. DocVQA questions are often answerable from local context
2. Bank extraction requires **global structural consistency** (every row must align)
3. A single row misalignment invalidates the entire extraction

## References

### Foundational Theory
- **No-Free-Lunch Theorem**: Wolpert, D.H. (1996). [The lack of a priori distinctions between learning algorithms](https://direct.mit.edu/neco/article-abstract/8/7/1341/6016/The-Lack-of-A-Priori-Distinctions-Between-Learning). Neural Computation.
- **Inductive Bias in Deep Learning**: Battaglia et al. (2018). [Relational inductive biases, deep learning, and graph networks](https://arxiv.org/abs/1806.01261). arXiv.

### Document Understanding Benchmarks
- **DocVQA**: Mathew et al. (2021). [DocVQA: A Dataset for VQA on Document Images](https://arxiv.org/abs/2007.00398). WACV 2021.
- **ChartQA**: Masry et al. (2022). [ChartQA: A Benchmark for Question Answering about Charts](https://arxiv.org/abs/2203.10244). ACL 2022.
- **IDP Leaderboard**: [Intelligent Document Processing Benchmark](https://idp-leaderboard.org/details/). 2024.

### VLM Architecture and Evaluation
- **DocVLM**: [Make Your VLM an Efficient Reader](https://arxiv.org/abs/2412.08746). arXiv, December 2024.
- **InternVL3**: [OpenGVLab InternVL Series](https://huggingface.co/OpenGVLab). Hugging Face.
- **Llama 3.2 Vision**: [Meta Llama Vision Models](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/). Meta AI, 2024.

### Benchmark Comparisons
- **LearnOpenCV**: [The Ultimate Guide To VLM Evaluation Metrics](https://learnopencv.com/vlm-evaluation-metrics/). 2024.
- **Nanonets**: [Best Vision Language Models for Document Data Extraction](https://nanonets.com/blog/vision-language-model-vlm-for-data-extraction/). 2024.
- **Dot Square Lab**: [AI Document Intelligence Benchmarking](https://dotsquarelab.com/resources/ai-document-intelligence-benchmark). 2024.
- **Dextra Labs**: [Top 10 Vision Language Models in 2025](https://dextralabs.com/blog/top-10-vision-language-models/). 2025.

### Document-Specific Architectures
- **LayoutLMv3**: Huang et al. (2022). [LayoutLMv3: Pre-training for Document AI](https://arxiv.org/abs/2204.08387). ACM MM 2022.
- **Donut**: Kim et al. (2022). [OCR-free Document Understanding Transformer](https://arxiv.org/abs/2111.15664). ECCV 2022.
