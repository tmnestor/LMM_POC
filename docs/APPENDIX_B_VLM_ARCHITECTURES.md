# Appendix B: Vision-Language Model Architectures

## Overview

This appendix describes the three Vision-Language Models (VLMs) evaluated in this study for business document information extraction. All models follow the common VLM paradigm: a vision encoder processes image patches into embeddings, which are then projected and concatenated with text embeddings for processing by a large language model.

---

## B.1 Llama-3.2-Vision-11B

### Architecture

| Component | Specification |
|:----------|:--------------|
| **Total Parameters** | 11 billion |
| **Vision Encoder** | ViT-H/14 (630M parameters) |
| **Language Model** | Llama-3.2 (10.6B parameters) |
| **Vision-Language Connector** | Cross-attention adapter layers |
| **Image Resolution** | Up to 1120 x 1120 pixels |
| **Context Length** | 128K tokens |

### Key Features

**Cross-Attention Integration**: Unlike many VLMs that simply concatenate visual and text tokens, Llama-3.2-Vision uses dedicated cross-attention layers that allow the language model to selectively attend to relevant image regions. This enables more nuanced visual grounding.

**Built-in Image Preprocessing**: The model includes native image preprocessing that handles resizing, normalization, and patch extraction without external dependencies. This simplifies deployment and ensures consistent image handling.

**Multi-Turn Conversation**: Supports interleaved image-text conversations, allowing follow-up questions about previously shown images—useful for iterative document analysis.

### Memory Requirements

| Precision | VRAM Required |
|:----------|:--------------|
| FP16 | ~22 GB |
| INT8 (quantized) | ~12 GB |
| INT4 (quantized) | ~6 GB |

### Reference

Meta AI. (2024). *Llama 3.2: Revolutionizing edge AI and vision with open, customizable models*. https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/

---

## B.2 InternVL3-2B

### Architecture

| Component | Specification |
|:----------|:--------------|
| **Total Parameters** | 2.2 billion |
| **Vision Encoder** | InternViT-300M |
| **Language Model** | InternLM2-1.8B |
| **Vision-Language Connector** | MLP Projector |
| **Dynamic Resolution** | Up to 4K pixels (via tiling) |
| **Max Image Tiles** | 12 (default), configurable |

### Key Features

**Native Multimodal Pre-training (NMP)**: Unlike models that separately pre-train vision and language components, InternVL3 jointly trains both modalities from initialization. This creates tighter vision-language alignment but may reduce specialization on document tasks.

**Variable Visual Position Encoding (V2PE)**: Uses fractional position increments (delta < 1) for visual tokens, allowing the model to distinguish between adjacent image patches at sub-token granularity. Critical for understanding tabular layouts.

**Dynamic High-Resolution Processing**: Automatically tiles large images into patches, processes each tile independently, and fuses results. Enables processing of high-resolution documents without resizing artifacts.

### Memory Requirements

| Precision | VRAM Required |
|:----------|:--------------|
| FP16 | ~5 GB |
| INT8 (quantized) | ~3 GB |
| INT4 (quantized) | ~2 GB |

### Reference

Chen, Z., et al. (2024). *InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks*. CVPR 2024. https://arxiv.org/abs/2312.14238

---

## B.3 InternVL3.5-8B

### Architecture

| Component | Specification |
|:----------|:--------------|
| **Total Parameters** | 8 billion |
| **Vision Encoder** | InternViT-6B |
| **Language Model** | InternLM2.5-7B |
| **Vision-Language Connector** | MLP Projector |
| **Dynamic Resolution** | Up to 4K pixels (via tiling) |
| **Max Image Tiles** | 12-40 (configurable) |

### Key Architectural Improvements over InternVL3

**Cascade Reinforcement Learning (Cascade RL)**: A two-stage training process that dramatically improves output consistency:

- **Stage 1 - Offline RL (MPO)**: Mixed Preference Optimization using pre-collected preference data. Provides stable convergence without the instability of online sampling.

- **Stage 2 - Online RL (GSPO)**: Group-wise Scaled Preference Optimization with live rollouts. Fine-tunes alignment using model-generated responses, improving edge-case handling.

This two-stage approach yields +16% improvement on reasoning benchmarks (MMMU, MathVista) and significantly reduces output variance—critical for structured document extraction where consistency matters as much as accuracy.

**Visual Resolution Router (ViR)**: Dynamically allocates visual token budget based on patch information density:

- Dense regions (text, tables): Higher token allocation
- Sparse regions (margins, whitespace): Lower token allocation
- Result: Up to 50% token reduction with <1% performance loss

**Decoupled Vision-Language Deployment (DvD)**: Enables running the vision encoder and language model on separate GPUs, improving resource utilization for production deployments.

### Memory Requirements

| Precision | VRAM Required |
|:----------|:--------------|
| FP16 | ~18 GB |
| INT8 (quantized) | ~10 GB |
| INT4 (quantized) | ~5 GB |

### Reference

Chen, Z., et al. (2025). *InternVL3: Exploring the Upper Bound of Multimodal Large Language Models*. Technical Report. https://internvl.github.io/blog/2025-01-25-InternVL3/

---

## B.4 Architecture Comparison

### Component Comparison

| Component | Llama-3.2-Vision-11B | InternVL3-2B | InternVL3.5-8B |
|:----------|:--------------------:|:------------:|:--------------:|
| Vision Encoder | ViT-H/14 | InternViT-300M | InternViT-6B |
| Vision Params | 630M | 300M | 6B |
| Language Model | Llama-3.2 | InternLM2-1.8B | InternLM2.5-7B |
| LLM Params | 10.6B | 1.8B | 7B |
| Connector Type | Cross-Attention | MLP | MLP |
| Dynamic Resolution | No | Yes | Yes |
| Reinforcement Learning | RLHF | No | Cascade RL |

### Benchmark Performance

| Benchmark | Llama-3.2-11B | InternVL3-2B | InternVL3.5-8B |
|:----------|:-------------:|:------------:|:--------------:|
| DocVQA | 88.4 | 88.3 | 93.0 |
| OCRBench | 805 | 835 | 822 |
| ChartQA | 83.4 | 76.7 | 84.8 |
| TextVQA | 73.1 | 70.5 | 77.5 |
| MMMU | 50.7 | 46.2 | 56.0 |

*Scores from official model documentation and published benchmarks.*

### Practical Trade-offs

| Criterion | Best Choice | Rationale |
|:----------|:------------|:----------|
| **Highest Accuracy** | InternVL3.5-8B | Best F1 on document extraction (73.7%) |
| **Lowest Memory** | InternVL3-2B | Runs on 4GB VRAM |
| **Best Speed** | InternVL3-2B | Smallest model, fastest inference |
| **Best Consistency** | InternVL3.5-8B | Cascade RL reduces output variance |
| **Multi-Turn Chat** | Llama-3.2-Vision | Native conversation support |

---

## B.5 Model Selection Guidance

### Recommended: InternVL3.5-8B

For production business document extraction, **InternVL3.5-8B** offers the best balance:

- Highest Mean F1 (73.7%) across all document types
- Lowest standard deviation (16.0%) indicating consistent performance
- Competitive processing speed (37s/doc, 1.6 docs/min)
- Cascade RL training produces reliable, structured outputs

### Alternative: InternVL3-2B

For resource-constrained deployments:

- Runs on consumer GPUs (4GB VRAM)
- Acceptable accuracy (60.9% Mean F1)
- Fastest throughput (1.7 docs/min)

### When to Consider Llama-3.2-Vision-11B

- Multi-turn document analysis workflows
- When cross-attention visual grounding is important
- Integration with existing Llama ecosystem

---

## References

1. Meta AI. (2024). *Llama 3.2: Revolutionizing edge AI and vision with open, customizable models*. Meta AI Blog.

2. Chen, Z., Wu, J., Wang, W., et al. (2024). *InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

3. Chen, Z., et al. (2025). *InternVL3: Exploring the Upper Bound of Multimodal Large Language Models*. Technical Report, OpenGVLab.

4. Zhu, D., Chen, J., Shen, X., et al. (2024). *MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models*. ICLR 2024.

5. Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2024). *Visual Instruction Tuning*. NeurIPS 2023.
