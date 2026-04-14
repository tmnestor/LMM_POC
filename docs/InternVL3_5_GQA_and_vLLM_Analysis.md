# InternVL3.5: Grouped Query Attention & vLLM vs HuggingFace Analysis

## 1. Grouped Query Attention Across Model Sizes

All three InternVL3.5 variants use Qwen3-series LLMs as their language backbone, each employing Grouped Query Attention (GQA) to reduce KV cache memory during inference.

### Configuration Summary

| Parameter | InternVL3_5-8B | InternVL3_5-14B | InternVL3_5-38B |
|---|---|---|---|
| Base LLM | Qwen3-8B | Qwen3-14B | Qwen3-32B |
| Query heads (`num_attention_heads`) | 32 | 40 | 64 |
| KV heads (`num_key_value_heads`) | 8 | 8 | 8 |
| GQA group ratio | 4:1 | 5:1 | 8:1 |
| Hidden size | 3,200 | 5,120 | 5,120 |
| Layers | 36 | 40 | 64 |
| Context length | 32K | 128K | 128K |
| Vision encoder | InternViT-300M | InternViT-300M | InternViT-6B |

### Key Observations

**Constant KV heads across all sizes.** All three models fix `num_key_value_heads = 8`, regardless of model size. As the model scales up, the KV cache cost stays relatively flat while the query-side capacity grows.

**Increasing compression ratio.** The GQA group ratio scales from 4:1 to 5:1 to 8:1. The 38B variant is the most aggressive: each KV head is shared across 8 query heads. The KV cache for the 38B model is the same size as the 8B model per layer (both have 8 KV heads), despite having twice the query heads.

**Practical implication for inference.** For long-context inference (especially with the 128K context models), GQA is what makes it tractable. The 38B model with standard multi-head attention (MHA) would need 64 KV heads per layer across 64 layers. With GQA it only needs 8, an 8x reduction in KV cache memory. This is particularly relevant when serving with vLLM or TGI and maximizing batch size under a memory budget.

**Vision encoder note.** The 8B and 14B variants use InternViT-300M while the 38B uses InternViT-6B. The GQA config only applies to the LLM backbone, not the vision encoder.

---

## 2. Why vLLM is Faster Than HuggingFace Transformers

### PagedAttention and KV Cache Management

This is where GQA and vLLM interact directly. With only 8 KV heads across all model sizes, the KV cache is already compact. vLLM's PagedAttention manages this cache in non-contiguous GPU memory blocks (like virtual memory paging), which eliminates the fragmentation and over-allocation that HF Transformers suffers from with its static, contiguous cache. The practical effect is significantly larger batch sizes, which drives throughput up.

### Continuous Batching

HF Transformers uses static batching: if you have 8 requests and one finishes early, those GPU cycles are wasted until the whole batch completes. vLLM slots in new requests as soon as a sequence finishes decoding, keeping the GPU saturated.

### Optimized Kernels

vLLM integrates FlashAttention-2/3 and fused CUDA kernels for the GQA pattern natively. The HF Transformers `generate()` loop has more Python overhead and less kernel fusion.

### DvD (Decoupled Vision-Language Deployment)

This is specific to InternVL3.5. The vision encoder (InternViT) and LLM (Qwen3) are deployed on separate GPUs, communicating via BF16 feature vectors over TCP/RDMA. In the HF Transformers pipeline, vision encoding blocks the LLM — it must wait for the ViT forward pass before prefilling begins. DvD overlaps these, achieving approximately 2x throughput from decoupling alone and up to 4.05x when combined with the Vision Retokenizer (ViR) compression.

---

## 3. Why vLLM Can Be Slightly More Accurate

### Image Preprocessing and Tile Selection

InternVL3.5 uses dynamic resolution tiling (448x448 tiles). The exact tile configuration selected for a given image can vary between implementations. vLLM's implementation may handle the `max_num` tile parameter or the aspect ratio selection logic differently than the HF reference code, potentially selecting a tiling that preserves more visual detail for certain benchmarks. Even a one-tile difference changes the visual tokens the LLM sees.

### FlashAttention Numerical Behavior

FlashAttention uses online softmax with higher-precision accumulators internally, which can be slightly more numerically stable than naive scaled dot-product attention in some edge cases. Over thousands of benchmark examples, this can nudge a few answers across the decision boundary.

### Chat Template and Prompt Formatting

vLLM's OpenAI-compatible server applies chat templates through its own tokenizer path. Subtle differences in how special tokens, system prompts, or image placeholder tokens are formatted can shift outputs. For InternVL3.5 specifically, the `<img>` and `</img>` token handling and the `IMG_CONTEXT_TOKEN` placement matter — a mismatch here between HF and vLLM has been a known source of discrepancies.

### Sampling Path

Even with `temperature=0`, the sampling implementations differ. vLLM's argmax operates on the raw logits after a fused softmax kernel, while HF Transformers may apply logit processors in a different order. Floating-point non-associativity means these can occasionally disagree on the top token when two logits are very close.

---

## 4. The Throughput–Accuracy Interaction

Throughput and accuracy are not independent here. Because vLLM can handle more tokens efficiently, it is more practical to run with higher `max_num` tile counts (more visual detail) without blowing up latency. In practice, people using HF Transformers often cap resolution to keep inference manageable, while vLLM users can afford to keep the full tiling — which directly helps on vision benchmarks like OCR, chart understanding, and document QA.

---

## Sources

- [InternVL3.5 paper (arXiv)](https://arxiv.org/abs/2508.18265)
- [InternVL3.5 blog](https://internvl.github.io/blog/2025-08-26-InternVL-3.5/)
- [DvD topic overview](https://www.emergentmind.com/topics/decoupled-vision-language-deployment-dvd)
- [vLLM InternVL3.5 recipes](https://docs.vllm.ai/projects/recipes/en/latest/InternVL/InternVL3_5.html)
- [InternVL3_5-8B (HuggingFace)](https://huggingface.co/OpenGVLab/InternVL3_5-8B)
- [InternVL3_5-14B (HuggingFace)](https://huggingface.co/OpenGVLab/InternVL3_5-14B)
- [InternVL3_5-38B (HuggingFace)](https://huggingface.co/OpenGVLab/InternVL3_5-38B)
- [Qwen3 documentation](https://huggingface.co/docs/transformers/en/model_doc/qwen3)
