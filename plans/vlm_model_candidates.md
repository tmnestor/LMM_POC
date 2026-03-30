# VLM Candidates for 2x L40S (96 GiB Total)

**Hardware**: 2x NVIDIA L40S, 48 GiB GDDR6 each (96 GiB total)
**Architecture**: Ada Lovelace (SM 8.9) - native FP8 hardware support
**Date**: 2026-03-31

---

## Summary Table

### BF16 (no quantization)

| Model | Params | VRAM (BF16) | HF Repo | Strengths |
|-------|--------|-------------|---------|-----------|
| InternVL3-8B | 8B | ~16 GB | `OpenGVLab/InternVL3-8B` | Already benchmarking. Proven SROIE pipeline. |
| InternVL3.5-38B | 38B | ~77 GB | `OpenGVLab/InternVL3_5-38B` | Already registered. Same API as 8B. |
| Qwen3.5-9B | 9B | ~18 GB | `Qwen/Qwen3.5-9B` | Native VLM (early fusion). 262k context. 201 languages. |
| Qwen3.5-27B | 27B | ~54 GB | `Qwen/Qwen3.5-27B` | Native VLM (early fusion). Dense model, strong multimodal. |
| Gemma 3 27B | 27B | ~54 GB | `google/gemma-3-27b-it` | Strong OCR, multilingual, 128k context. |
| Nemotron Nano 2 VL | 12B | ~24 GB | `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` | **#1 OCRBench v2.** Document understanding specialist. |
| Llama 3.2 Vision 11B | 11B | ~22 GB | `meta-llama/Llama-3.2-11B-Vision-Instruct` | Already have experience. Solid baseline. |

### FP8 Quantization (native L40S hardware)

| Model | Params | VRAM (FP8) | HF Repo | Strengths |
|-------|--------|------------|---------|-----------|
| InternVL3-78B | 78B | ~83 GB | `brandonbeiler/InternVL3-78B-FP8-Dynamic` | Top open-source VLM on MMMU (72.2). |
| Qwen2.5-VL-72B | 72B | ~80-90 GB | `Qwen/Qwen2.5-VL-72B-Instruct` | Video + multilingual. Tight fit - may need reduced context. Superseded by Qwen3.5. |

### NF4 Quantization (~0.5 bytes/param)

| Model | Params (total/active) | VRAM (NF4) | HF Repo | Strengths |
|-------|----------------------|------------|---------|-----------|
| Llama 4 Scout | 109B / 17B active | ~55 GB | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | Already registered. MoE. |
| Qwen3.5-35B-A3B | 35B / 3B active | ~17 GB | `Qwen/Qwen3.5-35B-A3B` | MoE, native VLM (early fusion). Supersedes Qwen3-VL-30B-A3B. |
| Qwen3.5-122B-A10B | 122B / 10B active | ~60 GB | `Qwen/Qwen3.5-122B-A10B` | MoE, frontier-class multimodal. 10B active params. |

---

## Top Picks for SROIE Benchmark

1. **Nemotron Nano 2 VL (12B)** - Purpose-built for document OCR/understanding. Tops OCRBench v2. Small enough to coexist with another model.
2. **InternVL3-78B (FP8)** - Highest overall VLM benchmark scores in open source. MMMU 72.2.
3. **Qwen3.5-27B** - Native VLM with early fusion. Dense 27B, 262k context, 201 languages.
4. **Qwen3.5-122B-A10B** - Frontier-class MoE. Only 10B active params at NF4 = ~60 GB. Fast and powerful.
5. **Gemma 3 27B** - Google's latest. Strong OCR, multilingual, efficient image tokenization (256 soft tokens).

---

## Installation Details

### Common Setup

```bash
# Ensure hf CLI is available (huggingface-cli is deprecated)
pip install -U huggingface_hub

# Set download directory (adjust to your NFS share)
export MODEL_DIR=/home/jovyan/nfs_share/models
```

---

### 1. NVIDIA Nemotron Nano 2 VL (12B) - BF16

**HF Repo**: [`nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16)

**Architecture**: Hybrid Transformer-Mamba (CRadioV2-H vision encoder + Mamba SSM)

**Extra dependencies** (required before loading):
```bash
pip install causal_conv1d "transformers>4.53,<4.54" timm "mamba-ssm==2.2.5" accelerate open_clip_torch
```

> **WARNING**: Requires `transformers >4.53, <4.54`. May conflict with other models if they need a different version. Consider a separate conda environment or test compatibility first.

**Download**:
```bash
hf download nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 \
  --local-dir $MODEL_DIR/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16
```

**Loading code**:
```python
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

model_path = "/home/jovyan/nfs_share/models/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
```

**FP8 variant also available**: `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8` (~12 GB)

---

### 2. InternVL3-78B (FP8 Dynamic)

**HF Repo**: [`brandonbeiler/InternVL3-78B-FP8-Dynamic`](https://huggingface.co/brandonbeiler/InternVL3-78B-FP8-Dynamic)

**VRAM**: ~83 GB (fits 2x L40S with ~13 GB headroom)

**Download**:
```bash
hf download brandonbeiler/InternVL3-78B-FP8-Dynamic \
  --local-dir $MODEL_DIR/InternVL3-78B-FP8-Dynamic
```

**Loading code** (vLLM recommended for FP8):
```python
from vllm import LLM, SamplingParams

model = LLM(
    model="/home/jovyan/nfs_share/models/InternVL3-78B-FP8-Dynamic",
    trust_remote_code=True,
    max_model_len=8192,
    tensor_parallel_size=2,  # Spread across 2x L40S
)
```

**Transformers loading** (if not using vLLM):
```python
import torch
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained(
    "/home/jovyan/nfs_share/models/InternVL3-78B-FP8-Dynamic",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# Same .chat() API as InternVL3-8B
```

> **Note**: FP8 Dynamic uses per-tensor scales (W8A8). Vision tower, embeddings, norm layers, and mlp1 are preserved at full precision.

---

### 3. Qwen3.5 Models (Native VLM, Early Fusion)

Qwen3.5 replaces Qwen3-VL. All sizes are natively multimodal (early fusion on text + image tokens), support 262k context, and cover 201 languages.

#### 3a. Qwen3.5-27B (Dense)

**HF Repo**: [`Qwen/Qwen3.5-27B`](https://huggingface.co/Qwen/Qwen3.5-27B)

**VRAM**: ~54 GB BF16 (fits single L40S or 2x L40S comfortably)

**Download**:
```bash
hf download Qwen/Qwen3.5-27B \
  --local-dir $MODEL_DIR/Qwen3.5-27B
```

#### 3b. Qwen3.5-35B-A3B (MoE)

**HF Repo**: [`Qwen/Qwen3.5-35B-A3B`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)

**VRAM**: ~70 GB BF16, ~17 GB NF4. MoE with 3B active params = fast inference.

**Download**:
```bash
hf download Qwen/Qwen3.5-35B-A3B \
  --local-dir $MODEL_DIR/Qwen3.5-35B-A3B
```

#### 3c. Qwen3.5-122B-A10B (MoE, Frontier)

**HF Repo**: [`Qwen/Qwen3.5-122B-A10B`](https://huggingface.co/Qwen/Qwen3.5-122B-A10B)

**VRAM**: ~60 GB NF4 (fits 2x L40S). Frontier-class multimodal, 10B active params.

**Download**:
```bash
hf download Qwen/Qwen3.5-122B-A10B \
  --local-dir $MODEL_DIR/Qwen3.5-122B-A10B
```

#### Qwen3.5 Loading Code (all sizes)

```python
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

model = AutoModelForCausalLM.from_pretrained(
    "/home/jovyan/nfs_share/models/Qwen3.5-27B",  # or 35B-A3B, 122B-A10B
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
).eval()
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
```

#### Qwen3.5 Inference

```python
messages = [
    {"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}
]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
).to(model.device)
output_ids = model.generate(**inputs, max_new_tokens=1024)
output_text = processor.batch_decode(
    output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
)[0]
```

---

### 4. Gemma 3 27B (Vision)

**HF Repo**: [`google/gemma-3-27b-it`](https://huggingface.co/google/gemma-3-27b-it)

**Architecture**: SigLIP vision encoder + "Pan & Scan" algorithm. Images become 256 compact soft tokens. 128k context.

**Access**: Requires accepting Google's license on HuggingFace.

**Download**:
```bash
# Accept license at https://huggingface.co/google/gemma-3-27b-it first
hf download google/gemma-3-27b-it \
  --local-dir $MODEL_DIR/gemma-3-27b-it
```

**Loading code**:
```python
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

model = Gemma3ForConditionalGeneration.from_pretrained(
    "/home/jovyan/nfs_share/models/gemma-3-27b-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
).eval()
processor = AutoProcessor.from_pretrained(model_path)
```

**Inference**:
```python
messages = [
    {"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}
]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
).to(model.device)
output_ids = model.generate(**inputs, max_new_tokens=1024)
output_text = processor.batch_decode(
    output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
)[0]
```

---

### 5. ~~Qwen2.5-VL-72B~~ (Superseded by Qwen3.5)

> **NOTE**: Qwen3.5-122B-A10B supersedes this model with better performance at comparable VRAM (NF4). Prefer Qwen3.5 models above. Kept here only for reference.

---

### 6. InternVL3.5-38B (BF16) - Already Registered

**HF Repo**: [`OpenGVLab/InternVL3_5-38B`](https://huggingface.co/OpenGVLab/InternVL3_5-38B)

**Download**:
```bash
hf download OpenGVLab/InternVL3_5-38B \
  --local-dir $MODEL_DIR/InternVL3_5-38B
```

Already registered as `internvl3-38b` in `models/registry.py`. Same `.chat()` API as 8B.

---

### 7. Llama 4 Scout (NF4) - Already Registered

**HF Repo**: [`meta-llama/Llama-4-Scout-17B-16E-Instruct`](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct)

**Access**: Requires accepting Meta's license on HuggingFace.

**Download**:
```bash
hf download meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --local-dir $MODEL_DIR/Llama-4-Scout-17B-16E-Instruct
```

Already registered as `llama4scout` in `models/registry.py` with NF4 quantization.

---

## Disk Space Requirements

| Model | Disk (approx) |
|-------|---------------|
| Nemotron Nano 2 VL BF16 | ~24 GB |
| InternVL3-78B FP8 | ~83 GB |
| Qwen3.5-27B | ~54 GB |
| Qwen3.5-35B-A3B | ~70 GB |
| Qwen3.5-122B-A10B | ~245 GB |
| Gemma 3 27B | ~54 GB |
| InternVL3.5-38B | ~77 GB |
| Llama 4 Scout | ~220 GB |

**Total if downloading all**: ~827 GB

---

## Sources

- [Top 10 Vision Language Models in 2026 | DataCamp](https://www.datacamp.com/blog/top-vision-language-models)
- [Top 10 Vision Language Models in 2026 | Dextra Labs](https://dextralabs.com/blog/top-10-vision-language-models/)
- [Best Open-Source VLMs of 2026 | Labellerr](https://www.labellerr.com/blog/top-open-source-vision-language-models/)
- [Multimodal AI: Open-Source VLMs | BentoML](https://www.bentoml.com/blog/multimodal-ai-a-guide-to-open-source-vision-language-models)
- [NVIDIA Nemotron Nano VLM Tops OCR Benchmark](https://developer.nvidia.com/blog/new-nvidia-llama-nemotron-nano-vision-language-model-tops-ocr-benchmark-for-accuracy/)
- [Nemotron Nano 2 VL BF16 | HuggingFace](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16)
- [InternVL3-78B-FP8-Dynamic | HuggingFace](https://huggingface.co/brandonbeiler/InternVL3-78B-FP8-Dynamic)
- [Qwen3.5: Towards Native Multimodal Agents | Qwen Blog](https://qwen.ai/blog?id=qwen3.5)
- [Qwen3.5-27B | HuggingFace](https://huggingface.co/Qwen/Qwen3.5-27B)
- [Qwen3.5-122B-A10B | HuggingFace](https://huggingface.co/Qwen/Qwen3.5-122B-A10B)
- [Gemma 3 27B IT | HuggingFace](https://huggingface.co/google/gemma-3-27b-it)
- [GPU Requirements Cheat Sheet 2026 | Spheron](https://www.spheron.network/blog/gpu-requirements-cheat-sheet-2026/)
- [Qwen3-VL Usage Guide | vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html)
- [Qwen2.5-VL-72B VRAM Needs | Novita AI](https://blogs.novita.ai/qwen2-5-vl-72b-vram/)
