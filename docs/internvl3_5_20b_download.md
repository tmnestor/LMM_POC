# InternVL3.5-GPT-OSS-20B-A4B-Preview — Download Instructions

**Model ID:** `OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview`

Mixture-of-Experts model — 21.2B total params, ~4B active per token.
Same `.chat()` API as InternVL3-8B.

## 1. Create a HuggingFace access token

1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **Create new token**
3. Select **Read** access (sufficient for downloading public models)
4. Copy the token

Then log in from the CLI:

```bash
huggingface-cli login
# Paste your token when prompted
```

## 2. Install/update dependencies

```bash
pip install -U transformers>=4.55.0 huggingface_hub
```

## 3. Download the model

```bash
# Full download (recommended — supports resume)
huggingface-cli download OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview \
  --local-dir /home/jovyan/nfs_share/models/InternVL3_5-GPT-OSS-20B-A4B-Preview
```

Or just the essential files:

```bash
huggingface-cli download OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview \
  --include "*.safetensors" "*.json" "*.py" "*.tiktoken" \
  --local-dir /path/to/your/models/InternVL3_5-GPT-OSS-20B-A4B-Preview
```

## 4. Quick smoke test

```python
import torch
from transformers import AutoTokenizer, AutoModel

path = "/path/to/your/models/InternVL3_5-GPT-OSS-20B-A4B-Preview"
model = AutoModel.from_pretrained(
    path,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False, fix_mistral_regex=True)
```

## Key details

| Detail | Value |
|--------|-------|
| License | Apache 2.0 |
| Total params | 21.2B (4B active per token) |
| Vision encoder | InternViT-300M |
| Language model | GPT-OSS-20B (MoE) |
| VRAM | Fits on a single A100-40GB at bf16 |
| Requires | `transformers >= 4.55.0` |
| `trust_remote_code` | Required |

## References

- [Model card on HuggingFace](https://huggingface.co/OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview)
- [InternVL3.5 blog post](https://internvl.github.io/blog/2025-08-26-InternVL-3.5/)
- [Paper (arXiv)](https://arxiv.org/abs/2508.18265)
