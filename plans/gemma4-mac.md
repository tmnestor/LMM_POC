# Plan: Gemma 4 on Mac M1 (MPS backend)

Branch: `feature/gemma4-mac`

## Goals

- Run `google/gemma-4-E4B-it` (primary) and optionally `google/gemma-4-27b-it`
  (stretch — needs 16 GB+ unified memory with 4-bit quantization) through
  the existing LMM_POC graph extraction pipeline.
- Use PyTorch MPS backend (Metal Performance Shaders) — no CUDA, no vLLM.
- Minimise changes: reuse `HFChatTemplateBackend`, `ModelSpec`, `GraphExecutor`,
  all post-processing. Add only what is Mac-specific.
- This branch is **never merged to main** — it is a standalone Mac dev branch.

---

## Model / Memory Analysis

**Confirmed: M1 with 16 GB unified memory.**

| Model          | Weights (F16) | INT4 (quanto) | Runtime total* | Fits 16 GB? |
|----------------|---------------|---------------|----------------|-------------|
| E4B (4.5B eff) | ~9 GB         | ~4.5 GB       | ~11 GB F16     | Yes — comfortable |
| 27B MoE        | ~54 GB        | ~14 GB        | ~16 GB INT4    | Tight — feasible |

*Runtime total includes weights + KV cache + activations + macOS overhead (~2–3 GB).

**E4B (primary)**: Use **float16**, no quantization needed. ~11 GB total leaves
~5 GB headroom for KV cache and activations — comfortable.

**27B MoE (stretch)**: INT4 is required. The 27B is a MoE model — only ~4B
parameters are active per forward pass, so KV cache/activation memory stays
manageable even at full 27B weight count. With INT4 (~14 GB weights) + 2 GB
OS + minimal activations, this is at the limit but feasible with batch_size=1
and short context (max_new_tokens ≤ 1000). Expect occasional swap to SSD if
macOS reclaims memory mid-run.

**bfloat16 on M1**: PyTorch MPS supports bf16 as of 2.0, but float16 is more
stable on base M1 and runs at the same speed. Use `dtype: float16` in config.

### Quantization on MPS

`bitsandbytes` does **not** support MPS. Use `transformers`-native `quanto`:

```python
from transformers import QuantoConfig
quantization_config = QuantoConfig(weights="int4")  # or "int8"
```

Quanto is CPU/MPS-compatible and integrates with `from_pretrained` the same
way as `BitsAndBytesConfig`.

---

## Architecture Impact

### Fully reusable unchanged
- `common/graph_executor.py` — model-agnostic `generate_fn` interface
- `common/bank_post_process.py`, `common/turn_parsers.py`, `common/bank_corrector.py`
- `models/backends/hf_chat_template.py` — minor MPS cache tweak only
- `cli.py` — `--model gemma4-e4b-mps` works via registry

### Requires Mac-aware changes
- `models/model_loader.py` — MPS device detection, no `force_single_gpu → cuda:0`
- `models/backends/hf_chat_template.py` — add `torch.mps.empty_cache()` cleanup
- `common/pipeline_config.py` — auto-detect MPS, set `device_map`

### New files
- `models/registry.py` — add `gemma4-e4b-mps` (and `gemma4-27b-mps`) `ModelSpec`
- `prompts/gemma4_prompts.yaml` — Mac prompt file (start as copy of internvl3)
- `config/gemma4_mac_config.yml` — run_config with Mac paths, no CUDA settings
- `config/gemma4_mac_env.yaml` — conda environment (PyTorch MPS, transformers,
  no flash-attn, no vllm, add quanto)

---

## Gemma4 Model API

Gemma4 uses `AutoProcessor` + `AutoModelForImageTextToText` (the image-text
class that maps to `Gemma4ForConditionalGeneration` under the hood).

**Message style**: `two_step` — identical to Qwen3VL / Nemotron:
1. `processor.apply_chat_template(messages, tokenize=False)` → text string
2. `processor(text=[text], images=[pil_image], return_tensors="pt")` → inputs
3. `model.generate(**inputs, **gen_kwargs)`

Content dict: `{"type": "image"}` with no key (the image is passed separately
to `processor()`, not inline in the message content). This matches the
`HFChatTemplateBackend._generate_two_step` path with `image_content_key=None`.

**No `enable_thinking`** for extraction (set `chat_template_kwargs={"enable_thinking": False}`).

---

## Implementation Steps

### Step 1 — Conda environment

New file `config/gemma4_mac_env.yaml`:

```yaml
name: gemma4_mac
channels: [pytorch, conda-forge, defaults]
dependencies:
  - python=3.12
  - pip
  - pip:
    - torch>=2.3          # includes MPS backend
    - torchvision
    - transformers>=4.51  # Gemma4 support landed in 4.51
    - accelerate
    - optimum[quanto]     # MPS-compatible quantization
    - Pillow
    - PyYAML
    - rich
    - typer
    - pandas
    - scikit-learn
```

No `flash-attn`, no `vllm`, no `bitsandbytes`.

### Step 2 — `models/model_loader.py` (3 changes)

**2a. MPS cache clear in loader finally block** (~line 235):
```python
finally:
    del model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
```

**2b. `force_single_gpu` fix** (~line 187) — currently hardcodes `cuda:0`:
```python
if spec.force_single_gpu:
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "cuda:0"
    else:
        load_kwargs["device_map"] = "mps:0"
    load_kwargs["dtype"] = torch.bfloat16
```
(E4B/27B don't use `force_single_gpu` but this prevents breakage for
other models like Granite4 if ever tested on Mac.)

**2c. dtype kwarg**: `AutoModelForImageTextToText` is already in the
`torch_dtype` branch (line 151-154) — no change needed. Gemma4 will use
`torch_dtype`.

### Step 3 — `models/backends/hf_chat_template.py` (1 change)

Add MPS cache clear after CUDA cleanup in `_generate_two_step` and
`_generate_one_step`:

```python
del inputs, output_ids, generated_ids
if torch.cuda.is_available():
    torch.cuda.empty_cache()
if hasattr(torch, "mps") and torch.backends.mps.is_available():
    torch.mps.empty_cache()
```

### Step 4 — `common/pipeline_config.py` (1 change)

Add MPS auto-detection alongside the CUDA path so `device_map` resolves
correctly when no CUDA is present:

```python
# In _resolve_device_map() or wherever device_map is set:
if device_map == "auto":
    import torch
    if torch.cuda.is_available():
        pass  # keep "auto"
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        device_map = "mps"
    else:
        device_map = "cpu"
```

Confirm the exact location after reading `pipeline_config.py`.

### Step 5 — Register Gemma4 models in `models/registry.py`

```python
# -- Gemma 4 (Mac MPS, HFChatTemplateBackend two_step) --

def _gemma4_post_load(model, processor, cfg):
    """Suppress generation_config warnings for Gemma4."""
    if hasattr(model, "generation_config"):
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        pad_id = getattr(processor, "eos_token_id", None)
        if pad_id is not None:
            model.generation_config.pad_token_id = pad_id


register_hf_model(
    ModelSpec(
        model_type="gemma4-e4b-mps",
        model_class="AutoModelForImageTextToText",
        processor_class="AutoProcessor",
        prompt_file="gemma4_prompts.yaml",
        description="Gemma 4 E4B (4.5B eff) — Mac MPS, float16",
        message_style="two_step",
        chat_template_kwargs={"enable_thinking": False},
        suppress_gen_warnings=("temperature", "top_p"),
        post_load=_gemma4_post_load,
    )
)

# 27B MoE with INT4 quantization (requires 16 GB+ unified memory)
def _gemma4_27b_quantization(cfg):
    from transformers import QuantoConfig
    return QuantoConfig(weights="int4")

register_hf_model(
    ModelSpec(
        model_type="gemma4-27b-mps",
        model_class="AutoModelForImageTextToText",
        processor_class="AutoProcessor",
        prompt_file="gemma4_prompts.yaml",
        description="Gemma 4 27B MoE — Mac MPS, INT4 (needs 16 GB+ RAM)",
        message_style="two_step",
        chat_template_kwargs={"enable_thinking": False},
        load_kwargs={"quantization_config": _gemma4_27b_quantization},
        suppress_gen_warnings=("temperature", "top_p"),
        post_load=_gemma4_post_load,
    )
)
```

### Step 6 — Prompt file `prompts/gemma4_prompts.yaml`

Start as a copy of `internvl3_prompts.yaml`. Adjust formatting if needed
after first test run — Gemma4's chat template wraps differently than InternVL3's.

### Step 7 — Mac run config `config/gemma4_mac_config.yml`

```yaml
model:
  type: gemma4-e4b-mps
  path: /Users/tod/PretrainedLLM/gemma-4-E4B-it
  flash_attn: false
  dtype: float16                     # M1 base: float16; M1 Pro/Max: bfloat16
  max_new_tokens: 2000

data:
  dir: ../evaluation_data/bank       # adjust as needed
  ground_truth: ../evaluation_data/bank/ground_truth_bank.csv
  max_images: 5                      # start small on Mac

output:
  dir: ../evaluation_data/output_mac
  skip_visualizations: false
  skip_reports: false

processing:
  batch_size: 1        # MPS: sequential only
  num_gpus: 0          # no CUDA GPUs
  bank_v2: true
  verbose: true
  debug: false

logging:
  log_dir: ../evaluation_data/output_mac/logs
```

---

## What is NOT Supported on This Branch

| Feature | Status |
|---|---|
| Batch inference | Not applicable — MPS sequential only |
| Flash Attention 2 | Not available on MPS |
| vLLM | Not available on MPS |
| bitsandbytes 4-bit | Not available on MPS (use quanto instead) |
| Data-parallel | Not applicable — single device |
| CUDA OOM recovery | Not applicable — use MPS memory monitoring |

---

## Open Questions Before Implementing

1. ~~How much unified memory does your M1 have?~~ **Resolved: 16 GB.**
   E4B at float16 = primary. 27B MoE at INT4 = stretch goal (tight but feasible).

2. ~~HuggingFace model location~~ **Resolved: `/Users/tod/PretrainedLLM`.**
   Models downloaded there via `huggingface-cli download --local-dir /Users/tod/PretrainedLLM/<model-name>`.

3. **`pipeline_config.py` device_map handling** — need to read the file to
   confirm where `device_map` is resolved before implementing Step 4.

4. **Prompt file** — reuse `internvl3_prompts.yaml` prompts directly (same
   schema, different model) or create a fresh `gemma4_prompts.yaml`?
   Recommendation: copy internvl3 file and rename — adjust if model outputs
   differ in formatting after first test run.
