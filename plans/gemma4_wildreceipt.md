# Plan: Gemma 4 31B-it on WildReceipt (vLLM)

## Overview

Wire up Gemma 4 31B-it as a vLLM model for the WildReceipt benchmark. Same pattern as InternVL3-vLLM and Qwen3.5-vLLM.

## Key Gemma 4 facts

- **Architecture**: `Gemma4ForConditionalGeneration` (model_type: `gemma4`)
- **Size**: ~58 GB BF16, fits 2x L40S (88.8 GB) with ~31 GB KV headroom
- **GQA**: 32 Q heads, 16 KV heads (2:1 ratio) — SDPA patch handles this
- **Vision tokens**: configurable via `mm_processor_kwargs={"max_soft_tokens": N}` and matching `hf_overrides`
  - 280 = default, 560 = high detail, 1120 = max (for dense OCR)
  - For receipt extraction, use **1120** (max detail)
- **Thinking mode**: must disable via `chat_template_kwargs={"enable_thinking": False}` (same as Qwen3.5)
- **vLLM**: day-one support, requires vLLM nightly and transformers >= 5.5.0
- **Sampling**: Google recommends `temperature=1.0, top_p=0.95, top_k=64` but we use `temperature=0` for deterministic extraction

## Changes required

### 1. `models/registry.py` — add loader + registration

New loader `_gemma4_vllm_loader()`:
```python
llm = LLM(
    model=str(cfg.model_path),
    tensor_parallel_size=tp_size,
    max_model_len=16384,
    gpu_memory_utilization=0.92,
    limit_mm_per_prompt={"image": 1},
    trust_remote_code=True,
    disable_log_stats=True,
    mm_processor_kwargs={"max_soft_tokens": 1120},
    hf_overrides={
        "vision_config": {"default_output_length": 1120},
        "vision_soft_tokens_per_image": 1120,
    },
)
```

Processor creator: reuse `DocumentAwareVllmProcessor` with `model_type_key="gemma4"`.

Register as `gemma4` model type.

### 2. `models/document_aware_vllm_processor.py` — add Gemma4 generation config routing

In `_configure_generation()`, add:
```python
elif self._model_type_key == "gemma4":
    self.gen_config = dict(GEMMA4_GENERATION_CONFIG)
```

In `generate()`, disable thinking mode for Gemma4:
```python
if self._model_type_key.startswith(("qwen35", "gemma4")):
    chat_kwargs["chat_template_kwargs"] = {"enable_thinking": False}
```

### 3. `common/model_config.py` — add Gemma4 generation config

```python
GEMMA4_GENERATION_CONFIG = {
    "max_new_tokens_base": 512,
    "max_new_tokens_per_field": 64,
    "do_sample": False,
}
```

### 4. `benchmark_sroie.py` — add to vLLM dispatch

Add `"gemma4"` to the vLLM model type tuple in `run_inference()`.

Add thinking mode disable in `_run_inference_vllm()`:
```python
if model_type.startswith(("qwen35", "gemma4")):
    chat_kwargs["chat_template_kwargs"] = {"enable_thinking": False}
```

### 5. `config/run_config.yml` — add default path

```yaml
gemma4: /home/jovyan/nfs_share/models/Gemma-4-31B-it
```

### 6. `scripts/run_wildreceipt_all.sh` — add benchmark job

```bash
run_job LMM_POC_VLLM gemma4 \
  "$OUTPUT_BASE/wildreceipt_gemma4" \
  env VLLM_LOGGING_LEVEL=WARNING
```

## Validation

On sandbox (2x L40S):
```bash
conda activate LMM_POC_VLLM
VLLM_LOGGING_LEVEL=WARNING PYTHONUNBUFFERED=1 python benchmark_wildreceipt.py \
  --model gemma4 \
  --data-dir ../data/wildreceipt \
  --output-dir evaluation_data/output/wildreceipt_gemma4 \
  --save-responses
```

## Environment check

Verify on sandbox:
```bash
python -c "import transformers; print(transformers.__version__)"  # needs >= 5.5.0
python -c "from vllm import LLM; print('vLLM OK')"               # needs nightly with Gemma4 support
```
