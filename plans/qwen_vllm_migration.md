# Plan: Add vLLM Support for Qwen3-VL-8B and Qwen3.5-27B

Both models have [official vLLM support](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html) (requires vllm >= 0.11.0). The existing `DocumentAwareVllmProcessor` already handles the OpenAI-compatible chat API with base64 image data URIs — the same pattern works for Qwen models.

## Current State

| Model | Registry key | HF class | Processor file |
|-------|-------------|----------|----------------|
| Qwen3-VL-8B | `qwen3vl` | `Qwen3VLForConditionalGeneration` | `document_aware_qwen3vl_processor.py` |
| Qwen3.5-27B | `qwen35` | `Qwen3_5ForConditionalGeneration` | `document_aware_qwen35_processor.py` |

Both use HuggingFace native inference with manual `torch.cuda.empty_cache()`, OOM retry, etc. The vLLM engine handles all of this internally.

## Design Decision: Reuse `DocumentAwareVllmProcessor`

The existing vLLM processor is model-agnostic — it uses the OpenAI-compatible chat API (`llm_engine.chat()` with base64 image data URIs). The only model-specific parts are:

1. **`_configure_generation()`** — selects generation config by `model_type_key`
2. **`get_model_info()`** — returns model type string

Both are already parameterized via `model_type_key`. No new processor file needed.

## Changes Required

### 1. `common/model_config.py` — Add generation configs

Add `QWEN3VL_VLLM_GENERATION_CONFIG` and `QWEN35_VLLM_GENERATION_CONFIG` (or reuse `QWEN3VL_GENERATION_CONFIG` since the shape is identical — just need the `model_type_key` routing in the vLLM processor).

Update `_configure_generation()` in `DocumentAwareVllmProcessor` to handle `qwen3vl` and `qwen35` keys.

### 2. `models/document_aware_vllm_processor.py` — Route Qwen generation configs

```python
def _configure_generation(self) -> None:
    from common.model_config import (
        INTERNVL3_GENERATION_CONFIG,
        LLAMA4SCOUT_GENERATION_CONFIG,
        QWEN3VL_GENERATION_CONFIG,
    )

    if self._model_type_key == "internvl3":
        self.gen_config = dict(INTERNVL3_GENERATION_CONFIG)
    elif self._model_type_key in ("qwen3vl", "qwen35"):
        self.gen_config = dict(QWEN3VL_GENERATION_CONFIG)
    else:
        self.gen_config = dict(LLAMA4SCOUT_GENERATION_CONFIG)
```

### 3. `models/registry.py` — Register vLLM variants

Add two new loader + registration blocks, following the InternVL3 vLLM pattern:

#### `_qwen3vl_vllm_loader(config)`
- Set `VLLM_WORKER_MULTIPROC_METHOD=spawn` before vLLM import
- Detect GPU count via `CUDA_VISIBLE_DEVICES` or `nvidia-smi`
- `max_model_len=8192` (8B model, plenty of headroom)
- `gpu_memory_utilization=0.92`
- `limit_mm_per_prompt={"image": 1}`
- `trust_remote_code=True`

#### `_qwen35_vllm_loader(config)`
- Same as above but `max_model_len=4096` (27B model needs more KV cache budget)
- Qwen3.5 thinking mode: vLLM respects `enable_thinking` via chat template kwargs, but we disable it at the prompt level (no `/think` token), so no special handling needed

#### Processor creators
Both use `_qwen_vllm_processor_creator` (shared) which instantiates `DocumentAwareVllmProcessor` with the appropriate `model_type_key`.

#### Registrations
```python
register_model(ModelRegistration(
    model_type="qwen3vl-vllm",
    loader=_qwen3vl_vllm_loader,
    processor_creator=_qwen_vllm_processor_creator,
    prompt_file="qwen3vl_prompts.yaml",
    description="Qwen3-VL-8B via vLLM (PagedAttention, tensor parallelism)",
    requires_sharding=True,
))

register_model(ModelRegistration(
    model_type="qwen35-vllm",
    loader=_qwen35_vllm_loader,
    processor_creator=_qwen_vllm_processor_creator,
    prompt_file="qwen3vl_prompts.yaml",  # same prompt works for both
    description="Qwen3.5-27B via vLLM (~54 GB BF16)",
    requires_sharding=True,
))
```

### 4. `config/run_config.yml` — Add default paths

```yaml
model_loading:
  default_paths:
    qwen3vl-vllm: /home/jovyan/nfs_share/models/Qwen3-VL-8B-Instruct
    qwen35-vllm: /home/jovyan/nfs_share/models/Qwen3.5-27B
```

### 5. Prompt files — No change needed

Both vLLM variants reuse the same `qwen3vl_prompts.yaml` as their HF counterparts. The vLLM engine applies the model's chat template internally.

## Qwen3.5 Thinking Mode

Qwen3.5 has a thinking/reasoning mode (`enable_thinking=True`). The HF processor disables it via `chat_template_kwargs={"enable_thinking": False}`. For vLLM:
- vLLM's chat API applies the model's chat template automatically
- We can pass `chat_template_kwargs={"enable_thinking": False}` to `llm.chat()` if needed
- Alternatively, the system prompt can include `/no_think` to suppress it
- **Test first without special handling** — if the model produces `<think>...</think>` blocks in output, add the kwarg

## Files to Modify

| File | Change |
|------|--------|
| `models/document_aware_vllm_processor.py` | Route `qwen3vl`/`qwen35` in `_configure_generation()` |
| `models/registry.py` | Add 2 loaders + 1 shared processor creator + 2 registrations |
| `config/run_config.yml` | Add `qwen3vl-vllm` and `qwen35-vllm` default paths |

No new files needed.

## Validation

1. Run SROIE benchmark with `--model qwen3vl-vllm` and `--model qwen35-vllm`
2. Compare F1 scores against HF variants to verify accuracy parity
3. Compare throughput (img/min) to quantify speedup
4. Run bank statement benchmark to check multi-turn accuracy (the KV cache issue seen with InternVL3)

## References

- [Qwen3-VL vLLM recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html)
- [Qwen3.5 vLLM recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
- [vLLM qwen3_vl model executor](https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/qwen3_vl/)
