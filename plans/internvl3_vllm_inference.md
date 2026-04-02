# Plan: InternVL3.5-8B vLLM Inference Engine

## Goal

Add a parallel `internvl3-vllm` model type that runs InternVL3.5-8B through vLLM's
offline inference engine instead of HuggingFace `AutoModel`. The existing `internvl3`
registration is unchanged — both paths coexist.

## Why

- **Faster throughput** — continuous batching, CUDA graphs, optimised kernels
- **No flash-attn required** — vLLM's built-in Triton attention backend works on
  production (4x A10G) where flash-attn compilation fails
- **Paves the way** for `internvl3-38b-vllm` with tensor parallelism (replacing
  pipeline-parallel `split_model`)

## Prerequisite: Production Environment

```bash
pip install vllm --no-cache-dir
```

Force Triton attention on production (no flash-attn):
```bash
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
```

---

## Changes

### 1. `models/registry.py` — New loader + registration

Add `_internvl3_vllm_loader` (copy the pattern from `_llama4scout_w4a16_loader`):

```python
def _internvl3_vllm_loader(config):
    from contextlib import contextmanager
    from rich.console import Console
    console = Console()

    @contextmanager
    def _loader(cfg):
        import os
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        from vllm import LLM

        llm = None
        try:
            tp_size = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").count(",")) + 1
            tp_size = max(1, tp_size)

            console.print(
                f"\n[bold]Loading InternVL3.5-8B via vLLM "
                f"(tensor_parallel_size={tp_size})[/bold]"
            )

            llm = LLM(
                model=str(cfg.model_path),
                tensor_parallel_size=tp_size,
                max_model_len=8192,
                gpu_memory_utilization=0.92,
                limit_mm_per_prompt={"image": 1},
                trust_remote_code=True,
            )

            console.print("[bold green]vLLM engine ready![/bold green]")
            yield llm, None

        finally:
            del llm

    return _loader(config)
```

Processor creator — reuse existing `DocumentAwareVllmProcessor`, but pass
`model_type_key="internvl3"` (for generation config / auto batch sizing):

```python
def _internvl3_vllm_processor_creator(
    model, tokenizer_or_processor, config, prompt_config,
    universal_fields, field_definitions,
):
    from models.document_aware_vllm_processor import DocumentAwareVllmProcessor
    return DocumentAwareVllmProcessor(
        field_list=universal_fields,
        model_path=str(config.model_path),
        debug=config.verbose,
        batch_size=config.batch_size,
        pre_loaded_model=model,
        pre_loaded_processor=tokenizer_or_processor,
        prompt_config=prompt_config,
        field_definitions=field_definitions,
        model_type_key="internvl3",         # <-- use InternVL3 generation config
    )
```

Register:
```python
register_model(ModelRegistration(
    model_type="internvl3-vllm",
    loader=_internvl3_vllm_loader,
    processor_creator=_internvl3_vllm_processor_creator,
    prompt_file="internvl3_prompts.yaml",   # same prompts as HF InternVL3
    description="InternVL3.5-8B via vLLM (PagedAttention, no flash-attn required)",
))
```

### 2. `models/document_aware_vllm_processor.py` — Accept `model_type_key` parameter

Currently hardcoded to `model_type_key="llama4scout"` (line 64). Make it a
constructor parameter with a default:

```python
def __init__(self, ..., model_type_key: str = "llama4scout"):
    ...
    self._init_shared(
        ...
        model_type_key=model_type_key,
    )
```

Also make `_configure_generation()` aware of different generation configs:

```python
def _configure_generation(self) -> None:
    from common.model_config import (
        INTERNVL3_GENERATION_CONFIG,
        LLAMA4SCOUT_GENERATION_CONFIG,
    )
    if self._model_type_key == "internvl3":
        self.gen_config = dict(INTERNVL3_GENERATION_CONFIG)
    else:
        self.gen_config = dict(LLAMA4SCOUT_GENERATION_CONFIG)
    ...
```

### 3. `config/run_config.yml` — Add default path

```yaml
default_paths:
  ...
  internvl3-vllm: /home/jovyan/nfs_share/models/InternVL3_5-8B  # same weights
```

### 4. `benchmark_sroie.py` — Route to vLLM inference

Add `internvl3-vllm` to the dispatch and config:

```python
# In run_inference():
if model_type == "internvl3-vllm":
    return _run_inference_vllm(
        model, tokenizer_or_processor, image, prompt, max_tokens
    )

# In run_benchmark() config_kwargs:
elif model_type == "internvl3-vllm":
    config_kwargs.setdefault("flash_attn", False)
    config_kwargs.setdefault("device_map", "auto")
```

---

## Files to Modify

| File | Action |
|------|--------|
| `models/registry.py` | Add `_internvl3_vllm_loader`, `_internvl3_vllm_processor_creator`, `register_model()` |
| `models/document_aware_vllm_processor.py` | Add `model_type_key` param, make generation config dynamic |
| `config/run_config.yml` | Add `internvl3-vllm` default path |
| `benchmark_sroie.py` | Add `internvl3-vllm` dispatch + config |

## Files Unchanged

| File | Why |
|------|-----|
| `models/document_aware_internvl3_processor.py` | HF path untouched |
| `internvl3_prompts.yaml` | Shared by both HF and vLLM paths |
| `cli.py` | Registry-driven — picks up new model type automatically |

## Usage

All three InternVL3.5 sizes are registered: `internvl3-vllm`, `internvl3-14b-vllm`, `internvl3-38b-vllm`.
They share the same loader, processor creator, and prompts. Use `LMM_POC_VLLM` conda env.

```bash
conda activate LMM_POC_VLLM

# --- InternVL3.5-8B vLLM ---
VLLM_LOGGING_LEVEL=WARNING VLLM_ATTENTION_BACKEND=TRITON_ATTN python cli.py \
  --model internvl3-vllm \
  --data-dir evaluation_data/bank \
  --ground-truth evaluation_data/bank/ground_truth_bank.csv \
  --output-dir evaluation_data/output/bank_ivl35_8b_vllm

# --- InternVL3.5-14B vLLM ---
VLLM_LOGGING_LEVEL=WARNING VLLM_ATTENTION_BACKEND=TRITON_ATTN python cli.py \
  --model internvl3-14b-vllm \
  --data-dir evaluation_data/bank \
  --ground-truth evaluation_data/bank/ground_truth_bank.csv \
  --output-dir evaluation_data/output/bank_ivl35_14b_vllm

# --- InternVL3.5-38B vLLM ---
VLLM_LOGGING_LEVEL=WARNING VLLM_ATTENTION_BACKEND=TRITON_ATTN python cli.py \
  --model internvl3-38b-vllm \
  --data-dir evaluation_data/bank \
  --ground-truth evaluation_data/bank/ground_truth_bank.csv \
  --output-dir evaluation_data/output/bank_ivl35_38b_vllm

# --- SROIE benchmarks ---
VLLM_LOGGING_LEVEL=WARNING VLLM_ATTENTION_BACKEND=TRITON_ATTN python benchmark_sroie.py \
  --model internvl3-vllm --data-dir data/sroie --output-dir evaluation_data/output/sroie_ivl35_8b_vllm

VLLM_LOGGING_LEVEL=WARNING VLLM_ATTENTION_BACKEND=TRITON_ATTN python benchmark_sroie.py \
  --model internvl3-14b-vllm --data-dir data/sroie --output-dir evaluation_data/output/sroie_ivl35_14b_vllm

VLLM_LOGGING_LEVEL=WARNING VLLM_ATTENTION_BACKEND=TRITON_ATTN python benchmark_sroie.py \
  --model internvl3-38b-vllm --data-dir data/sroie --output-dir evaluation_data/output/sroie_ivl35_38b_vllm
```
