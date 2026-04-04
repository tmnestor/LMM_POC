# Plan: IBM Granite 4.0 3B Vision on WildReceipt

## Overview

Wire up IBM Granite 4.0 3B Vision for the WildReceipt benchmark. This model is purpose-built for document key-value extraction (VAREX benchmark: 85.5% exact-match zero-shot at 3B params). Two paths: HF-native (recommended first) and vLLM (optional upgrade).

## Key Granite 4.0 3B Vision Facts

- **Architecture**: `Granite4VisionForConditionalGeneration` (SigLIP2 vision + WindowQFormer projectors + GraniteMoeHybrid LLM)
- **Size**: 4B total (3.5B base + 0.5B LoRA adapters), ~8 GB BF16 — fits single GPU easily
- **Vision encoder**: SigLIP2 (384×384 patches), supports multi-resolution up to 3840×384
- **Loading**: `AutoModelForImageTextToText` + `AutoProcessor` (requires `trust_remote_code=True`)
- **LoRA merge**: call `model.merge_lora_adapters()` after loading for faster inference
- **License**: Apache 2.0
- **KVP extraction**: schema-driven — you provide a JSON schema, model returns extracted JSON
- **Generation**: `max_new_tokens=4096`, `do_sample=False` (deterministic)
- **HuggingFace**: `ibm-granite/granite-4.0-3b-vision`

## Dependencies

```
transformers==4.57.6
peft==0.18.1
torch==2.10.0 (cu128)
tokenizers==0.22.2
pillow>=12.1.1
```

**NOTE**: `transformers==4.57.6` is the same version used by our existing LMM_POC_IVL3.5 env. Check compatibility — if it conflicts with InternVL3.5 requirements, create a separate env.

## Option A: HF-Native (Recommended First)

### 1. Conda Environment

**Try existing `LMM_POC_IVL3.5` env first** — Granite needs `transformers==4.57.6` which matches. Only additional dependency is `peft>=0.18.1`.

```bash
# Check if peft is already installed
conda activate LMM_POC_IVL3.5
pip list | grep peft

# If missing:
pip install peft==0.18.1
```

If the existing env doesn't work (e.g. torch version mismatch, model loading fails), create a dedicated env:

```yaml
# environment_granite4.yml
name: LMM_POC_GRANITE
channels:
  - nvidia
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pip
  - pip:
    - torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
    - transformers==4.57.6
    - peft==0.18.1
    - tokenizers==0.22.2
    - pillow>=12.1.1
    - typer
    - rich
    - pyyaml
```

### 2. Download Model

Download from a machine with internet access. Use `local_dir_use_symlinks=False`
to get real files (not cache symlinks) — required for copying to air-gapped
production via NFS.

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('ibm-granite/granite-4.0-3b-vision',
                  local_dir='/home/jovyan/nfs_share/models/granite-4.0-3b-vision',
                  local_dir_use_symlinks=False)
"
```

**Important**: `trust_remote_code=True` means transformers will try to fetch
updated `.py` files from HuggingFace at load time. On air-gapped production,
this is fine as long as the local snapshot is complete — `from_pretrained()`
falls back to the local directory. If you see download errors on prod, verify
all `.py` files (`modeling.py`, `downsampling.py`, `configuration.py`,
`processing.py`) exist in the model directory.

### 3. `models/registry.py` — add loader + registration

New loader `_granite4_vision_loader()`, following the Nemotron pattern (also uses `AutoModelForImageTextToText`):

```python
def _granite4_vision_loader(config):
    """Context manager for loading Granite 4.0 3B Vision.

    SigLIP2 + WindowQFormer + GraniteMoeHybrid LLM.
    ~8 GB BF16, fits single GPU. LoRA adapters merged at load time.
    """
    from contextlib import contextmanager

    import torch
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from transformers import AutoModelForImageTextToText, AutoProcessor

    console = Console()

    @contextmanager
    def _loader(cfg):
        model = None
        processor = None

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            console.print(
                f"\n[bold]Loading Granite 4.0 3B Vision from: {cfg.model_path}[/bold]"
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Loading processor...", total=None)

                processor = AutoProcessor.from_pretrained(
                    str(cfg.model_path), trust_remote_code=True
                )

                progress.update(task, description="Loading model weights...")

                model = AutoModelForImageTextToText.from_pretrained(
                    str(cfg.model_path),
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                ).eval()

                # Merge LoRA adapters for faster inference
                progress.update(task, description="Merging LoRA adapters...")
                model.merge_lora_adapters()

            _print_gpu_status(console)
            yield model, processor

        finally:
            del model, processor
            if torch.cuda.is_available():
                import gc
                gc.collect()
                torch.cuda.empty_cache()

    return _loader(config)
```

Processor creator: create `DocumentAwareGranite4Processor` or reuse a generic wrapper. The model uses standard `processor(text=..., images=..., return_tensors="pt")` + `model.generate()`, similar to Nemotron.

Register:
```python
register_model(
    ModelRegistration(
        model_type="granite4",
        loader=_granite4_vision_loader,
        processor_creator=_granite4_processor_creator,
        prompt_file="internvl3_prompts.yaml",  # same WildReceipt extraction prompt
        description="IBM Granite 4.0 3B Vision (~8 GB BF16)",
        requires_sharding=False,  # fits single GPU
    )
)
```

### 4. `benchmark_sroie.py` — add inference function

New `_run_inference_granite4()` function:

```python
def _run_inference_granite4(
    model, processor, image: Image.Image, prompt: str, max_tokens: int
) -> str:
    """Granite 4.0 3B Vision: AutoProcessor + AutoModelForImageTextToText."""
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(
        model.device
    )

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
    )

    # Trim input tokens
    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

Add `"granite4"` to dispatch in `run_inference()`:

```python
if model_type == "granite4":
    return _run_inference_granite4(
        model, tokenizer_or_processor, image, prompt, max_tokens
    )
```

### 5. `common/model_config.py` — add generation config

```python
GRANITE4_GENERATION_CONFIG = {
    "max_new_tokens_base": 1024,
    "max_new_tokens_per_field": 64,
    "do_sample": False,
}
```

### 6. `config/run_config.yml` — add default path

```yaml
granite4: /home/jovyan/nfs_share/models/granite-4.0-3b-vision
```

### 7. `scripts/run_wildreceipt_all.sh` — add benchmark job

```bash
# Determine which env has peft — use LMM_POC_IVL3.5 if compatible, else LMM_POC_GRANITE
run_job LMM_POC_IVL3.5 granite4 \
  "$OUTPUT_BASE/wildreceipt_granite4"
```

## Option B: vLLM (Optional — For Higher Throughput)

Granite 4.0 3B Vision provides an **out-of-tree vLLM model** — it ships `granite4_vision.py` in the HF repo. This is NOT natively in vLLM's model registry, so we need extra setup.

### Additional Setup

1. **Download the vLLM model file** from HF:
   ```bash
   # granite4_vision.py will be in the model directory after snapshot_download
   # It registers Granite4VisionForConditionalGeneration with vLLM's ModelRegistry
   ```

2. **vLLM loader** — register the custom model before creating LLM:
   ```python
   def _granite4_vllm_loader(config):
       from contextlib import contextmanager
       from rich.console import Console

       console = Console()

       @contextmanager
       def _loader(cfg):
           import os, sys
           os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

           from vllm import LLM, ModelRegistry

           # Register out-of-tree model from the HF repo
           model_dir = str(cfg.model_path)
           sys.path.insert(0, model_dir)
           ModelRegistry.register_model(
               "Granite4VisionForConditionalGeneration",
               "granite4_vision:Granite4VisionForConditionalGeneration",
           )

           llm = None
           try:
               llm = LLM(
                   model=model_dir,
                   max_model_len=4096,  # Small model, 4K is fine
                   gpu_memory_utilization=0.90,
                   limit_mm_per_prompt={"image": 1},
                   trust_remote_code=True,
                   disable_log_stats=True,
                   hf_overrides={"adapter_path": model_dir},  # Merge LoRA at load
               )
               console.print("[green]Granite 4.0 3B Vision vLLM engine ready[/green]")
               processor_cls = _get_vllm_processor_cls()
               processor = processor_cls(llm, model_type_key="granite4")
               yield llm, processor
           finally:
               if llm:
                   del llm
               sys.path.remove(model_dir)

       return _loader(config)
   ```

3. **Register as separate model type** (e.g. `granite4-vllm`) to keep HF and vLLM paths independent.

### vLLM Considerations

- The out-of-tree model file (`granite4_vision.py`) is ~800 lines and uses vLLM internal APIs that may break across vLLM versions
- At only 4B params, vLLM's overhead (model sharding, KV cache management) may not provide significant speedup over HF on a single GPU
- **Recommendation**: Start with HF-native (Option A). Only pursue vLLM if throughput is a bottleneck on the full 472-image benchmark.

## Validation

On sandbox:
```bash
conda activate LMM_POC_IVL3.5  # or LMM_POC_GRANITE
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Quick test (5 images)
PYTHONUNBUFFERED=1 python benchmark_wildreceipt.py \
  --model granite4 \
  --data-dir ../data/wildreceipt \
  --output-dir evaluation_data/output/wildreceipt_granite4 \
  --save-responses \
  -n 5

# Full benchmark (472 images)
PYTHONUNBUFFERED=1 python benchmark_wildreceipt.py \
  --model granite4 \
  --data-dir ../data/wildreceipt \
  --output-dir evaluation_data/output/wildreceipt_granite4 \
  --save-responses
```

## Expected Outcome

- **F1**: Unknown — VAREX benchmark (US government forms) ≠ WildReceipt (noisy receipts). The model's zero-shot KVP extraction strength should help, but WildReceipt has challenging OCR (crumpled receipts, faded text, non-English)
- **Speed**: Fast — 4B model on single GPU should be 10-20+ img/min
- **Memory**: ~8 GB VRAM — no multi-GPU needed

## Risk: Prompt Format

Our WildReceipt prompt asks for a specific JSON structure. Granite 4.0 was trained with a **JSON Schema input format** for KVP extraction. If the standard prompt underperforms, we may need to adapt the prompt to use Granite's native schema format:

```python
schema = {
    "type": "object",
    "properties": {
        "store_name": {"type": "string", "description": "Name of the store"},
        "store_address": {"type": "string", "description": "Address of the store"},
        "date": {"type": "string", "description": "Date on the receipt"},
        "total": {"type": "string", "description": "Total amount"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "quantity": {"type": "string"},
                    "price": {"type": "string"},
                }
            }
        },
        # ... etc
    }
}

prompt = f"""Extract structured data from this document.
Return a JSON object matching this schema:

{json.dumps(schema, indent=2)}

Return null for fields you cannot find.
Return ONLY valid JSON.
Return an instance of the JSON with extracted values, not the schema itself."""
```

This alternative prompt could be added as a `granite4_prompts.yaml` if needed after initial results.
