# Plan: vLLM Tensor Parallelism for Llama 4 Scout W4A16

## Problem

HuggingFace `device_map="auto"` uses **pipeline parallelism** — all prefill activations
concentrate on GPU 0 (~43.6 GB peak) regardless of model placement. Tested 6 configurations
across 2x L40S (44.4 GB each); all OOM by 1-160 MiB on GPU 0.

## Solution

vLLM with `tensor_parallel_size=2` splits **each layer** across both GPUs, distributing
model weight AND activation memory evenly: ~29 GB model + ~15 GB activations per GPU.

## Caveat

The vLLM blog (April 2025) marks INT4 (W4A16) for Scout as "work in progress."
compressed-tensors IS supported in vLLM for dense models. It may or may not work for
Scout's MoE architecture. Fallback: NVIDIA FP8 variant (`nvidia/Llama-4-Scout-17B-16E-Instruct-FP8`).

---

## Phase 1: Environment

Add vLLM to the scout conda env. vLLM bundles its own torch, so install it on top:

```bash
conda activate LMM_POC_VLLM
pip install vllm --no-cache-dir
```

vLLM pulls in `compressed-tensors` as a dependency (it's a vllm-project package).

## Phase 2: New Processor — `DocumentAwareVllmProcessor`

Create `models/document_aware_vllm_processor.py`.

Inherits from `BaseDocumentProcessor`. Only overrides `generate()`:

```python
class DocumentAwareVllmProcessor(BaseDocumentProcessor):
    """VLM processor backed by vLLM offline inference engine."""

    def __init__(self, llm_engine, field_list, prompt_config, field_definitions):
        # llm_engine is a vllm.LLM object
        self.llm_engine = llm_engine
        super().__init__(
            field_list=field_list,
            pre_loaded_model=llm_engine,      # satisfies base class
            pre_loaded_processor=None,         # vLLM handles tokenization
            prompt_config=prompt_config,
            field_definitions=field_definitions,
        )

    def generate(self, image: Image.Image, prompt: str, max_tokens: int = 2000) -> str:
        """Run inference via vLLM engine."""
        from vllm import SamplingParams

        sampling = SamplingParams(max_tokens=max_tokens, temperature=0)
        outputs = self.llm_engine.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": image},
            },
            sampling_params=sampling,
        )
        return outputs[0].outputs[0].text
```

The `BaseDocumentProcessor` handles:
- `detect_and_classify_document()` — loads image, calls `self.generate(image, detection_prompt)`
- `process_document_aware()` — loads image, routes to type-specific extraction prompt, calls `self.generate()`
- Prompt loading from YAML, field filtering, response parsing

So the vLLM processor only needs to implement `generate()`.

### Prompt Compatibility

vLLM handles chat template application internally. Two options:

**Option A**: Use `llm.chat()` with message-based API (cleaner):
```python
outputs = self.llm_engine.chat([
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": prompt},
    ]}
])
```

**Option B**: Use `llm.generate()` with raw prompt + `multi_modal_data` (simpler, matches existing pattern).
The processor would need to format the prompt with `<|image|>` placeholder matching Llama 4's template.

**Recommendation**: Option B — reuse existing `llama4scout_prompts.yaml` with a `<|image|>` token
prepended. The prompts already contain the extraction instructions; we just need the image placeholder.

## Phase 3: Registry Entry

Update `models/registry.py` — replace the existing `llama4scout-w4a16` loader:

```python
def _llama4scout_w4a16_loader(config):
    """vLLM-based loader with tensor parallelism for W4A16."""
    from contextlib import contextmanager
    from vllm import LLM

    @contextmanager
    def _loader(cfg):
        llm = LLM(
            model=str(cfg.model_path),
            tensor_parallel_size=2,
            max_model_len=8192,          # cap context to save memory
            gpu_memory_utilization=0.92,  # leave 8% for overhead
            limit_mm_per_prompt={"image": 1},
            trust_remote_code=True,
            # quantization="compressed-tensors",  # auto-detected from config
        )
        try:
            yield llm, None  # vLLM engine, no separate processor
        finally:
            del llm

    return _loader(config)
```

Processor creator:

```python
def _llama4scout_w4a16_processor_creator(model, _processor, config, prompt_config, fields, defs):
    from models.document_aware_vllm_processor import DocumentAwareVllmProcessor
    return DocumentAwareVllmProcessor(
        llm_engine=model,
        field_list=fields,
        prompt_config=prompt_config,
        field_definitions=defs,
    )
```

## Phase 4: Prompt Adaptation

Check if `llama4scout_prompts.yaml` detection/extraction prompts need an `<|image|>` token.
With vLLM `llm.generate()` + `multi_modal_data`, the image is injected based on the model's
chat template — may need `<|image|>` in the prompt text, or use `llm.chat()` which handles
image placement automatically.

**Test both approaches** — the chat API is safer for correct image token placement.

## Phase 5: Test Run

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python cli.py --model llama4scout-w4a16 \
    --data-dir evaluation_data/bank \
    --output-dir output/llama4scout_w4a16_bank_vllm
```

## Fallback: NVIDIA FP8

If W4A16 fails in vLLM (the "work in progress" issue):

1. Download: `huggingface-cli download nvidia/Llama-4-Scout-17B-16E-Instruct-FP8`
2. FP8 is ~54 GB, similar size to W4A16
3. Native vLLM support — no compressed-tensors needed
4. L40S (Ada Lovelace, cc 8.9) supports FP8 compute
5. Register as `llama4scout-fp8` in registry

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `models/document_aware_vllm_processor.py` | **Create** — new VLM processor wrapping vLLM |
| `models/registry.py` | **Modify** — replace W4A16 loader with vLLM-based loader |
| `config/vllm_env.yml` | **Modify** — add vllm to post-install instructions |

## Estimated Time

- Phase 1 (env): 10 min (pip install on remote)
- Phase 2 (processor): 30 min (new file, ~100 lines)
- Phase 3 (registry): 15 min (replace loader + processor creator)
- Phase 4 (prompts): 15 min (test image token placement)
- Phase 5 (test): 20 min (run bank evaluation)
- Total: ~90 min
