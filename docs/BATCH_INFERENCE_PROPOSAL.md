# Batch Inference Proposal: InternVL3.5-8B

## Motivation

The current pipeline processes images **sequentially (batch_size=1)**. Each image requires a full forward pass through the vision encoder and autoregressive token generation. With 88GB VRAM available, the GPU is significantly underutilised — the model weights occupy ~16GB in bfloat16, leaving ~72GB idle during single-image inference.

Increasing batch size allows the GPU to process multiple images in a single forward pass, amortising the fixed overhead of model execution across N images simultaneously.

## Current Architecture

```
process_batch()                          # batch_processor.py:234
  └─ for each image (sequential):
       └─ _process_internvl3_image()     # single image dispatch
            └─ process_single_image()    # single image inference
                 └─ model.chat()         # InternVL3 single-image API
```

**Key constraint:** `model.chat()` is a convenience wrapper that accepts one image and one prompt. It cannot process multiple images per call.

## VRAM Budget Analysis

| Component | Memory | Notes |
|-----------|--------|-------|
| Model weights (bfloat16) | ~16 GB | 8.5B parameters × 2 bytes |
| CUDA kernels + overhead | ~2 GB | Allocator, framework |
| **Available for inference** | **~70 GB** | For pixel values, KV cache, activations |
| Per-image pixel values (11 tiles) | ~0.1 GB | 11 × 3 × 448 × 448 × 2 bytes |
| Per-image KV cache + activations | ~2-4 GB | Depends on sequence length and max_new_tokens |

**Estimated batch capacity:** With ~70GB available and ~3-4GB per image, batch sizes of **8-16** are feasible. Conservative starting point: **batch_size=8**.

## Proposed Architecture

```
process_batch()                              # batch_processor.py
  └─ for each mini_batch of N images:        # grouped by batch_size
       └─ process_batch_images()             # NEW: multi-image method
            ├─ load + preprocess N images    # parallel image loading
            ├─ tokenize N prompts            # batched tokenisation
            ├─ pad + stack tensors           # uniform tensor shapes
            └─ model.generate()              # single batched forward pass
```

### Core Change: `model.chat()` → `model.generate()`

The `chat()` method is a single-image wrapper. Underneath, it calls `model.generate()`. Batching requires calling `generate()` directly with stacked inputs:

```python
# Current (batch_size=1)
response = model.chat(tokenizer, pixel_values, question, generation_config=...)

# Proposed (batch_size=N)
outputs = model.generate(
    input_ids=batched_input_ids,           # (N, max_seq_len)
    attention_mask=batched_attention_mask,  # (N, max_seq_len)
    pixel_values=batched_pixel_values,     # (N, max_tiles, 3, 448, 448)
    generation_config=generation_config,
)
responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

## Changes Required

### 1. Pipeline Config (`common/pipeline_config.py`)

Add `batch_size` field to `PipelineConfig` dataclass:

```python
@dataclass
class PipelineConfig:
    batch_size: int | None = None  # None = auto-detect from VRAM
```

### 2. CLI (`cli.py`)

Add `--batch-size` option:

```python
batch_size: int | None = typer.Option(
    None, "--batch-size", "-b",
    help="Images per batch (auto-detect from VRAM if omitted)",
)
```

### 3. Batch Processor (`common/batch_processor.py`)

Replace sequential loop with mini-batch grouping:

```python
# Current (line 234)
for idx, image_path in enumerate(image_paths, 1):
    result = self._process_internvl3_image(image_path, verbose)

# Proposed
for batch_start in range(0, len(image_paths), batch_size):
    batch_images = image_paths[batch_start:batch_start + batch_size]
    batch_results = self._process_internvl3_batch(batch_images, verbose)
```

### 4. Model Processor (`models/document_aware_internvl3_processor.py`)

New method `process_batch_images()` that:

1. Loads and preprocesses N images (can be parallelised with threading)
2. Tokenises N prompts with image token placeholders
3. Pads input_ids and attention_mask to uniform length
4. Stacks/pads pixel_values to uniform tile count
5. Calls `model.generate()` once for the batch
6. Decodes and parses N responses

### 5. Wire Existing Config (`common/config.py`)

The batch size auto-detection infrastructure already exists but is unused:

```python
# Already implemented — just needs to be connected
get_auto_batch_size("internvl3", available_memory)
get_batch_size_for_model("internvl3-8b", available_memory)
```

## Dynamic Tiling Consideration

InternVL3 uses **dynamic tiling** — different images produce different numbers of tiles based on aspect ratio and resolution. Within a batch, tile counts must be uniform. Two strategies:

| Strategy | Approach | Trade-off |
|----------|----------|-----------|
| **Pad to max tiles** | Pad all images in the batch to the highest tile count | Simple; wastes some compute on padding tiles |
| **Group by tile count** | Sort images by tile count, batch similar images together | More efficient; adds sorting complexity |

**Recommendation:** Start with pad-to-max. With `max_tiles=11`, the variance is small and padding overhead is minimal.

## Expected Performance Improvement

| Metric | batch_size=1 | batch_size=8 (est.) |
|--------|-------------|---------------------|
| GPU utilisation | ~20-30% | ~70-80% |
| Vision encoder passes | 1 per image | 8 per forward pass |
| Generation overhead | Full pipeline per image | Amortised across batch |
| **Estimated speedup** | **Baseline** | **3-5x throughput** |

The speedup is not linear (8x) because autoregressive generation is memory-bandwidth bound, not compute bound. The vision encoder benefits most from batching. Realistic expectation: **3-5x throughput improvement** on 88GB VRAM.

## Implementation Order

| Step | Scope | Risk |
|------|-------|------|
| 1. Add `batch_size` to `PipelineConfig` + CLI | Config only | None |
| 2. Wire existing `get_auto_batch_size()` | Config plumbing | Low |
| 3. Implement `process_batch_images()` using `model.generate()` | Model interface | Medium — requires understanding InternVL3's internal tokenisation |
| 4. Refactor batch processor loop | Orchestration | Low |
| 5. Test with batch_size=2, then scale up | Validation | Low |

Step 3 is the critical path. Steps 1, 2, and 4 are straightforward plumbing.

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| InternVL3 `generate()` may not handle batched pixel_values cleanly | Test with batch_size=2 first; fall back to sequential if needed |
| Variable-length outputs across batch | Use padding + `stopping_criteria`; post-process to strip padding |
| OOM at larger batch sizes | Auto-detection already exists in `common/config.py`; add OOM retry with halved batch size |
| Different document types in same batch need different prompts | Group by detected document type before batching, or use uniform prompt |

## Summary

With 88GB VRAM, the current batch_size=1 leaves ~70GB unused during inference. The batch size infrastructure (auto-detection, configuration) already exists in the codebase but is not connected. The primary engineering task is replacing `model.chat()` with a batched `model.generate()` call and restructuring the batch processor loop. Expected throughput improvement: **3-5x**.
