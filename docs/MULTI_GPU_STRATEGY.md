# Multi-GPU Strategy: Design Rationale

## Problem Statement

Vision-language model (VLM) inference on document images is slow. A single InternVL3.5-8B forward pass takes 2-8 seconds per image depending on tile count and token generation length. Bank statement extraction, which uses multi-turn sequential prompting, takes ~90 seconds per image. For production workloads of hundreds or thousands of documents, single-GPU sequential processing is impractical.

This document explains the scaling strategies we evaluated, the approach we chose, and why.

---

## Strategies Considered

### 1. Batched Inference (Single GPU)

**How it works:** Pack multiple images into a single `model.generate()` call. The vision encoder processes N images in one forward pass, amortising fixed overhead (kernel launch, memory allocation, scheduler latency) across the batch.

**Implementation:** InternVL3's `model.batch_chat()` API concatenates pixel values and passes a `num_patches_list` to handle variable tile counts across images. The pipeline groups images into mini-batches, runs detection in batch, then extraction in batch (standard documents) or sequentially (bank statements).

**Throughput gain:** 3-5x on a single GPU with batch_size=8. The speedup is sub-linear because autoregressive token generation is memory-bandwidth bound, not compute bound. The vision encoder benefits most from batching.

**Limitations:**
- Only works for models that expose a batched generation API (InternVL3 has `batch_chat()`; Llama does not)
- All images in a batch share the same prompt, so detection and extraction must be batched separately
- Bank statement multi-turn extraction cannot be batched (each turn depends on the previous response)
- Bounded by single-GPU VRAM: model weights (~16GB in bfloat16) + per-image KV cache and activations (~3-4GB each) limits practical batch size to 8-16 on an 80GB GPU

### 2. Pipeline Parallelism

**How it works:** Split the model across GPUs layer-by-layer. GPU 0 runs the vision encoder, GPU 1 runs transformer layers 0-15, GPU 2 runs layers 16-31, etc. Images flow through the pipeline sequentially, but different stages process different images concurrently.

**Why we rejected it:**
- **High inter-GPU communication cost.** Every layer boundary requires transferring the full hidden state (batch_size x seq_len x hidden_dim) across the PCIe/NVLink bus. For an 8B model with 4096 hidden dim, each transfer is ~32MB per image at bfloat16 — this happens at every pipeline stage boundary, dozens of times per generation step
- **Pipeline bubble.** The first and last GPUs idle while the pipeline fills and drains. With autoregressive generation (hundreds of sequential decode steps), the bubble overhead is severe
- **Complexity.** Requires model-specific sharding logic, careful load balancing across stages, and synchronisation between GPUs. The `transformers` library's `device_map="auto"` can shard automatically, but it optimises for fitting large models (that don't fit on one GPU) rather than for throughput
- **No throughput gain for models that fit on one GPU.** InternVL3.5-8B at 16GB in bfloat16 fits comfortably on a single 24GB L4 or 80GB H100. Pipeline parallelism solves the wrong problem — we're not memory-constrained, we're throughput-constrained

### 3. Tensor Parallelism

**How it works:** Split individual weight matrices across GPUs so each GPU computes a portion of every layer simultaneously. All GPUs work on the same image at the same time, reducing per-image latency.

**Why we rejected it:**
- **Requires all-reduce at every layer.** Each transformer layer needs an all-reduce operation (reduce-scatter + all-gather) to synchronise partial results across GPUs. For an 8B model with 32 layers, that's 64+ collective operations per forward pass, per generation step
- **Latency-bound, not throughput-bound.** Tensor parallelism reduces single-image latency (useful for real-time serving) but doesn't improve throughput better than data parallelism for batch workloads
- **NVLink dependency.** Practical tensor parallelism requires high-bandwidth GPU interconnect (NVLink at 600+ GB/s). On PCIe-connected GPUs (common in cloud L4/A10G configurations), the all-reduce overhead dominates and can actually reduce throughput
- **Framework complexity.** Requires either a specialised serving framework (vLLM, TensorRT-LLM) or manual weight sharding. Neither integrates cleanly with the `transformers` `AutoModel` API we use for model loading

### 4. Data Parallelism (Chosen Approach)

**How it works:** Load an independent, complete model copy on each GPU. Partition the input images into N chunks (one per GPU). Each GPU processes its chunk independently. Merge results when all GPUs finish.

**This is what we implemented.** See details below.

---

## Chosen Strategy: Data Parallelism + Batched Inference

Our approach combines two orthogonal optimisations:

1. **Data parallelism across GPUs** — each GPU gets its own model and processes a subset of images independently
2. **Batched inference within each GPU** — each GPU batches its image subset for higher per-GPU throughput

These compose multiplicatively:

| Strategy | Throughput | How |
|----------|-----------|-----|
| Sequential baseline | 1x | 1 GPU, 1 image at a time |
| Batched inference only | 3-5x | 1 GPU, 8 images per forward pass |
| Data parallelism only (4 GPUs) | ~4x | 4 GPUs, 1 image at a time each |
| **Both combined (4 GPUs, batch=4)** | **~12-20x** | **4 GPUs, each batching 4 images** |

For bank statements (which cannot be batched due to multi-turn dependencies), data parallelism is the **only** scaling mechanism — 4 GPUs process 4 bank statements concurrently, reducing wall-clock time by ~4x.

---

## Implementation Architecture

```
cli.py --num-gpus 0 (auto-detect)
  |
  +-- 1 GPU available --> Single-GPU path
  |     +-- BatchDocumentProcessor
  |           +-- Phase 1: Batched detection
  |           +-- Phase 2: Batched extraction (standard) / Sequential (bank)
  |
  +-- N GPUs available --> MultiGPUOrchestrator
        |
        +-- Phase 1: Sequential model loading (1 per GPU)
        |     +-- GPU 0: load_model(device_map="cuda:0")
        |     +-- GPU 1: load_model(device_map="cuda:1")
        |     +-- ...
        |
        +-- Phase 2: ThreadPoolExecutor(max_workers=N)
              +-- GPU 0: BatchDocumentProcessor(images[0:K])
              +-- GPU 1: BatchDocumentProcessor(images[K:2K])
              +-- ...
              +-- Merge results in original image order
```

### Why ThreadPoolExecutor (Not Multiprocessing)

PyTorch releases the GIL during CUDA kernel execution. This means Python threads achieve true GPU parallelism without the overhead of process-based parallelism:

- **No serialisation cost.** Multiprocessing requires pickling model objects, tensors, and results across process boundaries. A single model copy is ~16GB — serialising and deserialising this per-process is prohibitive
- **Shared address space.** Threads share memory, so the orchestrator can directly access per-GPU results without IPC
- **No import duplication.** Each process would need to re-import `transformers`, `torch`, and all dependencies. Thread-based parallelism imports once
- **Simpler cleanup.** Context managers for model lifecycle work naturally within threads. Process-based cleanup requires signal handling and IPC for graceful shutdown

The one caveat — `transformers` has lazy-import race conditions when multiple threads import simultaneously — is handled by serialising model loading with a `threading.Lock`, then running inference in parallel after all models are loaded.

### Image Partitioning

Images are split into contiguous chunks:

```python
# 100 images across 4 GPUs
# GPU 0: images[0:25]
# GPU 1: images[25:50]
# GPU 2: images[50:75]
# GPU 3: images[75:100]
```

If fewer images than GPUs, the orchestrator adapts by using only as many GPUs as there are images (one image per GPU, remaining GPUs idle).

### Result Merging

Results from all GPUs are collected using `as_completed()` (for progress reporting) but stored in GPU-index order to preserve original image ordering:

- **batch_results**: concatenated in image order
- **processing_times**: concatenated (for per-image timing)
- **document_type_counts**: summed across GPUs
- **batch_stats**: averaged across GPUs

### VRAM Requirements

Each GPU holds one complete model copy. Total VRAM requirement scales linearly:

| Model | Per-GPU (bfloat16) | 2 GPUs | 4 GPUs |
|-------|-------------------|--------|--------|
| InternVL3.5-8B | ~18 GB | ~36 GB | ~72 GB |
| Llama 3.2-11B | ~22 GB | ~44 GB | ~88 GB |

This means each GPU needs at least 24GB VRAM (L4, A10G, A100, H100, H200 all qualify). The trade-off is clear: we spend more total VRAM for proportionally more throughput.

---

## Configuration

### YAML (`config/run_config.yml`)

```yaml
processing:
  num_gpus: 0         # 0 = auto-detect all, 1 = single GPU, N = use N GPUs
  batch_size: null     # Per-GPU batch size (null = auto-detect from VRAM)
```

### CLI

```bash
# Auto-detect all available GPUs, auto batch size
python cli.py --num-gpus 0 -d ./data -o ./output

# Explicit: 4 GPUs, batch size 4 per GPU
python cli.py --num-gpus 4 --batch-size 4 -d ./data -o ./output

# Single GPU (disables multi-GPU even if more are available)
python cli.py --num-gpus 1 -d ./data -o ./output
```

### Config Cascade

CLI flags > YAML > dataclass defaults. The `num_gpus` resolution logic:

1. `num_gpus=0` (default): auto-detect via `torch.cuda.device_count()`
2. `num_gpus=N` where N > available: fatal error with clear diagnostic
3. `num_gpus=1`: forces single-GPU path regardless of available hardware

---

## Why Not vLLM / TensorRT-LLM?

Dedicated serving frameworks (vLLM, TensorRT-LLM, SGLang) offer continuous batching, PagedAttention, and optimised CUDA kernels. We evaluated these but chose native `transformers` inference for this project because:

1. **Model compatibility.** InternVL3.5's dynamic tiling and custom vision encoder require `trust_remote_code=True` and model-specific preprocessing. Serving frameworks often lag behind the latest model architectures
2. **Multi-turn bank extraction.** Our bank statement pipeline requires stateful multi-turn conversation with image context carried across turns. Serving frameworks optimise for stateless request-response patterns
3. **Development velocity.** We add new models regularly (InternVL3, Llama, Qwen3-VL). The Protocol + Registry pattern lets us integrate a new model in hours. Serving framework integration requires additional engineering per model
4. **Deployment simplicity.** Our target deployment (Kubeflow pipelines) already provides multi-GPU node allocation. A simple ThreadPoolExecutor is easier to operate than a separate model server process

For high-QPS production serving (many concurrent users, low-latency SLAs), a dedicated serving framework would be the right choice. For our use case — batch processing of document sets with prioritised accuracy — native inference with data parallelism provides sufficient throughput with lower operational complexity.

---

## Summary

| Criterion | Pipeline Parallel | Tensor Parallel | Data Parallel |
|-----------|:-:|:-:|:-:|
| Throughput scaling | Poor (pipeline bubbles) | Good (latency) | Excellent (throughput) |
| Inter-GPU communication | High (every layer) | High (all-reduce) | None |
| Works on PCIe (no NVLink) | Partially | Poorly | Fully |
| Model-agnostic | No (sharding logic) | No (weight splitting) | Yes (independent copies) |
| Bank statement support | No improvement | Marginal | Full linear speedup |
| Implementation complexity | High | High | Low |
| VRAM efficiency | High (shared weights) | High (shared weights) | Low (duplicated weights) |

Data parallelism trades VRAM efficiency for implementation simplicity, model-agnostic operation, and linear throughput scaling. For an 8B-parameter model that fits comfortably on a single 24GB GPU, the duplicated-weights cost is acceptable and the operational benefits are substantial.
