# Multi-GPU Orchestrator: Design Decisions

> Quick reference for the four key design choices behind `MultiGPUOrchestrator`.
> For full rationale, see [MULTI_GPU_STRATEGY.md](MULTI_GPU_STRATEGY.md).

---

## ThreadPoolExecutor

PyTorch releases the GIL during CUDA kernel execution, so threads achieve **true GPU parallelism** without inter-process communication overhead. No need to pickle 16 GB model copies across process boundaries — threads share the same address space.

## Sequential Loading

Model loading is serialised behind a `threading.Lock` to avoid race conditions in the `transformers` library's lazy-import machinery. Once all models are loaded, inference runs in parallel across GPUs.

## Contiguous Partitioning

Input images are split into equal, contiguous chunks — one chunk per GPU. Results are concatenated back in original image order after all GPUs finish.

```
100 images, 4 GPUs:
  GPU 0 → images[0:25]
  GPU 1 → images[25:50]
  GPU 2 → images[50:75]
  GPU 3 → images[75:100]
```

## Auto-Detection

| Flag | Behaviour |
|------|-----------|
| `--num-gpus 0` | Auto-detect all available GPUs via `torch.cuda.device_count()` |
| `--num-gpus 1` | Bypass orchestrator entirely — run single-GPU code path |
| `--num-gpus N` | Use exactly N GPUs (fatal error if N > available) |
