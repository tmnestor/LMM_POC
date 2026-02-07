# vLLM + KServe Deployment Architecture

## Context

The current pipeline loads InternVL3.5-8B directly into GPU memory via `transformers.AutoModel` and calls `model.batch_chat()` for inference. This works for evaluation but has fundamental throughput limitations for production workloads.

This document proposes replacing the in-process model with a **vLLM model server** behind **KServe**, integrating with our existing Kubeflow Pipeline infrastructure.

---

## Current Architecture

```
┌─────────────────────────────────────────────────────┐
│  Kubeflow Pipeline Step (single pod)                │
│                                                     │
│  ┌───────────┐    ┌──────────┐    ┌──────────────┐  │
│  │ ivl3_cli  │───>│ batch_   │───>│ InternVL3    │  │
│  │           │    │ processor│    │ model.batch_  │  │
│  │           │    │          │    │ chat()        │  │
│  └───────────┘    └──────────┘    └──────┬───────┘  │
│                                          │          │
│                                   ┌──────┴───────┐  │
│                                   │ 4x L4 GPUs   │  │
│                                   │ (88GB VRAM)   │  │
│                                   └──────────────┘  │
└─────────────────────────────────────────────────────┘
```

### Limitations

| Issue | Impact |
|-------|--------|
| **Static batching** | GPU idles while waiting for longest sequence in batch to finish |
| **Contiguous KV cache** | Pre-allocates max sequence length per image, wastes ~70% VRAM |
| **Coupled lifecycle** | Model loads/unloads with each pipeline run (~60s cold start) |
| **No request queuing** | Pipeline step must manage its own batching and OOM fallback |
| **Single consumer** | Only one pipeline run can use the model at a time |

---

## Proposed Architecture

```
┌──────────────────────────┐       ┌──────────────────────────────────┐
│  Kubeflow Pipeline Step  │       │  KServe InferenceService         │
│  (lightweight pod)       │       │  (persistent GPU pod)            │
│                          │       │                                  │
│  ┌───────────┐           │ HTTP  │  ┌─────────────────────────────┐ │
│  │ ivl3_cli  │──────────────────>│  │ vLLM Engine                 │ │
│  │           │           │  /v1/ │  │                             │ │
│  │ - detect  │           │  chat │  │ - Continuous batching       │ │
│  │ - extract │           │       │  │ - PagedAttention            │ │
│  │ - evaluate│           │       │  │ - Tensor parallelism (4 L4)│ │
│  └───────────┘           │       │  │ - Automatic scheduling      │ │
│                          │       │  └──────────┬──────────────────┘ │
│  No GPU required         │       │             │                    │
│                          │       │      ┌──────┴───────┐            │
└──────────────────────────┘       │      │ 4x L4 GPUs   │            │
                                   │      │ (88GB VRAM)   │            │
                                   │      └──────────────┘            │
                                   └──────────────────────────────────┘
```

### Key Changes

1. **Model runs as a persistent service** — not loaded per pipeline run
2. **Pipeline step becomes a lightweight HTTP client** — no GPU needed
3. **vLLM handles all batching and memory management** internally
4. **Multiple pipeline runs can share the same model server**

---

## Why vLLM

### Continuous Batching

Standard `batch_chat()` waits for all sequences in a batch to finish before returning. vLLM uses **iteration-level scheduling** — as soon as one sequence finishes, a new request is immediately slotted in:

```
batch_chat() — static batching:
  Image 1: ████░░░░░░░░░░░░░░░░  done, GPU idle
  Image 2: ██████████░░░░░░░░░░  done, GPU idle
  Image 3: ██████████████░░░░░░  done, GPU idle
  Image 4: ████████████████████  still generating...
  GPU utilization: ~40%

vLLM — continuous batching:
  Image 1: ████ → Image 5: ████████ → Image 9: ██████
  Image 2: ██████████ → Image 6: ██████████
  Image 3: ██████████████ → Image 7: ██████
  Image 4: ████████████████████
  GPU utilization: ~95%
```

### PagedAttention

The KV cache stores attention state during token generation. Standard inference pre-allocates contiguous memory blocks sized for the maximum sequence length — most of which goes unused:

```
Standard KV cache (contiguous):
  [Seq1: ████████████████████████████]  ← 2048 tokens reserved, 200 used
  [Seq2: ████████████████████████████]  ← 2048 tokens reserved, 500 used
  Wasted: ~82%

PagedAttention (paged, like OS virtual memory):
  [1][2][1][3][2][4][1][3]...  ← 4KB pages allocated on demand
  Wasted: <5%
```

Result: **2-4x more concurrent sequences** fit in the same VRAM.

### Tensor Parallelism

vLLM natively supports splitting the model across multiple GPUs. With 4x L4 (22GB each = 88GB total), InternVL3-8B in bfloat16 (~16GB weights) distributes evenly:

```
GPU 0: layers 0-7   + vision encoder
GPU 1: layers 8-15
GPU 2: layers 16-23
GPU 3: layers 24-31 + lm_head
```

This is handled automatically with `--tensor-parallel-size 4`.

---

## KServe InferenceService

### ServingRuntime (cluster-level, defined once)

```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: ClusterServingRuntime
metadata:
  name: vllm-internvl3
spec:
  supportedModelFormats:
    - name: vllm
      version: "1"
      autoSelect: true
  containers:
    - name: kserve-container
      image: vllm/vllm-openai:latest
      command: ["python", "-m", "vllm.entrypoints.openai.api_server"]
      args:
        - "--model=/mnt/models"
        - "--trust-remote-code"
        - "--tensor-parallel-size=4"
        - "--max-model-len=4096"
        - "--gpu-memory-utilization=0.90"
        - "--enable-prefix-caching"       # cache shared prompt prefixes
        - "--max-num-seqs=16"             # max concurrent sequences
      resources:
        requests:
          nvidia.com/gpu: "4"
          memory: "32Gi"
          cpu: "8"
        limits:
          nvidia.com/gpu: "4"
          memory: "64Gi"
          cpu: "16"
      volumeMounts:
        - name: model-store
          mountPath: /mnt/models
          readOnly: true
```

### InferenceService (per-model deployment)

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: internvl3-8b
  namespace: ml-serving
  annotations:
    # Scale to zero after 15 min idle (cost savings)
    serving.kserve.io/scaleTarget: "1"
    serving.kserve.io/minScale: "0"
    serving.kserve.io/maxScale: "1"
    serving.kserve.io/scaleMetric: "concurrency"
spec:
  predictor:
    model:
      modelFormat:
        name: vllm
      runtime: vllm-internvl3
      storageUri: "pvc://model-store/InternVL3_5-8B"
    minReplicas: 0
    maxReplicas: 1
```

### Model Storage

The model weights are stored on a PersistentVolumeClaim accessible to the KServe pod:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-store
  namespace: ml-serving
spec:
  accessModes: ["ReadOnlyMany"]
  storageClassName: efs-sc          # EFS for shared access
  resources:
    requests:
      storage: 50Gi
```

---

## Pipeline Client Integration

The pipeline step replaces `model.batch_chat()` with HTTP calls to the vLLM OpenAI-compatible API.

### Detection Request

```python
import httpx

VLLM_URL = "http://internvl3-8b.ml-serving.svc.cluster.local/v1/chat/completions"

def detect_documents(image_paths: list[str]) -> list[dict]:
    """Send detection requests to vLLM server."""
    # vLLM handles batching internally — send individual requests concurrently
    async with httpx.AsyncClient(timeout=120) as client:
        tasks = [
            client.post(VLLM_URL, json={
                "model": "InternVL3_5-8B",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"file://{path}"}},
                            {"type": "text", "text": detection_prompt},
                        ],
                    }
                ],
                "max_tokens": 50,
                "temperature": 0.0,
            })
            for path in image_paths
        ]
        responses = await asyncio.gather(*tasks)

    return [parse_detection(r.json()) for r in responses]
```

### Key Differences from Current Pipeline

| Aspect | Current (`batch_chat`) | Proposed (vLLM + KServe) |
|--------|----------------------|--------------------------|
| **Batching** | Manual mini-batches with OOM fallback | Automatic continuous batching |
| **Memory** | Pre-allocated contiguous KV cache | PagedAttention, on-demand |
| **Concurrency** | Single pipeline run at a time | Multiple consumers, queued |
| **Cold start** | ~60s model load per run | 0s (model already warm) |
| **GPU pod** | Required for pipeline step | Only for KServe service |
| **Scaling** | Fixed | Scale-to-zero when idle |
| **Image handling** | Local file path | Base64 or shared storage URL |
| **OOM recovery** | Recursive batch halving | vLLM manages internally |

---

## Expected Performance

Based on vLLM benchmarks for comparable vision-language models:

| Metric | Current (batch_chat) | Projected (vLLM) | Improvement |
|--------|---------------------|-------------------|-------------|
| Throughput (images/min) | ~12 (batch=4) | ~30-40 | 2.5-3x |
| GPU utilization | ~40-60% | ~85-95% | ~2x |
| Avg latency/image | ~5s | ~2-3s | ~2x |
| Cold start | ~60s | 0s (warm) | eliminated |
| Max concurrent requests | 1 pipeline | 16+ queued | 16x |
| VRAM efficiency | ~50% (fragmentation) | ~95% (paged) | ~2x |

> These are estimates. Actual numbers depend on document complexity, tile counts, and output token lengths. Benchmark on representative workload before committing.

---

## Migration Path

### Phase 1: Validate vLLM Compatibility

- Confirm InternVL3.5-8B runs on vLLM with `--trust-remote-code`
- Verify vision input handling (multi-tile images)
- Compare extraction accuracy: `batch_chat()` vs vLLM output for same inputs
- Benchmark throughput on 4x L4

### Phase 2: Deploy KServe InferenceService

- Create ServingRuntime and InferenceService manifests
- Configure model storage (EFS PVC)
- Set up health checks and readiness probes
- Configure scale-to-zero policy

### Phase 3: Adapt Pipeline Client

- Replace `DocumentAwareInternVL3HybridProcessor` with HTTP client
- Send concurrent async requests (vLLM batches internally)
- Keep evaluation and reporting logic unchanged (CPU-only)
- Remove GPU resource requests from pipeline step pod

### Phase 4: Production Hardening

- Add request timeouts and retry logic
- Monitor vLLM metrics (queue depth, latency, GPU utilization)
- Set up Prometheus/Grafana dashboards
- Configure autoscaling thresholds if multiple replicas needed

---

## Risks and Considerations

| Risk | Mitigation |
|------|------------|
| vLLM may not support InternVL3.5 vision features | Test in Phase 1 before committing; fallback to current approach |
| Scale-to-zero cold start (~60-90s) | Set `minScale: 1` for latency-sensitive workloads |
| Network overhead (HTTP vs in-process) | Negligible vs model inference time; use cluster-internal DNS |
| Image transfer to vLLM pod | Use shared storage (EFS) — pass paths, not pixels |
| Accuracy drift between inference engines | Run comparison evaluation in Phase 1; vLLM uses same model weights |

---

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM Vision Language Model Support](https://docs.vllm.ai/en/latest/models/vlm.html)
- [KServe Documentation](https://kserve.github.io/website/)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [KServe + vLLM Integration](https://kserve.github.io/website/latest/modelserving/v1beta1/llm/vllm/)
