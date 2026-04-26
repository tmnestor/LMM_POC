# vLLM 0.19.0 Multi-GPU on PCIe-Only A10G with Limited `/dev/shm` — Investigation Notes

**Scope.** Decision document for moving `InternVL3.5-8B` off the current `tensor_parallel_size>1` (TP) configuration, which stalls on this cluster due to a small `/dev/shm`. Comparison of three viable paths and a verification checklist.

**Hardware constraint.** 4× A10G (24 GiB each), no NVLink, PCIe Gen4 between GPUs. Cluster ships containers with the default `/dev/shm` (64 MB) and the user does not necessarily control the pod spec.

**Software constraint.** vLLM 0.19.0 (post 0.7+ DP support; native data-parallel deployment is mature).

**Workload.** Offline batch inference (graph-based extraction over partitioned image chunks). The HF data-parallel implementation already works on this cluster — there is no defect on the DP path. Only the vLLM TP path stalls.

---

## 1. Why TP stalls on small `/dev/shm`

vLLM with `tensor_parallel_size > 1` invokes NCCL for the per-layer all-reduce / all-gather and PyTorch distributed for process-group setup. On no-NVLink hardware:

- NCCL falls back to its **SHM transport** for intra-node GPU-to-GPU traffic (next-best after NVLink, P2P, IB).
- NCCL also uses **CUDA IPC handles** to share tensors between worker processes; the IPC table sits on `/dev/shm`.
- PyTorch distributed process-group bootstrap stages buffers on `/dev/shm`.

Default container `/dev/shm` is 64 MB. NCCL's staging buffers and CUDA IPC tables can't fit, and you get exactly the symptoms observed: silent stalls or hangs at the first collective op (often during warmup; sometimes only under load when buffers grow with image-token sequence length).

The HF DP implementation does not trip this because **there are no collectives** — N independent forward passes, no NCCL, no IPC.

---

## 2. Why DP=N, TP=1 avoids the bottleneck

In data-parallel inference:

- Each GPU loads a full copy of the model weights.
- Each engine processes its own request stream.
- For **dense models** (InternVL3.5-8B is dense — non-MoE), DP ranks **do not synchronize per forward pass**. The vLLM docs note that MoE workloads require "empty 'dummy' forward passes" for cross-rank synchronization, but this requirement is MoE-specific.
- Inter-rank communication is via **ZMQ for the control plane** (request routing, scheduler coordination), not the forward pass.

For InternVL3.5-8B at DP=4, TP=1:

- No NCCL collectives during inference.
- No `/dev/shm` dependence for the hot path.
- Independent kernels per GPU; PCIe traffic only for incidental control-plane messages, which are tiny.

This mirrors exactly what your existing HF `MultiGPUOrchestrator` does: independent model per GPU, contiguous image partitioning, results merged in original order.

---

## 3. Memory budget for InternVL3.5-8B at DP=4, TP=1

| Item | Size |
|---|---|
| Model weights (8B in bf16) | ~16 GB |
| Activation buffers, image tile tensors | ~2–3 GB |
| KV cache headroom | ~5–6 GB |
| **Total per GPU** | **~24 GB → fits A10G 24 GB with tight headroom** |

If KV cache pressure is a problem under load (e.g. multi-turn workflows with long shared image prefixes plus many concurrent requests), consider:

- Lowering `gpu_memory_utilization` from default 0.9 → 0.85.
- Setting `max_model_len` to the actual maximum you need (don't accept the model's max).
- Enabling **prefix caching** (you already enabled this in commit `37b3500`) — frees KV blocks faster on multi-turn shared-prefix workloads.

If you ever need to deploy a model that does **not** fit on one A10G (Qwen3.5-27B, InternVL3.5-38B), DP=4, TP=1 won't work. Options for those:

- Move to AWQ / W4A16 quantized variants (4× smaller weights → fit on one GPU).
- Accept TP for those models specifically and fix `/dev/shm` (Option A below).

---

## 4. The three viable options

### Option A — Fix `/dev/shm` at the cluster level

**Cleanest, lowest code change. Try this first if cluster admin will allow it.**

Modify the pod spec to mount `/dev/shm` as a tmpfs `emptyDir`:

```yaml
spec:
  containers:
    - name: vllm
      ...
      volumeMounts:
        - name: dshm
          mountPath: /dev/shm
  volumes:
    - name: dshm
      emptyDir:
        medium: Memory
        sizeLimit: 16Gi
```

**Pros**
- Zero application code change. TP keeps working.
- Unblocks anything else on this cluster that needs SHM (PyTorch DataLoader workers, multiprocessing, NCCL, etc.).
- Standard remedy documented in vLLM, sglang, and Kubernetes inference deployment guides.

**Cons**
- Requires pod-spec edit. May not be possible if you're consuming a managed image / Helm chart you don't control.
- Doesn't fix the architectural concern that TP on PCIe-only hardware pays an all-reduce tax on every layer regardless of `/dev/shm` size. TP works after this fix, but it's still the wrong tool for PCIe-only multi-GPU.

**Recommended size**: 16 GiB. (1 GiB is the documented minimum; 16 GiB is the recommended default for "large model deployments" in 2026 guides.)

---

### Option B — vLLM native DP (`--data-parallel-size 4 --tensor-parallel-size 1`)

**Production-recommended path for InternVL3.5-8B on PCIe-only multi-GPU.**

For an HTTP server deployment:

```bash
vllm serve OpenGVLab/InternVL3_5-8B \
  --data-parallel-size 4 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --gpu-memory-utilization 0.85
```

For offline inference, follow the pattern in `examples/offline_inference/data_parallel.py`:

- Spawn N workers via `multiprocessing.Process` with `spawn` start method (vLLM forces `spawn` on ROCm; on CUDA it's the default safe choice).
- Each worker receives env vars: `VLLM_DP_RANK`, `VLLM_DP_RANK_LOCAL`, `VLLM_DP_SIZE`, `VLLM_DP_MASTER_IP`, `VLLM_DP_MASTER_PORT`.
- `CUDA_VISIBLE_DEVICES` for each rank is **set automatically inside the engine process** by vLLM based on the DP rank.
- Each worker constructs `LLM(tensor_parallel_size=1, ...)` and processes its slice.

**Pros**
- Officially supported pattern; well-documented and matures with each release.
- Workers are independent processes; for dense models there's no per-step inter-rank communication.
- Engine handles GPU pinning automatically.
- May offer scheduler-level optimizations in future releases (cross-rank load balancing for unbalanced batches, etc.).

**Cons**
- A `VLLM_DP_MASTER` coordinator process exists. Empirically (and per the docs' MoE caveat) it is not on the per-step hot path for dense models. **Worth verifying with `NCCL_DEBUG=INFO` that no NCCL transport is initialized at TP=1 in your environment** — the search results don't fully resolve whether the coordinator's existence touches `/dev/shm` in any code path.
- Known regression: [vLLM #17685](https://github.com/vllm-project/vllm/issues/17685) reported offline DP significantly slower in 0.8.2 vs 0.6.4/0.7.2. By 0.19.0 this is presumably resolved, but a benchmark vs a single-GPU baseline is wise before scaling out to 4 GPUs.
- Adds a coordinator dependency to your stack architecture.

---

### Option C — DIY DP via `multiprocessing.spawn` (maximum defense)

**Bulletproof fallback. Same shape as your existing HF `MultiGPUOrchestrator`.**

Spawn N independent Python subprocesses, each with `CUDA_VISIBLE_DEVICES` pinned to one GPU, each constructing a plain `LLM(tensor_parallel_size=1, ...)`. No vLLM-internal coordinator, no DP master process, no shared infrastructure between engines.

```python
import multiprocessing as mp

def worker(gpu_id: int, image_chunk: list, model_path: str, output_q):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
    )
    # ... run extraction graphs over image_chunk via llm.generate(...) ...
    output_q.put((gpu_id, results))

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    output_q = mp.Queue()
    procs = []
    for gpu_id, chunk in enumerate(partitioned_chunks):
        p = mp.Process(target=worker, args=(gpu_id, chunk, MODEL_PATH, output_q))
        p.start()
        procs.append(p)
    results = [output_q.get() for _ in procs]
    for p in procs:
        p.join()
    merged = merge_in_original_order(results)
```

**Pros**
- Zero vLLM-internal coordinator. No DP master. No ZMQ inter-rank traffic. Guaranteed not to touch `/dev/shm`.
- Architecturally identical to your existing `MultiGPUOrchestrator` — easy to plug in alongside the HF DP path. The graph engine doesn't need to change at all.
- Works on any vLLM version. No risk from native-DP regressions or behavior changes between minor releases.
- Easy to reason about: each process is a textbook single-GPU vLLM deployment.

**Cons**
- You manage the worker lifecycle (start, monitor, error handling, shutdown). Crash in one worker doesn't auto-recover.
- You forgo any cross-rank scheduler optimization native DP might offer (none currently relevant for offline batch with contiguous partitioning).
- More code to maintain — but you already have the HF version of this exact pattern, so the implementation effort is small.

---

## 5. Side-by-side comparison

| Dimension | A. Fix `/dev/shm` | B. Native DP | C. DIY DP |
|---|---|---|---|
| **Code change** | None | Moderate (engine config + offline DP launcher) | Small (mirrors HF pattern) |
| **Cluster change** | Pod spec edit (16 GiB tmpfs at `/dev/shm`) | None | None |
| **Touches NCCL?** | Yes (TP collectives still run) | TP=1 → no per-step NCCL. Coordinator init: needs verification. | No |
| **Touches `/dev/shm`?** | Yes (and now it has room) | Verify with `NCCL_DEBUG=INFO` | No |
| **Per-image latency** | Lowest in theory; PCIe overhead is significant | Higher per image, parallelism via DP | Same as B |
| **Aggregate throughput on 4 GPUs** | TP all-reduce tax on PCIe ~ noticeable | Near-linear scaling | Near-linear scaling |
| **VRAM per GPU** | ~5–6 GB (TP-sharded weights) | ~16 GB (full weight replica) | ~16 GB (full weight replica) |
| **Scales to 27B/38B models?** | Yes (TP shards the model) | No (8B fits on one GPU; 27B/38B don't) | No (same constraint as B) |
| **Risk of recurrence** | Low if SHM size is generous | Low for dense models, verify init | Lowest |
| **Operational complexity** | Lowest | Moderate (DP coordinator in stack) | Moderate (custom worker lifecycle) |

---

## 6. Decision framework

The decision hinges on three questions:

**1. Can you get the cluster admin to bump `/dev/shm` to 16 GiB?**

- Yes → **Option A** is the cleanest fix. Done. Reconsider DP later if you need throughput optimization on PCIe.
- No → continue.

**2. Will you deploy any model that doesn't fit on a single A10G (24 GB)?**

- Yes (e.g. InternVL3.5-38B, Qwen3.5-27B in bf16) → you'll need TP for those models, which means you need Option A regardless. Use TP for big models, DP for InternVL3.5-8B.
- No (8B is the only target) → DP is strictly better than TP on PCIe-only A10G.

**3. How much do you trust native DP to never touch `/dev/shm` for dense models?**

- High (verified with `NCCL_DEBUG=INFO`, no NCCL init observed at TP=1) → **Option B** (native DP). Cleanest production deployment.
- Low or want certainty → **Option C** (DIY DP). Mirrors your existing HF pattern, zero risk.

---

## 7. Recommended path

**Primary recommendation**: Start with **Option B (native DP)** and verify it doesn't touch `/dev/shm`. If verified clean, use it. If you observe any SHM-related stalls during verification, fall back to **Option C (DIY DP)**.

If both B and C end up being more than you want to maintain and the cluster admin is amenable, **Option A** stays valid as a low-effort alternative (with the caveat that TP on PCIe still pays an all-reduce tax — the stalls go away but throughput is suboptimal).

Three reasons for B-first:

1. It's the officially supported and documented vLLM pattern.
2. For dense models, the docs (and vLLM source) indicate no per-step cross-rank synchronization.
3. If a future workload needs DP+TP composition (e.g. a 30B model on 4 GPUs as DP=2, TP=2), native DP is the only path.

---

## 8. Verification checklist

Before committing to Option B in production, run these checks on a single image, then on a small batch:

### A. Confirm DP initialization doesn't touch NCCL/SHM at TP=1

```bash
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,COLL,P2P,SHM \
python your_offline_dp_script.py 2>&1 | tee dp_init.log

# Look for these strings:
grep -E "NCCL INFO Init|NCCL INFO comm|via SHM|via P2P" dp_init.log
```

If you see *any* `NCCL INFO Init` or transport selection entries, NCCL is being initialized — this is the signal to fall back to Option C.

### B. Confirm no `/dev/shm` files are created by the engines

```bash
# Before launch
ls -la /dev/shm

# During inference, in another shell
watch -n 1 'ls -la /dev/shm'

# After shutdown
ls -la /dev/shm
```

DIY DP (Option C) and a clean native DP (Option B) should produce no growth in `/dev/shm`.

### C. Benchmark against your single-GPU baseline

There's a known regression ([vLLM #17685](https://github.com/vllm-project/vllm/issues/17685)) where offline DP throughput dropped in 0.8.2. Confirm 0.19 doesn't carry forward any of those regressions:

```
Baseline: 1× GPU, single LLM(), 100 images
Native DP: 4× GPU via --data-parallel-size 4, 100 images
DIY DP: 4× independent LLM() processes, 100 images
```

You should see ~3–4× throughput on 4 GPUs vs single GPU. Anything below ~2.5× on a balanced workload suggests something's wrong — investigate before going to production.

### D. Run with prefix caching for multi-turn workloads

Already enabled in commit `37b3500`. Confirm KV reuse during your bank-statement and transaction-linking workflows by checking vLLM's prefix cache hit rate metrics. Multi-turn over the same image should hit very high cache rates (the image prefix is shared turn-to-turn).

---

## 9. Implementation impact on the codebase

Whichever option you pick, the changes are scoped:

### Option A (fix `/dev/shm`)
- No application code change.
- Pod spec / Helm chart edit (cluster-side).

### Option B (native DP)
- `models/backends/vllm_backend.py` — set `tensor_parallel_size=1` in the LLM constructor; remove any TP-specific config.
- `models/registry.py` — `VllmSpec` may need a `data_parallel_size` field to control DP from config.
- `common/multi_gpu.py` — likely needs a vLLM-aware path that uses native DP launcher rather than the HF per-thread model loader.
- `cli.py` — `--num-gpus` may need to map to `--data-parallel-size` for vLLM backends.
- New offline DP launcher script (or integrate the spawn pattern into `MultiGPUOrchestrator`).

### Option C (DIY DP)
- Extend `common/multi_gpu.py`'s `MultiGPUOrchestrator` to support a vLLM worker variant — same `multiprocessing.Process` shape it already uses for HF.
- Each worker sets `CUDA_VISIBLE_DEVICES` and constructs `LLM(tensor_parallel_size=1, ...)`.
- Image partitioning, result merging, and post-processing reuse the existing HF DP machinery.
- Result: vLLM and HF backends share the exact same multi-GPU orchestration layer. Architecturally cleaner than option B.

---

## 10. Open questions to resolve during investigation

1. **Does vLLM 0.19.0 native DP create a `VLLM_DP_MASTER` coordinator process even at DP=4 dense?** And does that coordinator touch `/dev/shm` for IPC?
2. **What is the actual size of `/dev/shm` on your KFP pod today?** (`df -h /dev/shm` inside a running pod.) If it's already 8+ GiB, the diagnosis may be wrong and the stall has another cause.
3. **What's the exact stall trace?** A backtrace at the hang would pin the problem precisely. Look for: `pynccl.allreduce`, `torch.distributed.init_process_group`, `cuda.IPC.openMemHandle`. Each implicates a different fix.
4. **Is the cluster `/dev/shm` configurable per-pod by you, or only by cluster admins?** Affects whether Option A is on the table.
5. **What vLLM minor version exactly?** "0.19.0" pins the major/minor, but patch versions can carry fixes for regressions like #17685.

---

## 11. References

### vLLM documentation
- [Data Parallel Deployment — vLLM docs](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/)
- [Parallelism and Scaling — vLLM docs](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [InternVL3.5 vLLM Recipe](https://docs.vllm.ai/projects/recipes/en/latest/InternVL/InternVL3_5.html)

### Source / examples
- [`examples/offline_inference/data_parallel.py` — vLLM repo](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/data_parallel.py)

### Forums and community discussion
- [Data Parallel vs Tensor Parallel — vLLM Forum](https://discuss.vllm.ai/t/data-parallel-or-tensor-parallel/1608)
- [InternVL3.5-8B deployment QPS thread — vLLM Forum](https://discuss.vllm.ai/t/what-is-the-recommended-method-to-start-up-the-vllm-server-engine-for-inferencing-for-internvl3-5-8b-getting-2-qps/1604)
- [How to Achieve Data-Parallel Offline Batch Inference — vLLM Discussion #14283](https://github.com/vllm-project/vllm/discussions/14283)

### Known issues
- [Document how to configure `/dev/shm` size — vllm-project/production-stack#44](https://github.com/vllm-project/production-stack/issues/44)
- [Offline DP slower in 0.8.2 — vLLM Issue #17685](https://github.com/vllm-project/vllm/issues/17685)
- [Crash with `--tensor-parallel-size` in docker container — vLLM Issue #1710](https://github.com/vllm-project/vllm/issues/1710)
- [Document how to configure shared memory for multi-GPU deployments — sgl-project/sglang#4259](https://github.com/sgl-project/sglang/issues/4259)

### Production deployment guides (2026)
- [vLLM Optimization Guide: How to Avoid Performance Pitfalls in Multi-GPU Inference — DatabaseMart](https://www.databasemart.com/blog/vllm-distributed-inference-optimization-guide)
- [Deploying vLLM at Scale on Kubernetes — dasroot.net (Feb 2026)](https://dasroot.net/posts/2026/02/deploying-vllm-scale-kubernetes/)
- [vLLM Production Deployment: Complete 2026 Guide — SitePoint](https://www.sitepoint.com/vllm-production-deployment-guide-2026/)
- [Multi-GPU and Tensor Parallel LLM Inference on Kubernetes — kubernetes.recipes](https://kubernetes.recipes/recipes/ai/multi-gpu-llm-inference/)

---

## 12. TL;DR for skim-readers

- **Diagnosis**: vLLM TP stalls because PCIe-only A10G falls back to NCCL SHM transport, and your container's `/dev/shm` is too small for the staging buffers and CUDA IPC tables. Dense-model DP avoids all per-step collectives, so it sidesteps the issue.
- **Best path for InternVL3.5-8B**: Native vLLM DP (`--data-parallel-size 4 --tensor-parallel-size 1`). Verify with `NCCL_DEBUG=INFO` that no transport is initialized at TP=1.
- **Bulletproof fallback**: DIY DP via `multiprocessing.spawn` — N independent `LLM(tensor_parallel_size=1)` processes, each with its own `CUDA_VISIBLE_DEVICES`. Mirrors your existing HF `MultiGPUOrchestrator` exactly. Recommended if any verification raises doubt.
- **If cluster admin will bump `/dev/shm` to 16 GiB**: TP works again. Lowest code change. But TP on PCIe still pays a per-layer all-reduce tax, so DP is architecturally better for this hardware regardless of whether the SHM fix is available.
- **Watch out for**: 27B/38B models won't fit single-GPU; for those you'd need TP (and therefore the SHM fix) or a quantized variant. Plan for the model mix you actually deploy.
