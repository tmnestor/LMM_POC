# Limited Shared Memory Problem

When vLLM runs with tensor parallelism (TP > 1) across multiple GPUs, it uses NCCL for inter-GPU communication. On our PCIe-only A10G cluster (no NVLink), NCCL falls back to its shared-memory (SHM) transport, staging buffers and CUDA IPC handles in `/dev/shm`. Our KFP pods ship with the default 64 MB `/dev/shm`, which is far too small for these buffers -- NCCL's staging tables grow with sequence length and can't fit, causing the inference process to silently stall or hang at the first collective operation. The fix is either to increase `/dev/shm` to 16 GiB via a tmpfs `emptyDir` volume mount in the pod spec (`emptyDir: {medium: Memory, sizeLimit: 16Gi}`), or -- as we've implemented on the application side -- to switch from tensor parallelism to data parallelism (DP=4, TP=1), where each GPU runs an independent model copy with no NCCL collectives and therefore no `/dev/shm` dependency. Ideally both: the pod spec fix unblocks any future workload that needs shared memory, while DP gives us better throughput on PCIe-only hardware regardless.

## vLLM Single-GPU Smoke Test

```bash
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
CUDA_VISIBLE_DEVICES=0 python -c "
from vllm import LLM
llm = LLM(
    model='/home/jovyan/nfs_share/models/InternVL3_5-8B',
    tensor_parallel_size=1,
    max_model_len=8192,
    trust_remote_code=True,
    enforce_eager=True,
)
print('Engine initialized OK')
del llm
"
```
