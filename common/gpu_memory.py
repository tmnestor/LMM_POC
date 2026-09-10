"""Inter-inference GPU memory cleanup.

One function. `get_available_memory` used to live here too, read by the
auto-batch-size machinery; that machinery is gone -- no backend ever
implemented batched inference -- and it went with it.
"""

import gc
import logging

logger = logging.getLogger(__name__)


def release_memory(*, threshold_gb: float = 1.0, device: str | None = None) -> None:
    """Release fragmented GPU memory if the gap exceeds threshold.

    Call this between inference calls (outside except blocks).
    Does nothing on CPU or when fragmentation is below threshold.

    Args:
        threshold_gb: GPU memory threshold in GB. Only act when
            (reserved - allocated) exceeds this value. Callers should pass
            ``app_config.get_infra("gpu_memory_threshold_gb")`` when available.
        device:
            ``None`` or ``"cuda"`` — aggregate across all visible GPUs and
            release on all of them (legacy behaviour; suitable for
            single-process / device_map="auto" models).
            ``"cuda:N"`` — only inspect and release GPU N. In multi-GPU
            worker threads this prevents ``empty_cache()`` /
            ``synchronize()`` from targeting the main thread's current
            device (cuda:0) and causing asymmetric fragmentation.
            ``"cpu"`` — no-op.
    """
    import torch

    if not torch.cuda.is_available() or device == "cpu":
        return

    gib = 1024**3

    # Per-device path: only inspect/release this one GPU.
    if device is not None and ":" in device:
        try:
            gpu_id = int(device.split(":")[-1])
        except ValueError:
            logger.warning("release_memory: unparsable device %r", device)
            return

        alloc = torch.cuda.memory_allocated(gpu_id) / gib
        reserved = torch.cuda.memory_reserved(gpu_id) / gib
        fragmentation = reserved - alloc
        if fragmentation <= threshold_gb:
            return

        logger.warning(
            "GPU %d fragmentation %.2f GB > %.2f GB threshold -- releasing",
            gpu_id,
            fragmentation,
            threshold_gb,
        )

        # Pin the current CUDA device so empty_cache / synchronize hit
        # the intended GPU, not whatever the calling thread inherited.
        prev = torch.cuda.current_device()
        try:
            torch.cuda.set_device(gpu_id)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        finally:
            torch.cuda.set_device(prev)
        return

    # Aggregate path (device is None or "cuda"): scan and release all GPUs.
    total_alloc = 0.0
    total_reserved = 0.0
    for gpu_id in range(torch.cuda.device_count()):
        total_alloc += torch.cuda.memory_allocated(gpu_id) / gib
        total_reserved += torch.cuda.memory_reserved(gpu_id) / gib

    fragmentation = total_reserved - total_alloc
    if fragmentation <= threshold_gb:
        return

    logger.warning(
        "GPU fragmentation %.2f GB > %.2f GB threshold -- releasing",
        fragmentation,
        threshold_gb,
    )

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
