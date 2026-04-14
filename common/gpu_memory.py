"""GPU memory utilities for batch sizing and inter-inference cleanup."""

import gc
import logging

logger = logging.getLogger(__name__)


def get_available_memory(device: str = "cuda") -> float:
    """Available GPU memory in GB, suitable for batch-size decisions.

    Args:
        device:
            "cuda"   -- min(free) * world_size (safe for device_map="auto").
            "cuda:N" -- free memory on GPU N only.

    Returns:
        Available memory in GB, or 0.0 on CPU / no CUDA.
    """
    import torch

    if not torch.cuda.is_available() or device == "cpu":
        return 0.0

    try:
        if ":" in device:
            idx = int(device.split(":")[-1])
            props = torch.cuda.get_device_properties(idx)
            return (props.total_memory - torch.cuda.memory_allocated(idx)) / (1024**3)

        world_size = torch.cuda.device_count()
        min_free = float("inf")
        for gpu_id in range(world_size):
            props = torch.cuda.get_device_properties(gpu_id)
            free = (props.total_memory - torch.cuda.memory_allocated(gpu_id)) / (
                1024**3
            )
            min_free = min(min_free, free)

        return min_free * max(world_size, 1)

    except Exception as e:
        logger.warning("GPU memory detection failed: %s", e)
        return 24.0  # Safe fallback for A10G / L4


def release_memory(*, threshold_gb: float = 1.0) -> None:
    """Release fragmented GPU memory if the gap exceeds threshold.

    Call this between inference calls (outside except blocks).
    Does nothing on CPU or when fragmentation is below threshold.

    Args:
        threshold_gb: Only act when (reserved - allocated) > this value.
    """
    import torch

    if not torch.cuda.is_available():
        return

    total_alloc = 0.0
    total_reserved = 0.0
    for gpu_id in range(torch.cuda.device_count()):
        total_alloc += torch.cuda.memory_allocated(gpu_id) / (1024**3)
        total_reserved += torch.cuda.memory_reserved(gpu_id) / (1024**3)

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
