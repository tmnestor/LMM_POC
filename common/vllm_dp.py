"""vLLM data-parallel orchestrator.

Spawns N independent vLLM engine processes, each pinned to one GPU
via CUDA_VISIBLE_DEVICES. No NCCL, no /dev/shm, no inter-process
communication during inference.
"""

import importlib
import logging
import math
import multiprocessing as mp
import os
import traceback
from pathlib import Path
from typing import Any

from common.pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)


def resolve_gpu_count(config: PipelineConfig) -> int:
    """Resolve effective GPU count for vLLM DP.

    Priority: data_parallel_size > num_gpus > auto-detect.
    """
    if config.data_parallel_size is not None:
        return config.data_parallel_size
    if config.num_gpus > 0:
        return config.num_gpus
    try:
        import torch

        return max(1, torch.cuda.device_count())
    except ImportError:
        return 1


def _partition_images(images: list[Path], n: int) -> list[list[Path]]:
    """Split images into n contiguous chunks."""
    n = min(n, len(images))
    chunk_size = math.ceil(len(images) / n)
    return [images[i : i + chunk_size] for i in range(0, len(images), chunk_size)]


def run_dp(
    *,
    num_gpus: int,
    images: list[Path],
    worker_fn: str,
    worker_kwargs: dict[str, Any],
    timeout: int = 3600,
) -> list[dict[str, Any]]:
    """Launch N workers, collect merged results in original image order.

    Args:
        num_gpus: Number of GPUs / DP ranks.
        images: Full image list (will be partitioned).
        worker_fn: Dotted path to a top-level function with signature:
            (gpu_id: int, image_paths: list[str], **worker_kwargs) -> list[dict]
        worker_kwargs: Extra kwargs forwarded to worker_fn (must be picklable).
        timeout: Max seconds to wait for all workers (default 1 hour).

    Returns:
        Merged list of result dicts in original image order.
    """
    chunks = _partition_images(images, num_gpus)
    actual_gpus = len(chunks)

    logger.info(
        "vLLM DP: distributing %d images across %d GPUs (TP=1 per GPU)",
        len(images),
        actual_gpus,
    )

    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    procs: list[Any] = []  # SpawnProcess (mp.Process subtype)

    for gpu_id, chunk in enumerate(chunks):
        p = ctx.Process(
            target=_worker_wrapper,
            args=(gpu_id, chunk, worker_fn, worker_kwargs, result_queue),
        )
        p.start()
        procs.append(p)

    # Collect results
    gpu_results: dict[int, list[dict]] = {}
    try:
        for _ in range(actual_gpus):
            gpu_id, payload = result_queue.get(timeout=timeout)
            if isinstance(payload, Exception):
                raise payload
            gpu_results[gpu_id] = payload
            logger.info("GPU %d finished: %d images", gpu_id, len(payload))
    except Exception:
        for p in procs:
            if p.is_alive():
                p.terminate()
        raise
    finally:
        for p in procs:
            p.join(timeout=60)

    # Check for worker crashes
    for i, p in enumerate(procs):
        if p.exitcode and p.exitcode != 0:
            msg = f"GPU {i} worker exited with code {p.exitcode}"
            raise RuntimeError(msg)

    # Merge in GPU order (preserves original image order since chunks are contiguous)
    merged: list[dict[str, Any]] = []
    for gpu_id in range(actual_gpus):
        merged.extend(gpu_results[gpu_id])

    return merged


def _worker_wrapper(
    gpu_id: int,
    image_chunk: list[Path],
    worker_fn_path: str,
    worker_kwargs: dict[str, Any],
    result_queue: mp.Queue,
) -> None:
    """Subprocess entry point. Pins GPU, imports worker_fn, runs it."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        module_path, fn_name = worker_fn_path.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        fn = getattr(mod, fn_name)

        results = fn(
            gpu_id=gpu_id,
            image_paths=[str(p) for p in image_chunk],
            **worker_kwargs,
        )
        result_queue.put((gpu_id, results))
    except Exception:
        tb = traceback.format_exc()
        result_queue.put((gpu_id, RuntimeError(f"GPU {gpu_id} worker failed:\n{tb}")))
