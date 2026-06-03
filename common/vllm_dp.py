"""vLLM data-parallel orchestrator.

Spawns N independent vLLM engine processes, each pinned to one GPU
via CUDA_VISIBLE_DEVICES. No NCCL, no /dev/shm, no inter-process
communication during inference.
"""

import importlib
import logging
import multiprocessing as mp
import os
import traceback
from pathlib import Path
from queue import Empty
from typing import Any

from common.app_config import AppConfig
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
    """Deal images round-robin into n balanced chunks (stride partition).

    Images arrive sorted by document type (expensive bank statements first,
    per ``extraction_order``). Contiguous chunking would pile every bank
    statement onto GPU 0 — it becomes the straggler while the other GPUs race
    through cheap receipts. Dealing round-robin (``images[i::n]``) spreads each
    type evenly across GPUs, so every GPU gets roughly the same number of
    banks / invoices / receipts and thus a balanced workload.

    Per-GPU prefix-cache locality is preserved: the extract worker re-derives
    its processing order from the global type-sorted list, so each GPU still
    processes its banks together, then its invoices, then its receipts.
    """
    n = min(n, len(images))
    return [images[i::n] for i in range(n)]


def _reconstruct_order(
    gpu_results: dict[int, list[dict[str, Any]]], actual_gpus: int
) -> list[dict[str, Any]]:
    """Reassemble round-robin chunk results back into original image order.

    Chunk ``g`` (from :func:`_partition_images`) holds the images at original
    indices ``g, g + actual_gpus, g + 2*actual_gpus, ...`` and the worker
    returns its results in that same order. Interleave them back so the merged
    list matches the input ``images`` order.
    """
    indexed: list[tuple[int, dict[str, Any]]] = []
    for gpu_id in range(actual_gpus):
        for pos, rec in enumerate(gpu_results[gpu_id]):
            indexed.append((pos * actual_gpus + gpu_id, rec))
    indexed.sort(key=lambda x: x[0])
    return [rec for _, rec in indexed]


def run_dp(
    *,
    num_gpus: int,
    images: list[Path],
    worker_fn: str,
    worker_kwargs: dict[str, Any],
    app_config: AppConfig | None = None,
) -> list[dict[str, Any]]:
    """Launch N workers, collect merged results in original image order.

    No global timeout: workers run until they finish or die. A slow-but-alive
    worker (e.g. a GPU grinding through dense bank statements) is never falsely
    killed — the collect loop polls the queue every ``poll_interval`` seconds
    and only fails if a worker is found dead (OOM/SIGKILL) without having
    written a result.

    Args:
        num_gpus: Number of GPUs / DP ranks.
        images: Full image list (will be partitioned).
        worker_fn: Dotted path to a top-level function with signature:
            (gpu_id: int, image_paths: list[str], **worker_kwargs) -> list[dict]
        worker_kwargs: Extra kwargs forwarded to worker_fn (must be picklable).
        app_config: Optional application config. When supplied, the process
            join timeout is read from ``app_config.get_infra("dp_join_timeout")``.

    Returns:
        Merged list of result dicts in original image order.
    """
    join_timeout = int(app_config.get_infra("dp_join_timeout")) if app_config else 60
    poll_interval = 10  # seconds between liveness checks
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

    # Collect results — no global timeout. Poll the queue on a short interval
    # and check worker liveness. Workers that complete normally put results on
    # the queue; workers that hit a Python-level error put an exception. The
    # only unhandled case is a worker killed by the OS (OOM, SIGKILL) without
    # writing to the queue — detected via is_alive() + exitcode so a slow but
    # healthy worker is never falsely timed out.
    gpu_results: dict[int, list[dict]] = {}
    try:
        remaining = actual_gpus
        while remaining > 0:
            try:
                gpu_id, payload = result_queue.get(timeout=poll_interval)
            except Empty:
                for i, p in enumerate(procs):
                    if not p.is_alive() and p.exitcode != 0 and i not in gpu_results:
                        msg = f"GPU {i} worker died (exit code {p.exitcode}) without returning results"
                        raise RuntimeError(msg) from None
                continue
            if isinstance(payload, Exception):
                raise payload
            gpu_results[gpu_id] = payload
            remaining -= 1
            logger.info("GPU %d finished: %d images", gpu_id, len(payload))
    except Exception:
        for p in procs:
            if p.is_alive():
                p.terminate()
        raise
    finally:
        for p in procs:
            p.join(timeout=join_timeout)

    # Check for worker crashes
    for i, p in enumerate(procs):
        if p.exitcode and p.exitcode != 0:
            msg = f"GPU {i} worker exited with code {p.exitcode}"
            raise RuntimeError(msg)

    # Chunks were dealt round-robin (images[i::n]); interleave the per-GPU
    # results back into the original image order.
    return _reconstruct_order(gpu_results, actual_gpus)


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
