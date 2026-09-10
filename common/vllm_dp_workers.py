"""vLLM data-parallel worker functions.

Each function is a top-level callable invoked in a subprocess by
``vllm_dp.run_dp``. Workers build their own vLLM engine (TP=1)
inside a process pinned to a single GPU via CUDA_VISIBLE_DEVICES.

All arguments must be picklable (strings, dicts, ints -- no model
objects, no Path objects across the process boundary).
"""

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def quality_screen_worker(
    gpu_id: int,
    image_paths: list[str],
    *,
    config_path: str | None,
    cli_overrides: dict[str, Any],
    variant: str | None = None,
    tile_extra: dict[str, Any] | None = None,
    screened_at: str | None = None,
) -> list[dict[str, Any]]:
    """Worker: build vLLM engine + processor, screen each image's quality.

    Delegates to the same `run_quality_screen` the single-GPU path uses, so
    the record shape is shared by construction rather than duplicated here.
    A DP run and a single-GPU run must produce interchangeable output --
    evaluate cannot tell which produced the file it reads.

    Args:
        gpu_id: GPU rank (for logging).
        image_paths: Absolute paths to images (strings).
        config_path: Path to run_config.yml (or None).
        cli_overrides: CLI args dict for AppConfig.load().
        variant: Prompt variant override. Must be threaded through from the
            parent: a worker falling back to the configured variant while the
            parent was told to run another one would shard a single run across
            two different prompts.
        tile_extra: Tile budget, threaded from the parent for the same reason.
            A worker without it skips pre-tiling, and its shard would be judged
            at a different resolution from every other shard.

    Returns:
        List of quality-screen record dicts.
    """
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(levelname)s [GPU {gpu_id}] %(name)s: %(message)s",
    )

    from common.app_config import AppConfig
    from common.pipeline_ops import create_processor, load_model
    from common.quality_screen_parser import load_screen_vocabulary
    from stages.quality_screen import orchestrator_inference, run_quality_screen

    # Force single-GPU in this worker
    overrides = dict(cli_overrides)
    overrides["num_gpus"] = 1

    cfg_path = Path(config_path) if config_path else None
    app_cfg = AppConfig.load(overrides, config_path=cfg_path)
    config = app_cfg.pipeline

    screen_cfg = app_cfg.quality_screen_config
    resolved_variant = variant or screen_cfg["variant"]
    vocabulary = load_screen_vocabulary(Path(screen_cfg["prompt_file"]), variant=resolved_variant)
    max_tokens = app_cfg.get_token_budget("quality_screen")

    logger.info("Loading model: %s (GPU %d)", config.model_type, gpu_id)
    model_cm = load_model(config, app_config=app_cfg)
    model, tokenizer = model_cm.__enter__()

    try:
        processor = create_processor(model, tokenizer, config, app_config=app_cfg)

        started = time.time()
        records = run_quality_screen(
            image_paths,
            infer=orchestrator_inference(
                processor, max_tokens, verbose=config.verbose, tile_extra=tile_extra
            ),
            vocabulary=vocabulary,
            variant=resolved_variant,
            # Stamped here as well as on the single-GPU path. A DP run whose
            # records carried no tile budget would look like a settings change
            # to the next run's resume check, which would then rescreen the
            # whole corpus -- silently, and every time.
            tiling=tile_extra,
            # Threaded from the parent, not generated here: each worker starting
            # its own clock would stamp one run with several timestamps and make
            # a single sharded run look like several resumed ones.
            screened_at=screened_at,
        )
        elapsed = time.time() - started

        malformed = sum(1 for record in records if record["malformed"])
        logger.info(
            "[GPU %d] screened %d images in %.1fs (%d malformed)",
            gpu_id,
            len(records),
            elapsed,
            malformed,
        )

        # Tag every record with the rank that produced it. Workers run
        # concurrently, so the parent needs this to report the SLOWEST
        # worker's inference time -- summing across workers would give total
        # compute and understate throughput by roughly the GPU count.
        for record in records:
            record["gpu_id"] = gpu_id
        return records
    finally:
        model_cm.__exit__(None, None, None)
