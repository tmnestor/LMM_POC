"""SROIE benchmark stage.

Runs the ICDAR 2019 SROIE Task 3 key-information benchmark through the
repo's own model stack, so the tiling, tokenizer and chat-template
handling under test are the ones production uses.

The stage is deliberately thin: everything model-independent lives in
``common/sroie/`` and is unit-tested without a GPU. Swapping the model is a
``bootstrap.model.type`` change in run_config.yml, which is what keeps the
InternVL3 and Gemma 4 runs comparable — same prompt, same parser, same
scorer, byte for byte.

Run it through the entrypoint, never directly:

    KFP_TASK=sroie bash entrypoint.sh
"""

import logging
import time
from pathlib import Path
from typing import Any

import typer

from common.sroie.config import SroieSettings
from common.sroie.ground_truth import SroieRecord, load_sroie_split
from common.sroie.prompt import SROIE_PROMPT
from common.sroie.report import (
    execution_summary_rows,
    render_execution_summary,
    write_per_image_csv,
    write_summary_json,
)
from common.sroie.runner import benchmark_from_worker_responses, run_benchmark
from common.sroie.scoring import MatchPolicy, score_records

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)

# run_dp resolves this by dotted string in a subprocess, so a typo here
# would only surface minutes into a GPU run. Named as a constant so a test
# can import it and check it resolves.
SROIE_WORKER_FN = "common.vllm_dp_workers.sroie_worker"


def run(
    *,
    model_type: str | None = None,
    max_images: int | None = None,
    config_path: Path | None = None,
) -> Path:
    """Run the SROIE benchmark and write its artefacts.

    Args:
        model_type: Model type override. None = bootstrap.model.type.
        max_images: Score only the first N records. None = the whole split.
        config_path: Optional path to run_config.yml.

    Returns:
        Path to the summary JSON.

    Raises:
        SroieConfigError: If the pipeline.sroie block is missing a key.
        SroieDatasetError: If the split on disk is unusable.
    """
    from common.app_config import AppConfig
    from common.pipeline_config import load_yaml_config
    from common.pipeline_ops import create_processor, load_model
    from common.vllm_dp import run_dp
    from PIL import Image

    from cli import load_pipeline_configs

    cli_args: dict[str, object] = {}
    if model_type:
        cli_args["model_type"] = model_type

    app_cfg = AppConfig.load(cli_args, config_path=config_path)
    config = app_cfg.pipeline

    # AppConfig keeps the raw mapping local to load(), so read it here
    # rather than widening shared config code for one stage. Same file,
    # same resolution order.
    resolved_config = config_path or Path(__file__).parent.parent / "config" / "run_config.yml"
    _, raw_config = load_yaml_config(resolved_config)
    settings = SroieSettings.from_raw(raw_config, config_path=resolved_config)

    records = load_sroie_split(settings.data_dir)
    if max_images is not None:
        records = records[:max_images]
    logger.info("SROIE: %d records from %s", len(records), settings.data_dir)

    # SROIE is 100% receipts, so the tile budget is the RECEIPT budget from
    # inference.tiling.budgets — the same one production uses. It only takes
    # effect when inference.tiling.pre_tiling is enabled; with pre-tiling off,
    # vLLM tiles internally and this is inert. Gemma 4 cannot pre-tile at all
    # (its loader rejects it), so each model is measured as it is served.
    tile_budget = app_cfg.get_image_budget("receipt")
    logger.info(
        "Receipt tile budget: %s (applies only when pre-tiling is enabled)",
        tile_budget,
    )

    # -- vLLM data-parallel fast path -----------------------------------------
    # 347 receipts are independent single-image requests, so one engine per
    # GPU (DP) beats one model split across GPUs (TP): the work partitions
    # perfectly and nothing is shared. Mirrors stages/classify.py.
    dp_gpus = _resolve_dp_ranks(config)
    if dp_gpus:
        started = time.time()
        worker_records = run_dp(
            num_gpus=dp_gpus,
            images=[record.image_path for record in records],
            worker_fn=SROIE_WORKER_FN,
            worker_kwargs={
                "config_path": str(config_path) if config_path else None,
                "cli_overrides": cli_args,
                "max_new_tokens": settings.max_new_tokens,
                "max_tiles": tile_budget["max_tiles"],
            },
            app_config=app_cfg,
        )
        elapsed = time.time() - started
        result = benchmark_from_worker_responses(records, worker_records)
        return _finish(
            records=records,
            result=result,
            settings=settings,
            model_name=config.model_type,
            elapsed=elapsed,
            execution_mode=f"data-parallel ({dp_gpus} GPUs)",
        )

    # -- Single-engine path ---------------------------------------------------
    prompt_config, universal_fields, field_definitions = load_pipeline_configs(config.model_type)

    logger.info("Loading model: %s", config.model_type)
    model_cm = load_model(config, app_config=app_cfg)
    model, tokenizer = model_cm.__enter__()

    try:
        processor = create_processor(
            model,
            tokenizer,
            config,
            prompt_config,
            universal_fields,
            field_definitions,
            app_config=app_cfg,
        )

        def generate(record: SroieRecord) -> str:
            """Ask the model for one receipt's four fields."""
            with Image.open(record.image_path) as image:
                return processor.generate(
                    image.convert("RGB"),
                    SROIE_PROMPT,
                    max_tokens=settings.max_new_tokens,
                    extra={"max_tiles": tile_budget["max_tiles"]},
                )

        started = time.time()
        result = run_benchmark(records, generate)
        elapsed = time.time() - started
    finally:
        model_cm.__exit__(None, None, None)

    return _finish(
        records=records,
        result=result,
        settings=settings,
        model_name=config.model_type,
        elapsed=elapsed,
        execution_mode="single-engine",
    )


def _resolve_dp_ranks(config: Any) -> int | None:
    """Return the DP rank count, or None for the single-engine path.

    Prefers ``resolve_dp_gpus``, which also refuses DP for models that
    cannot hold one whole engine per GPU. That helper exists only on the
    branches carrying Gemma 4; elsewhere fall back to the plain GPU count,
    which is what stages/classify.py uses. Keeping both here lets this
    file stay byte-identical across the model branches — the thing the
    whole comparison rests on.
    """
    from models.registry import is_vllm_model

    if not is_vllm_model(config.model_type):
        return None

    try:
        # Absent on the InternVL3 line, so mypy cannot see it from here.
        from common.vllm_dp import resolve_dp_gpus  # type: ignore[attr-defined]
    except ImportError:
        from common.vllm_dp import resolve_gpu_count

        gpus = resolve_gpu_count(config)
        return gpus if gpus > 1 else None

    ranks: int | None = resolve_dp_gpus(config, config.model_type)
    return ranks


def _finish(
    *,
    records: list[SroieRecord],
    result: Any,
    settings: SroieSettings,
    model_name: str,
    elapsed: float,
    execution_mode: str,
) -> Path:
    """Score, write both artefacts, and surface any inference failures."""
    scores = {policy: score_records(records, result.predictions, policy=policy) for policy in MatchPolicy}

    csv_path = settings.output_dir / "sroie_per_image.csv"
    summary_path = settings.output_dir / "sroie_summary.json"
    write_per_image_csv(csv_path, records, result.predictions)
    write_summary_json(
        summary_path,
        model_name=model_name,
        scores=scores,
        image_count=len(records),
        elapsed_seconds=elapsed,
        execution_mode=execution_mode,
    )

    for policy in MatchPolicy:
        logger.info("SROIE %s overall F1: %.4f", policy.value, scores[policy].overall_f1)
    logger.info("Wrote %s and %s", csv_path, summary_path)

    render_execution_summary(
        execution_summary_rows(
            image_count=len(records),
            elapsed_seconds=elapsed,
            scores=scores,
            output_dir=settings.output_dir,
            execution_mode=execution_mode,
            failed_images=len(result.errors),
        )
    )

    # Artefacts are written first so a partly-failed run is still
    # inspectable, then the failure is surfaced. A run with inference
    # errors must not be read as a clean measurement of the model.
    if result.errors:
        logger.error(
            "%d image(s) failed inference: %s",
            len(result.errors),
            ", ".join(sorted(result.errors)[:10]),
        )
        raise typer.Exit(1) from None

    return summary_path


@app.command()
def main(
    model_type: str = typer.Option(None, "--model-type", help="Override bootstrap.model.type."),
    max_images: int = typer.Option(None, "--max-images", help="Score only the first N records."),
    config_path: Path = typer.Option(None, "--config", help="Path to run_config.yml."),
) -> None:
    """Run the SROIE benchmark stage."""
    # Per-image progress is always at INFO — it is the only sign the user
    # has that inference is advancing. A 347-image run takes ~35 minutes
    # and writes its artefacts only at the end, so without this the stage
    # is indistinguishable from a hang.
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    run(model_type=model_type, max_images=max_images, config_path=config_path)


if __name__ == "__main__":
    app()
