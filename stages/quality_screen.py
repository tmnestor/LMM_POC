"""Stage: run the image-quality screen over a directory of images.

This is the `classify` half of a two-stage `classify -> evaluate` path. There
is no `clean` stage between them: `clean` exists to normalise free-text field
values before comparison, and this screen's answers are already canonical
tokens (YES/NO, and one of the declared OVERALL levels), so there is nothing
to normalise.

Inference is injected rather than constructed here, so the stage's bookkeeping
is testable on CPU. What that bookkeeping owes the run:

  * one record per image, in input order
  * malformed responses kept as records, never dropped -- an image that
    silently vanishes between the corpus and the report shrinks the
    denominator and flatters the score
  * the raw response on every record, so a malformed verdict can be audited
    without paying for inference again
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from common.quality_screen_parser import (
    ScreenVocabulary,
    load_screen_vocabulary,
    parse_quality_response,
)

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)

# The DP worker, named for `run_dp` to import by string. A constant rather than
# an inline literal so a test can resolve it: a typo here is invisible until a
# GPU run has already loaded the model.
DP_WORKER = "common.vllm_dp_workers.quality_screen_worker"

# Takes image paths and one prompt, returns one raw response per image, in the
# same order. The real implementation batches through the model backend; tests
# pass a function over canned text.
InferenceFn = Callable[[list[str], str], list[str]]


def run_quality_screen(
    image_paths: list[str],
    *,
    infer: InferenceFn,
    vocabulary: ScreenVocabulary,
    variant: str | None = None,
) -> list[dict]:
    """Screen every image and return one record each.

    Args:
        image_paths: Images to screen, in the order they should be reported.
        infer: Callable running the prompt over the images.
        vocabulary: The prompt, its criteria and its permitted OVERALL levels.
        variant: Name of the variant that produced these answers, stamped on
            every record. Without it the evaluate stage has to guess from
            config, and a run screened with one prompt can be scored against
            another's criteria and polarity -- which fails loudly on a
            criteria mismatch and silently on a polarity one.

    Returns:
        One record per image, in input order.

    Raises:
        ValueError: Inference returned a different number of responses than
            images, which would silently misalign every record after the gap.
    """
    if not image_paths:
        return []

    responses = infer(image_paths, vocabulary.prompt)

    if len(responses) != len(image_paths):
        raise ValueError(
            f"Inference returned the wrong number of responses.\n"
            f"  What:        {len(image_paths)} images were sent but {len(responses)} responses "
            f"came back, so responses and images cannot be paired.\n"
            f"  Where:       the inference callable passed to run_quality_screen.\n"
            f"  Expected:    one response per image, in the same order.\n"
            f"  How to fix:  return a response for every image, including a placeholder for "
            f"any that failed, rather than omitting it."
        )

    records = []
    # `strict=True` is deliberately redundant with the length check above, and
    # no test can reach it while that check stands. It is kept as the backstop:
    # if the check is ever removed, this fails loudly instead of truncating to
    # the shorter list and scoring each image against another image's answers.
    for path, raw in zip(image_paths, responses, strict=True):
        result = parse_quality_response(
            raw, criteria=vocabulary.criteria, overall_levels=vocabulary.overall_levels
        )
        records.append(
            {
                "image_path": path,
                "image_name": Path(path).name,
                "variant": variant,
                "answers": result.answers,
                "overall": result.overall,
                "malformed": result.malformed,
                "malformed_reason": result.malformed_reason,
                "think_drift": result.think_drift,
                "raw_response": raw,
            }
        )
    return records


def orchestrator_inference(
    orchestrator,
    max_tokens: int,
    *,
    verbose: bool = False,
    tile_extra: dict | None = None,
) -> InferenceFn:
    """Adapt a loaded orchestrator to the injected-inference seam.

    Thin by design: everything the stage does with the responses is tested on
    CPU against fake text, and everything below this line needs a GPU. Keeping
    the adapter to one call means the untestable part stays one line long.

    Args:
        orchestrator: A loaded DocumentOrchestrator.
        max_tokens: Generation budget for one response.
        verbose: Whether to log per-batch progress.

    Returns:
        An inference callable for `run_quality_screen`.
    """

    def _infer(image_paths: list[str], prompt: str) -> list[str]:
        return orchestrator.screen_batch(
            image_paths, prompt, max_tokens, verbose=verbose, tile_extra=tile_extra
        )

    return _infer


def write_screen_records(records: list[dict], output_path: Path) -> Path:
    """Write the stage's records as JSONL.

    Args:
        records: Records from `run_quality_screen`.
        output_path: File to write.

    Returns:
        The path written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    output_path.write_text("\n".join(lines) + "\n" if lines else "")
    return output_path


def run(
    image_dir: Path,
    output_path: Path,
    *,
    model_type: str | None = None,
    batch_size: int | None = None,
    verbose: bool | None = None,
    config_path: Path | None = None,
    variant: str | None = None,
    min_tiles: int | None = None,
    max_tiles: int | None = None,
) -> Path:
    """Screen every image in a directory, write quality_screen.jsonl.

    Args:
        image_dir: Directory containing images.
        output_path: Path to write the screen records to.
        model_type: Model type (e.g. "internvl3-vllm").
        batch_size: Images per batch (None = auto-detect, 1 = sequential).
        verbose: Tier B output. None = read from YAML.
        config_path: Optional path to run_config.yml.
        variant: Prompt variant to run. None = the one declared in YAML.
            An override rather than a default: comparing prompts is the whole
            reason several variants exist, and editing config between runs
            makes it easy to lose track of which produced which output.
        min_tiles: Override the configured tile floor. The floor is the lever
            for small images; sweeping it is how its effect gets measured.
        max_tiles: Override the configured tile ceiling.

    Returns:
        Path to the written records.
    """
    from common.pipeline_prompts import load_pipeline_configs
    from common.app_config import AppConfig
    from common.pipeline_config import discover_images
    from common.pipeline_ops import create_processor, load_model

    # Same config cascade as the other stages: CLI > YAML > defaults, with None
    # meaning "let YAML win".
    cli_args: dict[str, Any] = {
        "data_dir": str(image_dir),
        "output_dir": str(output_path.parent),
    }
    if model_type is not None:
        cli_args["model_type"] = model_type
    if verbose is not None:
        cli_args["verbose"] = verbose
    if batch_size is not None:
        cli_args["batch_size"] = batch_size

    app_cfg = AppConfig.load(cli_args, config_path=config_path)
    config = app_cfg.pipeline
    screen_cfg = app_cfg.quality_screen_config
    resolved_variant = variant or screen_cfg["variant"]
    if variant:
        logger.info("Prompt variant overridden on the command line: %s", resolved_variant)

    vocabulary = load_screen_vocabulary(Path(screen_cfg["prompt_file"]), variant=resolved_variant)
    max_tokens = app_cfg.get_token_budget("quality_screen")

    tile_extra = dict(screen_cfg["tiling"])
    if min_tiles is not None:
        tile_extra["min_tiles"] = min_tiles
    if max_tiles is not None:
        tile_extra["max_tiles"] = max_tiles
    logger.info("Tile budget: %s", tile_extra)

    images = list(discover_images(config.data_dir))
    if not images:
        msg = f"No images found in {config.data_dir}"
        raise FileNotFoundError(msg)
    logger.info("Screening %d images with %s", len(images), resolved_variant)

    # -- vLLM data-parallel fast path -----------------------------------------
    # Same shape as the classify stage: shard the images across GPUs, each
    # worker building its own TP=1 engine. This is where the throughput comes
    # from -- no backend in this repo implements `generate_batch`, so
    # `supports_batch` is false everywhere and every worker runs sequentially
    # within its shard. Parallelism is across GPUs, not within a call.
    from models.registry import is_vllm_model

    if is_vllm_model(config.model_type):
        from common.vllm_dp import resolve_gpu_count, run_dp

        resolved_gpus = resolve_gpu_count(config)
        if resolved_gpus > 1:
            logger.info("vLLM data-parallel: sharding %d images across %d GPUs", len(images), resolved_gpus)
            dp_records = run_dp(
                num_gpus=resolved_gpus,
                images=images,
                worker_fn=DP_WORKER,
                worker_kwargs={
                    "config_path": str(config_path) if config_path else None,
                    "cli_overrides": cli_args,
                    "variant": resolved_variant,
                    "tile_extra": tile_extra,
                },
                app_config=app_cfg,
            )
            written = write_screen_records(dp_records, output_path)
            _log_screen_summary(dp_records, written)
            return written

    # -- Single-GPU / HF path -------------------------------------------------
    logger.info("Loading model: %s", config.model_type)
    prompt_config, universal_fields, field_definitions = load_pipeline_configs(config.model_type)
    model_cm = load_model(config, app_config=app_cfg)
    model, tokenizer = model_cm.__enter__()

    try:
        orchestrator = create_processor(
            model,
            tokenizer,
            config,
            prompt_config,
            universal_fields,
            field_definitions,
            app_config=app_cfg,
        )
        records = run_quality_screen(
            [str(path) for path in images],
            infer=orchestrator_inference(
                orchestrator, max_tokens, verbose=config.verbose, tile_extra=tile_extra
            ),
            vocabulary=vocabulary,
            variant=resolved_variant,
        )
    finally:
        model_cm.__exit__(None, None, None)

    written = write_screen_records(records, output_path)
    _log_screen_summary(records, written)
    return written


def _log_screen_summary(records: list[dict], written: Path) -> None:
    """Report what the run produced, on either path.

    Shared by the DP and single-GPU paths so a run's summary does not depend
    on how it was sharded.
    """
    malformed = sum(1 for record in records if record["malformed"])
    drifted = sum(1 for record in records if record["think_drift"])
    logger.info(
        "Screened %d images: %d malformed, %d with reasoning drift -> %s",
        len(records),
        malformed,
        drifted,
        written,
    )
    if malformed:
        # Loud, because a high malformed rate invalidates the run's scores and
        # is invisible in the per-criterion numbers themselves.
        logger.warning(
            "%d of %d responses were unreadable (%.1f%%). These are excluded from the "
            "metrics and counted separately; a high rate means the score describes a "
            "subset, not the corpus.",
            malformed,
            len(records),
            100.0 * malformed / len(records),
        )


@app.command()
def main(
    image_dir: Path = typer.Option(..., "--data-dir", "-d", help="Directory containing images"),
    output: Path = typer.Option(..., "--output", "-o", help="Path to write quality_screen.jsonl"),
    model: str | None = typer.Option(None, "--model", help="Model type"),
    batch_size: int | None = typer.Option(None, "--batch-size", help="Images per batch"),
    config: Path | None = typer.Option(None, "--config", help="YAML configuration file"),
    verbose: bool | None = typer.Option(None, "--verbose/--no-verbose", help="Tier B output"),
    variant: str | None = typer.Option(
        None,
        "--variant",
        help="Prompt variant to run, overriding the one in run_config.yml.",
    ),
    min_tiles: int | None = typer.Option(None, "--min-tiles", help="Override the tile floor."),
    max_tiles: int | None = typer.Option(None, "--max-tiles", help="Override the tile ceiling."),
) -> None:
    """Stage 1: screen image quality for every image in a directory."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(
        image_dir,
        output,
        model_type=model,
        batch_size=batch_size,
        verbose=verbose,
        config_path=config,
        variant=variant,
        min_tiles=min_tiles,
        max_tiles=max_tiles,
    )


if __name__ == "__main__":
    app()
