# ruff: noqa: B008 - typer.Option in defaults is the standard Typer pattern
"""Stage 1: Document type classification (GPU).

Detects document types for all images and writes classifications.jsonl.

Usage:
    python -m stages.classify --image-dir /data --output /artifacts/classifications.jsonl
    python -m stages.classify --image-dir /data --output /artifacts/classifications.jsonl --model llama
"""

import logging
import time
from pathlib import Path
from typing import Any

import typer

from .io import write_jsonl

logger = logging.getLogger(__name__)
app = typer.Typer()


def run(
    image_dir: Path,
    output_path: Path,
    *,
    model_type: str | None = None,
    batch_size: int | None = None,
    verbose: bool | None = None,
    debug: bool | None = None,
    config_path: Path | None = None,
) -> Path:
    """Detect document types for all images, write classifications.jsonl.

    Args:
        image_dir: Directory containing images.
        output_path: Path to write classifications.jsonl.
        model_type: Model type (e.g. "internvl3", "llama").
        batch_size: Images per batch (None = auto-detect, 1 = sequential).
        verbose: Tier B output (init/config details). None = read from YAML.
        debug: Tier C output (dev-noise: PARSING DEBUG, prompt dumps). None = YAML.
        config_path: Optional path to run_config.yml.

    Returns:
        Path to the written classifications.jsonl.
    """
    from cli import load_pipeline_configs
    from common.app_config import AppConfig
    from common.pipeline_config import discover_images
    from common.pipeline_ops import create_processor, load_model

    # Build config through the standard cascade. Only inject verbose/debug
    # when the caller passed an explicit bool — None means "let YAML win".
    cli_args: dict[str, Any] = {
        "data_dir": str(image_dir),
        "output_dir": str(output_path.parent),
    }
    if model_type is not None:
        cli_args["model_type"] = model_type
    if verbose is not None:
        cli_args["verbose"] = verbose
    if debug is not None:
        cli_args["debug"] = debug
    if batch_size is not None:
        cli_args["batch_size"] = batch_size

    app_cfg = AppConfig.load(cli_args, config_path=config_path)
    config = app_cfg.pipeline
    # Tier B gate — resolved from CLI > YAML > defaults.
    effective_verbose = config.verbose

    # Discover images
    images = list(discover_images(config.data_dir))
    if not images:
        msg = f"No images found in {config.data_dir}"
        raise FileNotFoundError(msg)

    logger.info("Found %d images in %s", len(images), config.data_dir)

    # -- Single-GPU / TP path --------------------------------------------------

    # Load pipeline configs (prompts, fields)
    prompt_config, universal_fields, field_definitions = load_pipeline_configs(
        config.model_type
    )

    # Load model and run detection
    logger.info("Loading model: %s", config.model_type)
    model_cm = load_model(config)
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

        image_paths = [str(img) for img in images]
        records: list[dict[str, Any]] = []

        effective_batch = config.batch_size or 1
        can_batch = getattr(processor, "supports_batch", False) and effective_batch > 1

        start = time.time()

        if can_batch:
            for batch_start in range(0, len(image_paths), effective_batch):
                batch_end = min(batch_start + effective_batch, len(image_paths))
                batch = image_paths[batch_start:batch_end]

                logger.info(
                    "Detecting batch [%d-%d] / %d",
                    batch_start + 1,
                    batch_end,
                    len(image_paths),
                )

                results = processor.detect_batch(batch, verbose=effective_verbose)
                for i, result in enumerate(results):
                    image_name = Path(batch[i]).name
                    records.append(
                        {
                            "image_path": batch[i],
                            "image_name": image_name,
                            "document_type": result["document_type"],
                            "confidence": result.get("confidence", 1.0),
                            "raw_response": result.get("raw_response", ""),
                            "prompt_used": result.get("prompt_used", "detection"),
                        }
                    )
                    # Tier A: always-on per-image progress.
                    logger.info(
                        "[%d/%d] %s -> %s",
                        batch_start + i + 1,
                        len(image_paths),
                        image_name,
                        result["document_type"],
                    )
        else:
            for idx, image_path in enumerate(image_paths):
                logger.info(
                    "Sending image %d/%d to engine: %s",
                    idx + 1,
                    len(image_paths),
                    Path(image_path).name,
                )
                result = processor.detect_and_classify_document(
                    image_path,
                    verbose=effective_verbose,
                )
                image_name = Path(image_path).name
                records.append(
                    {
                        "image_path": image_path,
                        "image_name": image_name,
                        "document_type": result["document_type"],
                        "confidence": result.get("confidence", 1.0),
                        "raw_response": result.get("raw_response", ""),
                        "prompt_used": result.get("prompt_used", "detection"),
                    }
                )
                # Tier A: always-on per-image progress.
                logger.info(
                    "[%d/%d] %s -> %s",
                    idx + 1,
                    len(image_paths),
                    image_name,
                    result["document_type"],
                )

        elapsed = time.time() - start
        logger.info(
            "Classification complete: %d images in %.1fs (%.2fs/image)",
            len(records),
            elapsed,
            elapsed / len(records) if records else 0,
        )
    finally:
        model_cm.__exit__(None, None, None)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = write_jsonl(output_path, records)
    logger.info("Wrote %d classifications to %s", count, output_path)

    return output_path


@app.command()
def main(
    image_dir: Path = typer.Option(
        ..., "--data-dir", "-d", help="Directory containing images"
    ),
    output: Path = typer.Option(
        ..., "--output-dir", "-o", help="Path to write classifications.jsonl"
    ),
    model: str | None = typer.Option(None, "--model", help="Model type"),
    batch_size: int | None = typer.Option(
        None, "--batch-size", help="Images per detection batch"
    ),
    config: Path | None = typer.Option(
        None, "--config", help="YAML configuration file"
    ),
    verbose: bool | None = typer.Option(
        None,
        "--verbose/--no-verbose",
        help="Tier B output (init details). Default: read YAML processing.verbose.",
    ),
    debug: bool | None = typer.Option(
        None,
        "--debug/--no-debug",
        help="Tier C output (PARSING DEBUG, prompt dumps). Default: YAML processing.debug.",
    ),
) -> None:
    """Stage 1: Classify document types for all images."""
    # Tier A (per-image progress, phase headings) is always at INFO — it's the
    # only indication the user has that inference is progressing. Verbose/debug
    # are independent switches for Tiers B/C inside the orchestrator.
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    run(
        image_dir,
        output,
        model_type=model,
        batch_size=batch_size,
        verbose=verbose,
        debug=debug,
        config_path=config,
    )


if __name__ == "__main__":
    app()
