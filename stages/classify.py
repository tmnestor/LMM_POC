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
    model_type: str = "internvl3",
    batch_size: int | None = None,
    verbose: bool = True,
    config_path: Path | None = None,
) -> Path:
    """Detect document types for all images, write classifications.jsonl.

    Args:
        image_dir: Directory containing images.
        output_path: Path to write classifications.jsonl.
        model_type: Model type (e.g. "internvl3", "llama").
        batch_size: Images per batch (None = auto-detect, 1 = sequential).
        verbose: Enable verbose logging.
        config_path: Optional path to run_config.yml.

    Returns:
        Path to the written classifications.jsonl.
    """
    from cli import load_pipeline_configs
    from common.app_config import AppConfig
    from common.pipeline_config import discover_images
    from common.pipeline_ops import create_processor, load_model

    # Build config through the standard cascade
    cli_args: dict[str, Any] = {
        "data_dir": str(image_dir),
        "output_dir": str(output_path.parent),
        "model_type": model_type,
        "verbose": verbose,
    }
    if batch_size is not None:
        cli_args["batch_size"] = batch_size

    app_cfg = AppConfig.load(cli_args, config_path=config_path)
    config = app_cfg.pipeline

    # Discover images
    images = list(discover_images(config.data_dir))
    if not images:
        msg = f"No images found in {config.data_dir}"
        raise FileNotFoundError(msg)

    logger.info("Found %d images in %s", len(images), config.data_dir)

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

                results = processor.detect_batch(batch, verbose=verbose)
                for i, result in enumerate(results):
                    records.append(
                        {
                            "image_path": batch[i],
                            "image_name": Path(batch[i]).name,
                            "document_type": result["document_type"],
                            "confidence": result.get("confidence", 1.0),
                            "raw_response": result.get("raw_response", ""),
                            "prompt_used": result.get("prompt_used", "detection"),
                        }
                    )
        else:
            for image_path in image_paths:
                result = processor.detect_and_classify_document(
                    image_path,
                    verbose=verbose,
                )
                records.append(
                    {
                        "image_path": image_path,
                        "image_name": Path(image_path).name,
                        "document_type": result["document_type"],
                        "confidence": result.get("confidence", 1.0),
                        "raw_response": result.get("raw_response", ""),
                        "prompt_used": result.get("prompt_used", "detection"),
                    }
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
    model: str = typer.Option("internvl3", "--model", help="Model type"),
    batch_size: int | None = typer.Option(
        None, "--batch-size", help="Images per detection batch"
    ),
    config: Path | None = typer.Option(
        None, "--config", help="YAML configuration file"
    ),
    verbose: bool = typer.Option(True, "--verbose/--no-verbose"),
) -> None:
    """Stage 1: Classify document types for all images."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    run(
        image_dir,
        output,
        model_type=model,
        batch_size=batch_size,
        verbose=verbose,
        config_path=config,
    )


if __name__ == "__main__":
    app()
