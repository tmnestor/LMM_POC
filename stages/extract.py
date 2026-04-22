# ruff: noqa: B008 - typer.Option in defaults is the standard Typer pattern
"""Stage 2: Field extraction (GPU).

Reads classifications.jsonl, runs type-specific extraction on each image,
and writes RAW model responses to raw_extractions.jsonl.

The raw response is the most valuable debugging artifact -- you can re-parse,
re-clean, and re-evaluate without touching the GPU.

Supports resumption: if the output file already has partial results from a
crashed run, skips already-processed images and appends.

Usage:
    python -m stages.extract \
        --classifications /artifacts/classifications.jsonl \
        --image-dir /data \
        --output /artifacts/raw_extractions.jsonl
"""

import logging
import time
from pathlib import Path
from typing import Any

import typer

from .io import StreamingJsonlWriter, read_completed_images, read_jsonl

logger = logging.getLogger(__name__)
app = typer.Typer()


def run(
    classifications_path: Path,
    image_dir: Path,
    output_path: Path,
    *,
    model_type: str = "internvl3",
    batch_size: int | None = None,
    bank_v2: bool = True,
    balance_correction: bool = True,
    verbose: bool | None = None,
    debug: bool | None = None,
    config_path: Path | None = None,
) -> Path:
    """Extract fields from classified images, write raw_extractions.jsonl.

    Writes the **raw model response string** per image, not parsed/cleaned
    data.  The clean stage handles parsing.

    Supports resumption: if output_path exists with partial results, skips
    already-processed images and appends new results.

    Args:
        classifications_path: Path to classifications.jsonl from Stage 1.
        image_dir: Directory containing the original images.
        output_path: Path to write raw_extractions.jsonl.
        model_type: Model type (e.g. "internvl3", "llama").
        batch_size: Images per batch (None = auto-detect, 1 = sequential).
        bank_v2: Use UnifiedBankExtractor for bank statements.
        balance_correction: Enable balance validation in bank extraction.
        verbose: Tier B output (init/config details). None = read from YAML.
        debug: Tier C output (dev-noise: PARSING DEBUG, prompt dumps). None = YAML.
        config_path: Optional path to run_config.yml.

    Returns:
        Path to the written raw_extractions.jsonl.
    """
    from cli import load_pipeline_configs
    from common.app_config import AppConfig
    from common.pipeline_ops import create_processor, load_model
    from common.unified_bank_extractor import UnifiedBankExtractor

    # Read classifications
    classifications = read_jsonl(classifications_path)
    if not classifications:
        msg = f"No classifications found in {classifications_path}"
        raise FileNotFoundError(msg)

    logger.info(
        "Read %d classifications from %s", len(classifications), classifications_path
    )

    # Check for resumption
    completed = read_completed_images(output_path)
    if completed:
        original_count = len(classifications)
        classifications = [
            c for c in classifications if c["image_name"] not in completed
        ]
        logger.info(
            "Resuming: %d already done, %d remaining (of %d total)",
            len(completed),
            len(classifications),
            original_count,
        )

    if not classifications:
        logger.info("All images already processed -- nothing to do")
        return output_path

    # Build config through the standard cascade. Only inject verbose/debug
    # when the caller passed an explicit bool — None means "let YAML win".
    cli_args: dict[str, Any] = {
        "data_dir": str(image_dir),
        "output_dir": str(output_path.parent),
        "model_type": model_type,
        "bank_v2": bank_v2,
        "balance_correction": balance_correction,
    }
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

    prompt_config, universal_fields, field_definitions = load_pipeline_configs(
        config.model_type
    )

    # Load model
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

        # Set up bank adapter if needed
        bank_adapter = None
        has_bank = any(
            c["document_type"].upper() == "BANK_STATEMENT" for c in classifications
        )
        if (
            has_bank
            and config.bank_v2
            and getattr(processor, "supports_multi_turn", True)
        ):
            bank_adapter = UnifiedBankExtractor(
                generate_fn=processor.generate,
                verbose=effective_verbose,
                use_balance_correction=config.balance_correction,
            )
            logger.info("Bank adapter enabled")

        # Process images with streaming writer (flush per record for crash recovery)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        start = time.time()
        count = 0
        total = len(classifications)

        with StreamingJsonlWriter(output_path) as writer:
            for classification in classifications:
                image_path = classification["image_path"]
                image_name = classification["image_name"]
                doc_type = classification["document_type"]

                img_start = time.time()

                try:
                    if (
                        doc_type.upper() == "BANK_STATEMENT"
                        and bank_adapter is not None
                    ):
                        _extract_bank_with_adapter(
                            writer, bank_adapter, image_path, image_name, doc_type
                        )
                    else:
                        _extract_standard(
                            writer,
                            processor,
                            image_path,
                            image_name,
                            doc_type,
                            classification,
                            effective_verbose,
                        )
                except Exception as e:
                    logger.error("Error extracting %s: %s", image_name, e)
                    img_time = time.time() - img_start
                    writer.write(
                        {
                            "image_name": image_name,
                            "image_path": image_path,
                            "document_type": doc_type,
                            "raw_response": "",
                            "processing_time": img_time,
                            "prompt_used": "error",
                            "error": str(e),
                        }
                    )

                count += 1
                img_time = time.time() - img_start
                logger.info(
                    "[%d/%d] %s: %s (%.1fs)",
                    count,
                    total,
                    image_name,
                    doc_type,
                    img_time,
                )

        elapsed = time.time() - start
        logger.info("Extraction complete: %d images in %.1fs", count, elapsed)
    finally:
        model_cm.__exit__(None, None, None)

    return output_path


def _extract_bank_with_adapter(
    writer: StreamingJsonlWriter,
    bank_adapter,
    image_path: str,
    image_name: str,
    doc_type: str,
) -> None:
    """Extract bank statement via UnifiedBankExtractor, write raw responses."""
    start = time.time()
    schema_fields, metadata = bank_adapter.extract_bank_statement(image_path)
    img_time = time.time() - start

    # Serialize structured fields into flat FIELD: value text that the
    # standard hybrid parser already handles.  This keeps the stage contract
    # clean: extract produces a parseable raw_response, clean parses it.
    raw_response_str = "\n".join(
        f"{field}: {value}" for field, value in schema_fields.items()
    )

    strategy = metadata.get("strategy_used", "unknown")

    writer.write(
        {
            "image_name": image_name,
            "image_path": image_path,
            "document_type": doc_type,
            "raw_response": raw_response_str,
            "processing_time": img_time,
            "prompt_used": f"unified_bank_{strategy}",
            "error": None,
        }
    )


def _extract_standard(
    writer: StreamingJsonlWriter,
    processor,
    image_path: str,
    image_name: str,
    doc_type: str,
    classification: dict[str, Any],
    verbose: bool,
) -> None:
    """Extract standard document, write raw response."""
    start = time.time()
    result = processor.process_document_aware(
        image_path, classification, verbose=verbose
    )
    img_time = time.time() - start

    writer.write(
        {
            "image_name": image_name,
            "image_path": image_path,
            "document_type": doc_type,
            "raw_response": result.get("raw_response", ""),
            "processing_time": img_time,
            "prompt_used": doc_type.lower(),
            "error": None,
        }
    )


@app.command()
def main(
    classifications: Path = typer.Option(
        ..., "--classifications", help="Path to classifications.jsonl from Stage 1"
    ),
    image_dir: Path = typer.Option(
        ..., "--data-dir", "-d", help="Directory containing images"
    ),
    output: Path = typer.Option(
        ..., "--output-dir", "-o", help="Path to write raw_extractions.jsonl"
    ),
    model: str = typer.Option("internvl3", "--model", help="Model type"),
    batch_size: int | None = typer.Option(
        None, "--batch-size", help="Images per extraction batch"
    ),
    bank_v2: bool = typer.Option(True, "--bank-v2/--no-bank-v2"),
    balance_correction: bool = typer.Option(
        True, "--balance-correction/--no-balance-correction"
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
    """Stage 2: Extract fields from classified images (raw responses)."""
    # Tier A (per-image progress, phase headings) is always at INFO — it's the
    # only indication the user has that inference is progressing. Verbose/debug
    # are independent switches for Tiers B/C inside the orchestrator.
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    run(
        classifications,
        image_dir,
        output,
        model_type=model,
        batch_size=batch_size,
        bank_v2=bank_v2,
        balance_correction=balance_correction,
        verbose=verbose,
        debug=debug,
        config_path=config,
    )


if __name__ == "__main__":
    app()
