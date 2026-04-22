# ruff: noqa: B008 - typer.Option in defaults is the standard Typer pattern
"""Stage: Transaction linking via graph executor.

Reads a pairs CSV mapping receipt images to bank statement images,
runs the cross-image ``transaction_link`` workflow via ``GraphExecutor``,
and writes results to raw_extractions.jsonl.

Compatible with existing clean + evaluate stages downstream.

Usage:
    python -m stages.link \
        --pairs pairs.csv \
        --data-dir /data \
        --output /artifacts/raw_extractions.jsonl \
        --model internvl3-vllm
"""

import csv
import logging
import time
from pathlib import Path
from typing import Any

import typer
import yaml

from .io import StreamingJsonlWriter, read_completed_images

logger = logging.getLogger(__name__)
app = typer.Typer()


def _read_pairs(pairs_path: Path) -> list[dict[str, str]]:
    """Read receipt-to-bank-statement pairs from CSV.

    Expected columns: ``receipt_file``, ``bank_statement_file``.
    """
    pairs: list[dict[str, str]] = []
    with pairs_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "receipt_file" not in row or "bank_statement_file" not in row:
                msg = (
                    f"Pairs CSV must have 'receipt_file' and "
                    f"'bank_statement_file' columns.  Got: {list(row.keys())}"
                )
                raise ValueError(msg)
            pairs.append(dict(row))
    return pairs


def _load_workflow(workflow_path: Path) -> dict[str, Any]:
    """Load a YAML workflow definition."""
    with workflow_path.open() as f:
        return yaml.safe_load(f)


def run(
    pairs_path: Path,
    image_dir: Path,
    output_path: Path,
    *,
    workflow_path: Path | None = None,
    model_type: str = "internvl3-vllm",
    config_path: Path | None = None,
) -> Path:
    """Stage: Transaction linking via graph executor.

    Reads image pairs, runs cross-image workflow, writes
    raw_extractions.jsonl.  Compatible with existing clean +
    evaluate stages downstream.

    Supports resumption: if output_path exists with partial
    results, skips already-processed pairs and appends.

    Args:
        pairs_path: CSV with receipt_file, bank_statement_file columns.
        image_dir: Directory containing the images.
        output_path: Path to write raw_extractions.jsonl.
        workflow_path: Path to workflow YAML (default: built-in).
        model_type: Model type for loading (e.g. "internvl3-vllm").
        config_path: Optional path to run_config.yml.

    Returns:
        Path to the written raw_extractions.jsonl.
    """
    from common.graph_executor import GraphExecutor
    from common.graph_generate import make_vllm_generate_fn
    from common.pipeline_ops import load_model
    from common.turn_parsers import build_parser_registry

    # Load pairs
    pairs = _read_pairs(pairs_path)
    if not pairs:
        msg = f"No pairs found in {pairs_path}"
        raise FileNotFoundError(msg)
    logger.info("Read %d pairs from %s", len(pairs), pairs_path)

    # Check for resumption
    completed = read_completed_images(output_path)
    if completed:
        original_count = len(pairs)
        pairs = [p for i, p in enumerate(pairs) if f"pair_{i:03d}" not in completed]
        logger.info(
            "Resuming: %d already done, %d remaining (of %d total)",
            len(completed),
            len(pairs),
            original_count,
        )

    if not pairs:
        logger.info("All pairs already processed -- nothing to do")
        return output_path

    # Load workflow definition
    if workflow_path is None:
        workflow_path = Path("prompts/workflows/transaction_link.yaml")
    workflow_def = _load_workflow(workflow_path)

    # Build pipeline config for model loading
    from common.app_config import AppConfig

    cli_args: dict[str, Any] = {
        "data_dir": str(image_dir),
        "output_dir": str(output_path.parent),
        "model_type": model_type,
    }
    app_cfg = AppConfig.load(cli_args, config_path=config_path)
    config = app_cfg.pipeline

    # Load model
    logger.info("Loading model: %s", config.model_type)
    model_cm = load_model(config)
    model, tokenizer = model_cm.__enter__()

    try:
        # Build generate_fn wrapper
        chat_kwargs: dict[str, Any] = {}
        if config.model_type.startswith(("qwen35", "gemma4")):
            chat_kwargs = {"enable_thinking": False}

        generate_fn = make_vllm_generate_fn(
            model,
            chat_template_kwargs=chat_kwargs or None,
        )
        parsers = build_parser_registry()
        executor = GraphExecutor(generate_fn, parsers)

        # Process pairs
        output_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.time()
        count = 0
        total = len(pairs)

        with StreamingJsonlWriter(output_path) as writer:
            for i, pair in enumerate(pairs):
                pair_name = f"pair_{i:03d}"
                receipt_path = str(image_dir / pair["receipt_file"])
                bank_path = str(image_dir / pair["bank_statement_file"])
                pair_start = time.time()

                try:
                    session = executor.run(
                        document_type="TRANSACTION_LINK",
                        definition=workflow_def,
                        images={
                            "receipt": receipt_path,
                            "bank_statement": bank_path,
                        },
                        image_name=pair_name,
                        extra_fields={
                            "BANK_STATEMENT_FILE": pair["bank_statement_file"],
                        },
                    )
                    writer.write(session.to_record())
                except Exception as exc:
                    logger.error("Error processing %s: %s", pair_name, exc)
                    pair_time = time.time() - pair_start
                    writer.write(
                        {
                            "image_name": pair_name,
                            "image_path": receipt_path,
                            "document_type": "TRANSACTION_LINK",
                            "raw_response": "",
                            "processing_time": pair_time,
                            "error": str(exc),
                        }
                    )

                count += 1
                pair_time = time.time() - pair_start
                logger.info("[%d/%d] %s: %.1fs", count, total, pair_name, pair_time)

        elapsed = time.time() - start
        logger.info("Linking complete: %d pairs in %.1fs", count, elapsed)
    finally:
        model_cm.__exit__(None, None, None)

    return output_path


@app.command()
def main(
    pairs: Path = typer.Option(
        ..., "--pairs", help="CSV with receipt_file, bank_statement_file columns"
    ),
    image_dir: Path = typer.Option(
        ..., "--data-dir", "-d", help="Directory containing images"
    ),
    output: Path = typer.Option(
        ..., "--output", "-o", help="Path to write raw_extractions.jsonl"
    ),
    model: str = typer.Option("internvl3-vllm", "--model", help="Model type"),
    workflow: Path | None = typer.Option(
        None, "--workflow", help="Path to workflow YAML"
    ),
    config: Path | None = typer.Option(
        None, "--config", help="YAML configuration file"
    ),
) -> None:
    """Stage: Link receipts to bank statement transactions."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    run(
        pairs,
        image_dir,
        output,
        workflow_path=workflow,
        model_type=model,
        config_path=config,
    )


if __name__ == "__main__":
    app()
