# ruff: noqa: B008 - typer.Option in defaults is the standard Typer pattern
"""Stage 4: Evaluation against ground truth (CPU).

Reads cleaned_extractions.jsonl, compares against a ground truth CSV,
computes F1 scores, and writes evaluation_results.jsonl.

No GPU needed -- this stage runs on CPU nodes.

Usage:
    python -m stages.evaluate \
        --input /artifacts/cleaned_extractions.jsonl \
        --ground-truth /data/ground_truth.csv \
        --output-dir /artifacts/evaluation
"""

import logging
from pathlib import Path
from typing import Any

import typer

from common.batch_types import ExtractionOutput
from common.extraction_evaluator import ExtractionEvaluator
from common.field_schema import get_field_schema

from .io import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)
app = typer.Typer()


def run(
    cleaned_extractions_path: Path,
    ground_truth_csv: Path,
    output_dir: Path,
    *,
    enable_math_enhancement: bool = False,
) -> Path:
    """Evaluate cleaned extractions against ground truth.

    For each record in cleaned_extractions.jsonl:
      1. Construct an ExtractionOutput from the cleaned data
      2. Run ExtractionEvaluator.evaluate()
      3. Write per-image evaluation results to evaluation_results.jsonl

    Args:
        cleaned_extractions_path: Path to cleaned_extractions.jsonl from Stage 3.
        ground_truth_csv: Path to ground truth CSV.
        output_dir: Directory for evaluation output files.
        enable_math_enhancement: Apply bank statement balance calculations.

    Returns:
        Path to the written evaluation_results.jsonl.
    """
    # Read cleaned extractions
    records = read_jsonl(cleaned_extractions_path)
    if not records:
        msg = f"No records found in {cleaned_extractions_path}"
        raise FileNotFoundError(msg)

    logger.info(
        "Read %d cleaned extractions from %s",
        len(records),
        cleaned_extractions_path,
    )

    # Set up evaluator
    schema = get_field_schema()
    field_definitions = schema.get_all_doc_type_fields()

    evaluator = ExtractionEvaluator(
        ground_truth_csv=str(ground_truth_csv),
        field_definitions=field_definitions,
        enable_math_enhancement=enable_math_enhancement,
    )

    if not evaluator.has_ground_truth:
        logger.warning("No ground truth loaded -- evaluation will be empty")

    # Evaluate each record
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_results: list[dict[str, Any]] = []

    for record in records:
        image_name = record["image_name"]
        error = record.get("error")

        if error:
            eval_results.append(
                {
                    "image_name": image_name,
                    "document_type": record["document_type"],
                    "error": error,
                    "overall_accuracy": 0,
                    "median_f1": 0,
                }
            )
            continue

        # Reconstruct ExtractionOutput for the evaluator
        extraction = ExtractionOutput(
            image_path=record.get("image_path", ""),
            image_name=image_name,
            document_type=record["document_type"],
            extracted_data=record.get("extracted_data", {}),
            processing_time=0,
            prompt_used="staged_pipeline",
        )

        evaluation = evaluator.evaluate(extraction)

        eval_results.append(
            {
                "image_name": image_name,
                "document_type": record["document_type"],
                **evaluation,
            }
        )

        if evaluation and "median_f1" in evaluation:
            logger.debug(
                "%s: median_f1=%.3f, overall_accuracy=%.3f",
                image_name,
                evaluation.get("median_f1", 0),
                evaluation.get("overall_accuracy", 0),
            )

    # Write evaluation results
    output_path = output_dir / "evaluation_results.jsonl"
    count = write_jsonl(output_path, eval_results)
    logger.info("Wrote %d evaluation results to %s", count, output_path)

    # Summary statistics
    scored = [r for r in eval_results if "median_f1" in r and not r.get("error")]
    if scored:
        avg_f1 = sum(r["median_f1"] for r in scored) / len(scored)
        avg_acc = sum(r.get("overall_accuracy", 0) for r in scored) / len(scored)
        logger.info(
            "Summary: %d images scored, avg median_f1=%.3f, avg accuracy=%.3f",
            len(scored),
            avg_f1,
            avg_acc,
        )
    else:
        logger.warning("No images scored -- check ground truth alignment")

    return output_path


@app.command()
def main(
    input_path: Path = typer.Option(
        ..., "--input", "-i", help="Path to cleaned_extractions.jsonl from Stage 3"
    ),
    ground_truth: Path = typer.Option(
        ..., "--ground-truth", "-g", help="Path to ground truth CSV"
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", "-o", help="Directory for evaluation output"
    ),
    math_enhancement: bool = typer.Option(
        False,
        "--math-enhancement/--no-math-enhancement",
        help="Enable bank balance calculations",
    ),
) -> None:
    """Stage 4: Evaluate extractions against ground truth (CPU only)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    run(
        input_path,
        ground_truth,
        output_dir,
        enable_math_enhancement=math_enhancement,
    )


if __name__ == "__main__":
    app()
