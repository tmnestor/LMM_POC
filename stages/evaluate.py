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
import time
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

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
    wall_clock_start: float | None = None,
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
        wall_clock_start: Epoch seconds when the pipeline started. When set,
            the Execution Summary reports true wall-clock across all phases
            (time.time() - start) instead of just stage-2 inference time.

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

    # Rich Execution Summary table (equivalent to pre-staged-pipeline output)
    wall_clock_s = (
        time.time() - wall_clock_start if wall_clock_start is not None else None
    )
    _print_summary_table(eval_results, records, output_dir, wall_clock_s)

    return output_path


def _print_summary_table(
    eval_results: list[dict[str, Any]],
    cleaned_records: list[dict[str, Any]],
    output_dir: Path,
    wall_clock_seconds: float | None = None,
) -> None:
    """Render an Execution Summary table and document-type breakdown.

    Mimics the pre-staged-pipeline `cli.print_summary` output so runs still
    end with the familiar throughput/accuracy summary.
    """
    console = Console()

    num = len(eval_results)
    total_inference = sum(r.get("processing_time", 0.0) for r in cleaned_records)
    # Throughput is computed against wall-clock when available (true
    # end-to-end rate), otherwise falls back to stage-2 inference time.
    throughput_denom = (
        wall_clock_seconds
        if wall_clock_seconds is not None and wall_clock_seconds > 0
        else total_inference
    )
    throughput = (num / throughput_denom * 60.0) if throughput_denom > 0 else 0.0

    scored = [r for r in eval_results if "median_f1" in r and not r.get("error")]
    avg_acc = (
        sum(r.get("overall_accuracy", 0.0) for r in scored) / len(scored)
        if scored
        else 0.0
    )
    avg_f1 = (
        sum(r.get("median_f1", 0.0) for r in scored) / len(scored) if scored else 0.0
    )

    doc_types: dict[str, int] = {}
    for r in eval_results:
        dt = r.get("document_type", "UNKNOWN")
        doc_types[dt] = doc_types.get(dt, 0) + 1

    table = Table(title="Execution Summary", show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Images Processed", str(num))
    if wall_clock_seconds is not None:
        table.add_row("Wall Clock Time", f"{wall_clock_seconds:.1f}s")
    table.add_row("Inference Time", f"{total_inference:.1f}s")
    table.add_row("Throughput", f"{throughput:.2f} images/min")
    if scored:
        table.add_row("Avg Accuracy", f"{avg_acc:.1%}")
        table.add_row("Avg F1 (median)", f"{avg_f1:.3f}")
    table.add_row("Output Directory", str(output_dir))

    console.print()
    console.print(table)

    if doc_types:
        console.print("\n[bold]Document Types:[/bold]")
        for doc_type, count in sorted(doc_types.items()):
            console.print(f"  {doc_type}: {count}")


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
    wall_clock_start: float | None = typer.Option(
        None,
        "--wall-clock-start",
        help=(
            "Epoch seconds when the pipeline started (set by entrypoint.sh "
            "before Phase 1). Used to report true end-to-end wall-clock in "
            "the Execution Summary instead of just stage-2 inference time."
        ),
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
        wall_clock_start=wall_clock_start,
    )


if __name__ == "__main__":
    app()
