"""Classification stage: detect document types from images.

Produces a CSV with one row per image:
    image_path, document_type, confidence, raw_response, prompt_used, error, timestamp

Supports both batched (InternVL3) and sequential (Llama/Qwen) detection paths.
"""

import csv
import logging
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from .io_schemas import ClassificationOutput, ClassificationRow

logger = logging.getLogger(__name__)


def classify_images(
    processor,
    image_paths: list[str | Path],
    batch_size: int = 1,
    verbose: bool = False,
    console: Console | None = None,
) -> ClassificationOutput:
    """Run document type classification on all images.

    Handles both batched (BatchCapableProcessor) and sequential
    (DocumentProcessor) detection paths based on processor capabilities
    and batch_size.

    Args:
        processor: Model processor implementing detect_and_classify_document()
            and optionally batch_detect_documents().
        image_paths: List of image file paths to classify.
        batch_size: Images per batch. 1 = sequential.
        verbose: Enable debug output.
        console: Rich console for progress display.

    Returns:
        ClassificationOutput with one ClassificationRow per image.
    """
    from models.protocol import BatchCapableProcessor

    console = console or Console()
    str_paths = [str(p) for p in image_paths]
    total_images = len(str_paths)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info(
        "Classification stage: %d images, batch_size=%d", total_images, batch_size
    )

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[cyan]{task.fields[current]}[/cyan]"),
        console=console,
        transient=False,
    )
    progress_task = progress.add_task("Classifying", total=total_images, current="")

    rows: list[ClassificationRow] = []

    if batch_size > 1 and isinstance(processor, BatchCapableProcessor):
        rows = _classify_batched(
            processor,
            str_paths,
            batch_size,
            verbose,
            progress,
            progress_task,
            console,
        )
    else:
        rows = _classify_sequential(
            processor,
            str_paths,
            verbose,
            progress,
            progress_task,
            console,
        )

    progress.update(progress_task, current="done")
    console.print(progress.get_renderable())

    # Stamp timestamps
    now = datetime.now().isoformat()
    for row in rows:
        if not row.timestamp:
            row.timestamp = now

    # Log summary
    type_counts: dict[str, int] = {}
    for row in rows:
        type_counts[row.document_type] = type_counts.get(row.document_type, 0) + 1
    for doc_type, count in sorted(type_counts.items()):
        logger.info("  %s: %d", doc_type, count)

    model_type = getattr(processor, "model_type", "unknown")
    return ClassificationOutput(rows=rows, model_type=model_type, timestamp=timestamp)


def _classify_batched(
    processor,
    image_paths: list[str],
    batch_size: int,
    verbose: bool,
    progress: Progress,
    progress_task,
    console: Console,
) -> list[ClassificationRow]:
    """Batched classification via processor.batch_detect_documents()."""
    rows: list[ClassificationRow] = []
    total = len(image_paths)

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_paths = image_paths[batch_start:batch_end]

        logger.debug("Detecting batch [%d-%d] / %d", batch_start + 1, batch_end, total)

        batch_results = processor.batch_detect_documents(batch_paths, verbose=verbose)

        for i, result in enumerate(batch_results):
            path = batch_paths[i]
            rows.append(
                ClassificationRow(
                    image_path=path,
                    document_type=result.get("document_type", "UNKNOWN"),
                    confidence=result.get("confidence", 0.0),
                    raw_response=result.get("raw_response", ""),
                    prompt_used=result.get("prompt_used", ""),
                    error=result.get("error", ""),
                )
            )
            progress.update(progress_task, advance=1, current=Path(path).name)
            console.print(progress.get_renderable())

    return rows


def _classify_sequential(
    processor,
    image_paths: list[str],
    verbose: bool,
    progress: Progress,
    progress_task,
    console: Console,
) -> list[ClassificationRow]:
    """Sequential classification via processor.detect_and_classify_document()."""
    rows: list[ClassificationRow] = []

    for path in image_paths:
        image_name = Path(path).name
        progress.update(progress_task, current=image_name)
        console.print(progress.get_renderable())

        try:
            result = processor.detect_and_classify_document(path, verbose=verbose)
            rows.append(
                ClassificationRow(
                    image_path=path,
                    document_type=result.get("document_type", "UNKNOWN"),
                    confidence=result.get("confidence", 0.0),
                    raw_response=result.get("raw_response", ""),
                    prompt_used=result.get("prompt_used", ""),
                    error=result.get("error", ""),
                )
            )
        except Exception as e:
            logger.error("Classification failed for %s: %s", image_name, e)
            rows.append(
                ClassificationRow(
                    image_path=path,
                    document_type="UNKNOWN",
                    confidence=0.0,
                    raw_response="",
                    prompt_used="",
                    error=str(e),
                )
            )

        progress.update(progress_task, advance=1)

    return rows


# ---------------------------------------------------------------------------
# CSV serialization
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "image_path",
    "document_type",
    "confidence",
    "raw_response",
    "prompt_used",
    "error",
    "timestamp",
]


def write_classification_csv(output: ClassificationOutput, output_path: Path) -> Path:
    """Write classification results to CSV.

    Args:
        output: ClassificationOutput from classify_images().
        output_path: Path to write the CSV file.

    Returns:
        The path written to.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for row in output.rows:
            writer.writerow(
                {
                    "image_path": row.image_path,
                    "document_type": row.document_type,
                    "confidence": row.confidence,
                    "raw_response": row.raw_response,
                    "prompt_used": row.prompt_used,
                    "error": row.error,
                    "timestamp": row.timestamp,
                }
            )

    logger.info("Wrote classification CSV: %s (%d rows)", output_path, len(output.rows))
    return output_path


def read_classification_csv(csv_path: Path) -> ClassificationOutput:
    """Read classification CSV back into ClassificationOutput.

    Used by the extract stage when running standalone.

    Args:
        csv_path: Path to classification CSV file.

    Returns:
        ClassificationOutput with rows populated from CSV.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If CSV structure is invalid.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Classification CSV not found: {csv_path}")

    rows: list[ClassificationRow] = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames or "image_path" not in reader.fieldnames:
            raise ValueError(
                f"Invalid classification CSV: missing 'image_path' column in {csv_path}"
            )

        for csv_row in reader:
            rows.append(
                ClassificationRow(
                    image_path=csv_row["image_path"],
                    document_type=csv_row.get("document_type", "UNKNOWN"),
                    confidence=float(csv_row.get("confidence", 0.0)),
                    raw_response=csv_row.get("raw_response", ""),
                    prompt_used=csv_row.get("prompt_used", ""),
                    error=csv_row.get("error", ""),
                    timestamp=csv_row.get("timestamp", ""),
                )
            )

    logger.info("Read classification CSV: %s (%d rows)", csv_path, len(rows))
    return ClassificationOutput(rows=rows, model_type="unknown", timestamp="")
