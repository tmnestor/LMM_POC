"""Extraction stage: extract fields from classified document images.

Reads classification output (CSV or in-memory ClassificationOutput).
Produces JSON with nested extracted_data per image.

Handles bank statements (sequential, multi-turn) separately from
standard documents (batchable).
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from .io_schemas import ClassificationOutput, ExtractionOutput, ExtractionRecord

logger = logging.getLogger(__name__)


def extract_documents(
    processor,
    classifications: ClassificationOutput,
    bank_adapter=None,
    batch_size: int = 1,
    verbose: bool = False,
    console: Console | None = None,
) -> ExtractionOutput:
    """Run field extraction on classified images.

    Partitions into bank (sequential) vs standard (batchable).
    Uses classifications to determine prompts and field lists.

    Args:
        processor: Model processor implementing process_document_aware()
            and optionally batch_extract_documents().
        classifications: Output from classify_images() or read_classification_csv().
        bank_adapter: Optional BankStatementAdapter for multi-turn bank extraction.
        batch_size: Images per batch. 1 = sequential.
        verbose: Enable debug output.
        console: Rich console for progress display.

    Returns:
        ExtractionOutput with one ExtractionRecord per image.
    """
    from models.protocol import BatchCapableProcessor

    console = console or Console()
    total_images = len(classifications.rows)

    logger.info("Extraction stage: %d images, batch_size=%d", total_images, batch_size)

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[cyan]{task.fields[current]}[/cyan]"),
        console=console,
        transient=False,
    )
    progress_task = progress.add_task("Extracting", total=total_images, current="")

    # Partition into bank vs standard by index
    bank_indices: list[int] = []
    standard_indices: list[int] = []
    for i, row in enumerate(classifications.rows):
        if row.document_type.upper() == "BANK_STATEMENT":
            bank_indices.append(i)
        else:
            standard_indices.append(i)

    logger.debug(
        "Standard documents: %d, Bank statements: %d",
        len(standard_indices),
        len(bank_indices),
    )

    # Pre-allocate results in original order
    records: list[ExtractionRecord | None] = [None] * total_images

    use_batch = (
        batch_size > 1
        and isinstance(processor, BatchCapableProcessor)
        and len(standard_indices) > 0
    )

    # --- Standard documents ---
    if standard_indices:
        if use_batch:
            _extract_standard_batched(
                processor,
                classifications,
                standard_indices,
                records,
                batch_size,
                verbose,
                progress,
                progress_task,
                console,
            )
        else:
            _extract_standard_sequential(
                processor,
                classifications,
                standard_indices,
                records,
                verbose,
                progress,
                progress_task,
                console,
            )

    # --- Bank statements (always sequential, multi-turn) ---
    if bank_indices:
        _extract_bank_sequential(
            processor,
            classifications,
            bank_indices,
            records,
            bank_adapter,
            verbose,
            progress,
            progress_task,
            console,
        )

    progress.update(progress_task, current="done")
    console.print(progress.get_renderable())

    # Convert None entries to error records (shouldn't happen, but defensive)
    final_records: list[ExtractionRecord] = []
    for i, rec in enumerate(records):
        if rec is None:
            row = classifications.rows[i]
            final_records.append(
                ExtractionRecord(
                    image_path=row.image_path,
                    image_name=Path(row.image_path).name,
                    document_type=row.document_type,
                    extracted_data={},
                    processing_time=0.0,
                    prompt_used="",
                    field_count=0,
                    fields_found=0,
                    timestamp=datetime.now().isoformat(),
                    error="Extraction produced no result",
                )
            )
        else:
            final_records.append(rec)

    metadata = {
        "model_type": classifications.model_type,
        "timestamp": classifications.timestamp,
        "total_images": total_images,
        "standard_count": len(standard_indices),
        "bank_count": len(bank_indices),
        "batch_size": batch_size,
    }

    return ExtractionOutput(records=final_records, metadata=metadata)


# ---------------------------------------------------------------------------
# Standard document extraction
# ---------------------------------------------------------------------------


def _extract_standard_batched(
    processor,
    classifications: ClassificationOutput,
    indices: list[int],
    records: list[ExtractionRecord | None],
    batch_size: int,
    verbose: bool,
    progress: Progress,
    progress_task,
    console: Console,
) -> None:
    """Batched extraction for standard (non-bank) documents."""
    for batch_start in range(0, len(indices), batch_size):
        batch_end = min(batch_start + batch_size, len(indices))
        batch_indices = indices[batch_start:batch_end]

        batch_paths = [classifications.rows[i].image_path for i in batch_indices]
        batch_class_infos = [
            {
                "document_type": classifications.rows[i].document_type,
                "confidence": classifications.rows[i].confidence,
                "raw_response": classifications.rows[i].raw_response,
                "prompt_used": classifications.rows[i].prompt_used,
            }
            for i in batch_indices
        ]

        logger.debug(
            "Extracting standard batch [%d-%d] / %d",
            batch_start + 1,
            batch_end,
            len(indices),
        )

        batch_start_time = time.time()
        extraction_results = processor.batch_extract_documents(
            batch_paths, batch_class_infos, verbose=verbose
        )
        batch_elapsed = time.time() - batch_start_time
        per_image_time = batch_elapsed / len(batch_indices) if batch_indices else 0

        for j, orig_idx in enumerate(batch_indices):
            row = classifications.rows[orig_idx]
            image_name = Path(row.image_path).name
            result = extraction_results[j]
            extracted_data = result.get("extracted_data", {})

            records[orig_idx] = ExtractionRecord(
                image_path=row.image_path,
                image_name=image_name,
                document_type=row.document_type,
                extracted_data=extracted_data,
                processing_time=per_image_time,
                prompt_used=f"batch_{row.document_type.lower()}",
                field_count=result.get("field_count", len(extracted_data)),
                fields_found=sum(
                    1 for v in extracted_data.values() if v != "NOT_FOUND"
                ),
                timestamp=datetime.now().isoformat(),
                raw_response=result.get("raw_response", ""),
            )

            progress.update(progress_task, advance=1, current=image_name)
            console.print(progress.get_renderable())


def _extract_standard_sequential(
    processor,
    classifications: ClassificationOutput,
    indices: list[int],
    records: list[ExtractionRecord | None],
    verbose: bool,
    progress: Progress,
    progress_task,
    console: Console,
) -> None:
    """Sequential extraction for standard (non-bank) documents."""
    for orig_idx in indices:
        row = classifications.rows[orig_idx]
        image_name = Path(row.image_path).name

        progress.update(progress_task, current=image_name)
        console.print(progress.get_renderable())

        start_time = time.time()

        try:
            classification_info = {
                "document_type": row.document_type,
                "confidence": row.confidence,
                "raw_response": row.raw_response,
                "prompt_used": row.prompt_used,
            }

            result = processor.process_document_aware(
                row.image_path,
                classification_info,
                verbose=verbose,
            )

            elapsed = time.time() - start_time
            extracted_data = result.get("extracted_data", {})

            records[orig_idx] = ExtractionRecord(
                image_path=row.image_path,
                image_name=image_name,
                document_type=row.document_type,
                extracted_data=extracted_data,
                processing_time=elapsed,
                prompt_used=row.document_type.lower(),
                field_count=result.get("field_count", len(extracted_data)),
                fields_found=sum(
                    1 for v in extracted_data.values() if v != "NOT_FOUND"
                ),
                timestamp=datetime.now().isoformat(),
                raw_response=result.get("raw_response", ""),
            )
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error("Extraction failed for %s: %s", image_name, e)
            records[orig_idx] = ExtractionRecord(
                image_path=row.image_path,
                image_name=image_name,
                document_type=row.document_type,
                extracted_data={},
                processing_time=elapsed,
                prompt_used="",
                field_count=0,
                fields_found=0,
                timestamp=datetime.now().isoformat(),
                error=str(e),
            )

        progress.update(progress_task, advance=1)


# ---------------------------------------------------------------------------
# Bank statement extraction (always sequential, multi-turn)
# ---------------------------------------------------------------------------


def _extract_bank_sequential(
    processor,
    classifications: ClassificationOutput,
    indices: list[int],
    records: list[ExtractionRecord | None],
    bank_adapter,
    verbose: bool,
    progress: Progress,
    progress_task,
    console: Console,
) -> None:
    """Sequential extraction for bank statements.

    Routes to BankStatementAdapter when available, falls back to
    standard process_document_aware().
    """
    for orig_idx in indices:
        row = classifications.rows[orig_idx]
        image_name = Path(row.image_path).name

        logger.info("BANK STATEMENT (sequential): %s", image_name)
        progress.update(progress_task, current=image_name)
        console.print(progress.get_renderable())

        start_time = time.time()

        try:
            # Try BankStatementAdapter first
            if bank_adapter is not None:
                record = _extract_bank_with_adapter(
                    row.image_path,
                    image_name,
                    row.document_type,
                    bank_adapter,
                )
                if record is not None:
                    record.processing_time = time.time() - start_time
                    records[orig_idx] = record
                    progress.update(progress_task, advance=1)
                    continue
                # Adapter failed, fall through to standard extraction

            # Standard extraction fallback
            classification_info = {
                "document_type": row.document_type,
                "confidence": row.confidence,
                "raw_response": row.raw_response,
                "prompt_used": row.prompt_used,
            }

            result = processor.process_document_aware(
                row.image_path,
                classification_info,
                verbose=verbose,
            )

            elapsed = time.time() - start_time
            extracted_data = result.get("extracted_data", {})

            records[orig_idx] = ExtractionRecord(
                image_path=row.image_path,
                image_name=image_name,
                document_type=row.document_type,
                extracted_data=extracted_data,
                processing_time=elapsed,
                prompt_used=row.document_type.lower(),
                field_count=result.get("field_count", len(extracted_data)),
                fields_found=sum(
                    1 for v in extracted_data.values() if v != "NOT_FOUND"
                ),
                timestamp=datetime.now().isoformat(),
                raw_response=result.get("raw_response", ""),
            )

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error("Error processing bank statement %s: %s", image_name, e)
            records[orig_idx] = ExtractionRecord(
                image_path=row.image_path,
                image_name=image_name,
                document_type=row.document_type,
                extracted_data={},
                processing_time=elapsed,
                prompt_used="",
                field_count=0,
                fields_found=0,
                timestamp=datetime.now().isoformat(),
                error=str(e),
            )

        progress.update(progress_task, advance=1)


def _extract_bank_with_adapter(
    image_path: str,
    image_name: str,
    document_type: str,
    bank_adapter,
) -> ExtractionRecord | None:
    """Try extracting via BankStatementAdapter. Returns None on failure."""
    try:
        schema_fields, metadata = bank_adapter.extract_bank_statement(image_path)

        strategy = metadata.get("strategy_used", "unknown")
        raw_response = metadata.get("raw_responses", {}).get("turn1", "")

        tx_count = (
            len(schema_fields.get("TRANSACTION_DATES", "").split("|"))
            if schema_fields.get("TRANSACTION_DATES") != "NOT_FOUND"
            else 0
        )
        logger.debug("Bank adapter strategy: %s, transactions: %d", strategy, tx_count)

        return ExtractionRecord(
            image_path=image_path,
            image_name=image_name,
            document_type=document_type,
            extracted_data=schema_fields,
            processing_time=0.0,  # Caller sets this
            prompt_used=f"unified_bank_{strategy}",
            field_count=len(schema_fields),
            fields_found=sum(1 for v in schema_fields.values() if v != "NOT_FOUND"),
            timestamp=datetime.now().isoformat(),
            raw_response=raw_response,
            skip_math_enhancement=True,
        )
    except Exception as e:
        logger.warning("BankStatementAdapter failed for %s: %s", image_name, e)
        logger.warning("Falling back to standard extraction...")
        return None


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


def write_extraction_json(output: ExtractionOutput, output_path: Path) -> Path:
    """Write extraction results to JSON.

    Args:
        output: ExtractionOutput from extract_documents().
        output_path: Path to write the JSON file.

    Returns:
        The path written to.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "metadata": output.metadata,
        "results": [
            {
                "image_path": r.image_path,
                "image_name": r.image_name,
                "document_type": r.document_type,
                "extracted_data": r.extracted_data,
                "processing_time": r.processing_time,
                "prompt_used": r.prompt_used,
                "field_count": r.field_count,
                "fields_found": r.fields_found,
                "timestamp": r.timestamp,
                "error": r.error,
            }
            for r in output.records
        ],
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(
        "Wrote extraction JSON: %s (%d records)", output_path, len(output.records)
    )
    return output_path


def read_extraction_json(json_path: Path) -> ExtractionOutput:
    """Read extraction JSON back into ExtractionOutput.

    Used by the evaluate stage when running standalone.

    Args:
        json_path: Path to extraction JSON file.

    Returns:
        ExtractionOutput with records populated from JSON.

    Raises:
        FileNotFoundError: If json_path does not exist.
        ValueError: If JSON structure is invalid.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Extraction JSON not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "results" not in data:
        raise ValueError(
            f"Invalid extraction JSON: missing 'results' key in {json_path}"
        )

    records: list[ExtractionRecord] = []
    for item in data["results"]:
        records.append(
            ExtractionRecord(
                image_path=item["image_path"],
                image_name=item.get("image_name", Path(item["image_path"]).name),
                document_type=item.get("document_type", "UNKNOWN"),
                extracted_data=item.get("extracted_data", {}),
                processing_time=float(item.get("processing_time", 0)),
                prompt_used=item.get("prompt_used", ""),
                field_count=int(item.get("field_count", 0)),
                fields_found=int(item.get("fields_found", 0)),
                timestamp=item.get("timestamp", ""),
                error=item.get("error", ""),
            )
        )

    metadata = data.get("metadata", {})
    logger.info("Read extraction JSON: %s (%d records)", json_path, len(records))
    return ExtractionOutput(records=records, metadata=metadata)
