# ruff: noqa: B008 - typer.Option in defaults is the standard Typer pattern
"""Stage 3: Parse and clean raw model responses (CPU).

Reads raw_extractions.jsonl, runs ResponseHandler.handle() on each raw
response to produce structured field dicts, and writes cleaned_extractions.jsonl.

No GPU needed -- this stage runs on CPU nodes.

Usage:
    python -m stages.clean \
        --input /artifacts/raw_extractions.jsonl \
        --output /artifacts/cleaned_extractions.jsonl
"""

import logging
from pathlib import Path
from typing import Any

import typer

from common.field_schema import get_field_schema
from common.response_handler import create_response_handler

from .io import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)
app = typer.Typer()


def run(
    raw_extractions_path: Path,
    output_path: Path,
    *,
    debug: bool = False,
) -> Path:
    """Parse, clean, and validate raw model responses.

    For each record in raw_extractions.jsonl:
      1. Look up the expected fields for the document type
      2. Run ResponseHandler.handle(raw_response, expected_fields)
      3. Write the cleaned field dict to cleaned_extractions.jsonl

    Args:
        raw_extractions_path: Path to raw_extractions.jsonl from Stage 2.
        output_path: Path to write cleaned_extractions.jsonl.
        debug: Enable debug logging in response handler.

    Returns:
        Path to the written cleaned_extractions.jsonl.
    """
    # Read raw extractions
    raw_records = read_jsonl(raw_extractions_path)
    if not raw_records:
        msg = f"No records found in {raw_extractions_path}"
        raise FileNotFoundError(msg)

    logger.info(
        "Read %d raw extractions from %s", len(raw_records), raw_extractions_path
    )

    # Set up response handler (CPU-only, no model)
    schema = get_field_schema()
    handler = create_response_handler(schema=schema, debug=debug)

    # Build field lookup: doc_type -> list of expected fields
    all_fields = schema.get_all_doc_type_fields()

    cleaned_records: list[dict[str, Any]] = []

    for record in raw_records:
        image_name = record["image_name"]
        doc_type = record["document_type"]
        raw_response = record.get("raw_response", "")
        error = record.get("error")

        # Skip records with extraction errors
        if error:
            logger.warning("Skipping %s: extraction error: %s", image_name, error)
            cleaned_records.append(
                {
                    "image_name": image_name,
                    "image_path": record.get("image_path", ""),
                    "document_type": doc_type,
                    "extracted_data": {},
                    "field_count": 0,
                    "extracted_fields_count": 0,
                    "error": error,
                }
            )
            continue

        # Resolve expected fields for this document type
        doc_type_lower = doc_type.lower()
        expected_fields = all_fields.get(doc_type_lower)

        if expected_fields is None:
            # Try stripping structure suffixes (bank_statement_flat -> bank_statement)
            base_type = doc_type_lower.replace("_flat", "").replace("_date_grouped", "")
            expected_fields = all_fields.get(base_type, [])

        if not expected_fields:
            logger.warning(
                "No field definitions for doc_type '%s' -- using empty list", doc_type
            )

        # Parse + clean + validate
        extracted_data = handler.handle(raw_response, expected_fields)

        found_count = sum(1 for v in extracted_data.values() if v != "NOT_FOUND")

        cleaned_records.append(
            {
                "image_name": image_name,
                "image_path": record.get("image_path", ""),
                "document_type": doc_type,
                "extracted_data": extracted_data,
                "field_count": len(expected_fields),
                "extracted_fields_count": found_count,
            }
        )

        logger.debug(
            "%s: %d/%d fields extracted",
            image_name,
            found_count,
            len(expected_fields),
        )

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = write_jsonl(output_path, cleaned_records)
    logger.info("Wrote %d cleaned extractions to %s", count, output_path)

    # Summary
    total_fields = sum(r.get("field_count", 0) for r in cleaned_records)
    total_found = sum(r.get("extracted_fields_count", 0) for r in cleaned_records)
    error_count = sum(1 for r in cleaned_records if r.get("error"))
    logger.info(
        "Summary: %d records, %d/%d fields found, %d errors",
        len(cleaned_records),
        total_found,
        total_fields,
        error_count,
    )

    return output_path


@app.command()
def main(
    input_path: Path = typer.Option(
        ..., "--input", help="Path to raw_extractions.jsonl from Stage 2"
    ),
    output: Path = typer.Option(
        ..., "--output", help="Path to write cleaned_extractions.jsonl"
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    """Stage 3: Parse and clean raw model responses (CPU only)."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    run(input_path, output, debug=debug)


if __name__ == "__main__":
    app()
