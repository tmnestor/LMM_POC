# ruff: noqa: B008 - typer.Option in defaults is the standard Typer pattern
"""Stage: Trust distribution compliance cleaning (CPU).

Reads raw_extractions.jsonl from the trust extraction stage,
re-parses per-node raw_response strings via FieldValueParser,
builds WorkflowState with NodeResult per extraction node,
and runs trust compliance validation.

No GPU needed -- this stage runs on CPU nodes.

Usage:
    python -m stages.trust_clean \
        --input /artifacts/raw_extractions.jsonl \
        --output /artifacts/trust_compliance_results.jsonl
"""

import logging
from pathlib import Path
from typing import Any

import typer

from common.extraction_types import NodeResult, WorkflowState
from common.trust_compliance import run_trust_compliance
from common.turn_parsers import FieldValueParser, ParseError

from .io import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)
app = typer.Typer()

_EXTRACTION_NODES = (
    "extract_trust_return",
    "extract_distribution_stmt",
    "extract_income_schedule",
    "extract_beneficiary_itr",
)


def _build_workflow_state(nodes: list[dict[str, Any]]) -> WorkflowState:
    """Reconstruct WorkflowState from serialized nodes[] array.

    Re-parses each node's raw_response through FieldValueParser to
    recover the parsed dict (not serialized by ExtractionSession.to_record).
    """
    parser = FieldValueParser()
    dummy_context = WorkflowState()
    state = WorkflowState()

    for node in nodes:
        key = node.get("key", "")
        if key not in _EXTRACTION_NODES:
            continue

        raw_response = node.get("raw_response", "")
        try:
            parsed = parser.parse(raw_response, dummy_context)
        except ParseError:
            logger.warning("Failed to parse node %s raw_response, using empty dict", key)
            parsed = {}

        state.node_results[key] = NodeResult(
            key=key,
            image_ref=node.get("image_ref", ""),
            prompt_sent="",
            raw_response=raw_response,
            parsed=parsed,
            elapsed=node.get("elapsed", 0.0),
            attempt=node.get("attempt", 1),
            edge_taken=node.get("edge_taken", "ok"),
        )

    return state


def run(
    input_path: Path,
    output_path: Path,
) -> Path:
    """Parse trust extraction nodes and run compliance validation.

    For each record in raw_extractions.jsonl:
      1. Extract per-node raw_responses from record["nodes"]
      2. Re-parse each with FieldValueParser
      3. Build WorkflowState with NodeResult per extraction node
      4. Call run_trust_compliance(state)
      5. Write record with compliance fields in extracted_data

    Args:
        input_path: Path to raw_extractions.jsonl from trust extraction stage.
        output_path: Path to write trust_compliance_results.jsonl.

    Returns:
        Path to the written trust_compliance_results.jsonl.
    """
    records = read_jsonl(input_path)
    if not records:
        msg = f"No records found in {input_path}"
        raise FileNotFoundError(msg)

    logger.info("Read %d raw extractions from %s", len(records), input_path)

    cleaned_records: list[dict[str, Any]] = []

    for record in records:
        image_name = record.get("image_name", "")
        error = record.get("error")

        if error:
            logger.warning("Skipping %s: extraction error: %s", image_name, error)
            cleaned_records.append(
                {
                    "image_name": image_name,
                    "image_path": record.get("image_path", ""),
                    "document_type": record.get("document_type", ""),
                    "extracted_data": {},
                    "processing_time": record.get("processing_time", 0.0),
                    "error": error,
                }
            )
            continue

        nodes = record.get("nodes", [])
        state = _build_workflow_state(nodes)

        _all_match, result = run_trust_compliance(state)

        # Remove field_comparisons from extracted_data (internal detail)
        result.pop("field_comparisons", None)

        cleaned_records.append(
            {
                "image_name": image_name,
                "image_path": record.get("image_path", ""),
                "document_type": record.get("document_type", ""),
                "extracted_data": result,
                "processing_time": record.get("processing_time", 0.0),
                "error": None,
            }
        )

        logger.debug(
            "%s: compliance=%s discrepancy=%s",
            image_name,
            result.get("COMPLIANCE_STATUS"),
            result.get("DISCREPANCY_TYPE"),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = write_jsonl(output_path, cleaned_records)
    logger.info("Wrote %d trust compliance results to %s", count, output_path)

    compliant = sum(
        1 for r in cleaned_records if r.get("extracted_data", {}).get("COMPLIANCE_STATUS") == "compliant"
    )
    non_compliant = sum(
        1
        for r in cleaned_records
        if r.get("extracted_data", {}).get("COMPLIANCE_STATUS") == "non_compliant"
    )
    errors = sum(1 for r in cleaned_records if r.get("error"))
    logger.info(
        "Summary: %d records — %d compliant, %d non-compliant, %d errors",
        len(cleaned_records),
        compliant,
        non_compliant,
        errors,
    )

    return output_path


@app.command()
def main(
    input_path: Path = typer.Option(
        ..., "--input", "-i", help="Path to raw_extractions.jsonl from trust extraction stage"
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Path to write trust_compliance_results.jsonl"),
) -> None:
    """Stage: Trust distribution compliance cleaning (CPU only)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    run(input_path, output)


if __name__ == "__main__":
    app()
