#!/usr/bin/env python3
"""Generate an extraction ground-truth CSV from the per-type synthetic YAMLs.

Merges ``bank_statements.yml`` + ``invoices.yml`` + ``receipts.yml`` (the
Synthetic_Doc_Generation ground truth) into a single CSV in the shape the
standard ``stages/evaluate.py`` stage expects: one row per image keyed by
``image_file``, columns = the union of the schema's per-doc-type extraction
fields, values formatted to match the model-output convention (monetary fields
carry ``$``, multi-value fields use `` | `` separators, missing fields are
``NOT_FOUND``).

Image filenames are reconstructed as ``{CASE}_{layout}.png`` from each YAML
entry's ``layout`` field, matching the synthetic_transaction_linking dataset.

Columns and field-type formatting are derived from ``common.field_schema`` (the
single source of truth) — not hardcoded — so the CSV tracks schema changes.

Usage:
    python3 scripts/generate_extraction_gt.py \
        --yaml-dir /Users/tod/Desktop/Synthetic_Doc_Generation/ground_truth \
        --output   ../evaluation_data/synthetic_transaction_linking/ground_truth_extraction.csv \
        --data-dir ../evaluation_data/synthetic_transaction_linking
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import yaml

# Allow running as a plain script from anywhere: put the repo root on sys.path
# so ``common.*`` resolves (mirrors how the stages are invoked as modules).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.field_schema import get_field_schema  # noqa: E402

# Source YAML file -> the doc-type key whose schema fields it carries.
_SOURCE_FILES: dict[str, str] = {
    "bank_statements.yml": "bank_statement",
    "invoices.yml": "invoice",
    "receipts.yml": "receipt",
}

# YAML field name -> canonical schema field name (only where they differ).
_FIELD_ALIASES: dict[str, str] = {
    "TRANSACTION_DESCRIPTIONS": "LINE_ITEM_DESCRIPTIONS",  # bank statements
}


def _diagnostic(what: str, where: str, example: str, fix: str) -> str:
    """Assemble a 4-element diagnostic error message."""
    return f"What: {what}\nWhere: {where}\nExpected: {example}\nHow to fix: {fix}"


def _build_columns() -> list[str]:
    """Derive the CSV column order from the schema (image_file first).

    Union of the invoice/receipt and bank-statement extraction fields, ordered
    invoice-fields-first then any bank-only extras, with ``image_file`` prepended.
    """
    schema = get_field_schema()
    invoice_fields = list(schema.get_document_type_fields("invoice"))
    bank_fields = list(schema.get_document_type_fields("bank_statement"))
    columns = ["image_file", *invoice_fields]
    for field in bank_fields:
        if field not in columns:
            columns.append(field)
    return columns


def _normalize_pipes(value: str) -> list[str]:
    """Split a pipe-delimited value into stripped items (no spacing assumed)."""
    return [item.strip() for item in value.split("|")]


def _format_value(field: str, raw: Any, monetary: frozenset[str], boolean: frozenset[str]) -> str:
    """Format one YAML field value to the evaluate-stage CSV convention.

    - Multi-value fields are re-joined with `` | `` (space-pipe-space).
    - Monetary fields get a ``$`` prefix per item (NOT_FOUND items untouched).
    - Boolean fields are lowercased.
    """
    text = str(raw).strip()
    if not text:
        return "NOT_FOUND"

    items = _normalize_pipes(text)

    if field in monetary:
        items = [it if it.upper() == "NOT_FOUND" or it.startswith("$") else f"${it}" for it in items]
    elif field in boolean:
        items = [it.lower() for it in items]

    return " | ".join(items)


def _row_for_entry(
    case_id: str,
    entry: dict[str, Any],
    columns: list[str],
    monetary: frozenset[str],
    boolean: frozenset[str],
) -> tuple[str, dict[str, str]]:
    """Build (image_file, row dict) for one CASE entry of a source YAML."""
    layout = entry.get("layout")
    if not layout:
        msg = _diagnostic(
            what=f"case {case_id} has no 'layout' field — cannot build the image filename.",
            where=f"source YAML entry {case_id}",
            example="each entry must carry layout: <name> (image = {CASE}_{layout}.png).",
            fix="regenerate the ground-truth YAML with a 'layout' per case.",
        )
        raise ValueError(msg)

    image_file = f"{case_id}_{layout}.png"
    fields = entry.get("fields", {}) or {}

    # Apply aliases (e.g. bank TRANSACTION_DESCRIPTIONS -> LINE_ITEM_DESCRIPTIONS).
    resolved: dict[str, Any] = {}
    for key, val in fields.items():
        resolved[_FIELD_ALIASES.get(key, key)] = val

    row: dict[str, str] = {"image_file": image_file}
    for col in columns:
        if col == "image_file":
            continue
        if col in resolved and str(resolved[col]).strip():
            row[col] = _format_value(col, resolved[col], monetary, boolean)
        else:
            row[col] = "NOT_FOUND"
    return image_file, row


def generate(yaml_dir: Path, output: Path, data_dir: Path | None) -> int:
    """Generate the extraction ground-truth CSV. Returns the row count."""
    if not yaml_dir.is_dir():
        msg = _diagnostic(
            what="the source YAML directory does not exist.",
            where=str(yaml_dir),
            example="a directory containing bank_statements.yml, invoices.yml, receipts.yml.",
            fix="pass --yaml-dir pointing at the Synthetic_Doc_Generation/ground_truth directory.",
        )
        raise FileNotFoundError(msg)

    schema = get_field_schema()
    monetary = schema.monetary_fields
    boolean = schema.boolean_fields
    columns = _build_columns()

    rows: list[dict[str, str]] = []
    image_files: list[str] = []
    for filename, _doc_type in _SOURCE_FILES.items():
        path = yaml_dir / filename
        if not path.is_file():
            msg = _diagnostic(
                what=f"required source file '{filename}' is missing.",
                where=str(path),
                example=f"{yaml_dir}/{filename} (a mapping of CASE id -> {{layout, fields}}).",
                fix=f"ensure {filename} exists in --yaml-dir.",
            )
            raise FileNotFoundError(msg)

        data = yaml.safe_load(path.read_text()) or {}
        for case_id, entry in data.items():
            image_file, row = _row_for_entry(case_id, entry, columns, monetary, boolean)
            rows.append(row)
            image_files.append(image_file)

    rows.sort(key=lambda r: r["image_file"])

    # Optional: warn if generated filenames don't exist in the dataset.
    if data_dir is not None:
        missing = [name for name in image_files if not (data_dir / name).is_file()]
        if missing:
            print(
                f"WARNING: {len(missing)}/{len(image_files)} image filenames not found in "
                f"{data_dir} (first few: {missing[:5]})",
                file=sys.stderr,
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yaml-dir",
        type=Path,
        default=Path("/Users/tod/Desktop/Synthetic_Doc_Generation/ground_truth"),
        help="Directory with bank_statements.yml / invoices.yml / receipts.yml.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the extraction ground-truth CSV.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional image directory to validate generated filenames against.",
    )
    args = parser.parse_args()

    count = generate(args.yaml_dir, args.output, args.data_dir)
    print(f"Wrote {count} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
