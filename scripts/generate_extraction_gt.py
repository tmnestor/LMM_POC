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

Columns and field-type formatting are read directly from LMM_POC's
``config/field_definitions.yaml`` (the single source of truth, via ``--schema``)
-- not hardcoded -- so the CSV tracks schema changes. The script imports nothing
from the LMM_POC package, so it can live in the Synthetic_Doc_Generation repo and
run standalone; ``--schema`` just points back at the authoritative file.

Usage:
    python3 generate_extraction_gt.py \
        --yaml-dir /Users/tod/Desktop/Synthetic_Doc_Generation/ground_truth \
        --schema   /Users/tod/Desktop/LMM_POC/config/field_definitions.yaml \
        --output   /Users/tod/Desktop/evaluation_data/synthetic_transaction_linking/ground_truth_extraction.csv \
        --data-dir /Users/tod/Desktop/evaluation_data/synthetic_transaction_linking
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import yaml

# Fields that exist for validation but are excluded from the evaluation CSV.
# Mirrors ``common.field_schema._VALIDATION_ONLY`` in LMM_POC -- the only piece
# of schema policy not expressible in field_definitions.yaml. Keep in sync.
_VALIDATION_ONLY: frozenset[str] = frozenset({"TRANSACTION_AMOUNTS_RECEIVED", "ACCOUNT_BALANCE"})

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


def _doc_type_fields(doc_fields: dict[str, Any], doc_type: str, schema_path: Path) -> list[str]:
    """Per-doc-type extraction field list from the schema YAML, validation-only excluded."""
    cfg = doc_fields.get(doc_type)
    if not isinstance(cfg, dict) or not cfg.get("fields"):
        raise ValueError(
            _diagnostic(
                what=f"document type '{doc_type}' is missing or has no non-empty 'fields' list.",
                where=f"{schema_path} -> document_fields.{doc_type}.fields",
                example="document_fields:\n  invoice:\n    fields: [SUPPLIER_NAME, TOTAL_AMOUNT, ...]",
                fix=f"ensure '{doc_type}' with a non-empty 'fields' list exists in the schema YAML.",
            )
        )
    return [f for f in cfg["fields"] if f not in _VALIDATION_ONLY]


def _load_schema(schema_path: Path) -> dict[str, Any]:
    """Read LMM_POC's field_definitions.yaml -- the column / field-type contract.

    Minimal, read-only mirror of ``common.field_schema.get_field_schema()`` so this
    generator runs standalone (outside LMM_POC) while still tracking that YAML as the
    single source of truth. Returns invoice/bank field lists + monetary/boolean sets.
    """
    if not schema_path.is_file():
        raise FileNotFoundError(
            _diagnostic(
                what="the field-definitions schema YAML does not exist.",
                where=str(schema_path),
                example="LMM_POC/config/field_definitions.yaml (the column / field-type contract).",
                fix="pass --schema pointing at LMM_POC's config/field_definitions.yaml.",
            )
        )
    raw = yaml.safe_load(schema_path.read_text()) or {}
    doc_fields = raw.get("document_fields", {})
    field_types = raw.get("evaluation", {}).get("field_types", {})
    return {
        "invoice_fields": _doc_type_fields(doc_fields, "invoice", schema_path),
        "bank_fields": _doc_type_fields(doc_fields, "bank_statement", schema_path),
        "monetary": frozenset(field_types.get("monetary", [])),
        "boolean": frozenset(field_types.get("boolean", [])),
    }


def _build_columns(schema: dict[str, Any]) -> list[str]:
    """Derive the CSV column order from the schema (image_file first).

    Union of the invoice/receipt and bank-statement extraction fields, ordered
    invoice-fields-first then any bank-only extras, with ``image_file`` prepended.
    """
    columns = ["image_file", *schema["invoice_fields"]]
    for field in schema["bank_fields"]:
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


def generate(yaml_dir: Path, output: Path, data_dir: Path | None, schema: dict[str, Any]) -> int:
    """Generate the extraction ground-truth CSV. Returns the row count."""
    if not yaml_dir.is_dir():
        msg = _diagnostic(
            what="the source YAML directory does not exist.",
            where=str(yaml_dir),
            example="a directory containing bank_statements.yml, invoices.yml, receipts.yml.",
            fix="pass --yaml-dir pointing at the Synthetic_Doc_Generation/ground_truth directory.",
        )
        raise FileNotFoundError(msg)

    monetary = schema["monetary"]
    boolean = schema["boolean"]
    columns = _build_columns(schema)

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
        "--schema",
        type=Path,
        default=Path("/Users/tod/Desktop/LMM_POC/config/field_definitions.yaml"),
        help="LMM_POC config/field_definitions.yaml (column / field-type contract).",
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

    schema = _load_schema(args.schema)
    count = generate(args.yaml_dir, args.output, args.data_dir, schema)
    print(f"Wrote {count} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
