"""Convert ground truth CSV to JSONL format.

Reads a CSV where every row has all columns (cross-schema fields filled
with NOT_FOUND), and writes JSONL where each record carries only its
document type's fields.  Uses field_definitions.yaml to determine which
fields belong to each type.

Usage:
    python -m scripts.csv_to_jsonl \
        --csv ../evaluation_data/synthetic/ground_truth_synthetic.csv \
        --output ../evaluation_data/synthetic/ground_truth.jsonl
"""

import json
import sys
from pathlib import Path

import typer
import yaml

app = typer.Typer()

# Resolve field_definitions.yaml relative to this script
_FIELD_DEFS = (
    Path(__file__).resolve().parent.parent / "config" / "field_definitions.yaml"
)

# Keys that identify the filename column (tried in order)
_FILENAME_KEYS = ("filename", "image_name", "image_file", "file")

# typer.Option singletons (B008)
_OPT_CSV: Path = typer.Option(..., "--csv", "-c", help="Input ground truth CSV")  # type: ignore[assignment]
_OPT_OUTPUT: Path = typer.Option(..., "--output", "-o", help="Output JSONL path")  # type: ignore[assignment]
_OPT_FIELD_DEFS: Path = typer.Option(  # type: ignore[assignment]
    _FIELD_DEFS, "--field-defs", "-f", help="Path to field_definitions.yaml"
)


def _load_type_fields(field_defs_path: Path) -> dict[str, list[str]]:
    """Load document_fields from field_definitions.yaml.

    Returns mapping of UPPERCASE doc type -> list of field names.
    """
    with field_defs_path.open() as f:
        defs = yaml.safe_load(f)

    type_fields: dict[str, list[str]] = {}
    for doc_type, info in defs.get("document_fields", {}).items():
        fields = info.get("fields", [])
        type_fields[doc_type.upper()] = fields
    return type_fields


@app.command()
def main(
    csv: Path = _OPT_CSV,
    output: Path = _OPT_OUTPUT,
    field_defs: Path = _OPT_FIELD_DEFS,
) -> None:
    """Convert ground truth CSV to per-type JSONL."""
    import pandas as pd

    if not csv.exists():
        print(f"ERROR: CSV not found: {csv}", file=sys.stderr)
        raise typer.Exit(1) from None

    if not field_defs.exists():
        print(f"ERROR: field_definitions.yaml not found: {field_defs}", file=sys.stderr)
        raise typer.Exit(1) from None

    type_fields = _load_type_fields(field_defs)
    df = pd.read_csv(csv, dtype=str).fillna("NOT_FOUND")

    # Find filename column
    filename_col = None
    for key in _FILENAME_KEYS:
        if key in df.columns:
            filename_col = key
            break
    if filename_col is None:
        print(
            f"ERROR: No filename column found. Expected one of: {_FILENAME_KEYS}",
            file=sys.stderr,
        )
        raise typer.Exit(1) from None

    # Find document type column
    doc_type_col = None
    for candidate in ("DOCUMENT_TYPE", "document_type"):
        if candidate in df.columns:
            doc_type_col = candidate
            break
    if doc_type_col is None:
        print("ERROR: No DOCUMENT_TYPE column found", file=sys.stderr)
        raise typer.Exit(1) from None

    records_written = 0
    skipped_types: dict[str, int] = {}

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for _, row in df.iterrows():
            doc_type = str(row[doc_type_col]).strip().upper()
            filename = str(row[filename_col]).strip()

            # Look up per-type fields
            fields = type_fields.get(doc_type)
            if fields is None:
                skipped_types[doc_type] = skipped_types.get(doc_type, 0) + 1
                continue

            record: dict[str, str] = {"filename": filename}
            for field in fields:
                val = str(row.get(field, "NOT_FOUND")).strip()
                record[field] = val

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            records_written += 1

    print(f"Wrote {records_written} records to {output}")
    if skipped_types:
        for dtype, count in sorted(skipped_types.items()):
            print(f"  SKIPPED {count} records with unknown type: {dtype}")


if __name__ == "__main__":
    app()
