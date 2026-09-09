"""Tests for scripts/generate_extraction_gt.py (YAML -> extraction CSV)."""

import csv
from pathlib import Path

import yaml

from scripts.generate_extraction_gt import (
    _build_columns,
    _format_value,
    _load_schema,
    _row_for_entry,
    generate,
)

_SCHEMA = _load_schema(Path("config/field_definitions.yaml"))
_MONETARY = _SCHEMA["monetary"]
_BOOLEAN = _SCHEMA["boolean"]


# ---------------------------------------------------------------------------
# columns
# ---------------------------------------------------------------------------


def test_build_columns_image_file_first_and_union():
    cols = _build_columns(_SCHEMA)
    assert cols[0] == "image_file"
    # union spans both invoice and bank-only fields
    for expected in (
        "DOCUMENT_TYPE",
        "TOTAL_AMOUNT",
        "LINE_ITEM_DESCRIPTIONS",
        "STATEMENT_DATE_RANGE",
        "TRANSACTION_DATES",
        "TRANSACTION_AMOUNTS_PAID",
    ):
        assert expected in cols
    assert len(cols) == len(set(cols))  # no duplicates


# ---------------------------------------------------------------------------
# value formatting
# ---------------------------------------------------------------------------


def test_format_value_monetary_prefixes_dollar():
    out = _format_value("TOTAL_AMOUNT", "34.16", _MONETARY, _BOOLEAN)
    assert out == "$34.16"


def test_format_value_monetary_pipe_items_and_not_found():
    out = _format_value("TRANSACTION_AMOUNTS_PAID", "NOT_FOUND|212.93|51.04", _MONETARY, _BOOLEAN)
    assert out == "NOT_FOUND | $212.93 | $51.04"


def test_format_value_does_not_double_dollar():
    out = _format_value("TOTAL_AMOUNT", "$5.00", _MONETARY, _BOOLEAN)
    assert out == "$5.00"


def test_format_value_boolean_lowercased():
    assert _format_value("IS_GST_INCLUDED", "True", _MONETARY, _BOOLEAN) == "true"


def test_format_value_pipe_normalized_for_text():
    out = _format_value("LINE_ITEM_DESCRIPTIONS", "A|B|C", _MONETARY, _BOOLEAN)
    assert out == "A | B | C"


def test_format_value_empty_is_not_found():
    assert _format_value("SUPPLIER_NAME", "", _MONETARY, _BOOLEAN) == "NOT_FOUND"


# ---------------------------------------------------------------------------
# row building
# ---------------------------------------------------------------------------


def test_row_for_entry_builds_image_file_and_maps_alias():
    cols = _build_columns(_SCHEMA)
    entry = {
        "layout": "cba_standard",
        "fields": {
            "DOCUMENT_TYPE": "BANK_STATEMENT",
            "TRANSACTION_DESCRIPTIONS": "EFTPOS A|BPAY B",  # alias -> LINE_ITEM_DESCRIPTIONS
            "TRANSACTION_AMOUNTS_PAID": "212.93|51.04",
            "STATEMENT_DATE_RANGE": "01/01/2023 - 31/01/2023",
        },
    }
    image_file, row = _row_for_entry("CASE001", entry, cols, _MONETARY, _BOOLEAN)
    assert image_file == "CASE001_cba_standard.png"
    assert row["LINE_ITEM_DESCRIPTIONS"] == "EFTPOS A | BPAY B"
    assert row["TRANSACTION_AMOUNTS_PAID"] == "$212.93 | $51.04"
    assert row["TOTAL_AMOUNT"] == "NOT_FOUND"  # not applicable to a bank statement


# ---------------------------------------------------------------------------
# end-to-end generate()
# ---------------------------------------------------------------------------


def test_generate_end_to_end(tmp_path):
    yaml_dir = tmp_path / "gt"
    yaml_dir.mkdir()
    (yaml_dir / "receipts.yml").write_text(
        yaml.safe_dump(
            {
                "CASE001": {
                    "layout": "receipt_thermal_80mm",
                    "fields": {
                        "DOCUMENT_TYPE": "RECEIPT",
                        "SUPPLIER_NAME": "Bunnings",
                        "TOTAL_AMOUNT": "34.16",
                        "IS_GST_INCLUDED": "true",
                    },
                }
            }
        )
    )
    (yaml_dir / "invoices.yml").write_text(
        yaml.safe_dump(
            {
                "CASE001": {
                    "layout": "tax_invoice_standard",
                    "fields": {"DOCUMENT_TYPE": "INVOICE", "TOTAL_AMOUNT": "6493.84"},
                }
            }
        )
    )
    (yaml_dir / "bank_statements.yml").write_text(
        yaml.safe_dump(
            {
                "CASE001": {
                    "layout": "cba_standard",
                    "fields": {
                        "DOCUMENT_TYPE": "BANK_STATEMENT",
                        "TRANSACTION_DESCRIPTIONS": "EFTPOS A|BPAY B",
                        "TRANSACTION_AMOUNTS_PAID": "212.93|51.04",
                    },
                }
            }
        )
    )

    out = tmp_path / "gt_extraction.csv"
    count = generate(yaml_dir, out, data_dir=None, schema=_SCHEMA)
    assert count == 3

    rows = list(csv.DictReader(out.open()))
    assert rows[0].keys().__contains__("image_file")
    by_name = {r["image_file"]: r for r in rows}
    assert set(by_name) == {
        "CASE001_receipt_thermal_80mm.png",
        "CASE001_tax_invoice_standard.png",
        "CASE001_cba_standard.png",
    }
    assert by_name["CASE001_receipt_thermal_80mm.png"]["TOTAL_AMOUNT"] == "$34.16"
    assert by_name["CASE001_cba_standard.png"]["LINE_ITEM_DESCRIPTIONS"] == "EFTPOS A | BPAY B"


def test_generate_missing_source_file_diagnostic(tmp_path, assert_diagnostic_error):
    import pytest

    yaml_dir = tmp_path / "gt"
    yaml_dir.mkdir()
    (yaml_dir / "receipts.yml").write_text(yaml.safe_dump({}))
    # invoices.yml / bank_statements.yml missing -> diagnostic FileNotFoundError
    with pytest.raises(FileNotFoundError) as exc:
        generate(yaml_dir, tmp_path / "o.csv", data_dir=None, schema=_SCHEMA)
    assert_diagnostic_error(str(exc.value))
