"""Tests for scripts/relabel_evaluation_set.py (raw synthetic set -> YAML + JSONL)."""

import json
from pathlib import Path

import pytest
import yaml

from common.field_schema import get_field_schema
from scripts.relabel_evaluation_set import (
    BACKUP_NAME,
    JSONL_NAME,
    MAPPING_NAME,
    NOT_FOUND,
    YAML_NAME,
    format_value,
    load_blocks,
    main,
    plan_documents,
    project_fields,
)

_SCHEMA = get_field_schema()

# A bank block as the generator emits it: metadata, non-schema fields, and the
# TRANSACTION_DESCRIPTIONS spelling the pipeline does not use.
_BANK_FIELDS = {
    "DOCUMENT_TYPE": "BANK_STATEMENT",
    "SUPPLIER_NAME": "Commonwealth Bank",
    "STATEMENT_DATE_RANGE": "01/07/2024 - 29/07/2024",
    "TRANSACTION_DATES": "01/07/2024|02/07/2024",
    "TRANSACTION_DESCRIPTIONS": "ATM WITHDRAWAL Alexandria|EFTPOS ASHBY LEGAL",
    "TRANSACTION_AMOUNTS_PAID": "219.04|NOT_FOUND",
    "TRANSACTION_AMOUNTS_RECEIVED": "NOT_FOUND|906.72",
    "ACCOUNT_BALANCE": "14064.96",
    "PAYER_NAME": "Robin Wood",
}

_RECEIPT_FIELDS = {
    "DOCUMENT_TYPE": "RECEIPT",
    "SUPPLIER_NAME": "Ravensdale Health Store",
    "BUSINESS_ABN": "79 104 332 181",
    "BUSINESS_ADDRESS": "400 Stewart Rd, South Yarra VIC 3141",
    "INVOICE_DATE": "07/07/2024",
    "IS_GST_INCLUDED": True,
    "GST_AMOUNT": "1.24",
    "TOTAL_AMOUNT": "13.60",
    "LINE_ITEM_DESCRIPTIONS": "Dishwashing Liquid|Bandaids 40pk",
    "LINE_ITEM_QUANTITIES": "1|1",
    "LINE_ITEM_PRICES": "4.73|8.87",
    "LINE_ITEM_TOTAL_PRICES": "4.73|8.87",
}

_INVOICE_FIELDS = {
    "DOCUMENT_TYPE": "INVOICE",
    "SUPPLIER_NAME": "Ashby Legal",
    "BUSINESS_ABN": "06 082 698 025",
    "BUSINESS_ADDRESS": "481 Bourke Street Perth WA 6000",
    "INVOICE_DATE": "05/08/2025",
    "IS_GST_INCLUDED": False,
    "GST_AMOUNT": "8.62",
    "TOTAL_AMOUNT": "94.87",
    "LINE_ITEM_DESCRIPTIONS": "Consulting",
    "LINE_ITEM_QUANTITIES": "3",
    "LINE_ITEM_PRICES": "15.00",
    "LINE_ITEM_TOTAL_PRICES": "45.00",
    "PAYER_NAME": "Robert Taylor",
    "PAYER_ADDRESS": "243 Adelaide Street Perth WA 6000",
}


def _raw_set(tmp_path: Path, cases: int = 2) -> Path:
    """Build a raw evaluation set: duplicate CASE keys + {CASE}_{layout}.png images."""
    eval_dir = tmp_path / "synthetic_set"
    eval_dir.mkdir()

    layouts = {
        "BANK_STATEMENT": ("cba_standard", _BANK_FIELDS),
        "INVOICE": ("tax_invoice_mixed", _INVOICE_FIELDS),
        "RECEIPT": ("receipt_fuel", _RECEIPT_FIELDS),
    }

    chunks = []
    for index in range(1, cases + 1):
        case_id = f"CASE{index:03d}"
        for layout, fields in layouts.values():
            block = {
                case_id: {
                    "layout": layout,
                    "degradation_seed": 1000 + index,
                    "fields": fields,
                }
            }
            chunks.append(yaml.safe_dump(block, sort_keys=False, width=10**6))
            (eval_dir / f"{case_id}_{layout}.png").write_bytes(b"\x89PNG\r\n")

    (eval_dir / YAML_NAME).write_text("".join(chunks))
    return eval_dir


# ---------------------------------------------------------------------------
# value formatting
# ---------------------------------------------------------------------------


def test_format_value_monetary_prefixes_dollar():
    assert format_value("TOTAL_AMOUNT", "34.16", _SCHEMA) == "$34.16"


def test_format_value_monetary_pipe_items_leaves_not_found():
    out = format_value("TRANSACTION_AMOUNTS_PAID", "NOT_FOUND|212.93|51.04", _SCHEMA)
    assert out == "NOT_FOUND | $212.93 | $51.04"


def test_format_value_does_not_double_dollar():
    assert format_value("TOTAL_AMOUNT", "$5.00", _SCHEMA) == "$5.00"


def test_format_value_boolean_lowercased():
    assert format_value("IS_GST_INCLUDED", True, _SCHEMA) == "true"


def test_format_value_text_respaces_pipes():
    out = format_value("LINE_ITEM_DESCRIPTIONS", "Car Wash|Coffee Large", _SCHEMA)
    assert out == "Car Wash | Coffee Large"


def test_format_value_empty_becomes_not_found():
    assert format_value("SUPPLIER_NAME", "   ", _SCHEMA) == NOT_FOUND


# ---------------------------------------------------------------------------
# schema projection
# ---------------------------------------------------------------------------


def test_project_bank_emits_exactly_the_schema_fields():
    out = project_fields("CASE001", _BANK_FIELDS, "BANK_STATEMENT", _SCHEMA, Path("s.yml"))
    assert list(out) == _SCHEMA.get_extraction_fields("bank_statement")


def test_project_bank_applies_the_descriptions_alias():
    out = project_fields("CASE001", _BANK_FIELDS, "BANK_STATEMENT", _SCHEMA, Path("s.yml"))
    assert out["LINE_ITEM_DESCRIPTIONS"] == "ATM WITHDRAWAL Alexandria | EFTPOS ASHBY LEGAL"
    assert "TRANSACTION_DESCRIPTIONS" not in out


def test_project_bank_drops_non_schema_and_validation_only_fields():
    out = project_fields("CASE001", _BANK_FIELDS, "BANK_STATEMENT", _SCHEMA, Path("s.yml"))
    for dropped in ("SUPPLIER_NAME", "PAYER_NAME", "ACCOUNT_BALANCE", "TRANSACTION_AMOUNTS_RECEIVED"):
        assert dropped not in out


def test_project_receipt_fills_absent_schema_fields_with_not_found():
    out = project_fields("CASE001", _RECEIPT_FIELDS, "RECEIPT", _SCHEMA, Path("s.yml"))
    assert out["PAYER_NAME"] == NOT_FOUND
    assert out["PAYER_ADDRESS"] == NOT_FOUND
    assert out["TOTAL_AMOUNT"] == "$13.60"


def test_project_invoice_matches_schema_exactly():
    out = project_fields("CASE001", _INVOICE_FIELDS, "INVOICE", _SCHEMA, Path("s.yml"))
    assert list(out) == _SCHEMA.get_extraction_fields("invoice")
    assert out["IS_GST_INCLUDED"] == "false"


def test_project_unknown_doc_type_is_diagnostic(assert_diagnostic_error):
    with pytest.raises(SystemExit) as exc:
        project_fields("CASE001", {"DOCUMENT_TYPE": "PAYSLIP"}, "PAYSLIP", _SCHEMA, Path("s.yml"))
    assert_diagnostic_error(str(exc.value))


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------


def test_plan_produces_one_unique_key_per_document(tmp_path):
    eval_dir = _raw_set(tmp_path)
    yaml_path = eval_dir / YAML_NAME
    blocks = load_blocks(yaml_path)
    plan = plan_documents(blocks, eval_dir, _SCHEMA, yaml_path)

    assert len(blocks) == 6
    assert len({case_id for case_id, _ in blocks}) == 2  # duplicate keys in the source
    assert len({item["new_key"] for item in plan}) == 6


def test_plan_missing_layout_is_diagnostic(tmp_path, assert_diagnostic_error):
    eval_dir = _raw_set(tmp_path)
    yaml_path = eval_dir / YAML_NAME
    yaml_path.write_text(yaml.safe_dump({"CASE001": {"fields": _BANK_FIELDS}}, width=10**6))

    with pytest.raises(SystemExit) as exc:
        plan_documents(load_blocks(yaml_path), eval_dir, _SCHEMA, yaml_path)
    assert_diagnostic_error(str(exc.value))


def test_plan_key_collision_is_diagnostic(tmp_path, assert_diagnostic_error):
    eval_dir = _raw_set(tmp_path)
    yaml_path = eval_dir / YAML_NAME
    twice = yaml.safe_dump({"CASE001": {"layout": "cba_standard", "fields": _BANK_FIELDS}}, width=10**6)
    other = yaml.safe_dump({"CASE001": {"layout": "anz_modern", "fields": _BANK_FIELDS}}, width=10**6)
    yaml_path.write_text(twice + other)

    with pytest.raises(SystemExit) as exc:
        plan_documents(load_blocks(yaml_path), eval_dir, _SCHEMA, yaml_path)
    assert_diagnostic_error(str(exc.value))


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------


def test_apply_renames_images_and_writes_both_artefacts(tmp_path, monkeypatch, capsys):
    eval_dir = _raw_set(tmp_path)
    monkeypatch.setattr("sys.argv", ["relabel", "--dir", str(eval_dir), "--apply"])
    main()

    assert (eval_dir / "CASE001_bank_statement.png").is_file()
    assert (eval_dir / "CASE001_invoice.png").is_file()
    assert (eval_dir / "CASE001_receipt.png").is_file()
    assert not (eval_dir / "CASE001_cba_standard.png").exists()

    assert (eval_dir / BACKUP_NAME).is_file()
    assert (eval_dir / MAPPING_NAME).is_file()
    assert "✅ verified" in capsys.readouterr().out


def test_apply_yaml_has_unique_keys_and_no_metadata(tmp_path, monkeypatch):
    eval_dir = _raw_set(tmp_path)
    monkeypatch.setattr("sys.argv", ["relabel", "--dir", str(eval_dir), "--apply"])
    main()

    reloaded = yaml.safe_load((eval_dir / YAML_NAME).read_text())
    assert len(reloaded) == 6
    for key, block in reloaded.items():
        assert set(block) == {"fields"}, f"{key} kept generator metadata"
    assert list(reloaded["CASE001_bank_statement"]["fields"]) == _SCHEMA.get_extraction_fields(
        "bank_statement"
    )


def test_apply_jsonl_records_carry_filename_plus_schema_fields(tmp_path, monkeypatch):
    eval_dir = _raw_set(tmp_path)
    monkeypatch.setattr("sys.argv", ["relabel", "--dir", str(eval_dir), "--apply"])
    main()

    records = [json.loads(line) for line in (eval_dir / JSONL_NAME).read_text().splitlines()]
    assert len(records) == 6

    by_name = {record["filename"]: record for record in records}
    bank = by_name["CASE001_bank_statement.png"]
    assert sorted(bank) == sorted(["filename", *_SCHEMA.get_extraction_fields("bank_statement")])
    assert bank["TRANSACTION_AMOUNTS_PAID"] == "$219.04 | NOT_FOUND"

    receipt = by_name["CASE001_receipt.png"]
    assert receipt["PAYER_NAME"] == NOT_FOUND
    assert receipt["IS_GST_INCLUDED"] == "true"


def test_apply_jsonl_and_yaml_agree_field_for_field(tmp_path, monkeypatch):
    eval_dir = _raw_set(tmp_path)
    monkeypatch.setattr("sys.argv", ["relabel", "--dir", str(eval_dir), "--apply"])
    main()

    reloaded = yaml.safe_load((eval_dir / YAML_NAME).read_text())
    for line in (eval_dir / JSONL_NAME).read_text().splitlines():
        record = json.loads(line)
        key = Path(record.pop("filename")).stem
        assert reloaded[key]["fields"] == record


def test_apply_preserves_every_source_field_in_the_backup(tmp_path, monkeypatch):
    eval_dir = _raw_set(tmp_path)
    monkeypatch.setattr("sys.argv", ["relabel", "--dir", str(eval_dir), "--apply"])
    main()

    backup = load_blocks(eval_dir / BACKUP_NAME)
    assert len(backup) == 6
    bank = next(block for _, block in backup if block["fields"]["DOCUMENT_TYPE"] == "BANK_STATEMENT")
    assert bank["fields"]["ACCOUNT_BALANCE"] == "14064.96"
    assert bank["layout"] == "cba_standard"


def test_second_apply_refuses_to_clobber_the_backup(tmp_path, monkeypatch, assert_diagnostic_error):
    eval_dir = _raw_set(tmp_path)
    monkeypatch.setattr("sys.argv", ["relabel", "--dir", str(eval_dir), "--apply"])
    main()

    with pytest.raises(SystemExit) as exc:
        main()
    assert_diagnostic_error(str(exc.value))
    # The pristine copy still holds the original duplicate-key document.
    assert len(load_blocks(eval_dir / BACKUP_NAME)) == 6


def test_rebuild_regenerates_from_the_backup_without_renaming(tmp_path, monkeypatch):
    eval_dir = _raw_set(tmp_path)
    monkeypatch.setattr("sys.argv", ["relabel", "--dir", str(eval_dir), "--apply"])
    main()
    first = (eval_dir / JSONL_NAME).read_text()

    (eval_dir / JSONL_NAME).unlink()
    monkeypatch.setattr("sys.argv", ["relabel", "--dir", str(eval_dir), "--rebuild"])
    main()

    assert (eval_dir / JSONL_NAME).read_text() == first
    assert (eval_dir / "CASE001_bank_statement.png").is_file()


def test_rebuild_without_a_backup_is_diagnostic(tmp_path, monkeypatch, assert_diagnostic_error):
    eval_dir = _raw_set(tmp_path)
    monkeypatch.setattr("sys.argv", ["relabel", "--dir", str(eval_dir), "--rebuild"])

    with pytest.raises(SystemExit) as exc:
        main()
    assert_diagnostic_error(str(exc.value))


def test_dry_run_changes_nothing(tmp_path, monkeypatch, capsys):
    eval_dir = _raw_set(tmp_path)
    before = sorted(p.name for p in eval_dir.iterdir())
    monkeypatch.setattr("sys.argv", ["relabel", "--dir", str(eval_dir)])
    main()

    assert sorted(p.name for p in eval_dir.iterdir()) == before
    assert "DRY RUN" in capsys.readouterr().out


def test_missing_image_is_diagnostic(tmp_path, monkeypatch, assert_diagnostic_error):
    eval_dir = _raw_set(tmp_path)
    (eval_dir / "CASE001_cba_standard.png").unlink()
    monkeypatch.setattr("sys.argv", ["relabel", "--dir", str(eval_dir), "--apply"])

    with pytest.raises(SystemExit) as exc:
        main()
    assert_diagnostic_error(str(exc.value))


def test_missing_yaml_is_diagnostic(tmp_path, monkeypatch, assert_diagnostic_error):
    eval_dir = tmp_path / "empty_set"
    eval_dir.mkdir()
    monkeypatch.setattr("sys.argv", ["relabel", "--dir", str(eval_dir), "--apply"])

    with pytest.raises(SystemExit) as exc:
        main()
    assert_diagnostic_error(str(exc.value))
