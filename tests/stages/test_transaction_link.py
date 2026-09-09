"""Tests for stages.transaction_link orchestration + fail-fast config."""

import json

import pytest
import yaml

from stages import transaction_link as tl

_VALID_LINKING = {
    "case_key_pattern": "^(?P<case>[^_]+)_",
    "vlm_prompt": "single_receipt_link",
    "vlm_max_tokens": 4096,
    "vlm_temperature": 0.0,
    "hybrid_amount_tolerance": 0.01,
    "hybrid_date_window_days": 5,
    "hybrid_description_threshold": 0.3,
    "hybrid_min_confidence": "LOW",
    "data_dir": "/tmp/x",
    "output": "/tmp/x/transaction_links.jsonl",
    "ground_truth": "/tmp/x/gt.yml",
    "evaluation_dir": "/tmp/x/eval",
    "log_dir": "/tmp/x/logs",
}


def _write_config(tmp_path, linking=None, omit=None):
    cfg = {"bootstrap": {"model": {"type": "internvl3-vllm"}}, "pipeline": {}}
    if linking is not False:
        block = dict(linking) if isinstance(linking, dict) else dict(_VALID_LINKING)
        if omit:
            block.pop(omit)
        cfg["pipeline"]["linking"] = block
    path = tmp_path / "run_config.yml"
    path.write_text(yaml.safe_dump(cfg))
    return path


# ---------------------------------------------------------------------------
# Config fail-fast diagnostics
# ---------------------------------------------------------------------------


def test_config_missing_file_is_diagnostic(tmp_path, assert_diagnostic_error):
    with pytest.raises(FileNotFoundError) as exc:
        tl._load_linking_config(tmp_path / "does_not_exist.yml")
    assert_diagnostic_error(str(exc.value))


def test_config_missing_linking_section_is_diagnostic(tmp_path, assert_diagnostic_error):
    path = _write_config(tmp_path, linking=False)
    with pytest.raises(ValueError) as exc:
        tl._load_linking_config(path)
    assert_diagnostic_error(str(exc.value))


@pytest.mark.parametrize(
    "key",
    [
        "case_key_pattern",
        "vlm_prompt",
        "vlm_max_tokens",
        "vlm_temperature",
        "hybrid_amount_tolerance",
        "hybrid_date_window_days",
        "hybrid_description_threshold",
        "hybrid_min_confidence",
    ],
)
def test_config_missing_key_is_diagnostic(tmp_path, assert_diagnostic_error, key):
    path = _write_config(tmp_path, omit=key)
    with pytest.raises(ValueError) as exc:
        tl._load_linking_config(path)
    assert_diagnostic_error(str(exc.value))


def test_config_bad_regex_is_diagnostic(tmp_path, assert_diagnostic_error):
    block = dict(_VALID_LINKING, case_key_pattern="(?P<case>[")
    path = _write_config(tmp_path, linking=block)
    with pytest.raises(ValueError) as exc:
        tl._load_linking_config(path)
    assert_diagnostic_error(str(exc.value))


def test_config_regex_without_case_group_is_diagnostic(tmp_path, assert_diagnostic_error):
    block = dict(_VALID_LINKING, case_key_pattern="^[^_]+_")
    path = _write_config(tmp_path, linking=block)
    with pytest.raises(ValueError) as exc:
        tl._load_linking_config(path)
    assert_diagnostic_error(str(exc.value))


def test_config_nonzero_temperature_is_diagnostic(tmp_path, assert_diagnostic_error):
    block = dict(_VALID_LINKING, vlm_temperature=0.7)
    path = _write_config(tmp_path, linking=block)
    with pytest.raises(ValueError) as exc:
        tl._load_linking_config(path)
    assert_diagnostic_error(str(exc.value))


def test_config_bad_min_confidence_is_diagnostic(tmp_path, assert_diagnostic_error):
    block = dict(_VALID_LINKING, hybrid_min_confidence="MAYBE")
    path = _write_config(tmp_path, linking=block)
    with pytest.raises(ValueError) as exc:
        tl._load_linking_config(path)
    assert_diagnostic_error(str(exc.value))


def test_config_valid_loads(tmp_path):
    path = _write_config(tmp_path)
    cfg = tl._load_linking_config(path)
    assert cfg["vlm_prompt"] == "single_receipt_link"


# ---------------------------------------------------------------------------
# Helpers: amount parsing, echo detection, confidence levels
# ---------------------------------------------------------------------------


def test_parse_amount_str():
    assert tl._parse_amount_str("$1,234.56") == 1234.56
    assert tl._parse_amount_str("-50.00") == 50.00
    assert tl._parse_amount_str("NOT_FOUND") is None


def test_is_bank_row_echo():
    assert tl._is_bank_row_echo(
        {"RECEIPT_STORE": "WOOLWORTHS 2847", "TRANSACTION_DESCRIPTION": "WOOLWORTHS 2847"}
    )
    assert not tl._is_bank_row_echo({"RECEIPT_STORE": "Woolworths", "TRANSACTION_DESCRIPTION": "WW 2847"})


def test_confidence_level():
    assert tl._confidence_level("HIGH") > tl._confidence_level("LOW")
    assert tl._confidence_level("NONE") == 0


# ---------------------------------------------------------------------------
# run(): matcher-only path (no model load)
# ---------------------------------------------------------------------------


def _write_jsonl(path, records):
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_run_matcher_only(tmp_path):
    config_path = _write_config(tmp_path)
    extractions = tmp_path / "cleaned.jsonl"
    _write_jsonl(
        extractions,
        [
            {
                "image_name": "CASE001_bank.png",
                "document_type": "BANK_STATEMENT",
                "extracted_data": {
                    "TRANSACTION_DATES": "18/03/2024",
                    "LINE_ITEM_DESCRIPTIONS": "WOOLWORTHS 2847",
                    "TRANSACTION_AMOUNTS_PAID": "127.35",
                },
            },
            {
                "image_name": "CASE001_receipt.png",
                "document_type": "RECEIPT",
                "extracted_data": {
                    "TOTAL_AMOUNT": "$127.35",
                    "SUPPLIER_NAME": "Woolworths",
                    "INVOICE_DATE": "18/03/2024",
                },
            },
        ],
    )
    out = tmp_path / "links.jsonl"
    tl.run(extractions, out, data_dir=tmp_path, config_path=config_path)

    results = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(results) == 1
    assert results[0]["matched"] is True
    assert results[0]["bank_transaction_amount"] == 127.35


def test_run_no_bank_statements(tmp_path):
    config_path = _write_config(tmp_path)
    extractions = tmp_path / "cleaned.jsonl"
    _write_jsonl(
        extractions,
        [
            {
                "image_name": "CASE001_receipt.png",
                "document_type": "RECEIPT",
                "extracted_data": {"TOTAL_AMOUNT": "$99.00", "SUPPLIER_NAME": "X", "INVOICE_DATE": ""},
            }
        ],
    )
    out = tmp_path / "links.jsonl"
    tl.run(extractions, out, data_dir=tmp_path, config_path=config_path)
    results = [json.loads(line) for line in out.read_text().splitlines()]
    assert results[0]["matched"] is False
    assert "No bank statements" in results[0]["reasoning"]


def test_run_empty_input_is_diagnostic(tmp_path, assert_diagnostic_error):
    config_path = _write_config(tmp_path)
    extractions = tmp_path / "empty.jsonl"
    extractions.write_text("")
    with pytest.raises(FileNotFoundError) as exc:
        tl.run(extractions, tmp_path / "o.jsonl", data_dir=tmp_path, config_path=config_path)
    assert_diagnostic_error(str(exc.value))


# ---------------------------------------------------------------------------
# VLM fallback amount gate
# ---------------------------------------------------------------------------


def _receipt(total):
    from common.transaction_matcher import ReceiptSummary

    return ReceiptSummary(
        image_name="CASE001_receipt.png",
        supplier_name="STORE FULL NAME",
        date=None,
        total=total,
        document_type="RECEIPT",
    )


def _found_match(amount="$2216.00"):
    return {
        "MATCHED_TRANSACTION": "FOUND",
        "TRANSACTION_AMOUNT": amount,
        "TRANSACTION_DATE": "15/01/2024",
        "TRANSACTION_DESCRIPTION": "SOME DEBIT",
        "RECEIPT_STORE": "STORE FULL NAME",
        "CONFIDENCE": "HIGH",
        "REASONING": "looks right",
    }


def test_passes_amount_gate():
    # Comparable mismatch beyond tolerance -> reject (the CASE037-shaped FP).
    assert tl._passes_amount_gate(2196.58, _found_match("$2216.00"), 0.01) is False
    # Exact (modulo formatting) -> accept.
    assert tl._passes_amount_gate(2196.58, _found_match("$2,196.58"), 0.01) is True
    # Within a loosened tolerance -> accept.
    assert tl._passes_amount_gate(100.0, _found_match("$100.50"), 1.00) is True
    # No receipt total -> nothing to verify against -> pass through.
    assert tl._passes_amount_gate(None, _found_match(), 0.01) is True
    # Known total but unverifiable VLM amount -> fail closed.
    assert tl._passes_amount_gate(100.0, _found_match("NOT_FOUND"), 0.01) is False
    # Negative receipt total compares on magnitude.
    assert tl._passes_amount_gate(-83.48, _found_match("$83.48"), 0.01) is True


def _attempt(tmp_path, monkeypatch, *, total, vlm_amount, tolerance=0.01):
    (tmp_path / "bank.png").touch()
    receipt = _receipt(total)
    record = tl._base_record(receipt, "CASE001")
    monkeypatch.setattr(tl, "call_vlm_linker", lambda *a, **k: [_found_match(vlm_amount)])
    found = tl._attempt_on_image(
        record,
        receipt,
        "bank.png",
        None,
        generate_fn=None,
        data_dir=tmp_path,
        prompt=None,
        max_tokens=16,
        amount_tolerance=tolerance,
    )
    return found, record


def test_attempt_on_image_gate_rejects_mismatched_found(tmp_path, monkeypatch):
    found, record = _attempt(tmp_path, monkeypatch, total=2196.58, vlm_amount="$2216.00")
    assert found is False
    assert record["matched"] is False
    assert record.get("bank_statement_file") in (None, "")


def test_attempt_on_image_gate_accepts_matching_amount(tmp_path, monkeypatch):
    found, record = _attempt(tmp_path, monkeypatch, total=2196.58, vlm_amount="$2,196.58")
    assert found is True
    assert record["matched"] is True
    assert record["bank_transaction_amount"] == 2196.58


def test_attempt_on_image_gate_passes_unknown_total(tmp_path, monkeypatch):
    found, record = _attempt(tmp_path, monkeypatch, total=None, vlm_amount="$50.00")
    assert found is True
    assert record["matched"] is True
