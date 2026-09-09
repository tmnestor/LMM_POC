"""Tests for the scorer embedded in notebooks/field_f1_standalone.ipynb.

The notebook is deliberately standalone — it imports nothing from this repo — so the
functions are loaded by executing its code cells into a namespace, with the input paths
left unbound. Any cell that touches the filesystem is skipped.
"""

import json
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks/field_f1_standalone.ipynb"


@pytest.fixture(scope="module")
def nb():
    """Execute the notebook's definition cells and return their namespace."""
    cells = [c for c in json.loads(NOTEBOOK.read_text())["cells"] if c["cell_type"] == "code"]
    namespace: dict = {}
    for cell in cells:
        source = "".join(cell["source"])
        # Stop before the cell that reads the input files.
        if "read_raw_extractions(RAW_PATH)" in source:
            definitions = source.split("raw_records = read_raw_extractions")[0]
            exec(compile(definitions, "<nb>", "exec"), namespace)
            break
        exec(compile(source, "<nb>", "exec"), namespace)
    return namespace


def assert_diagnostic_error(message: str) -> None:
    """Assert a fail-fast message carries what, where, expected and remedy."""
    assert "Where to fix:" in message
    assert "Expected:" in message
    assert "To recover:" in message
    assert message.splitlines()[0].strip()


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def test_parses_key_value_lines(nb):
    raw = "DOCUMENT_TYPE: INVOICE\nTOTAL_AMOUNT: $10.00"
    assert nb["parse_raw_response"](raw, ["DOCUMENT_TYPE", "TOTAL_AMOUNT"]) == {
        "DOCUMENT_TYPE": "INVOICE",
        "TOTAL_AMOUNT": "$10.00",
    }


def test_parses_json_response(nb):
    raw = '{"DOCUMENT_TYPE": "RECEIPT", "TOTAL_AMOUNT": "$5.00"}'
    parsed = nb["parse_raw_response"](raw, ["DOCUMENT_TYPE", "TOTAL_AMOUNT"])
    assert parsed["DOCUMENT_TYPE"] == "RECEIPT"


def test_strips_markdown_emphasis(nb):
    raw = "*   **SUPPLIER_NAME:** Acme Pty Ltd"
    assert nb["parse_raw_response"](raw, ["SUPPLIER_NAME"])["SUPPLIER_NAME"] == "Acme Pty Ltd"


def test_unemitted_field_defaults_to_not_found(nb):
    parsed = nb["parse_raw_response"]("DOCUMENT_TYPE: INVOICE", ["DOCUMENT_TYPE", "GST_AMOUNT"])
    assert parsed["GST_AMOUNT"] == "NOT_FOUND"


def test_emitted_keys_sees_unasked_fields(nb):
    raw = "DOCUMENT_TYPE: BANK_STATEMENT\nACCOUNT_BALANCE: $1.00 | $2.00"
    assert "ACCOUNT_BALANCE" in nb["emitted_keys"](raw)


# ---------------------------------------------------------------------------
# field selection
# ---------------------------------------------------------------------------


def test_legacy_field_never_scored(nb):
    gt_row = {"DOCUMENT_TYPE": "BANK_STATEMENT", "ACCOUNT_BALANCE": "$1.00"}
    assert "ACCOUNT_BALANCE" not in nb["scored_fields_for"]("BANK_STATEMENT", gt_row)


def test_scored_set_is_the_asked_set(nb):
    gt_row = dict.fromkeys(nb["FIELDS_ASKED_BY_DOC_TYPE"]["BANK_STATEMENT"], "x")
    assert (
        nb["scored_fields_for"]("BANK_STATEMENT", gt_row)
        == (nb["FIELDS_ASKED_BY_DOC_TYPE"]["BANK_STATEMENT"])
    )


def test_receipt_and_invoice_ask_for_the_same_fields(nb):
    asked = nb["FIELDS_ASKED_BY_DOC_TYPE"]
    assert asked["RECEIPT"] == asked["INVOICE"]


def test_gt_field_never_asked_is_not_scored(nb):
    gt_row = dict.fromkeys(nb["FIELDS_ASKED_BY_DOC_TYPE"]["BANK_STATEMENT"], "x")
    gt_row["SUPPLIER_NAME"] = "Big Bank"
    assert "SUPPLIER_NAME" not in nb["scored_fields_for"]("BANK_STATEMENT", gt_row)


def test_undeclared_doc_type_fails_with_diagnostic(nb):
    with pytest.raises(ValueError) as excinfo:
        nb["scored_fields_for"]("LOGBOOK", {"DOCUMENT_TYPE": "LOGBOOK"})
    assert_diagnostic_error(str(excinfo.value))


# ---------------------------------------------------------------------------
# NOT_FOUND as an expected answer
# ---------------------------------------------------------------------------


def test_correct_not_found_scores_one_with_no_counts(nb):
    score = nb["score_field"]("PAYER_NAME", "NOT_FOUND", "NOT_FOUND")
    assert score == {"f1_score": 1.0, "tp": 0, "fp": 0, "fn": 0}


def test_value_where_gt_is_not_found_is_a_false_positive(nb):
    score = nb["score_field"]("PAYER_NAME", "David Jones", "NOT_FOUND")
    assert score["f1_score"] == 0.0
    assert (score["fp"], score["fn"]) == (1, 0)


def test_case016_swap_produces_fn_on_supplier_and_fp_on_payer(nb):
    """The supplier's name landing in PAYER_NAME must cost on both fields."""
    supplier = nb["score_field"]("SUPPLIER_NAME", "NOT_FOUND", "David Jones")
    payer = nb["score_field"]("PAYER_NAME", "David Jones", "NOT_FOUND")
    assert (supplier["fn"], supplier["fp"]) == (1, 0)
    assert (payer["fp"], payer["fn"]) == (1, 0)


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------


def test_document_type_is_scored_not_assumed(nb):
    assert nb["score_field"]("DOCUMENT_TYPE", "INVOICE", "RECEIPT")["f1_score"] == 0.0
    assert nb["score_field"]("DOCUMENT_TYPE", "RECEIPT", "RECEIPT")["f1_score"] == 1.0


def test_monetary_tolerance_and_currency_symbols(nb):
    assert nb["values_match"]("monetary", "$1,234.56", "1234.56")
    assert not nb["values_match"]("monetary", "$1,234.56", "1244.56")


def test_boolean_polarity_mismatch_fails(nb):
    assert nb["values_match"]("boolean", "Yes", "true")
    assert not nb["values_match"]("boolean", "Yes", "false")


def test_abn_ignores_spacing(nb):
    assert nb["values_match"]("id", "12 345 678 901", "12345678901")


# ---------------------------------------------------------------------------
# list scoring, order normalisation and GT row filtering
# ---------------------------------------------------------------------------


def test_list_f1_is_position_aware(nb):
    aligned = nb["score_field"]("TRANSACTION_DATES", "01/01/2023 | 02/01/2023", "01/01/2023 | 02/01/2023")
    swapped = nb["score_field"]("TRANSACTION_DATES", "02/01/2023 | 01/01/2023", "01/01/2023 | 02/01/2023")
    assert aligned["f1_score"] == 1.0
    assert swapped["tp"] == 0


def test_surplus_and_missing_items_are_counted(nb):
    score = nb["score_field"]("LINE_ITEM_DESCRIPTIONS", "a | b | c", "a | b")
    assert (score["tp"], score["fp"], score["fn"]) == (2, 1, 0)


def test_chronological_sort_reorders_parallel_fields(nb):
    data = {
        "TRANSACTION_DATES": "02/01/2023 | 01/01/2023",
        "TRANSACTION_AMOUNTS_PAID": "$2.00 | $1.00",
        "LINE_ITEM_DESCRIPTIONS": "second | first",
    }
    result = nb["normalise_order"](data, "BANK_STATEMENT")
    assert result["TRANSACTION_DATES"] == "01/01/2023 | 02/01/2023"
    assert result["TRANSACTION_AMOUNTS_PAID"] == "$1.00 | $2.00"
    assert result["LINE_ITEM_DESCRIPTIONS"] == "first | second"


def test_sort_skips_fields_whose_length_disagrees(nb):
    data = {
        "TRANSACTION_DATES": "02/01/2023 | 01/01/2023",
        "LINE_ITEM_DESCRIPTIONS": "only one",
    }
    assert nb["normalise_order"](data, "BANK_STATEMENT")["LINE_ITEM_DESCRIPTIONS"] == "only one"


def test_sort_does_not_apply_to_invoices(nb):
    data = {"TRANSACTION_DATES": "02/01/2023 | 01/01/2023"}
    assert nb["normalise_order"](data, "INVOICE") == data


def test_credit_rows_are_dropped_from_bank_ground_truth(nb):
    truth = {
        "TRANSACTION_AMOUNTS_PAID": "NOT_FOUND | $10.00 | NOT_FOUND",
        "TRANSACTION_DATES": "01/01/2023 | 02/01/2023 | 03/01/2023",
        "LINE_ITEM_DESCRIPTIONS": "salary | groceries | refund",
    }
    filtered = nb["filter_gt_rows"](truth, "BANK_STATEMENT")
    assert filtered["TRANSACTION_AMOUNTS_PAID"] == "$10.00"
    assert filtered["TRANSACTION_DATES"] == "02/01/2023"
    assert filtered["LINE_ITEM_DESCRIPTIONS"] == "groceries"


def test_row_filter_makes_a_correct_debit_extraction_score_one(nb):
    """Without the filter the credit row shifts every position and F1 collapses."""
    truth = {
        "TRANSACTION_AMOUNTS_PAID": "NOT_FOUND | $10.00 | $20.00",
        "TRANSACTION_DATES": "01/01/2023 | 02/01/2023 | 03/01/2023",
    }
    filtered = nb["filter_gt_rows"](truth, "BANK_STATEMENT")
    predicted = "$10.00 | $20.00"
    assert (
        nb["score_field"]("TRANSACTION_AMOUNTS_PAID", predicted, truth["TRANSACTION_AMOUNTS_PAID"])[
            "f1_score"
        ]
        < 1.0
    )
    assert (
        nb["score_field"]("TRANSACTION_AMOUNTS_PAID", predicted, filtered["TRANSACTION_AMOUNTS_PAID"])[
            "f1_score"
        ]
        == 1.0
    )


def test_row_filter_does_not_apply_to_invoices(nb):
    truth = {"TRANSACTION_AMOUNTS_PAID": "NOT_FOUND | $10.00"}
    assert nb["filter_gt_rows"](truth, "INVOICE") == truth
