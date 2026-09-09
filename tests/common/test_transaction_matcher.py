"""Unit tests for common.transaction_matcher (pure algorithmic matcher)."""

import re
from datetime import date

from common.transaction_matcher import (
    BankTransaction,
    ReceiptSummary,
    build_receipt_summaries,
    build_transaction_index,
    description_score,
    extract_case_id,
    group_by_case,
    match_all_receipts,
    match_receipt,
    normalize_date,
    parse_amount,
)

CASE_PATTERN = re.compile(r"^(?P<case>[^_]+)_")


# ---------------------------------------------------------------------------
# parse_amount
# ---------------------------------------------------------------------------


def test_parse_amount_currency_and_commas():
    assert parse_amount("$127.35") == 127.35
    assert parse_amount("1,234.56") == 1234.56
    assert parse_amount("£99.00") == 99.00


def test_parse_amount_negatives():
    assert parse_amount("-$127.35") == -127.35
    assert parse_amount("(50.00)") == -50.00


def test_parse_amount_not_found_and_garbage():
    assert parse_amount("NOT_FOUND") is None
    assert parse_amount("") is None
    assert parse_amount("abc") is None


# ---------------------------------------------------------------------------
# normalize_date
# ---------------------------------------------------------------------------


def test_normalize_date_formats():
    assert normalize_date("2024-03-18") == date(2024, 3, 18)
    assert normalize_date("18/03/2024") == date(2024, 3, 18)
    assert normalize_date("18/03/24") == date(2024, 3, 18)
    assert normalize_date("18 Mar 2024") == date(2024, 3, 18)
    assert normalize_date("22 December 23") == date(2023, 12, 22)


def test_normalize_date_invalid():
    assert normalize_date("NOT_FOUND") is None
    assert normalize_date("") is None
    assert normalize_date("garbage") is None
    assert normalize_date("99/99/2024") is None


# ---------------------------------------------------------------------------
# description_score
# ---------------------------------------------------------------------------


def test_description_score_full_and_partial():
    assert description_score("WOOLWORTHS", "WOOLWORTHS 2847 SYDNEY") == 1.0
    assert description_score("OFFICE SUPPLIES", "OFFICE PLUS") == 0.5


def test_description_score_ampersand_and_empty():
    assert description_score("BLACK & WHITE", "BLACK AND WHITE CO") == 1.0
    assert description_score("NOT_FOUND", "ANYTHING") == 0.0
    assert description_score("STORE", "") == 0.0


# ---------------------------------------------------------------------------
# case grouping
# ---------------------------------------------------------------------------


def test_extract_case_id():
    assert extract_case_id("CASE001_receipt.png", CASE_PATTERN) == "CASE001"
    assert extract_case_id("nounderscorehere.png", CASE_PATTERN) is None


def test_group_by_case_groups_and_ungrouped():
    records = [
        {"image_name": "CASE001_a.png"},
        {"image_name": "CASE001_b.png"},
        {"image_name": "CASE002_c.png"},
        {"image_name": "noprefix.png"},
    ]
    groups = group_by_case(records, CASE_PATTERN)
    assert len(groups["CASE001"]) == 2
    assert len(groups["CASE002"]) == 1
    assert len(groups["_ungrouped"]) == 1


# ---------------------------------------------------------------------------
# build_transaction_index
# ---------------------------------------------------------------------------


def test_build_transaction_index_pipe_delimited():
    bank = [
        {
            "image_name": "CASE001_bank.png",
            "extracted_data": {
                "TRANSACTION_DATES": "18 Mar 2024 | 19 Mar 2024",
                "LINE_ITEM_DESCRIPTIONS": "WOOLWORTHS 2847 | TELSTRA",
                "TRANSACTION_AMOUNTS_PAID": "127.35 | -48.50",
            },
        }
    ]
    index = build_transaction_index(bank)
    assert len(index) == 2
    assert index[0].amount == 127.35
    assert index[0].date == date(2024, 3, 18)
    # amounts are stored absolute
    assert index[1].amount == 48.50
    assert index[1].row_index == 1


def test_build_transaction_index_skips_not_found_amounts():
    bank = [
        {
            "image_name": "CASE001_bank.png",
            "extracted_data": {
                "TRANSACTION_DATES": "NOT_FOUND",
                "LINE_ITEM_DESCRIPTIONS": "NOT_FOUND",
                "TRANSACTION_AMOUNTS_PAID": "NOT_FOUND",
            },
        }
    ]
    assert build_transaction_index(bank) == []


# ---------------------------------------------------------------------------
# build_receipt_summaries
# ---------------------------------------------------------------------------


def test_build_receipt_summaries_single():
    record = {
        "image_name": "CASE001_receipt.png",
        "document_type": "RECEIPT",
        "extracted_data": {
            "TOTAL_AMOUNT": "$127.35",
            "SUPPLIER_NAME": "Woolworths",
            "INVOICE_DATE": "18/03/2024",
        },
    }
    summaries = build_receipt_summaries(record)
    assert len(summaries) == 1
    assert summaries[0].total == 127.35
    assert summaries[0].supplier_name == "Woolworths"
    assert summaries[0].date == date(2024, 3, 18)


def test_build_receipt_summaries_multi_receipt():
    record = {
        "image_name": "CASE001_multi.png",
        "document_type": "RECEIPT",
        "extracted_data": {
            "TOTAL_AMOUNT": "$83.48 | $39.70",
            "SUPPLIER_NAME": "Office Plus | Cafe Roma",
            "INVOICE_DATE": "15/01/2024 | 16/01/2024",
        },
    }
    summaries = build_receipt_summaries(record)
    assert len(summaries) == 2
    assert summaries[1].total == 39.70
    assert summaries[1].supplier_name == "Cafe Roma"


# ---------------------------------------------------------------------------
# match_receipt confidence tiers
# ---------------------------------------------------------------------------


def _txn(amount, d, desc, idx=0):
    return BankTransaction(
        date=d, description=desc, amount=amount, source_image="CASE001_bank.png", row_index=idx
    )


def test_match_receipt_none_when_no_total():
    receipt = ReceiptSummary("r.png", "Store", date(2024, 3, 18), None, "RECEIPT")
    result = match_receipt(receipt, [_txn(127.35, date(2024, 3, 18), "STORE")])
    assert result.confidence == "NONE"
    assert result.matched is False


def test_match_receipt_none_when_no_amount_candidate():
    receipt = ReceiptSummary("r.png", "Store", date(2024, 3, 18), 999.99, "RECEIPT")
    result = match_receipt(receipt, [_txn(127.35, date(2024, 3, 18), "STORE")])
    assert result.matched is False
    assert result.confidence == "NONE"


def test_match_receipt_low_amount_only():
    receipt = ReceiptSummary("r.png", "Mismatch", None, 127.35, "RECEIPT")
    result = match_receipt(receipt, [_txn(127.35, None, "SOMETHING ELSE")], description_threshold=0.5)
    assert result.matched is True
    assert result.confidence == "LOW"


def test_match_receipt_medium_amount_plus_date():
    receipt = ReceiptSummary("r.png", "Mismatch", date(2024, 3, 18), 127.35, "RECEIPT")
    result = match_receipt(
        receipt, [_txn(127.35, date(2024, 3, 18), "SOMETHING ELSE")], description_threshold=0.5
    )
    assert result.confidence == "MEDIUM"


def test_match_receipt_high_amount_date_desc():
    receipt = ReceiptSummary("r.png", "WOOLWORTHS", date(2024, 3, 18), 127.35, "RECEIPT")
    result = match_receipt(
        receipt,
        [_txn(127.35, date(2024, 3, 18), "WOOLWORTHS 2847")],
        description_threshold=0.5,
    )
    assert result.confidence == "HIGH"


# ---------------------------------------------------------------------------
# match_all_receipts one-to-one constraint
# ---------------------------------------------------------------------------


def test_match_all_receipts_one_to_one():
    # Two receipts with the same amount; two bank rows with that amount.
    # Each row may match only one receipt.
    receipts = [
        ReceiptSummary("r1.png", "WOOLWORTHS", date(2024, 3, 18), 50.00, "RECEIPT"),
        ReceiptSummary("r2.png", "TELSTRA", date(2024, 3, 20), 50.00, "RECEIPT"),
    ]
    index = [
        _txn(50.00, date(2024, 3, 18), "WOOLWORTHS 2847", idx=0),
        _txn(50.00, date(2024, 3, 20), "TELSTRA CORP", idx=1),
    ]
    results = match_all_receipts(receipts, index, description_threshold=0.5)
    matched_rows = {r.transaction.row_index for r in results if r.matched}
    assert matched_rows == {0, 1}  # distinct rows consumed
    assert all(r.matched for r in results)
