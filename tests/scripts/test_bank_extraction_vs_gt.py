"""Tests for the bank-extraction-vs-ground-truth diff harness.

tests/ is gitignored — local-only. The harness quantifies dense bank-table
extraction quality by comparing PROD cleaned_extractions amounts against the
synthetic ground-truth TRANSACTION_AMOUNTS_PAID, classifying each GT amount as
matched / missing (dropped) / spurious (misread). See
memory thousands-comma-cleaner-bug (CORRECTION) for why this exists.
"""

from scripts.bank_extraction_vs_gt import (
    StmtResult,
    diff_amounts,
    extract_case_id,
    parse_amount_list,
    summarize_by_layout,
)


class TestParseAmountList:
    def test_skips_not_found_and_parses(self) -> None:
        assert parse_amount_list("306.68|NOT_FOUND|351.56") == [306.68, 351.56]

    def test_strips_dollar_and_thousands_comma(self) -> None:
        assert parse_amount_list("$1,234.56 | $8,026.87") == [1234.56, 8026.87]

    def test_empty_and_not_found_yield_empty(self) -> None:
        assert parse_amount_list("") == []
        assert parse_amount_list("NOT_FOUND") == []


class TestExtractCaseId:
    def test_extracts_case_prefix(self) -> None:
        assert extract_case_id("CASE002_cba_date_grouped.png") == "CASE002"

    def test_none_when_no_case(self) -> None:
        assert extract_case_id("random_image.png") is None


class TestDiffAmounts:
    def test_all_matched(self) -> None:
        matched, missing, spurious = diff_amounts([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert matched == [1.0, 2.0, 3.0]
        assert missing == []
        assert spurious == []

    def test_dropped_row(self) -> None:
        matched, missing, spurious = diff_amounts([1.0, 2.0, 3.0], [1.0, 3.0])
        assert missing == [2.0]
        assert spurious == []

    def test_misread_row(self) -> None:
        # 8026.87 dropped, 3396.88 appears instead — the CASE002 signature.
        matched, missing, spurious = diff_amounts([558.63, 8026.87, 423.75], [558.63, 3396.88, 423.75])
        assert missing == [8026.87]
        assert spurious == [3396.88]
        assert sorted(matched) == [423.75, 558.63]

    def test_duplicate_amounts_multiset(self) -> None:
        matched, missing, spurious = diff_amounts([245.09, 245.81, 245.09], [245.09, 245.81])
        assert missing == [245.09]
        assert spurious == []

    def test_tolerance(self) -> None:
        matched, missing, spurious = diff_amounts([100.00], [100.004])
        assert missing == []
        assert matched == [100.00]


class TestSummarizeByLayout:
    def test_sums_miss_and_spur_per_layout(self) -> None:
        results = [
            StmtResult("CASE1", "nab_classic", gt=10, match=6, miss=4, spur=2),
            StmtResult("CASE2", "nab_classic", gt=5, match=5, miss=0, spur=1),
            StmtResult("CASE3", "cba_standard", gt=8, match=4, miss=4, spur=3),
        ]
        summary = summarize_by_layout(results)
        assert summary["nab_classic"].gt == 15
        assert summary["nab_classic"].match == 11
        assert summary["nab_classic"].miss == 4
        assert summary["nab_classic"].spur == 3
        assert summary["cba_standard"].miss == 4
        assert summary["cba_standard"].spur == 3
