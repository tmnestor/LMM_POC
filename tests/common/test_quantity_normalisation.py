"""Item quantities are a count, not a count wearing a multiplier.

tests/ is gitignored -- local-only.

Found 2026-08-12. Gemma writes "3x | 1x | 2x" where ground truth writes
"3 | 1 | 2". Every value is correct, but the scorer compares quantities as text,
so the whole field scored 0.000 on 29 of 55 receipts. Measured effect of
stripping the decoration, on runs already collected:

    tier      Gemma stored -> normalised    InternVL
    clean            0.333    0.764         unchanged
    light            0.304    0.709         unchanged
    moderate         0.435    0.688         unchanged
    heavy            0.389    0.645         +0.003

The receipt prompt now also asks for digits only. That stops it being emitted;
this stops it mattering. Neither depends on the other holding.
"""

import pytest

from common.extraction_parser import hybrid_parse_response, normalise_quantity_items

FIELD = "LINE_ITEM_QUANTITIES"


class TestDecorationIsStripped:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("3x | 1x | 2x", "3 | 1 | 2"),  # the measured Gemma failure
            ("3x", "3"),
            ("3X", "3"),
            ("3 x", "3"),
            ("x3 | x1", "3 | 1"),  # leading multiplier
            ("2 EACH | 1 EACH", "2 | 1"),  # pre-existing behaviour, preserved
            ("2 each", "2"),
            ("3x | 1 EACH | 2", "3 | 1 | 2"),  # mixed forms in one list
        ],
    )
    def test_strips(self, raw: str, expected: str) -> None:
        assert normalise_quantity_items(raw) == expected

    def test_bare_digits_are_untouched(self) -> None:
        """InternVL already writes these; the fix must be a no-op for it."""
        assert normalise_quantity_items("3 | 1 | 2") == "3 | 1 | 2"

    def test_spacing_is_normalised_to_the_pipe_convention(self) -> None:
        assert normalise_quantity_items("3|1|2") == "3 | 1 | 2"


class TestItDoesNotMangleOtherValues:
    """Both patterns are anchored on a digit, deliberately."""

    def test_not_found_passes_through(self) -> None:
        assert normalise_quantity_items("NOT_FOUND") == "NOT_FOUND"
        assert normalise_quantity_items("not_found") == "not_found"

    def test_empty_passes_through(self) -> None:
        assert normalise_quantity_items("") == ""

    @pytest.mark.parametrize("value", ["BOX", "6 PACK", "EACH", "1.5", "12"])
    def test_leaves_these_alone(self, value: str) -> None:
        # "BOX" keeps its X because no digit precedes it; a bare "EACH" with no
        # count is not silently turned into an empty string.
        assert normalise_quantity_items(value) == value

    def test_decimal_quantity_survives(self) -> None:
        assert normalise_quantity_items("1.5x | 2.25") == "1.5 | 2.25"


class TestBothParsePathsNormalise:
    """The JSON fast path returns before parse_extraction_response runs.

    hybrid_parse_response tries JSON first and returns immediately on success,
    skipping every per-field normalisation below it. Without an explicit call
    there, a JSON response would keep its "3x" while an otherwise identical
    text response would not -- the parser would disagree with itself.
    """

    FIELDS = ["DOCUMENT_TYPE", FIELD, "LINE_ITEM_DESCRIPTIONS"]

    def test_json_response_is_normalised(self) -> None:
        response = '{"DOCUMENT_TYPE": "RECEIPT", "LINE_ITEM_QUANTITIES": "3x | 1x | 2x"}'
        assert hybrid_parse_response(response, self.FIELDS)[FIELD] == "3 | 1 | 2"

    def test_text_response_is_normalised(self) -> None:
        response = "DOCUMENT_TYPE: RECEIPT\nLINE_ITEM_QUANTITIES: 3x | 1x | 2x"
        assert hybrid_parse_response(response, self.FIELDS)[FIELD] == "3 | 1 | 2"

    def test_the_two_paths_agree(self) -> None:
        json_out = hybrid_parse_response(
            '{"DOCUMENT_TYPE": "RECEIPT", "LINE_ITEM_QUANTITIES": "2 EACH | 3x"}', self.FIELDS
        )
        text_out = hybrid_parse_response(
            "DOCUMENT_TYPE: RECEIPT\nLINE_ITEM_QUANTITIES: 2 EACH | 3x", self.FIELDS
        )
        assert json_out[FIELD] == text_out[FIELD] == "2 | 3"
