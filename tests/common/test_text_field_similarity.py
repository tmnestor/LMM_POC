"""Text fields must score by edit-distance similarity, not exact match.

tests/ is gitignored -- local-only.

Locks in the 2026-08-12 finding. `calculate_field_accuracy_f1` scores text fields
(addresses, names) ANLS-style: 1 - edit_distance/max_len, with a 0.5 floor. But
the implementation imported `Levenshtein` inside a try/except and fell back to
EXACT MATCH on ImportError -- and `Levenshtein` is declared in no environment
file, so the fallback was always the live path and the ANLS branch was dead code.

Cost: `BUSINESS_ADDRESS` scored 0.000 for BOTH models at EVERY degradation tier
because ground truth writes `384 Bailey Cres, Paddington QLD 4064` and the models
return the same string without the comma. One character, whole field lost. With
ANLS it scores 1 - 1/36 = 0.972.

The fix computes the distance in pure Python when the fast package is absent, so
the score never depends on whether an optional dependency happens to be present.
"""

import pytest

from common.evaluation_metrics import _edit_distance, calculate_field_accuracy_f1

ADDRESS = "BUSINESS_ADDRESS"


class TestEditDistance:
    """The pure-Python distance must be a real Levenshtein distance."""

    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            ("", "", 0),
            ("abc", "abc", 0),
            ("", "abc", 3),
            ("abc", "", 3),
            ("kitten", "sitting", 3),  # the textbook case
            ("flaw", "lawn", 2),
            ("a,b", "ab", 1),  # one deletion -- the address case in miniature
        ],
    )
    def test_known_distances(self, a: str, b: str, expected: int) -> None:
        assert _edit_distance(a, b) == expected

    def test_symmetric(self) -> None:
        assert _edit_distance("paddington", "padington") == _edit_distance("padington", "paddington")


class TestAddressScoring:
    """The regression that motivated this: a comma must not cost the field."""

    GT = "384 Bailey Cres, Paddington QLD 4064"
    EXTRACTED = "384 Bailey Cres Paddington QLD 4064"

    def test_missing_comma_scores_near_one(self) -> None:
        score = calculate_field_accuracy_f1(self.EXTRACTED, self.GT, ADDRESS)["f1_score"]
        assert score > 0.95, f"one comma should barely dent the score, got {score}"

    def test_missing_comma_is_not_zero(self) -> None:
        # The exact-match fallback returned 0.0 here for both models at every tier.
        assert calculate_field_accuracy_f1(self.EXTRACTED, self.GT, ADDRESS)["f1_score"] > 0.0

    def test_counts_as_a_true_positive(self) -> None:
        result = calculate_field_accuracy_f1(self.EXTRACTED, self.GT, ADDRESS)
        assert result["tp"] == 1
        assert result["fp"] == 0

    def test_identical_scores_one(self) -> None:
        assert calculate_field_accuracy_f1(self.GT, self.GT, ADDRESS)["f1_score"] == 1.0

    def test_case_insensitive(self) -> None:
        assert calculate_field_accuracy_f1(self.GT.upper(), self.GT.lower(), ADDRESS)["f1_score"] == 1.0


class TestStillDiscriminates:
    """Partial credit must not become credit for anything."""

    def test_unrelated_address_scores_zero(self) -> None:
        score = calculate_field_accuracy_f1(
            "1 Nowhere Road Perth WA 6000", "384 Bailey Cres, Paddington QLD 4064", ADDRESS
        )["f1_score"]
        assert score == 0.0, "below the 0.5 ANLS floor must collapse to 0"

    def test_different_supplier_scores_zero(self) -> None:
        score = calculate_field_accuracy_f1("Woolworths", "Capital Legal Group", "SUPPLIER_NAME")[
            "f1_score"
        ]
        assert score == 0.0

    def test_wrong_street_number_still_penalised(self) -> None:
        # Same shape, one digit out -- should score high but strictly below 1.0.
        score = calculate_field_accuracy_f1(
            "385 Bailey Cres, Paddington QLD 4064",
            "384 Bailey Cres, Paddington QLD 4064",
            ADDRESS,
        )["f1_score"]
        assert 0.9 < score < 1.0


class TestOtherFieldTypesUnaffected:
    """The change touches the text branch only."""

    def test_abn_still_requires_exact_match(self) -> None:
        # ID fields bypass fuzzy matching entirely -- a digit out must be 0.
        assert calculate_field_accuracy_f1("57773872148", "57773872149", "BUSINESS_ABN")["f1_score"] == 0.0

    def test_abn_ignores_spacing(self) -> None:
        assert (
            calculate_field_accuracy_f1("57 773 872 148", "57773872148", "BUSINESS_ABN")["f1_score"] == 1.0
        )

    def test_total_amount_still_monetary(self) -> None:
        assert calculate_field_accuracy_f1("$216.89", "216.89", "TOTAL_AMOUNT")["f1_score"] == 1.0
