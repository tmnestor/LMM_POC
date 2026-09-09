"""Transaction descriptions belong to the register and need text similarity.

tests/ is gitignored -- local-only.

Two coupled defects, found 2026-08-12 while comparing debit rows position by
position:

1. `LINE_ITEM_DESCRIPTIONS` was absent from `field_types.transaction_list` even
   though it is the third index-aligned column of the same bank register as
   TRANSACTION_DATES and TRANSACTION_AMOUNTS_PAID -- and even though
   `_TRANSACTION_SORT_GROUPS` already treats it as part of that register. The
   schema and the sort group disagreed.

2. Adding it alone would have made scoring WORSE, not better:
   `_transaction_item_matches` fell through to EXACT lowercase match for any
   field without AMOUNT/DATE/BALANCE in its name. Measured on the 55-statement
   corpus, descriptions scored exact 30.7% / Jaccard 34.3% for InternVL and
   61.5% / 66.4% for Gemma -- so the "fix" would have cost ~5 points.

The text branch now matches if EITHER edit-distance similarity >= 0.90 OR the
pre-existing Jaccard word overlap >= 0.75 passes.

3. Replacing Jaccard with similarity (rather than unioning them) was measured and
   was WRONG: receipts fell 85.3% -> 34.2% for InternVL and 94.0% -> 72.8% for
   Gemma, because receipts prefix the quantity ('2x USB Hub 4-port' vs
   'USB Hub 4-port') -- one extra token barely moves a set overlap but costs
   0.176 of edit-distance similarity on a 14-character string.

The two measures fail on opposite error types, so the union is the fix:

    error type            example                                Jaccard  similarity
    token added           '2x USB Hub 4-port' / 'USB Hub 4-port'  PASS      0.824 fail
    character confused    'DO COASTAL ...'    / 'DD COASTAL ...'  fail      0.970 PASS

Real near-misses the similarity half rescues, all scored ZERO before:
    'DO COASTAL CHAR REF74032 MHF 5114'   vs 'DD COASTAL CHAR REF74032 MHF 5114'
    'EFTPOS JSHY LEGAL Alexandria AUS'    vs 'EFTPOS ASHBY LEGAL Alexandria AUS'
    'EFTPOS CALLOWAY GEN Leaderville AUS' vs 'EFTPOS CALLOWAY GEN Leederville AUS'
"""

import pytest

from common.evaluation_metrics import _transaction_item_matches
from common.field_schema import get_field_schema

DESC = "LINE_ITEM_DESCRIPTIONS"


class TestDescriptionsAreRegisterFields:
    """The schema must agree with the sort group about what the register is."""

    def test_descriptions_are_transaction_list_fields(self) -> None:
        assert DESC in get_field_schema().transaction_list_fields

    def test_the_other_two_register_columns_are_too(self) -> None:
        fields = get_field_schema().transaction_list_fields
        assert {"TRANSACTION_DATES", "TRANSACTION_AMOUNTS_PAID"} <= fields

    def test_schema_matches_the_sort_group(self) -> None:
        # _TRANSACTION_SORT_GROUPS already listed all three; the schema did not.
        from common.extraction_evaluator import _TRANSACTION_SORT_GROUPS

        group = set(_TRANSACTION_SORT_GROUPS["bank_statement"]["fields"])
        schema = get_field_schema().transaction_list_fields
        # Every bank register column the sorter reorders must also be a
        # transaction_list field, or the two disagree about the register again.
        assert DESC in group and DESC in schema


class TestNearMissesNowMatch:
    """Single-character OCR-style confusions must not zero a whole row."""

    @pytest.mark.parametrize(
        ("extracted", "truth"),
        [
            ("DO COASTAL CHAR REF74032 MHF 5114", "DD COASTAL CHAR REF74032 MHF 5114"),
            ("EFTPOS JSHY LEGAL Alexandria AUS", "EFTPOS ASHBY LEGAL Alexandria AUS"),
            ("EFTPOS CALLOWAY GEN Leaderville AUS", "EFTPOS CALLOWAY GEN Leederville AUS"),
            ("ATM WITHDRAWAL Leaderville", "ATM WITHDRAWAL Leederville"),
            ("BPAY VERRALL REF30033", "BPAY VERRALL REF30O33"),
        ],
    )
    def test_single_character_confusions_match(self, extracted: str, truth: str) -> None:
        assert _transaction_item_matches(extracted, truth, DESC)

    def test_identical_matches(self) -> None:
        assert _transaction_item_matches("BPAY VERRALL REF30033", "BPAY VERRALL REF30033", DESC)

    def test_case_insensitive(self) -> None:
        assert _transaction_item_matches("bpay verrall ref30033", "BPAY VERRALL REF30033", DESC)

    def test_whitespace_tolerant(self) -> None:
        assert _transaction_item_matches("  ATM WITHDRAWAL Perth  ", "ATM WITHDRAWAL Perth", DESC)


class TestQuantityPrefixesStillMatch:
    """Receipts prefix the quantity; the Jaccard half of the union covers it.

    These are the exact pairs that regressed when similarity REPLACED Jaccard
    instead of joining it. Each fails the 0.90 similarity threshold, so if the
    union is ever collapsed back to a single measure these tests go red.
    """

    @pytest.mark.parametrize(
        ("extracted", "truth"),
        [
            ("2x USB Hub 4-port", "USB Hub 4-port"),
            ("3x HDMI Cable 2m", "HDMI Cable 2m"),
            ("3x Bread White 700g", "Bread White 700g"),
            ("2x Wine Red 750ml", "Wine Red 750ml"),
        ],
    )
    def test_quantity_prefix_matches(self, extracted: str, truth: str) -> None:
        assert _transaction_item_matches(extracted, truth, DESC)

    @pytest.mark.parametrize(
        ("extracted", "truth"),
        [
            ("2x USB Hub 4-port", "USB Hub 4-port"),
            ("3x Bread White 700g", "Bread White 700g"),
        ],
    )
    def test_these_fail_the_similarity_half(self, extracted: str, truth: str) -> None:
        """Guard the premise: they pass only because the union has a second arm."""
        from common.evaluation_metrics import _text_similarity

        threshold = get_field_schema().get_threshold("transaction_text_similarity", 0.90)
        assert _text_similarity(extracted, truth) < threshold


class TestUnionOfBothMeasures:
    """Each arm is load-bearing for a different error type."""

    def test_character_confusion_needs_the_similarity_arm(self) -> None:
        from common.evaluation_metrics import _fuzzy_text_match

        pair = ("DO COASTAL CHAR REF74032 MHF 5114", "DD COASTAL CHAR REF74032 MHF 5114")
        assert not _fuzzy_text_match(*pair)  # Jaccard alone would score this zero
        assert _transaction_item_matches(*pair, DESC)

    def test_token_addition_needs_the_jaccard_arm(self) -> None:
        from common.evaluation_metrics import _text_similarity

        pair = ("2x USB Hub 4-port", "USB Hub 4-port")
        threshold = get_field_schema().get_threshold("transaction_text_similarity", 0.90)
        assert _text_similarity(*pair) < threshold  # similarity alone would score this zero
        assert _transaction_item_matches(*pair, DESC)


class TestStillDiscriminates:
    """Partial credit must not become credit for anything."""

    @pytest.mark.parametrize(
        ("extracted", "truth"),
        [
            ("ATM WITHDRAWAL Perth", "BPAY ORIGIN ENERGY CRN 203848421"),
            ("Salary PAYROLL REF47880", "EFTPOS METRO BOTTLE Brunswick AUS"),
            ("Transfer To Tanner NetBank", "VISA DEBIT PURCHASE CARD 7511"),
        ],
    )
    def test_unrelated_descriptions_do_not_match(self, extracted: str, truth: str) -> None:
        assert not _transaction_item_matches(extracted, truth, DESC)

    def test_empty_against_populated_does_not_match(self) -> None:
        assert not _transaction_item_matches("", "ATM WITHDRAWAL Perth", DESC)

    @pytest.mark.parametrize(
        ("extracted", "truth", "similarity"),
        [
            # These share a long prefix and differ only in the part that
            # IDENTIFIES the transaction. They are the reason the threshold is
            # 0.90 and not ANLS's 0.50 floor -- the worst of them sits at 0.857.
            ("BPAY VERRALL REF30033", "BPAY VERRALL REF74032", 0.857),
            ("EFTPOS COLES Alexandria AUS", "EFTPOS WOOLWORTHS Alexandria AUS", 0.781),
            ("Transfer To Tanner NetBank", "Transfer To Wilson NetBank", 0.769),
            ("ATM WITHDRAWAL Perth", "ATM WITHDRAWAL Sydney", 0.714),
        ],
    )
    def test_same_prefix_different_transaction_does_not_match(
        self, extracted: str, truth: str, similarity: float
    ) -> None:
        from common.evaluation_metrics import _text_similarity

        # Guard the premise as well as the behaviour: if similarity drifts above
        # the threshold this test would silently stop testing anything.
        assert _text_similarity(extracted, truth) == pytest.approx(similarity, abs=0.01)
        assert not _transaction_item_matches(extracted, truth, DESC)


class TestOtherRegisterColumnsUnaffected:
    """The change touches only the text branch of the matcher."""

    def test_amounts_still_monetary(self) -> None:
        assert _transaction_item_matches("$1,234.56", "1234.56", "TRANSACTION_AMOUNTS_PAID")

    def test_wrong_amount_still_fails(self) -> None:
        # NOTE monetary_tolerance is RELATIVE (0.01 = 1%), so $1,234.56 vs
        # $1,234.65 DOES match -- 1% of $1,234 is +/- $12. Pre-existing and out
        # of scope here, but it means digit transpositions can pass on large
        # amounts. Use a clearly different value to test the negative case.
        assert not _transaction_item_matches("$1,234.56", "$2,500.00", "TRANSACTION_AMOUNTS_PAID")

    def test_dates_still_date_compared(self) -> None:
        assert _transaction_item_matches("02/03/2023", "02/03/2023", "TRANSACTION_DATES")

    def test_wrong_date_still_fails(self) -> None:
        assert not _transaction_item_matches("02/03/2023", "05/03/2023", "TRANSACTION_DATES")

    def test_balance_still_monetary(self) -> None:
        assert _transaction_item_matches("$223.38 CR", "223.38", "ACCOUNT_BALANCE")


class TestThresholdComesFromYaml:
    """The cut-off is operator-visible config, not a Python constant."""

    def test_threshold_is_declared_in_yaml(self) -> None:
        assert get_field_schema().get_threshold("transaction_text_similarity", -1.0) == pytest.approx(0.90)

    def test_threshold_sits_between_the_two_populations(self) -> None:
        """0.90 is not arbitrary: it separates measured near-misses from
        measured different-transactions, which overlap nowhere in between."""
        from common.evaluation_metrics import _text_similarity

        near_miss = _text_similarity(
            "EFTPOS JSHY LEGAL Alexandria AUS", "EFTPOS ASHBY LEGAL Alexandria AUS"
        )
        different = _text_similarity("BPAY VERRALL REF30033", "BPAY VERRALL REF74032")
        threshold = get_field_schema().get_threshold("transaction_text_similarity", 0.90)
        assert different < threshold <= near_miss
