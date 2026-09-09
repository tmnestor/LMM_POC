"""Regression tests: travel docs must classify as TRAVEL, not INVOICE.

tests/ is gitignored — local-only. Travel docs (flight ticket / e-ticket /
boarding pass / itinerary) carry NO transaction table — the model reports
``COLUMNS: NONE`` — so they are invisible to column-based rules and fell through
to the INVOICE default. The fix adds a dedicated ``TRAVEL: YES/NO`` evidence
question to the detection prompt + a ``travel: true`` rule.

Response strings below mirror the real classify ``raw_response`` captured from a
GPU run (terse and verbose-CoT formats), with the new TRAVEL field appended.
These tests load the REAL shipped YAML so they fail until it carries the rule.
"""

from common.prompt_catalog import PromptCatalog
from common.turn_parsers import ClassificationParser, _evaluate_classification

# Terse format (most itineraries answered exactly like this on GPU).
_ITINERARY_TERSE = "1. COLUMNS: NONE\n2. PAID: NO\n3. ROWS: 3\n4. TRAVEL: YES\n"

# Verbose chain-of-thought format (some tickets drifted to this on GPU); the
# parser must still recover COLUMNS=NONE and TRAVEL=YES from markdown headings.
_TICKET_VERBOSE = (
    "Let's analyze the document image step by step.\n\n"
    "### 1. COLUMNS\nThe document is a boarding pass and does not contain a "
    "transaction table. **Answer:** NONE\n\n"
    "### 2. PAID\nNo payment method or EFTPOS details are shown. Answer: NO\n\n"
    "### 3. ROWS\nIt is a boarding pass with flight details. Answer: 0\n\n"
    "### 4. TRAVEL\nThis is a boarding pass for an airline flight. Answer: YES\n"
)

# A paid e-ticket — payment evidence must NOT downgrade it to RECEIPT.
_ETICKET_PAID = "1. COLUMNS: NONE\n2. PAID: YES\n3. ROWS: 2\n4. TRAVEL: YES\n"


def _evidence() -> dict:
    return PromptCatalog().get_classification_evidence()


class TestTravelRuleInShippedYaml:
    """The shipped rule set must derive TRAVEL from the travel boolean."""

    def test_travel_flag_derives_travel(self) -> None:
        assert _evaluate_classification(None, False, True, _evidence()) == "TRAVEL"

    def test_travel_wins_over_paid(self) -> None:
        assert _evaluate_classification(None, True, True, _evidence()) == "TRAVEL"

    def test_no_travel_flag_does_not_force_travel(self) -> None:
        # An ordinary unpaid invoice (no travel flag) must not become TRAVEL.
        assert _evaluate_classification({"gst": "GST"}, False, False, _evidence()) == "INVOICE"


class TestParseEnrichedTravel:
    """End-to-end through ClassificationParser._parse_enriched on real YAML."""

    def _parse(self, response: str) -> dict:
        parser = ClassificationParser(fallback_type="UNIVERSAL")
        result = parser._parse_enriched(response)
        assert result is not None
        return result

    def test_terse_itinerary_classified_travel(self) -> None:
        assert self._parse(_ITINERARY_TERSE)["DOCUMENT_TYPE"] == "TRAVEL"

    def test_verbose_ticket_classified_travel(self) -> None:
        assert self._parse(_TICKET_VERBOSE)["DOCUMENT_TYPE"] == "TRAVEL"

    def test_paid_eticket_classified_travel(self) -> None:
        assert self._parse(_ETICKET_PAID)["DOCUMENT_TYPE"] == "TRAVEL"

    def test_travel_evidence_surfaced(self) -> None:
        assert self._parse(_ITINERARY_TERSE)["travel_evidence"] is True

    def test_columns_none_without_travel_flag_defaults_invoice(self) -> None:
        # Pre-fix behaviour for a genuine non-travel COLUMNS:NONE doc is preserved.
        result = self._parse("1. COLUMNS: NONE\n2. PAID: NO\n3. ROWS: 0\n4. TRAVEL: NO\n")
        assert result["DOCUMENT_TYPE"] == "INVOICE"


class TestVerboseRecoverySafety:
    """The verbose recovery must not misfire on echoes or prose mentions."""

    def _parse(self, response: str) -> dict:
        parser = ClassificationParser(fallback_type="UNIVERSAL")
        result = parser._parse_enriched(response)
        assert result is not None
        return result

    def test_echoed_instruction_only_is_not_an_answer(self) -> None:
        # The model echoes "Answer YES or NO" but states no conclusion.
        resp = "1. COLUMNS: NONE\n2. PAID: NO\n3. ROWS: 0\n### 4. TRAVEL\nAnswer YES or NO.\n"
        assert self._parse(resp)["travel_evidence"] is False

    def test_prose_travel_mention_does_not_flag(self) -> None:
        # A doc whose COLUMNS reasoning says "not a travel document" but has no
        # TRAVEL field: the prose mention must not be read as a TRAVEL answer.
        resp = (
            "### 1. COLUMNS\nThis is not a travel document. NONE\n"
            "### 2. PAID\nAnswer: YES\n### 3. ROWS\n0\n"
        )
        result = self._parse(resp)
        assert result["travel_evidence"] is False
        assert result["DOCUMENT_TYPE"] != "TRAVEL"

    def test_verbose_travel_no_stays_invoice(self) -> None:
        resp = "### 1. COLUMNS\nNONE\n### 4. TRAVEL\nAfter review, the answer is NO\n"
        assert self._parse(resp)["DOCUMENT_TYPE"] == "INVOICE"


class TestDetectionPromptAsksTravel:
    """The detection prompt must actually request the TRAVEL evidence field."""

    def test_prompt_contains_travel_question(self) -> None:
        cfg = PromptCatalog().get_detection_config()
        prompt = cfg["prompts"]["detection"]["prompt"]
        assert "TRAVEL:" in prompt
