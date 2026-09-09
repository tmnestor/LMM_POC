"""Tests for the YAML-driven classification-evidence evaluator.

tests/ is gitignored — local-only. Covers the pure rule-evaluation logic
(`_evaluate_classification` / `_when_matches`) independent of YAML loading.
"""

from common.turn_parsers import _evaluate_classification, _when_matches

# A representative rule set (mirrors the enabled shape, default deferring).
_EVIDENCE = {
    "rules": [
        {"type": "BANK_STATEMENT", "when": {"any_roles": ["debit", "credit", "balance"]}},
        {"type": "LOGBOOK", "when": {"any_roles": ["distance", "odometer", "purpose"]}},
        {"type": "TRAVEL", "when": {"travel": True}},
        {"type": "INVOICE", "when": {"any_roles": ["gst", "unit_price", "quantity"]}},
        {"type": "RECEIPT", "when": {"paid": True}},
    ],
    "default": "none",
}


def _map(*roles: str) -> dict[str, str | None]:
    """Build a column_mapping where the given roles are 'present'."""
    return {role: f"<{role}>" for role in roles}


class TestWhenMatches:
    def test_any_roles_hit(self) -> None:
        assert _when_matches({"any_roles": ["debit"]}, {"debit"}, False, False) is True

    def test_any_roles_miss(self) -> None:
        assert _when_matches({"any_roles": ["debit"]}, {"date"}, False, False) is False

    def test_all_roles_requires_every(self) -> None:
        assert _when_matches({"all_roles": ["a", "b"]}, {"a"}, False, False) is False
        assert _when_matches({"all_roles": ["a", "b"]}, {"a", "b"}, False, False) is True

    def test_paid_flag(self) -> None:
        assert _when_matches({"paid": True}, set(), True, False) is True
        assert _when_matches({"paid": True}, set(), False, False) is False

    def test_travel_flag(self) -> None:
        assert _when_matches({"travel": True}, set(), False, True) is True
        assert _when_matches({"travel": True}, set(), False, False) is False

    def test_anded_conditions(self) -> None:
        when = {"any_roles": ["gst"], "paid": False}
        assert _when_matches(when, {"gst"}, False, False) is True
        assert _when_matches(when, {"gst"}, True, False) is False


class TestEvaluateClassification:
    def test_bank_columns_win(self) -> None:
        assert (
            _evaluate_classification(_map("debit", "balance"), True, False, _EVIDENCE) == "BANK_STATEMENT"
        )

    def test_logbook_columns(self) -> None:
        assert _evaluate_classification(_map("odometer", "distance"), False, False, _EVIDENCE) == "LOGBOOK"

    def test_invoice_columns(self) -> None:
        assert _evaluate_classification(_map("gst", "quantity"), False, False, _EVIDENCE) == "INVOICE"

    def test_paid_receipt(self) -> None:
        assert _evaluate_classification(_map("date", "description"), True, False, _EVIDENCE) == "RECEIPT"

    def test_travel_flag_derives_travel(self) -> None:
        # No columns, not paid — the TRAVEL boolean is the only signal.
        assert _evaluate_classification(None, False, True, _EVIDENCE) == "TRAVEL"

    def test_travel_wins_over_paid(self) -> None:
        # A paid e-ticket: TRAVEL rule precedes the RECEIPT (paid) rule.
        assert _evaluate_classification(None, True, True, _EVIDENCE) == "TRAVEL"

    def test_precedence_bank_over_paid(self) -> None:
        # debit present AND paid -> bank rule comes first.
        assert _evaluate_classification(_map("debit"), True, False, _EVIDENCE) == "BANK_STATEMENT"

    def test_default_none_returns_none(self) -> None:
        assert _evaluate_classification(_map("date"), False, False, _EVIDENCE) is None

    def test_none_mapping_unpaid_defers(self) -> None:
        assert _evaluate_classification(None, False, False, _EVIDENCE) is None

    def test_hard_default_returned(self) -> None:
        evidence = {
            "rules": [{"type": "BANK_STATEMENT", "when": {"any_roles": ["debit"]}}],
            "default": "INVOICE",
        }
        assert _evaluate_classification(_map("date"), False, False, evidence) == "INVOICE"
