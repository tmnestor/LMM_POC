"""The routing decision: send this image on to extraction, or send it back.

The per-criterion table and the severity matrix describe how the photograph
looks. Neither is the decision. The decision combines severity with composition,
and the combination is a policy choice declared in run_config.yml rather than
something the scorer can derive: downstream extraction does not handle a
photograph of several receipts reliably, clean or damaged, so a collage is sent
back however good the picture is.

The case these tests exist to pin: a CLEAN collage graded POOR. Scored against
severity alone it is a false alarm and the screen looks imprecise. Scored
against the gate it is a correct decision reached by a different route.
"""

import pytest

from common.quality_screen_parser import QualityResponse
from stages.evaluate_quality_screen import format_report, score_routing

CONDITION_TO_LEVEL = {"clean": "GOOD", "moderate": "FAIR", "heavy": "POOR"}
REJECT = {"pass_levels": ["GOOD"], "multiple_documents": "reject"}
ALLOW = {"pass_levels": ["GOOD"], "multiple_documents": "allow"}


def answered(overall, composition):
    return QualityResponse({}, overall, False, composition=composition)


def truth(name, condition, composition):
    return {"filename": name, "condition": condition, "composition": composition}


class TestTheCollageRule:
    def test_a_clean_collage_sent_back_is_correct_not_a_false_alarm(self):
        """The measurement this gate exists for.

        The photograph is undamaged, so severity scoring calls the POOR verdict
        a false alarm. But extraction cannot read a plate of receipts at any
        quality, so sending it back is right. Under the gate it is a true
        positive and precision is unharmed.
        """
        responses = {"c.png": answered("POOR", "MULTIPLE")}
        truths = [truth("c.png", "clean", "MULTIPLE")]

        result = score_routing(responses, truths, CONDITION_TO_LEVEL, REJECT)

        assert result["sent_back_correctly"] == 1
        assert result["sent_back_wrongly"] == 0
        assert result["precision"] == 1.0

    def test_a_clean_collage_graded_good_is_still_sent_back_on_composition(self):
        """Composition overrides severity. A sharp, evenly lit plate of four
        receipts is unprocessable, and the screen must catch it on the other
        axis."""
        responses = {"c.png": answered("GOOD", "MULTIPLE")}
        truths = [truth("c.png", "clean", "MULTIPLE")]

        result = score_routing(responses, truths, CONDITION_TO_LEVEL, REJECT)

        assert result["sent_back_correctly"] == 1
        assert result["passed_wrongly"] == 0

    def test_a_collage_missed_on_both_axes_is_a_miss(self):
        responses = {"c.png": answered("GOOD", "SINGLE")}
        truths = [truth("c.png", "clean", "MULTIPLE")]

        result = score_routing(responses, truths, CONDITION_TO_LEVEL, REJECT)

        assert result["passed_wrongly"] == 1
        assert result["recall"] == 0.0

    def test_allow_lets_a_clean_collage_through(self):
        """The escape hatch for a pipeline whose extraction can split a plate.
        Ours cannot, but the behaviour must follow the config rather than a
        constant, or the config is decoration."""
        responses = {"c.png": answered("GOOD", "MULTIPLE")}
        truths = [truth("c.png", "clean", "MULTIPLE")]

        result = score_routing(responses, truths, CONDITION_TO_LEVEL, ALLOW)

        assert result["passed_correctly"] == 1
        assert result["sent_back_correctly"] == 0


class TestTheSeverityRule:
    def test_a_degraded_single_page_is_sent_back(self):
        responses = {"a.png": answered("POOR", "SINGLE")}
        truths = [truth("a.png", "heavy", "SINGLE")]

        assert score_routing(responses, truths, CONDITION_TO_LEVEL, REJECT)["sent_back_correctly"] == 1

    def test_a_clean_single_page_is_passed(self):
        responses = {"a.png": answered("GOOD", "SINGLE")}
        truths = [truth("a.png", "clean", "SINGLE")]

        assert score_routing(responses, truths, CONDITION_TO_LEVEL, REJECT)["passed_correctly"] == 1

    def test_a_clean_single_page_sent_back_is_the_real_false_alarm(self):
        """This one still counts against precision, and should: it is a
        taxpayer asked to re-photograph a receipt that was fine."""
        responses = {"a.png": answered("POOR", "SINGLE")}
        truths = [truth("a.png", "clean", "SINGLE")]

        result = score_routing(responses, truths, CONDITION_TO_LEVEL, REJECT)

        assert result["sent_back_wrongly"] == 1
        assert result["precision"] == 0.0

    def test_pass_levels_come_from_config_not_a_constant(self):
        """A pipeline willing to accept mild damage declares it. FAIR then
        passes without touching Python."""
        lenient = {"pass_levels": ["GOOD", "FAIR"], "multiple_documents": "reject"}
        responses = {"a.png": answered("FAIR", "SINGLE")}
        truths = [truth("a.png", "moderate", "SINGLE")]

        assert score_routing(responses, truths, CONDITION_TO_LEVEL, lenient)["passed_correctly"] == 1
        assert score_routing(responses, truths, CONDITION_TO_LEVEL, REJECT)["sent_back_correctly"] == 1


class TestUnmeasurable:
    def test_an_unlabelled_corpus_returns_none_rather_than_half_the_rule(self):
        """With multiple_documents=reject and no composition labels, the gate
        cannot be applied. A number computed from severity alone would look
        like a measurement of the gate and would not be one."""
        responses = {"a.png": answered("GOOD", "SINGLE")}
        truths = [{"filename": "a.png", "condition": "clean"}]

        assert score_routing(responses, truths, CONDITION_TO_LEVEL, REJECT) is None

    def test_without_the_collage_rule_an_unlabelled_corpus_still_scores(self):
        """multiple_documents=allow never consults composition, so its absence
        costs nothing."""
        responses = {"a.png": answered("GOOD", "SINGLE")}
        truths = [{"filename": "a.png", "condition": "clean"}]

        assert score_routing(responses, truths, CONDITION_TO_LEVEL, ALLOW)["passed_correctly"] == 1

    def test_a_malformed_response_is_skipped_not_counted_as_a_miss(self):
        responses = {"a.png": QualityResponse(None, None, True, composition=None)}
        truths = [truth("a.png", "heavy", "SINGLE")]

        assert score_routing(responses, truths, CONDITION_TO_LEVEL, REJECT)["scored"] == 0


class TestRendering:
    def _report(self, routing):
        return {
            "counts": {"total": 1, "scored": 1, "malformed": 0, "missing": 0, "think_drift": 0},
            "per_criterion": {},
            "by_document_type": {},
            "overall_confusion": {},
            "routing": routing,
        }

    def test_the_routing_block_is_printed_before_the_criterion_table(self):
        """It is the only line answering what the screen is for. A reader who
        takes the first table as the result reads a diagnostic as an outcome."""
        responses = {"c.png": answered("POOR", "MULTIPLE")}
        truths = [truth("c.png", "clean", "MULTIPLE")]
        text = format_report(self._report(score_routing(responses, truths, CONDITION_TO_LEVEL, REJECT)))

        assert "ROUTING" in text
        assert text.index("ROUTING") < text.index("CRITERION")

    def test_the_gate_is_stated_so_the_number_can_be_interpreted(self):
        responses = {"c.png": answered("POOR", "MULTIPLE")}
        truths = [truth("c.png", "clean", "MULTIPLE")]
        text = format_report(self._report(score_routing(responses, truths, CONDITION_TO_LEVEL, REJECT)))

        assert "pass GOOD" in text
        assert "MULTIPLE always sent back" in text

    def test_an_unmeasurable_gate_prints_nothing(self):
        assert "ROUTING" not in format_report(self._report(None))


@pytest.mark.parametrize("gate", [REJECT, ALLOW])
def test_the_gate_echoes_its_own_configuration_into_the_report(gate):
    """So a report read months later says which policy produced it."""
    responses = {"a.png": answered("GOOD", "SINGLE")}
    truths = [truth("a.png", "clean", "SINGLE")]

    result = score_routing(responses, truths, CONDITION_TO_LEVEL, gate)

    assert result["multiple_documents"] == gate["multiple_documents"]
    assert result["pass_levels"] == sorted(gate["pass_levels"])
