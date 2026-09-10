"""COMPOSITION: how many documents are in the picture.

tests/ is gitignored — local-only.

Taxpayers photograph several receipts laid out on a table in one shot, and
downstream extraction handles those badly. That is a different failure from
everything the six criteria ask about: a four-receipt table photo can be sharp,
evenly lit and undamaged, and is still unprocessable. The remedy differs too --
a blurred receipt is re-photographed, a collage is split.

So it is a SEPARATE axis, not a seventh criterion and not a severity level.
These tests pin that separation, and pin that adding it to v13 left v12 --
the measured variant -- untouched.
"""

from pathlib import Path

import pytest

from common.quality_screen_parser import load_screen_vocabulary, parse_quality_response

PROMPTS = Path("prompts/quality_screen.yaml")
CRITERIA = ["blur", "shadow", "crease", "faded", "tilt", "speckle"]


def response(*, overall="GOOD", composition=None):
    lines = [f"{i}. {name.upper()}: NO" for i, name in enumerate(CRITERIA, start=1)]
    lines.append(f"7. OVERALL: {overall}")
    if composition is not None:
        lines.append(f"8. COMPOSITION: {composition}")
    return "\n".join(lines)


@pytest.fixture
def v13():
    return load_screen_vocabulary(PROMPTS, variant="quality_screen_v13")


@pytest.fixture
def v12():
    return load_screen_vocabulary(PROMPTS, variant="quality_screen_v12")


def parse(text, vocabulary):
    return parse_quality_response(
        text,
        criteria=vocabulary.criteria,
        overall_levels=vocabulary.overall_levels,
        composition_levels=vocabulary.composition_levels,
    )


class TestV12IsUntouched:
    """v13 is additive. v12 carries the measured numbers and must not move."""

    def test_v12_does_not_ask_about_composition(self, v12):
        assert v12.composition_levels is None

    def test_a_v12_response_still_reads(self, v12):
        result = parse(response(overall="FAIR"), v12)

        assert not result.malformed
        assert result.composition is None

    def test_the_two_variants_share_criteria_and_polarity(self, v12, v13):
        assert v12.criteria == v13.criteria
        assert v12.polarity == v13.polarity
        assert v12.overall_levels == v13.overall_levels

    def test_questions_one_to_seven_are_byte_identical(self, v12, v13):
        """The point of appending COMPOSITION at 8 rather than inserting it at
        7. Identical text through OVERALL is what makes a v13 run comparable to
        the measured v12 baseline -- if the six criteria move, the new question
        disturbed them."""
        shared_v12 = v12.prompt.split("Fill in this template")[0].rstrip()
        shared_v13 = v13.prompt.split("8. COMPOSITION:")[0].rstrip()

        assert shared_v12 == shared_v13


class TestReadingTheAnswer:
    def test_single_and_multiple_both_read(self, v13):
        for value in ("SINGLE", "MULTIPLE"):
            result = parse(response(composition=value), v13)
            assert not result.malformed, result.malformed_reason
            assert result.composition == value

    def test_a_missing_answer_is_unreadable_not_an_absent_field(self, v13):
        """A variant that asks and gets no answer has a broken response. If it
        were treated as optional, a run could screen nothing for collages and
        still report a full set of records."""
        result = parse(response(composition=None), v13)

        assert result.malformed
        assert "composition" in result.malformed_reason

    def test_a_value_outside_the_vocabulary_is_unreadable(self, v13):
        result = parse(response(composition="TWO"), v13)

        assert result.malformed
        assert "TWO" in result.malformed_reason

    def test_yes_is_not_accepted(self, v13):
        """The answer spaces are disjoint on purpose. v11 lost 15 of 60
        responses to the model confusing two of them; a third vocabulary keeps
        that property only if it is enforced."""
        result = parse(response(composition="YES"), v13)

        assert result.malformed

    def test_the_answer_is_case_folded_like_the_others(self, v13):
        result = parse(response(composition="multiple"), v13)

        assert not result.malformed
        assert result.composition == "MULTIPLE"


class TestItIsItsOwnAxis:
    def test_composition_does_not_appear_among_the_criteria(self, v13):
        assert "composition" not in v13.criteria
        assert "composition" not in v13.polarity

    def test_a_good_photo_can_still_be_a_collage(self, v13):
        """The case that motivates the whole thing, and the reason it is not
        folded into OVERALL: nothing is wrong with the photograph."""
        result = parse(response(overall="GOOD", composition="MULTIPLE"), v13)

        assert result.overall == "GOOD"
        assert result.composition == "MULTIPLE"
        assert not any(result.answers.values()), "no defect was reported"


class TestScoring:
    def test_an_unlabelled_corpus_scores_none_not_zero(self):
        """Unmeasured must not read as measured. Reported as 0.0 it looks
        broken; as 1.0 it looks like it works."""
        from stages.evaluate_quality_screen import score_composition
        from common.quality_screen_parser import QualityResponse

        responses = {"a.png": QualityResponse({}, "GOOD", False, composition="SINGLE")}
        truths = [{"image_name": "a.png", "condition": "clean"}]

        assert score_composition(responses, truths) is None

    def test_a_labelled_corpus_is_scored(self):
        from stages.evaluate_quality_screen import score_composition
        from common.quality_screen_parser import QualityResponse

        responses = {
            "a.png": QualityResponse({}, "GOOD", False, composition="SINGLE"),
            "b.png": QualityResponse({}, "GOOD", False, composition="SINGLE"),
        }
        truths = [
            {"image_name": "a.png", "composition": "SINGLE"},
            {"image_name": "b.png", "composition": "MULTIPLE"},
        ]

        score = score_composition(responses, truths)

        assert score["labelled"] == 2
        assert score["correct"] == 1
        assert score["accuracy"] == 0.5
        assert score["confusion"] == {"MULTIPLE->SINGLE": 1, "SINGLE->SINGLE": 1}

    def test_a_malformed_response_is_not_counted_as_wrong(self):
        """It is unscored, like everywhere else -- counting it as an error
        would blame the criterion for a reading failure."""
        from stages.evaluate_quality_screen import score_composition
        from common.quality_screen_parser import QualityResponse

        responses = {"a.png": QualityResponse(None, None, True, composition=None)}
        truths = [{"image_name": "a.png", "composition": "MULTIPLE"}]

        score = score_composition(responses, truths)

        assert score["labelled"] == 1
        assert score["scored"] == 0
        assert score["accuracy"] is None


class TestReport:
    def _report(self, tally, scored):
        return {
            "variant": "quality_screen_v13",
            "counts": {"total": 2, "scored": 2, "malformed": 0, "missing": 0, "think_drift": 0},
            "per_criterion": {},
            "by_document_type": {},
            "overall_confusion": {},
            "composition_tally": tally,
            "composition": scored,
        }

    def test_an_unscored_run_says_so(self):
        from stages.evaluate_quality_screen import format_report

        text = format_report(self._report({"SINGLE": 2}, None))

        assert "NOT SCORED" in text
        assert "any MULTIPLE is a false positive" in text

    def test_a_scored_run_prints_accuracy(self):
        from stages.evaluate_quality_screen import format_report

        text = format_report(
            self._report(
                {"SINGLE": 1, "MULTIPLE": 1},
                {
                    "labelled": 2,
                    "scored": 2,
                    "correct": 2,
                    "accuracy": 1.0,
                    "confusion": {"MULTIPLE->MULTIPLE": 1, "SINGLE->SINGLE": 1},
                },
            )
        )

        assert "accuracy 1.000" in text
        assert "NOT SCORED" not in text

    def test_a_variant_that_does_not_ask_prints_nothing(self):
        """v12 reports must not gain an empty section."""
        from stages.evaluate_quality_screen import format_report

        assert "COMPOSITION" not in format_report(self._report({}, None))

    def test_composition_is_printed_under_its_own_heading(self):
        """Not inside the severity table. A reader skimming must not take a
        MULTIPLE for a severity."""
        from stages.evaluate_quality_screen import format_report

        text = format_report(self._report({"MULTIPLE": 2}, None))

        assert "OVERALL severity" in text
        assert text.index("OVERALL severity") < text.index("COMPOSITION")
