"""The severity matrix, split by how many documents are in the photograph.

The single matrix averages two populations built for different reasons:
photographs of one document, and photographs of several laid out on a plate. A
plate carries inter-receipt shadow and a separate tilt per receipt, so a *clean*
collage can look damaged to a screen tuned on flat single pages. Averaged
together, that reads as the screen getting worse rather than the corpus getting
harder, and the split is what tells the two apart.
"""

import pytest

from common.quality_screen_parser import QualityResponse
from stages.evaluate_quality_screen import (
    format_report,
    score_composition,
    severity_by_composition,
)

CONDITION_TO_LEVEL = {"clean": "GOOD", "moderate": "FAIR", "heavy": "POOR"}


def answered(overall, composition):
    return QualityResponse({}, overall, False, composition=composition)


class TestSeverityByComposition:
    def test_an_unlabelled_corpus_returns_none_not_an_empty_split(self):
        """Same UNMEASURED convention as score_composition. An empty dict would
        render as a split that found nothing, which is a different claim."""
        responses = {"a.png": answered("GOOD", None)}
        truths = [{"filename": "a.png", "condition": "clean"}]

        assert severity_by_composition(responses, truths, CONDITION_TO_LEVEL) is None

    def test_each_composition_gets_its_own_matrix(self):
        responses = {
            "single_ok.png": answered("GOOD", "SINGLE"),
            "collage_ok.png": answered("GOOD", "MULTIPLE"),
            "collage_flagged.png": answered("POOR", "MULTIPLE"),
        }
        truths = [
            {"filename": "single_ok.png", "condition": "clean", "composition": "SINGLE"},
            {"filename": "collage_ok.png", "condition": "clean", "composition": "MULTIPLE"},
            {"filename": "collage_flagged.png", "condition": "clean", "composition": "MULTIPLE"},
        ]

        split = severity_by_composition(responses, truths, CONDITION_TO_LEVEL)

        assert split == {
            "MULTIPLE": {"GOOD->GOOD": 1, "GOOD->POOR": 1},
            "SINGLE": {"GOOD->GOOD": 1},
        }

    def test_the_split_separates_a_clean_collage_false_alarm_from_the_single_pages(self):
        """The measurement this exists for.

        Every single-page call is right and every clean collage is flagged POOR.
        The combined matrix shows 2 of 4 correct and reads as a screen that has
        half broken; the split shows single pages perfect and collages at zero,
        which is a corpus finding, not a regression.
        """
        responses = {
            "p1.png": answered("GOOD", "SINGLE"),
            "p2.png": answered("GOOD", "SINGLE"),
            "c1.png": answered("POOR", "MULTIPLE"),
            "c2.png": answered("POOR", "MULTIPLE"),
        }
        truths = [
            {"filename": n, "condition": "clean", "composition": c}
            for n, c in (
                ("p1.png", "SINGLE"),
                ("p2.png", "SINGLE"),
                ("c1.png", "MULTIPLE"),
                ("c2.png", "MULTIPLE"),
            )
        ]

        split = severity_by_composition(responses, truths, CONDITION_TO_LEVEL)

        assert split["SINGLE"] == {"GOOD->GOOD": 2}
        assert split["MULTIPLE"] == {"GOOD->POOR": 2}

    def test_a_malformed_response_is_skipped_rather_than_counted_wrong(self):
        responses = {"a.png": QualityResponse(None, None, True, composition=None)}
        truths = [{"filename": "a.png", "condition": "clean", "composition": "MULTIPLE"}]

        assert severity_by_composition(responses, truths, CONDITION_TO_LEVEL) == {}

    def test_an_unmappable_condition_is_skipped_not_crashed_on(self):
        """The criterion scorer raises a diagnostic for this; the split must not
        be the thing that fails first, or the diagnostic never gets printed."""
        responses = {"a.png": answered("GOOD", "SINGLE")}
        truths = [{"filename": "a.png", "condition": "pristine", "composition": "SINGLE"}]

        assert severity_by_composition(responses, truths, CONDITION_TO_LEVEL) == {}


class TestRendering:
    def _report(self, split):
        return {
            "counts": {"total": 1, "scored": 1, "malformed": 0, "missing": 0, "think_drift": 0},
            "per_criterion": {},
            "by_document_type": {},
            "overall_confusion": {"GOOD->GOOD": 1},
            "severity_by_composition": split,
        }

    def test_a_single_composition_corpus_prints_no_split(self):
        """It would restate the matrix above it. A table that only ever repeats
        its neighbour trains the reader to skip both."""
        text = format_report(self._report({"SINGLE": {"GOOD->GOOD": 1}}))

        assert "split by composition" not in text

    def test_two_compositions_print_a_split_with_an_exact_match_rate(self):
        text = format_report(
            self._report(
                {
                    "MULTIPLE": {"GOOD->GOOD": 1, "GOOD->POOR": 3},
                    "SINGLE": {"GOOD->GOOD": 4},
                }
            )
        )

        assert "split by composition" in text
        assert "MULTIPLE (4 images, exact 0.250)" in text
        assert "SINGLE (4 images, exact 1.000)" in text

    def test_an_unmeasured_split_prints_nothing(self):
        assert "split by composition" not in format_report(self._report(None))


class TestNamedMisses:
    def test_a_wrong_answer_is_named_not_just_counted(self):
        """A count cannot say whether a SINGLE->MULTIPLE error fell on the
        folded-receipt hard negative -- which looks like two receipts, so losing
        on it is the check working as designed -- or on an ordinary page."""
        responses = {
            "FOLDED001_receipt.png": answered("GOOD", "MULTIPLE"),
            "CASE001_receipt.png": answered("GOOD", "SINGLE"),
        }
        truths = [
            {"filename": "FOLDED001_receipt.png", "composition": "SINGLE"},
            {"filename": "CASE001_receipt.png", "composition": "SINGLE"},
        ]

        score = score_composition(responses, truths)

        assert score["misses"] == [
            {"filename": "FOLDED001_receipt.png", "truth": "SINGLE", "predicted": "MULTIPLE"}
        ]

    def test_a_perfect_run_names_nothing(self):
        responses = {"a.png": answered("GOOD", "SINGLE")}
        truths = [{"filename": "a.png", "composition": "SINGLE"}]

        assert score_composition(responses, truths)["misses"] == []

    def test_the_printed_report_caps_the_named_misses(self):
        """A badly broken run must print a finding, not several hundred
        filenames burying the counts above them."""
        from stages.evaluate_quality_screen import _MAX_NAMED_MISSES

        responses = {f"x{i}.png": answered("GOOD", "MULTIPLE") for i in range(_MAX_NAMED_MISSES + 5)}
        truths = [{"filename": n, "composition": "SINGLE"} for n in responses]

        score = score_composition(responses, truths)
        report = {
            "counts": {"total": 1, "scored": 1, "malformed": 0, "missing": 0, "think_drift": 0},
            "per_criterion": {},
            "by_document_type": {},
            "overall_confusion": {},
            "composition": score,
            "composition_tally": {"MULTIPLE": len(responses)},
        }

        text = format_report(report)

        assert f"wrong on {_MAX_NAMED_MISSES + 5}" in text
        assert "and 5 more" in text
        assert text.count("SINGLE -> MULTIPLE") == _MAX_NAMED_MISSES


@pytest.mark.parametrize("missing_key", ["composition", "condition"])
def test_a_truth_record_missing_either_key_is_skipped_not_crashed_on(missing_key):
    """Ground truth is generated by a separate repository on its own branch. A
    record that predates one of these keys must be skipped, not raise."""
    record = {"filename": "a.png", "condition": "clean", "composition": "SINGLE"}
    del record[missing_key]

    result = severity_by_composition({"a.png": answered("GOOD", "SINGLE")}, [record], CONDITION_TO_LEVEL)

    assert result in (None, {})
