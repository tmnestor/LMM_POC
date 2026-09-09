"""Scoring the quality screen against the corpus's defect labels.

Two reports, because they answer different questions: per-criterion
precision/recall/F1 over the six booleans, and a confusion matrix over the
graded OVERALL. Malformed responses are counted apart from both -- a run with
40% unreadable responses and 0.95 F1 on the remainder is not a 0.95 run.

The fixtures here are hand-built and tiny. The point is to pin the arithmetic
and the edge cases where a scorer lies quietly: a criterion that never occurs,
a model that never predicts one, and responses that could not be read at all.
"""

import json
from pathlib import Path

import pytest

from common.quality_screen_parser import QualityResponse
from common.quality_screen_scorer import score_quality_screen

# The generated corpus lives outside the repo, so these tests skip rather than
# fail on a machine that has not generated it.
CORPUS = Path("../evaluation_data/quality_20260909/quality_ground_truth.jsonl")

CRITERIA = ["blur", "shadow"]
# The generator records a condition; the prompt answers a severity level. The
# two vocabularies are different words for the same ladder and the mapping
# between them is a real contract, so it is passed in rather than assumed.
CONDITION_TO_LEVEL = {"clean": "NONE", "moderate": "MODERATE", "heavy": "HEAVY"}


def truth(name, *, blur, shadow, condition="moderate", doc_type="receipt"):
    return {
        "filename": name,
        "document_type": doc_type,
        "condition": condition,
        "defects": {"blur": blur, "shadow": shadow},
    }


def predicted(*, blur, shadow, overall="MODERATE"):
    return QualityResponse(answers={"blur": blur, "shadow": shadow}, overall=overall, malformed=False)


def score(predictions, truths):
    return score_quality_screen(
        predictions,
        truths,
        criteria=CRITERIA,
        condition_to_level=CONDITION_TO_LEVEL,
    )


def test_counts_true_and_false_positives_per_criterion():
    """One true positive, one false positive, one false negative for blur."""
    truths = [
        truth("a.png", blur=True, shadow=False),
        truth("b.png", blur=False, shadow=False),
        truth("c.png", blur=True, shadow=False),
    ]
    predictions = {
        "a.png": predicted(blur=True, shadow=False),
        "b.png": predicted(blur=True, shadow=False),
        "c.png": predicted(blur=False, shadow=False),
    }

    result = score(predictions, truths)
    blur = result.per_criterion["blur"]

    assert (blur.true_positives, blur.false_positives, blur.false_negatives) == (1, 1, 1)
    assert blur.precision == pytest.approx(0.5)
    assert blur.recall == pytest.approx(0.5)
    assert blur.f1 == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Where a scorer lies quietly
# --------------------------------------------------------------------------


def test_a_criterion_that_never_occurs_reports_undefined_not_perfect():
    """Nothing to find and nothing claimed is not a perfect score.

    Reporting 1.0 would say the model aced a criterion it was never tested on;
    reporting 0.0 would say it failed one. Both are assertions the data cannot
    support, and either would be averaged into a headline number.
    """
    truths = [truth("a.png", blur=False, shadow=False)]
    predictions = {"a.png": predicted(blur=False, shadow=False)}

    blur = score(predictions, truths).per_criterion["blur"]

    assert (blur.true_positives, blur.false_positives, blur.false_negatives) == (0, 0, 0)
    assert blur.precision is None
    assert blur.recall is None
    assert blur.f1 is None


def test_a_criterion_never_predicted_has_undefined_precision_but_real_recall():
    """The model claimed no blur anywhere. Precision has no denominator, but
    recall is a genuine zero -- there were defects and it found none."""
    truths = [truth("a.png", blur=True, shadow=False)]
    predictions = {"a.png": predicted(blur=False, shadow=False)}

    blur = score(predictions, truths).per_criterion["blur"]

    assert blur.precision is None
    assert blur.recall == pytest.approx(0.0)
    assert blur.f1 is None


def test_getting_every_prediction_wrong_scores_zero_rather_than_crashing():
    """Precision and recall are both a defined 0.0 here -- the model predicted
    a defect where there was none and missed the one there was. F1's usual
    formula divides by their sum, so this is the case that raises."""
    truths = [
        truth("a.png", blur=True, shadow=False),
        truth("b.png", blur=False, shadow=False),
    ]
    predictions = {
        "a.png": predicted(blur=False, shadow=False),
        "b.png": predicted(blur=True, shadow=False),
    }

    blur = score(predictions, truths).per_criterion["blur"]

    assert blur.precision == pytest.approx(0.0)
    assert blur.recall == pytest.approx(0.0)
    assert blur.f1 == pytest.approx(0.0)


def test_malformed_responses_are_counted_and_excluded_from_the_metrics():
    """Scoring an unreadable response as six wrong answers would blame the
    model's judgement for a formatting failure."""
    truths = [
        truth("a.png", blur=True, shadow=True),
        truth("b.png", blur=True, shadow=True),
    ]
    predictions = {
        "a.png": predicted(blur=True, shadow=True),
        "b.png": QualityResponse(answers=None, overall=None, malformed=True, malformed_reason="refusal"),
    }

    result = score(predictions, truths)

    assert result.malformed == 1
    assert result.scored == 1
    assert result.per_criterion["blur"].true_positives == 1
    assert result.per_criterion["blur"].false_negatives == 0


def test_a_response_claiming_to_be_readable_but_carrying_nothing_is_malformed():
    """A contract violation, not a crash.

    `malformed=False` with no answers should be impossible, but nothing in the
    type enforces it. Reaching into a None here would fail partway through a
    330-image run with an index error and no indication of which image or why;
    counting it as unreadable keeps the run going and puts it in the tally.
    """
    truths = [truth("a.png", blur=True, shadow=True)]
    predictions = {
        "a.png": QualityResponse(answers=None, overall=None, malformed=False),
    }

    result = score(predictions, truths)

    assert result.malformed == 1
    assert result.scored == 0


def test_scores_are_broken_out_by_document_type():
    """The corpus degrades receipts and invoices with separately calibrated
    ladders, and they differ 5x in page scale. A screen that reads thermal
    receipts but not A4 invoices is a result to see, not to average away.
    """
    truths = [
        truth("r.png", blur=True, shadow=False, doc_type="receipt"),
        truth("i.png", blur=True, shadow=False, doc_type="invoice"),
    ]
    predictions = {
        "r.png": predicted(blur=True, shadow=False),
        "i.png": predicted(blur=False, shadow=False),
    }

    result = score(predictions, truths)

    assert result.by_document_type["receipt"]["blur"].true_positives == 1
    assert result.by_document_type["invoice"]["blur"].false_negatives == 1


def test_overall_is_reported_as_a_confusion_matrix():
    """A matrix, not an accuracy: calling heavy "moderate" is a different
    problem from calling clean "heavy", and one number hides which."""
    truths = [
        truth("a.png", blur=False, shadow=False, condition="clean"),
        truth("b.png", blur=True, shadow=True, condition="heavy"),
        truth("c.png", blur=True, shadow=True, condition="heavy"),
    ]
    predictions = {
        "a.png": predicted(blur=False, shadow=False, overall="NONE"),
        "b.png": predicted(blur=True, shadow=True, overall="HEAVY"),
        "c.png": predicted(blur=True, shadow=True, overall="MODERATE"),
    }

    result = score(predictions, truths)

    assert result.overall_confusion[("NONE", "NONE")] == 1
    assert result.overall_confusion[("HEAVY", "HEAVY")] == 1
    assert result.overall_confusion[("HEAVY", "MODERATE")] == 1


def test_a_condition_with_no_declared_level_fails_fast(assert_diagnostic_error):
    """The corpus and the prompt name the same ladder differently. If the
    mapping does not cover a condition, silently skipping those rows would
    drop a whole severity from the report without saying so.
    """
    truths = [truth("a.png", blur=False, shadow=False, condition="light")]
    predictions = {"a.png": predicted(blur=False, shadow=False, overall="NONE")}

    with pytest.raises(ValueError) as exc_info:
        score(predictions, truths)

    message = str(exc_info.value)
    assert_diagnostic_error(message)
    assert "light" in message


def test_a_model_that_answers_yes_to_everything_is_caught_by_the_clean_rows():
    """The corpus's own shape makes this the likeliest way to score well while
    seeing nothing.

    `shadow`, `faded` and `speckle` are true for essentially every degraded
    image, so a model that always answers YES gets perfect recall on them. Only
    the clean rows -- where every defect is false -- turn that into false
    positives. This is the concrete reason the combined directory carries the
    clean half, and why precision is reported per criterion rather than blended
    into one number.
    """
    truths = [
        truth("clean1.png", blur=False, shadow=False, condition="clean"),
        truth("clean2.png", blur=False, shadow=False, condition="clean"),
        truth("heavy1.png", blur=True, shadow=True, condition="heavy"),
        truth("heavy2.png", blur=True, shadow=True, condition="heavy"),
    ]
    always_yes = {
        name: predicted(blur=True, shadow=True, overall="HEAVY") for name in (r["filename"] for r in truths)
    }

    blur = score(always_yes, truths).per_criterion["blur"]

    assert blur.recall == pytest.approx(1.0), "always-YES finds every real defect"
    assert blur.precision == pytest.approx(0.5), "and is only caught by the clean rows"
    assert blur.f1 == pytest.approx(2 / 3)


def test_an_image_with_no_prediction_at_all_is_counted_as_missing():
    """A crashed or skipped image must not silently shrink the denominator."""
    truths = [
        truth("a.png", blur=True, shadow=True),
        truth("b.png", blur=True, shadow=True),
    ]
    predictions = {"a.png": predicted(blur=True, shadow=True)}

    result = score(predictions, truths)

    assert result.missing == 1
    assert result.scored == 1
    assert result.total == 2


# --------------------------------------------------------------------------
# Against the real corpus
# --------------------------------------------------------------------------
# The fixtures above pin the arithmetic; these pin the data shape. A scorer
# that only ever sees hand-built dicts can be wrong about the file it will
# actually be given.


@pytest.fixture
def real_truths():
    if not CORPUS.exists():
        pytest.skip(f"corpus not generated at {CORPUS}")
    return [json.loads(line) for line in CORPUS.read_text().splitlines() if line.strip()]


def test_the_real_ground_truth_has_the_shape_the_scorer_expects(real_truths):
    assert len(real_truths) == 330
    for record in real_truths:
        assert {"filename", "document_type", "condition", "defects"} <= set(record)


def test_a_perfect_predictor_scores_perfectly_on_the_real_corpus(real_truths):
    """End to end on 330 real rows: every defined F1 is 1.0, nothing is
    malformed or missing, and the confusion matrix is diagonal."""
    criteria = list(real_truths[0]["defects"])
    predictions = {
        record["filename"]: QualityResponse(
            answers=dict(record["defects"]),
            overall=CONDITION_TO_LEVEL[record["condition"]],
            malformed=False,
        )
        for record in real_truths
    }

    result = score_quality_screen(
        predictions, real_truths, criteria=criteria, condition_to_level=CONDITION_TO_LEVEL
    )

    assert (result.scored, result.malformed, result.missing) == (330, 0, 0)
    for name, criterion in result.per_criterion.items():
        assert criterion.f1 == pytest.approx(1.0), f"{name} should be perfect"
    off_diagonal = {k: v for k, v in result.overall_confusion.items() if k[0] != k[1]}
    assert not off_diagonal


def test_always_yes_is_visibly_imperfect_on_the_real_corpus(real_truths):
    """The always-YES model against the real class balance. Recall is perfect
    on every criterion; precision is not, and the gap is exactly what the clean
    third of the corpus buys."""
    criteria = list(real_truths[0]["defects"])
    predictions = {
        record["filename"]: QualityResponse(
            answers=dict.fromkeys(criteria, True), overall="HEAVY", malformed=False
        )
        for record in real_truths
    }

    result = score_quality_screen(
        predictions, real_truths, criteria=criteria, condition_to_level=CONDITION_TO_LEVEL
    )

    for name, criterion in result.per_criterion.items():
        assert criterion.recall == pytest.approx(1.0), f"{name} recall"
        assert criterion.precision < 1.0, f"{name} precision should be dragged down by clean rows"
