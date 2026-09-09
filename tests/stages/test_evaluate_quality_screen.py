"""The evaluate half of classify -> evaluate.

Reads what the classify stage wrote, scores it against the corpus's defect
labels, and renders a report. All CPU: the GPU pass already happened and its
output is on disk, which is the point of keeping the two stages apart -- a
threshold change or a scorer fix re-scores without paying for inference again.

The round-trip test is the one that matters most here. The two stages agree
only by convention about the shape of quality_screen.jsonl, and nothing but a
test crossing that boundary would notice if one side drifted.
"""

import json
from pathlib import Path

from common.quality_screen_parser import QualityResponse
from common.quality_screen_scorer import score_quality_screen
from stages.evaluate_quality_screen import build_report, format_report, load_screen_records
from stages.evaluate_quality_screen import run as stage_run
from stages.quality_screen import run_quality_screen, write_screen_records
from tests.stages.test_quality_screen import CRITERIA, VOCABULARY, fake_infer, response

CONDITION_TO_LEVEL = {"clean": "NONE", "moderate": "MODERATE", "heavy": "HEAVY"}


def truth(name, *, condition, doc_type="receipt", **defects):
    return {
        "filename": name,
        "document_type": doc_type,
        "condition": condition,
        "defects": {c: defects.get(c, False) for c in CRITERIA},
    }


def answered(*, overall, drift=False, **defects):
    return QualityResponse(
        answers={c: defects.get(c, False) for c in CRITERIA},
        overall=overall,
        malformed=False,
        think_drift=drift,
    )


def scored(predictions, truths):
    return score_quality_screen(
        predictions, truths, criteria=CRITERIA, condition_to_level=CONDITION_TO_LEVEL
    )


def test_records_written_by_classify_load_back_as_scoreable_responses(tmp_path):
    """The seam between the two stages, crossed in both directions."""
    images = ["/data/a.png", "/data/b.png"]
    responses = {"a.png": response(blur=True, overall="HEAVY"), "b.png": "I cannot help."}
    records = run_quality_screen(images, infer=fake_infer(responses), vocabulary=VOCABULARY)
    path = write_screen_records(records, tmp_path / "quality_screen.jsonl")

    loaded = load_screen_records(path)

    assert set(loaded) == {"a.png", "b.png"}
    assert loaded["a.png"].answers["blur"] is True
    assert loaded["a.png"].overall == "HEAVY"
    assert loaded["a.png"].malformed is False
    assert loaded["b.png"].malformed is True
    assert loaded["b.png"].answers is None


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------


def test_the_report_is_json_serialisable():
    """The confusion matrix is keyed by (truth, prediction) pairs, which JSON
    cannot represent as keys. If that is not converted the whole report fails
    to write -- after the GPU pass has been paid for and released."""
    truths = [truth("a.png", condition="heavy", blur=True)]
    predictions = {"a.png": answered(overall="HEAVY", blur=True)}

    report = build_report(scored(predictions, truths), predictions)

    json.dumps(report)  # must not raise


def test_the_report_carries_the_malformed_and_drift_rates():
    """Both belong beside the scores, not buried: a high malformed rate means
    the metrics describe a subset, and the drift rate is the measurement v5's
    wording exists to move."""
    truths = [truth("a.png", condition="heavy", blur=True), truth("b.png", condition="clean")]
    predictions = {
        "a.png": answered(overall="HEAVY", blur=True, drift=True),
        "b.png": QualityResponse(answers=None, overall=None, malformed=True, malformed_reason="refusal"),
    }

    report = build_report(scored(predictions, truths), predictions)

    assert report["counts"]["malformed"] == 1
    assert report["counts"]["scored"] == 1
    assert report["counts"]["think_drift"] == 1


def test_the_report_keeps_undefined_metrics_distinct_from_zero():
    """A criterion never predicted has no precision. Rendering that as 0.0
    would read as a measured failure."""
    truths = [truth("a.png", condition="clean")]
    predictions = {"a.png": answered(overall="NONE")}

    report = build_report(scored(predictions, truths), predictions)

    assert report["per_criterion"]["blur"]["precision"] is None


def test_the_rendered_report_shows_every_criterion_and_the_malformed_count():
    truths = [truth("a.png", condition="heavy", blur=True), truth("b.png", condition="clean")]
    predictions = {
        "a.png": answered(overall="HEAVY", blur=True),
        "b.png": answered(overall="NONE"),
    }

    text = format_report(build_report(scored(predictions, truths), predictions))

    for name in CRITERIA:
        assert name.upper() in text
    assert "malformed" in text.lower()
    assert "OVERALL" in text


def test_the_rendered_report_marks_undefined_rather_than_printing_a_number():
    truths = [truth("a.png", condition="clean")]
    predictions = {"a.png": answered(overall="NONE")}

    text = format_report(build_report(scored(predictions, truths), predictions))

    assert "n/a" in text.lower(), "an undefined rate must not render as a number"


def test_the_rendered_report_warns_when_images_went_unscored():
    """The scores describe the remainder, and saying so beside them is the
    difference between a partial run and a partial run mistaken for a whole
    one."""
    truths = [truth("a.png", condition="heavy", blur=True), truth("b.png", condition="clean")]
    predictions = {
        "a.png": answered(overall="HEAVY", blur=True),
        "b.png": QualityResponse(answers=None, overall=None, malformed=True, malformed_reason="x"),
    }

    text = format_report(build_report(scored(predictions, truths), predictions))

    assert "not scored" in text
    assert "50.0%" in text


# --------------------------------------------------------------------------
# The stage end to end
# --------------------------------------------------------------------------


def test_the_stage_scores_a_run_and_writes_a_report(tmp_path):
    """Everything but the model: real files in, real report out.

    This is the whole evaluate stage, and it needs no GPU -- which is the
    argument for splitting classify from evaluate in the first place.
    """
    images = ["/data/a.png", "/data/b.png", "/data/c.png"]
    responses = {
        "a.png": response(blur=True, overall="HEAVY"),
        "b.png": response(overall="NONE"),
        "c.png": "I'm sorry, I can't analyse this.",
    }
    records = run_quality_screen(images, infer=fake_infer(responses), vocabulary=VOCABULARY)
    screen_path = write_screen_records(records, tmp_path / "quality_screen.jsonl")

    truth_rows = [
        truth("a.png", condition="heavy", blur=True),
        truth("b.png", condition="clean"),
        truth("c.png", condition="moderate", shadow=True),
    ]
    gt_path = tmp_path / "quality_ground_truth.jsonl"
    gt_path.write_text("\n".join(json.dumps(row) for row in truth_rows) + "\n")

    report_path = stage_run(
        screen_path,
        gt_path,
        tmp_path / "eval",
        prompt_file=Path("prompts/quality_screen.yaml"),
        variant="quality_screen_v5",
        condition_to_level=CONDITION_TO_LEVEL,
    )

    report = json.loads(report_path.read_text())
    assert report["counts"] == {
        "total": 3,
        "scored": 2,
        "malformed": 1,
        "missing": 0,
        "think_drift": 0,
    }
    assert report["per_criterion"]["blur"]["true_positives"] == 1
    assert report["overall_confusion"]["HEAVY->HEAVY"] == 1


def test_the_stage_scores_with_the_variants_own_polarity(tmp_path):
    """End to end with a good-phrased criterion: the stage must read polarity
    from the prompt that produced the answers, not assume defect-phrasing."""
    import yaml as _yaml

    prompt = tmp_path / "inverted.yaml"
    prompt.write_text(
        _yaml.safe_dump(
            {
                "prompts": {
                    "inverted": {
                        # blur asks "is it perfectly sharp?" -> NO means blurred
                        "evidence": {c: (c != "blur") for c in CRITERIA},
                        "overall_levels": ["NONE", "MODERATE", "HEAVY"],
                        "prompt": "x",
                    }
                }
            }
        )
    )

    screen = tmp_path / "quality_screen.jsonl"
    screen.write_text(
        json.dumps(
            {
                "image_name": "a.png",
                "answers": {c: (c != "blur") for c in CRITERIA},
                "overall": "HEAVY",
                "malformed": False,
                "malformed_reason": None,
                "think_drift": False,
            }
        )
        + "\n"
    )
    gt = tmp_path / "gt.jsonl"
    gt.write_text(json.dumps(truth("a.png", condition="heavy", **dict.fromkeys(CRITERIA, True))) + "\n")

    report_path = stage_run(
        screen,
        gt,
        tmp_path / "eval",
        prompt_file=prompt,
        variant="inverted",
        condition_to_level=CONDITION_TO_LEVEL,
    )

    report = json.loads(report_path.read_text())
    assert report["per_criterion"]["blur"]["true_positives"] == 1, "NO on a good-phrased question"
    assert report["per_criterion"]["shadow"]["true_positives"] == 1, "YES on a defect-phrased one"


def test_an_image_the_screen_never_reached_is_reported_as_missing(tmp_path):
    """A crashed or interrupted classify pod must show up as a gap, not as a
    smaller corpus."""
    records = run_quality_screen(
        ["/data/a.png"],
        infer=fake_infer({"a.png": response(blur=True, overall="HEAVY")}),
        vocabulary=VOCABULARY,
    )
    screen_path = write_screen_records(records, tmp_path / "quality_screen.jsonl")

    truth_rows = [truth("a.png", condition="heavy", blur=True), truth("b.png", condition="clean")]
    gt_path = tmp_path / "gt.jsonl"
    gt_path.write_text("\n".join(json.dumps(row) for row in truth_rows) + "\n")

    report_path = stage_run(
        screen_path,
        gt_path,
        tmp_path / "eval",
        prompt_file=Path("prompts/quality_screen.yaml"),
        variant="quality_screen_v5",
        condition_to_level=CONDITION_TO_LEVEL,
    )

    counts = json.loads(report_path.read_text())["counts"]
    assert counts["missing"] == 1
    assert counts["total"] == 2
