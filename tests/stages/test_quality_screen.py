"""The quality-screen stage: run the prompt over images, record what came back.

Inference is injected, so every test here runs on CPU against fake responses.
That is deliberate -- the stage's job is bookkeeping (one record per image,
malformed responses preserved rather than dropped, raw text kept for audit),
and none of that needs a GPU to verify. The model's actual judgement is the
only part that does, and it is the last step in the plan.
"""

import json
from pathlib import Path

import pytest

from common.quality_screen_parser import ScreenVocabulary
from stages.quality_screen import run_quality_screen, write_screen_records

CRITERIA = ["blur", "shadow", "crease", "faded", "tilt", "speckle"]
LEVELS = ["NONE", "MODERATE", "HEAVY"]
VOCABULARY = ScreenVocabulary(criteria=CRITERIA, overall_levels=LEVELS, prompt="ask the questions")


def response(*, overall="MODERATE", **answers):
    """Build a well-formed model response."""
    lines = [
        f"{i}. {name.upper()}: {'YES' if answers.get(name, False) else 'NO'}"
        for i, name in enumerate(CRITERIA, start=1)
    ]
    lines.append(f"{len(CRITERIA) + 1}. OVERALL: {overall}")
    return "\n".join(lines)


def fake_infer(by_name):
    """An inference callable returning canned text, keyed by image name."""

    def _infer(image_paths: list[str], prompt: str) -> list[str]:
        assert prompt == VOCABULARY.prompt, "the stage must send the prompt it was given"
        return [by_name[Path(path).name] for path in image_paths]

    return _infer


def test_writes_one_record_per_image():
    images = ["/data/a.png", "/data/b.png"]
    responses = {"a.png": response(blur=True), "b.png": response(shadow=True)}

    records = run_quality_screen(images, infer=fake_infer(responses), vocabulary=VOCABULARY)

    assert [r["image_name"] for r in records] == ["a.png", "b.png"]
    assert records[0]["answers"]["blur"] is True
    assert records[1]["answers"]["shadow"] is True
    assert records[0]["malformed"] is False


def test_a_malformed_response_still_produces_a_record():
    """Dropping it would shrink the denominator and flatter the score: the
    image would vanish between the corpus and the report with nothing saying
    it had ever been screened."""
    images = ["/data/a.png", "/data/b.png"]
    responses = {"a.png": response(blur=True), "b.png": "I cannot help with that."}

    records = run_quality_screen(images, infer=fake_infer(responses), vocabulary=VOCABULARY)

    assert len(records) == 2
    assert records[1]["malformed"] is True
    assert records[1]["answers"] is None
    assert records[1]["malformed_reason"]


def test_every_record_keeps_the_raw_response():
    """So a malformed verdict can be argued with, or a parser bug found, after
    the GPU has been released."""
    images = ["/data/a.png"]
    raw = "I cannot help with that."

    records = run_quality_screen(images, infer=fake_infer({"a.png": raw}), vocabulary=VOCABULARY)

    assert records[0]["raw_response"] == raw


def test_reasoning_drift_is_carried_onto_the_record():
    """The rate is the reason v5 was worded the way it was; it has to survive
    into the output to be measurable."""
    images = ["/data/a.png"]
    drifted = f"<think>hmm</think>\n{response(blur=True)}"

    records = run_quality_screen(images, infer=fake_infer({"a.png": drifted}), vocabulary=VOCABULARY)

    assert records[0]["think_drift"] is True
    assert records[0]["malformed"] is False


def test_records_follow_input_order_not_response_order():
    """Records are paired positionally, so order is load-bearing."""
    images = ["/data/b.png", "/data/a.png"]
    responses = {"a.png": response(blur=True), "b.png": response(shadow=True)}

    records = run_quality_screen(images, infer=fake_infer(responses), vocabulary=VOCABULARY)

    assert [r["image_name"] for r in records] == ["b.png", "a.png"]
    assert records[0]["answers"]["shadow"] is True


def test_a_response_count_mismatch_fails_fast(assert_diagnostic_error):
    """Silently zipping to the shorter list would misalign every record after
    the gap -- each image scored against another image's answers."""

    def short_infer(image_paths, prompt):
        return [response(blur=True)]

    with pytest.raises(ValueError) as exc_info:
        run_quality_screen(["/data/a.png", "/data/b.png"], infer=short_infer, vocabulary=VOCABULARY)

    assert_diagnostic_error(str(exc_info.value))


def test_no_images_produces_no_records_and_no_inference():
    def refuse(image_paths, prompt):
        raise AssertionError("inference must not be called for an empty image list")

    assert run_quality_screen([], infer=refuse, vocabulary=VOCABULARY) == []


def test_records_round_trip_through_the_output_file(tmp_path):
    images = ["/data/a.png", "/data/b.png"]
    responses = {"a.png": response(blur=True), "b.png": "nonsense"}
    records = run_quality_screen(images, infer=fake_infer(responses), vocabulary=VOCABULARY)

    path = write_screen_records(records, tmp_path / "out" / "quality_screen.jsonl")
    loaded = [json.loads(line) for line in path.read_text().splitlines()]

    assert loaded == records
