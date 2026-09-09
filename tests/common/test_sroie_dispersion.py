"""Tests for dispersion reporting: per-field intervals, per-document F1."""

from pathlib import Path

from common.sroie.ground_truth import SroieRecord
from common.sroie.scoring import (
    MatchPolicy,
    per_document_f1,
    per_field_document_scores,
    score_records,
    wilson_interval,
)


def _record(image_id: str) -> SroieRecord:
    return SroieRecord(
        image_id=image_id,
        image_path=Path(f"{image_id}.jpg"),
        company="ACME",
        date="15/01/2019",
        address="27 JALAN",
        total="9.00",
    )


def test_interval_brackets_the_observed_proportion() -> None:
    low, high = wilson_interval(successes=296, trials=347)

    assert low < 296 / 347 < high


def test_a_smaller_sample_gives_a_wider_interval() -> None:
    """The whole point of publishing an interval: 8/10 is not 800/1000."""
    narrow = wilson_interval(successes=800, trials=1000)
    wide = wilson_interval(successes=8, trials=10)

    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_a_perfect_score_does_not_claim_certainty() -> None:
    """347/347 is not proof of 1.0 — the lower bound must stay below 1."""
    low, high = wilson_interval(successes=347, trials=347)

    assert low < 1.0
    assert high == 1.0


def test_interval_of_no_trials_is_the_whole_range() -> None:
    assert wilson_interval(successes=0, trials=0) == (0.0, 1.0)


def test_per_field_scores_give_one_value_per_document() -> None:
    """The mean and SD another team publishes are taken over documents,
    so each field needs its per-document score list."""
    records = [_record("X001"), _record("X002")]
    predictions = {
        "X001": {"company": "ACME", "total": "9.00"},
        "X002": {"company": "WRONG", "total": "9.00"},
    }

    by_field = per_field_document_scores(records, predictions, policy=MatchPolicy.STRICT)

    assert by_field["company"] == [1.0, 0.0]
    assert by_field["total"] == [1.0, 1.0]
    assert by_field["address"] == [0.0, 0.0]


def test_per_field_mean_equals_the_pooled_f1_when_every_field_is_answered() -> None:
    """Sanity tie-back: with one value per field per document, the mean of
    the per-document scores is the same number the pooled counts give."""
    records = [_record("X001"), _record("X002")]
    predictions = {
        "X001": {"company": "ACME"},
        "X002": {"company": "WRONG"},
    }

    by_field = per_field_document_scores(records, predictions, policy=MatchPolicy.STRICT)
    pooled = score_records(records, predictions, policy=MatchPolicy.STRICT)

    mean = sum(by_field["company"]) / len(by_field["company"])
    assert mean == pooled.per_field["company"].f1


def test_per_document_f1_is_the_share_of_fields_matched() -> None:
    """Two of four fields right is 0.5 for that receipt."""
    record = _record("X001")
    predictions = {"X001": {"company": "ACME", "date": "15/01/2019"}}

    scores = per_document_f1([record], predictions, policy=MatchPolicy.STRICT)

    assert scores == [0.5]


def test_per_document_f1_returns_one_score_per_receipt() -> None:
    records = [_record("X001"), _record("X002")]
    predictions = {"X001": {"company": "ACME"}}

    scores = per_document_f1(records, predictions, policy=MatchPolicy.STRICT)

    assert scores == [0.25, 0.0]
