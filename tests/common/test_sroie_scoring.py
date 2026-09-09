"""Tests for SROIE scoring under the strict and lenient policies."""

from pathlib import Path

import pytest

from common.sroie.ground_truth import SroieRecord
from common.sroie.normalise import SroieNormalisationError
from common.sroie.scoring import FieldCounts, MatchPolicy, field_matches, score_records


def _record(image_id: str = "X001") -> SroieRecord:
    return SroieRecord(
        image_id=image_id,
        image_path=Path(f"{image_id}.jpg"),
        company="ACME SDN BHD",
        date="15/01/2019",
        address="27, JALAN DEDAP 13, JOHOR.",
        total="193.00",
    )


def test_identical_values_match_under_both_policies() -> None:
    for policy in MatchPolicy:
        assert field_matches("company", "ACME SDN BHD", "ACME SDN BHD", policy=policy)


def test_case_and_spacing_are_free_under_both_policies() -> None:
    for policy in MatchPolicy:
        assert field_matches("company", "ACME  SDN BHD", "acme sdn bhd", policy=policy)


def test_punctuation_difference_separates_the_two_policies() -> None:
    """This is the whole reason both numbers are reported."""
    gt = "27, JALAN DEDAP 13, JOHOR."
    predicted = "27,JALAN DEDAP 13, JOHOR"

    assert not field_matches("address", gt, predicted, policy=MatchPolicy.STRICT)
    assert field_matches("address", gt, predicted, policy=MatchPolicy.LENIENT)


def test_date_formatting_difference_separates_the_two_policies() -> None:
    """The old scorer counted this as a model failure. It is a format
    difference: both sides name the same day."""
    assert not field_matches("date", "12-01-19", "12/01/2019", policy=MatchPolicy.STRICT)
    assert field_matches("date", "12-01-19", "12/01/2019", policy=MatchPolicy.LENIENT)


def test_different_dates_never_match() -> None:
    for policy in MatchPolicy:
        assert not field_matches("date", "12/01/2019", "13/01/2019", policy=policy)


def test_currency_prefix_on_a_total_separates_the_two_policies() -> None:
    assert not field_matches("total", "8.20", "RM8.20", policy=MatchPolicy.STRICT)
    assert field_matches("total", "8.20", "RM8.20", policy=MatchPolicy.LENIENT)


def test_different_totals_never_match() -> None:
    for policy in MatchPolicy:
        assert not field_matches("total", "8.20", "8.21", policy=policy)


def test_unparseable_prediction_is_a_miss_not_a_crash() -> None:
    """One malformed model answer must not abort a 347-image run."""
    assert not field_matches("date", "15/01/2019", "sometime tuesday", policy=MatchPolicy.LENIENT)
    assert not field_matches("total", "8.20", "about eight", policy=MatchPolicy.LENIENT)


def test_unparseable_ground_truth_still_raises() -> None:
    """A corrupt answer key is a dataset defect and must stop the run."""
    with pytest.raises(SroieNormalisationError):
        field_matches("date", "sometime tuesday", "15/01/2019", policy=MatchPolicy.LENIENT)


def test_correct_answer_counts_a_true_positive() -> None:
    counts = FieldCounts()
    counts.record(matched=True, answered=True)

    assert (counts.true_positives, counts.false_positives, counts.false_negatives) == (1, 0, 0)


def test_wrong_answer_counts_both_a_false_positive_and_a_false_negative() -> None:
    """A wrong answer is both a bad prediction and a missed gold value."""
    counts = FieldCounts()
    counts.record(matched=False, answered=True)

    assert (counts.true_positives, counts.false_positives, counts.false_negatives) == (0, 1, 1)


def test_declined_answer_counts_only_a_false_negative() -> None:
    """Saying nothing is not the same as saying something wrong: it costs
    recall but must not cost precision."""
    counts = FieldCounts()
    counts.record(matched=False, answered=False)

    assert (counts.true_positives, counts.false_positives, counts.false_negatives) == (0, 0, 1)


def test_f1_of_a_perfect_field_is_one() -> None:
    counts = FieldCounts()
    counts.record(matched=True, answered=True)

    assert counts.f1 == 1.0


def test_f1_of_a_field_never_answered_is_zero() -> None:
    counts = FieldCounts()
    counts.record(matched=False, answered=False)

    assert counts.f1 == 0.0


def test_scoring_a_perfect_run_gives_one_across_every_field() -> None:
    record = _record()
    predictions = {
        record.image_id: {
            "company": record.company,
            "date": record.date,
            "address": record.address,
            "total": record.total,
        }
    }

    score = score_records([record], predictions, policy=MatchPolicy.STRICT)

    assert score.overall_f1 == 1.0


def test_a_document_with_no_prediction_scores_zero_not_an_error() -> None:
    """A model that returns nothing for an image must be counted, not
    skipped — skipping it would shrink the denominator."""
    score = score_records([_record()], {}, policy=MatchPolicy.LENIENT)

    assert score.overall_f1 == 0.0
    assert score.per_field["company"].false_negatives == 1
    assert score.per_field["company"].false_positives == 0


def test_policies_disagree_on_the_same_run() -> None:
    """The two policies must be capable of producing different numbers for
    identical input, or reporting both is pointless."""
    record = _record()
    predictions = {
        record.image_id: {
            "company": record.company,
            "date": "15-01-19",
            "address": "27,JALAN DEDAP 13, JOHOR",
            "total": "RM193.00",
        }
    }

    strict = score_records([record], predictions, policy=MatchPolicy.STRICT)
    lenient = score_records([record], predictions, policy=MatchPolicy.LENIENT)

    assert strict.overall_f1 == 0.25  # company only
    assert lenient.overall_f1 == 1.0


def test_every_scored_field_appears_in_the_report() -> None:
    score = score_records([_record()], {}, policy=MatchPolicy.STRICT)

    assert set(score.per_field) == {"company", "date", "address", "total"}
