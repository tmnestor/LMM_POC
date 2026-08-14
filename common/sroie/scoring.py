"""Scoring of SROIE extractions under two explicit match policies.

Both numbers are reported for every run:

* ``STRICT`` follows the official SROIE protocol — the extracted string
  must equal the answer key, forgiving only case and whitespace runs.
* ``LENIENT`` asks whether the model read the receipt — dates compare as
  dates, totals as amounts, and text ignores punctuation.

Publishing one number alone hides the gap between them. That gap is not
noise: in the earlier 626-image run it was 8.8 F1 points, and it fell
almost entirely on two fields.
"""

from dataclasses import dataclass, field
from enum import Enum

from common.sroie.ground_truth import SROIE_FIELDS, SroieRecord
from common.sroie.normalise import (
    SroieNormalisationError,
    normalise_date,
    normalise_text_lenient,
    normalise_text_strict,
    normalise_total,
)


class MatchPolicy(Enum):
    """How closely an extraction must resemble the answer key."""

    STRICT = "strict"
    LENIENT = "lenient"


@dataclass
class FieldCounts:
    """Precision/recall counts for one field across a set of documents."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    def record(self, *, matched: bool, answered: bool) -> None:
        """Fold one document's outcome into the counts.

        Args:
            matched: Whether the extraction equalled the answer key.
            answered: Whether the model supplied a value at all.
        """
        if matched:
            self.true_positives += 1
            return
        # A wrong answer is both a bad prediction and a missed gold value.
        # A declined answer costs recall only — silence is not a claim.
        if answered:
            self.false_positives += 1
        self.false_negatives += 1

    @property
    def precision(self) -> float:
        """Share of supplied answers that were right."""
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        """Share of answer-key values the model recovered."""
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        precision, recall = self.precision, self.recall
        if not (precision + recall):
            return 0.0
        return 2 * precision * recall / (precision + recall)


@dataclass
class PolicyScore:
    """Per-field and overall scores under a single match policy."""

    policy: MatchPolicy
    per_field: dict[str, FieldCounts] = field(default_factory=dict)

    @property
    def overall_f1(self) -> float:
        """Unweighted mean of the per-field F1 scores."""
        if not self.per_field:
            return 0.0
        return sum(counts.f1 for counts in self.per_field.values()) / len(self.per_field)


def _canonical(value: str, field_name: str, policy: MatchPolicy, source: str | None) -> object:
    """Reduce a value to the form the policy compares on."""
    if policy is MatchPolicy.STRICT:
        return normalise_text_strict(value)
    if field_name == "date":
        return normalise_date(value, source=source)
    if field_name == "total":
        return normalise_total(value, source=source)
    return normalise_text_lenient(value)


def field_matches(
    field_name: str,
    ground_truth: str,
    predicted: str,
    *,
    policy: MatchPolicy,
    source: str | None = None,
) -> bool:
    """Decide whether an extraction matches the answer key.

    The two sides are treated differently on purpose. A ground-truth value
    that cannot be canonicalised is a dataset defect and raises. A
    prediction that cannot be canonicalised is simply a wrong answer — one
    malformed model reply must not abort the run.

    Args:
        field_name: One of the four SROIE fields.
        ground_truth: The answer-key value.
        predicted: The model's value.
        policy: Which match policy to apply.
        source: Image id, quoted in ground-truth errors.

    Returns:
        True when the two values agree under this policy.

    Raises:
        SroieNormalisationError: If the ground-truth value is unparseable.
    """
    expected = _canonical(ground_truth, field_name, policy, source)

    try:
        actual = _canonical(predicted, field_name, policy, source)
    except SroieNormalisationError:
        # An unreadable prediction is a wrong answer, not a dataset defect.
        return False

    return bool(expected == actual)


def score_records(
    records: list[SroieRecord],
    predictions: dict[str, dict[str, str]],
    *,
    policy: MatchPolicy,
) -> PolicyScore:
    """Score every record under one policy.

    Args:
        records: Ground-truth records for the split.
        predictions: Image id to parsed field values. A missing image id
            counts as a document the model answered nothing for.
        policy: Which match policy to apply.

    Returns:
        Per-field counts and the overall mean F1.
    """
    score = PolicyScore(policy=policy, per_field={name: FieldCounts() for name in SROIE_FIELDS})

    for record in records:
        answers = predictions.get(record.image_id, {})
        for field_name in SROIE_FIELDS:
            expected = getattr(record, field_name)
            actual = answers.get(field_name)
            matched = bool(
                actual
                and field_matches(
                    field_name,
                    expected,
                    actual,
                    policy=policy,
                    source=record.image_id,
                )
            )
            score.per_field[field_name].record(matched=matched, answered=bool(actual))

    return score


__all__ = [
    "FieldCounts",
    "MatchPolicy",
    "PolicyScore",
    "field_matches",
    "score_records",
]
