"""Score the image-quality screen against the corpus's defect labels."""

from dataclasses import dataclass

from common.quality_screen_parser import QualityResponse


@dataclass(frozen=True)
class CriterionScore:
    """Counts and rates for one defect criterion.

    Attributes:
        true_positives: Defect present and predicted present.
        false_positives: Defect absent but predicted present.
        false_negatives: Defect present but predicted absent.
    """

    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float | None:
        """Share of predicted defects that were real.

        `None` when the model never predicted this defect: there is no
        denominator, and reporting 1.0 would claim a perfect score on a
        criterion that was never tested while 0.0 would claim a failure.
        Neither is supported by the data, and either would be averaged into a
        headline number as though it were measured.
        """
        claimed = self.true_positives + self.false_positives
        return self.true_positives / claimed if claimed else None

    @property
    def recall(self) -> float | None:
        """Share of real defects that were found.

        `None` only when the defect never occurs in the truth. A model that
        missed every real defect scores a genuine 0.0, which is a measurement,
        not an absence of one.
        """
        present = self.true_positives + self.false_negatives
        return self.true_positives / present if present else None

    @property
    def f1(self) -> float | None:
        """Harmonic mean of precision and recall, or `None` if either is."""
        if self.precision is None or self.recall is None:
            return None
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)


@dataclass(frozen=True)
class QualityScore:
    """A whole run's result.

    Attributes:
        per_criterion: Criterion name -> its counts and rates.
        scored: Images whose response was read and compared.
        malformed: Images whose response could not be read. Counted rather than
            scored: an unreadable response is a formatting failure, and
            charging it to the model's judgement as six wrong answers hides
            which of the two went wrong.
        missing: Images in the truth with no prediction at all -- crashed,
            skipped, or never run. Counted so a shortfall cannot quietly shrink
            the denominator and flatter the result.
        total: Images in the ground truth.
    """

    per_criterion: dict[str, CriterionScore]
    by_document_type: dict[str, dict[str, CriterionScore]]
    overall_confusion: dict[tuple[str, str], int]
    scored: int
    malformed: int
    missing: int
    total: int


def score_quality_screen(
    predictions: dict[str, QualityResponse],
    truths: list[dict],
    *,
    criteria: list[str],
    condition_to_level: dict[str, str],
) -> QualityScore:
    """Score one run.

    Args:
        predictions: Image name -> the response read for it.
        truths: The corpus's quality ground-truth records.
        criteria: Criterion names to score, in prompt order.
        condition_to_level: Generator condition -> the OVERALL level that
            answers it. Passed in rather than assumed: the two vocabularies are
            a contract between the corpus and the prompt.

    Returns:
        The run's scores.
    """
    # (record, answers, overall) with both narrowed at the point of admission.
    # A response is only comparable if it actually carries answers, so nothing
    # downstream has to re-check and no None can reach the arithmetic.
    comparable: list[tuple[dict, dict[str, bool], str]] = []
    malformed = 0
    missing = 0

    for record in truths:
        prediction = predictions.get(record["filename"])
        if prediction is None:
            missing += 1
        elif prediction.malformed or prediction.answers is None or prediction.overall is None:
            # The second and third conditions should be unreachable: the reader
            # never returns readable-but-empty. Counted rather than trusted,
            # because a contract violation from a future caller should land in
            # the malformed tally instead of failing partway through a run.
            malformed += 1
        else:
            comparable.append((record, prediction.answers, prediction.overall))

    per_criterion = _score_criteria(comparable, criteria)

    document_types = sorted({record["document_type"] for record, _answers, _overall in comparable})
    by_document_type = {
        doc_type: _score_criteria(
            [row for row in comparable if row[0]["document_type"] == doc_type], criteria
        )
        for doc_type in document_types
    }

    overall_confusion: dict[tuple[str, str], int] = {}
    for record, _answers, overall in comparable:
        condition = record["condition"]
        if condition not in condition_to_level:
            raise ValueError(
                f"Quality ground truth uses a condition the scorer cannot map.\n"
                f"  What:        condition {condition!r} has no OVERALL level, so its rows "
                f"could not be scored.\n"
                f"  Where:       the condition_to_level mapping passed to "
                f"score_quality_screen; declared conditions are {sorted(condition_to_level)}.\n"
                f"  Expected:    every condition in the ground truth to name one of the "
                f"prompt's OVERALL levels.\n"
                f"  How to fix:  add {condition!r} to the mapping, or regenerate the corpus "
                f"so its conditions match the prompt's levels."
            )
        key = (condition_to_level[condition], overall)
        overall_confusion[key] = overall_confusion.get(key, 0) + 1

    return QualityScore(
        per_criterion=per_criterion,
        by_document_type=by_document_type,
        overall_confusion=overall_confusion,
        scored=len(comparable),
        malformed=malformed,
        missing=missing,
        total=len(truths),
    )


def _score_criteria(
    comparable: list[tuple[dict, dict[str, bool], str]], criteria: list[str]
) -> dict[str, CriterionScore]:
    """Count each criterion over one slice of the comparable rows."""
    scores: dict[str, CriterionScore] = {}
    for name in criteria:
        tp = fp = fn = 0
        for record, answers, _overall in comparable:
            actual = record["defects"][name]
            guess = answers[name]
            if guess and actual:
                tp += 1
            elif guess and not actual:
                fp += 1
            elif actual and not guess:
                fn += 1
        scores[name] = CriterionScore(tp, fp, fn)
    return scores
