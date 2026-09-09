"""Tests for the per-field F1 dispersion report.

Answers the gap named in plans/2026-08-13-field-f1-reporting.md: we publish
one number per field with no dispersion, no interval and no sample size.
"""

import math

from common.field_report import field_distributions


def _record(**field_f1: float) -> dict:
    """An eval record shaped like extraction_evaluator's payload."""
    return {"field_scores": {name: {"f1_score": v} for name, v in field_f1.items()}}


def test_mean_and_n_per_field() -> None:
    rows = field_distributions([_record(TOTAL_AMOUNT=1.0), _record(TOTAL_AMOUNT=0.0)])

    total = next(r for r in rows if r.field == "TOTAL_AMOUNT")
    assert total.mean == 0.5
    assert total.n == 2


def test_fields_may_have_different_sample_sizes() -> None:
    """Field sets differ by document type — TRANSACTION_DATES appears only
    on bank statements. Reporting n per field is what makes that legible
    instead of silently averaging over a different denominator."""
    rows = field_distributions(
        [
            _record(SUPPLIER_NAME=1.0, TRANSACTION_DATES=0.5),
            _record(SUPPLIER_NAME=1.0),
        ]
    )

    by_name = {r.field: r for r in rows}
    assert by_name["SUPPLIER_NAME"].n == 2
    assert by_name["TRANSACTION_DATES"].n == 1


def test_interval_narrows_as_the_sample_grows() -> None:
    """The whole point of publishing an interval."""
    few = field_distributions([_record(TOTAL_AMOUNT=v) for v in (1.0, 0.0, 1.0, 0.0)])[0]
    many = field_distributions([_record(TOTAL_AMOUNT=v) for v in (1.0, 0.0) * 50])[0]

    assert (many.ci_high - many.ci_low) < (few.ci_high - few.ci_low)


def test_interval_is_a_mean_interval_not_a_proportion_interval() -> None:
    """Production per-(doc, field) F1 is CONTINUOUS — list fields get
    partial credit — so the interval must be the normal approximation on
    the mean, sd/sqrt(n), not a Wilson interval for a proportion."""
    row = field_distributions([_record(TOTAL_AMOUNT=v) for v in (0.6, 0.4, 0.5, 0.5)])[0]

    expected_half_width = 1.96 * row.sd / math.sqrt(row.n)
    assert math.isclose(row.ci_high - row.mean, expected_half_width, rel_tol=1e-9)


def test_interval_is_clamped_to_the_unit_interval() -> None:
    """F1 cannot exceed 1, so an interval must not claim it might."""
    row = field_distributions([_record(TOTAL_AMOUNT=v) for v in (1.0, 1.0, 0.9)])[0]

    assert row.ci_high <= 1.0
    assert row.ci_low >= 0.0


def test_a_single_document_has_no_dispersion() -> None:
    """One observation gives no spread; it must not crash or invent one."""
    row = field_distributions([_record(TOTAL_AMOUNT=0.7)])[0]

    assert row.n == 1
    assert row.sd == 0.0


def test_weakest_field_is_reported_first() -> None:
    """The table is read to find what to fix."""
    rows = field_distributions([_record(GOOD=1.0, BAD=0.1, MIDDLING=0.5)])

    assert [r.field for r in rows] == ["BAD", "MIDDLING", "GOOD"]


def test_records_without_field_scores_are_skipped() -> None:
    """Errored documents carry no field_scores; they must not be counted
    as zeros, which would understate every field."""
    rows = field_distributions([_record(TOTAL_AMOUNT=1.0), {"error": "boom"}])

    assert rows[0].n == 1
    assert rows[0].mean == 1.0
