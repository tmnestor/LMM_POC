"""Tests for the SROIE benchmark report artefacts."""

import csv
import json
from pathlib import Path

from common.sroie.ground_truth import SroieRecord
from common.sroie.report import (
    execution_summary_rows,
    write_per_image_csv,
    write_summary_json,
)
from common.sroie.scoring import MatchPolicy, score_records


def _record(image_id: str = "X001") -> SroieRecord:
    return SroieRecord(
        image_id=image_id,
        image_path=Path(f"{image_id}.jpg"),
        company="ACME SDN BHD",
        date="15/01/2019",
        address="27, JALAN DEDAP 13, JOHOR.",
        total="193.00",
    )


_PREDICTIONS = {"X001": {"company": "ACME SDN BHD", "date": "15-01-19", "total": "RM193.00"}}


def _summary_rows(**overrides):
    records = [_record()]
    scores = {policy: score_records(records, _PREDICTIONS, policy=policy) for policy in MatchPolicy}
    kwargs = {
        "image_count": 347,
        "elapsed_seconds": 1003.69,
        "scores": scores,
        "output_dir": Path("/data/sroie/output_internvl3"),
        "execution_mode": "data-parallel (2 GPUs)",
        "failed_images": 0,
        "wall_clock_seconds": None,
    }
    kwargs.update(overrides)
    return dict(execution_summary_rows(**kwargs))


def test_summary_reports_throughput_in_images_per_minute() -> None:
    """347 images in 1003.69s is 20.74/min."""
    assert _summary_rows()["Throughput"] == "20.74 images/min"


def test_summary_reports_the_headline_metrics() -> None:
    rows = _summary_rows()

    assert rows["Images Processed"] == "347"
    assert rows["Inference Time"] == "1003.7s"
    assert rows["Execution Mode"] == "data-parallel (2 GPUs)"
    assert rows["Output Directory"] == "/data/sroie/output_internvl3"
    assert "Strict F1" in rows
    assert "Lenient F1" in rows


def test_wall_clock_is_shown_when_it_differs_from_inference_time() -> None:
    """On the data-parallel path, wall clock includes engine startup while
    Inference Time does not. Hiding the gap would make the two look like
    one number that quietly changed meaning between execution modes."""
    rows = _summary_rows(wall_clock_seconds=1180.0)

    assert rows["Total Wall Clock"] == "1180.0s"
    assert rows["Inference Time"] == "1003.7s"


def test_wall_clock_row_is_omitted_on_the_single_engine_path() -> None:
    """There it is the same measurement, so a second row would be noise."""
    assert "Total Wall Clock" not in _summary_rows()


def test_zero_elapsed_does_not_divide_by_zero() -> None:
    """A fully-cached or mocked run must not crash the summary."""
    assert _summary_rows(elapsed_seconds=0.0)["Throughput"] == "0.00 images/min"


def test_failed_images_are_shown_only_when_there_are_some() -> None:
    """A clean run should not carry a 'Failed Images: 0' line, but a run
    with failures must never hide them behind a headline score."""
    assert "Failed Images" not in _summary_rows()
    assert _summary_rows(failed_images=3)["Failed Images"] == "3"


def test_per_image_csv_has_one_row_per_record(tmp_path: Path) -> None:
    path = tmp_path / "per_image.csv"

    write_per_image_csv(path, [_record()], _PREDICTIONS)

    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 1
    assert rows[0]["image_id"] == "X001"


def test_per_image_csv_records_both_policies(tmp_path: Path) -> None:
    """The per-image file is where a disputed score gets checked by hand,
    so it must show which policy accepted which value."""
    path = tmp_path / "per_image.csv"

    write_per_image_csv(path, [_record()], _PREDICTIONS)

    row = next(iter(csv.DictReader(path.open())))
    assert row["date_gt"] == "15/01/2019"
    assert row["date_pred"] == "15-01-19"
    assert row["date_strict"] == "False"
    assert row["date_lenient"] == "True"


def test_per_image_csv_marks_an_unanswered_field(tmp_path: Path) -> None:
    """Address is absent from the prediction; the row must say so rather
    than leave it indistinguishable from a wrong answer."""
    path = tmp_path / "per_image.csv"

    write_per_image_csv(path, [_record()], _PREDICTIONS)

    row = next(iter(csv.DictReader(path.open())))
    assert row["address_pred"] == ""
    assert row["address_strict"] == "False"
    assert row["address_lenient"] == "False"


def test_summary_json_reports_both_policies(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    records = [_record()]
    scores = {policy: score_records(records, _PREDICTIONS, policy=policy) for policy in MatchPolicy}

    write_summary_json(
        path,
        model_name="internvl3-vllm",
        scores=scores,
        records=records,
        predictions=_PREDICTIONS,
        image_count=len(records),
        elapsed_seconds=12.5,
    )

    summary = json.loads(path.read_text())
    assert summary["model"] == "internvl3-vllm"
    assert summary["total_images"] == 1
    assert set(summary["policies"]) == {"strict", "lenient"}
    assert summary["policies"]["lenient"]["per_field"]["date"]["f1"] == 1.0
    assert summary["policies"]["strict"]["per_field"]["date"]["f1"] == 0.0


def test_summary_json_records_throughput(tmp_path: Path) -> None:
    """Throughput belongs in the artefact, not only on the terminal —
    comparing two models' speed should not need the scrollback."""
    path = tmp_path / "summary.json"
    records = [_record()]
    scores = {policy: score_records(records, _PREDICTIONS, policy=policy) for policy in MatchPolicy}

    write_summary_json(
        path,
        model_name="internvl3-vllm",
        scores=scores,
        records=records,
        predictions=_PREDICTIONS,
        image_count=60,
        elapsed_seconds=120.0,
        execution_mode="data-parallel (2 GPUs)",
    )

    summary = json.loads(path.read_text())
    assert summary["throughput_images_per_min"] == 30.0
    assert summary["execution_mode"] == "data-parallel (2 GPUs)"


def test_summary_json_reports_per_field_mean_and_sd(tmp_path: Path) -> None:
    """Another team publishes per-field mean and SD, so ours must be
    directly comparable with theirs."""
    path = tmp_path / "summary.json"
    records = [_record("X001"), _record("X002")]
    predictions = {"X001": dict(_PREDICTIONS["X001"]), "X002": {}}
    scores = {p: score_records(records, predictions, policy=p) for p in MatchPolicy}

    write_summary_json(
        path,
        model_name="internvl3-vllm",
        scores=scores,
        records=records,
        predictions=predictions,
        image_count=len(records),
        elapsed_seconds=12.5,
        execution_mode="single-engine",
    )

    company = json.loads(path.read_text())["policies"]["strict"]["per_field"]["company"]
    assert company["mean"] == 0.5  # right on one of two receipts
    assert company["sd"] == 0.5  # population SD of [1.0, 0.0]
    assert company["n"] == 2
    assert company["ci_low"] < 0.5 < company["ci_high"]


def test_summary_json_reports_the_per_document_distribution(tmp_path: Path) -> None:
    """Per-document F1 is a real distribution, so its mean, median and SD
    each carry information — unlike the per-field SD."""
    path = tmp_path / "summary.json"
    records = [_record("X001"), _record("X002")]
    predictions = {"X001": dict(_PREDICTIONS["X001"]), "X002": {}}
    scores = {p: score_records(records, predictions, policy=p) for p in MatchPolicy}

    write_summary_json(
        path,
        model_name="internvl3-vllm",
        scores=scores,
        records=records,
        predictions=predictions,
        image_count=len(records),
        elapsed_seconds=12.5,
        execution_mode="single-engine",
    )

    per_doc = json.loads(path.read_text())["policies"]["lenient"]["per_document"]
    assert per_doc["n"] == 2
    assert 0.0 <= per_doc["mean"] <= 1.0
    assert "median" in per_doc
    assert "sd" in per_doc


def test_summary_json_keeps_the_raw_counts(tmp_path: Path) -> None:
    """Discarding tp/fp/fn makes any pooled metric uncomputable later
    without re-running the benchmark."""
    path = tmp_path / "summary.json"
    records = [_record()]
    scores = {policy: score_records(records, _PREDICTIONS, policy=policy) for policy in MatchPolicy}

    write_summary_json(
        path,
        model_name="internvl3-vllm",
        scores=scores,
        records=records,
        predictions=_PREDICTIONS,
        image_count=len(records),
        elapsed_seconds=12.5,
    )

    counts = json.loads(path.read_text())["policies"]["lenient"]["per_field"]["address"]
    assert counts["true_positives"] == 0
    assert counts["false_positives"] == 0
    assert counts["false_negatives"] == 1
