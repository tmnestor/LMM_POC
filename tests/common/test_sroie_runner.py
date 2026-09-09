"""Tests for the SROIE benchmark loop, independent of any model."""

from pathlib import Path

from common.sroie.ground_truth import SroieRecord
from common.sroie.runner import (
    benchmark_from_worker_responses,
    inference_seconds_from_responses,
    run_benchmark,
)


def _records(count: int = 3) -> list[SroieRecord]:
    return [
        SroieRecord(
            image_id=f"X00{index}",
            image_path=Path(f"X00{index}.jpg"),
            company="ACME SDN BHD",
            date="15/01/2019",
            address="27, JALAN DEDAP 13, JOHOR.",
            total="193.00",
        )
        for index in range(count)
    ]


def _answer_for(record: SroieRecord) -> str:
    return (
        f"company: {record.company}\ndate: {record.date}\naddress: {record.address}\ntotal: {record.total}"
    )


def test_generates_once_per_record() -> None:
    records = _records()
    seen = []

    def generate(record: SroieRecord) -> str:
        seen.append(record.image_id)
        return _answer_for(record)

    result = run_benchmark(records, generate)

    assert seen == ["X000", "X001", "X002"]
    assert len(result.predictions) == 3


def test_parses_each_response_into_fields() -> None:
    records = _records(1)

    result = run_benchmark(records, _answer_for)

    assert result.predictions["X000"]["company"] == "ACME SDN BHD"
    assert result.predictions["X000"]["total"] == "193.00"


def test_keeps_the_raw_response_for_every_record() -> None:
    """A disputed score is settled by reading what the model actually said."""
    records = _records(1)

    result = run_benchmark(records, _answer_for)

    assert result.raw_responses["X000"].startswith("company: ACME")


def test_one_failing_image_does_not_abort_the_run() -> None:
    """Aborting at image 300 of 347 throws away an hour of GPU time."""
    records = _records(3)

    def generate(record: SroieRecord) -> str:
        if record.image_id == "X001":
            raise RuntimeError("CUDA hiccup")
        return _answer_for(record)

    result = run_benchmark(records, generate)

    assert len(result.predictions) == 3
    assert result.predictions["X001"] == {}
    assert result.predictions["X002"]["company"] == "ACME SDN BHD"


def test_failures_are_counted_and_named() -> None:
    """A run with inference errors must not report a clean score as if
    every image had been read."""
    records = _records(2)

    def generate(record: SroieRecord) -> str:
        if record.image_id == "X000":
            raise RuntimeError("CUDA hiccup")
        return _answer_for(record)

    result = run_benchmark(records, generate)

    assert result.errors == {"X000": "CUDA hiccup"}


def test_a_clean_run_reports_no_errors() -> None:
    result = run_benchmark(_records(2), _answer_for)

    assert result.errors == {}


def test_inference_seconds_is_the_slowest_worker_not_the_sum() -> None:
    """Workers run concurrently, so wall-clock inference is the slowest
    worker's total — summing them would report compute, not elapsed."""
    responses = [
        {"image_id": "A", "gpu_id": 0, "elapsed": 10.0},
        {"image_id": "B", "gpu_id": 0, "elapsed": 12.0},
        {"image_id": "C", "gpu_id": 1, "elapsed": 5.0},
        {"image_id": "D", "gpu_id": 1, "elapsed": 6.0},
    ]

    assert inference_seconds_from_responses(responses, fallback=999.0) == 22.0


def test_inference_seconds_falls_back_when_workers_did_not_time_themselves() -> None:
    """Never silently report 0.0 seconds — that would show as an absurd
    throughput rather than as missing data."""
    responses = [{"image_id": "A", "raw_response": "x"}]

    assert inference_seconds_from_responses(responses, fallback=99.5) == 99.5


def test_worker_responses_rebuild_the_same_run() -> None:
    """The data-parallel path returns raw responses from N GPU workers;
    reassembling them must give exactly what the serial loop would."""
    records = _records(2)
    responses = [{"image_id": r.image_id, "raw_response": _answer_for(r), "error": None} for r in records]

    rebuilt = benchmark_from_worker_responses(records, responses)
    serial = run_benchmark(records, _answer_for)

    assert rebuilt.predictions == serial.predictions
    assert rebuilt.raw_responses == serial.raw_responses
    assert rebuilt.errors == serial.errors


def test_a_worker_error_is_carried_through() -> None:
    records = _records(1)
    responses = [{"image_id": "X000", "raw_response": "", "error": "CUDA hiccup"}]

    rebuilt = benchmark_from_worker_responses(records, responses)

    assert rebuilt.errors == {"X000": "CUDA hiccup"}
    assert rebuilt.predictions["X000"] == {}


def test_a_record_missing_from_worker_output_is_an_error_not_a_gap() -> None:
    """A worker that dies mid-partition returns fewer records than it was
    given. Dropping those silently would shrink the denominator and make
    the score look better than the run earned."""
    records = _records(3)
    responses = [
        {"image_id": r.image_id, "raw_response": _answer_for(r), "error": None} for r in records[:2]
    ]

    rebuilt = benchmark_from_worker_responses(records, responses)

    assert set(rebuilt.predictions) == {"X000", "X001", "X002"}
    assert "X002" in rebuilt.errors
    assert rebuilt.predictions["X002"] == {}
