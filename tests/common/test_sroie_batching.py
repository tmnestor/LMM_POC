"""Tests for batched SROIE execution."""

from pathlib import Path

from common.sroie.ground_truth import SroieRecord
from common.sroie.runner import run_benchmark_batched


def _records(count: int) -> list[SroieRecord]:
    return [
        SroieRecord(
            image_id=f"X{index:03d}",
            image_path=Path(f"X{index:03d}.jpg"),
            company="ACME",
            date="15/01/2019",
            address="27 JALAN",
            total="9.00",
        )
        for index in range(count)
    ]


def _answer(record: SroieRecord) -> str:
    return f"company: {record.company}\ntotal: {record.total}"


def test_records_are_submitted_in_batches_of_the_configured_size() -> None:
    records = _records(7)
    sizes = []

    def generate_batch(batch: list[SroieRecord]) -> list[str]:
        sizes.append(len(batch))
        return [_answer(r) for r in batch]

    run_benchmark_batched(records, generate_batch, batch_size=3)

    assert sizes == [3, 3, 1]


def test_every_record_gets_its_own_response() -> None:
    """Responses are matched by position; an off-by-one here would attach
    every prediction to the wrong receipt."""
    records = _records(5)

    def generate_batch(batch: list[SroieRecord]) -> list[str]:
        return [f"company: {r.image_id}" for r in batch]

    result = run_benchmark_batched(records, generate_batch, batch_size=2)

    for record in records:
        assert result.predictions[record.image_id]["company"] == record.image_id


def test_a_short_response_list_never_misaligns_predictions() -> None:
    """A backend returning fewer responses than submitted must not have
    them matched by position — that would attach answers to the wrong
    receipts. Recovering via individual retry is fine; shifting is not."""
    records = _records(3)

    def generate_batch(batch: list[SroieRecord]) -> list[str]:
        if len(batch) > 1:
            return ["company: X000"]  # one response for three images
        return [f"company: {batch[0].image_id}"]

    result = run_benchmark_batched(records, generate_batch, batch_size=3)

    for record in records:
        assert result.predictions[record.image_id]["company"] == record.image_id


def test_a_short_response_list_that_cannot_be_recovered_errors() -> None:
    """If even the single-image retry misbehaves, the records must be
    recorded as failures rather than silently dropped."""
    records = _records(2)

    result = run_benchmark_batched(records, lambda batch: [], batch_size=2)

    assert set(result.errors) == {"X000", "X001"}
    assert result.predictions["X000"] == {}


def test_a_failed_batch_is_retried_one_at_a_time() -> None:
    """A batch OOM must not cost every receipt in it — only the one that
    actually fails."""
    records = _records(4)
    attempts = {"batched": 0}

    def generate_batch(batch: list[SroieRecord]) -> list[str]:
        if len(batch) > 1:
            attempts["batched"] += 1
            raise RuntimeError("CUDA out of memory")
        if batch[0].image_id == "X002":
            raise RuntimeError("genuinely bad image")
        return [_answer(batch[0])]

    result = run_benchmark_batched(records, generate_batch, batch_size=4)

    assert attempts["batched"] == 1
    assert set(result.errors) == {"X002"}
    assert result.predictions["X000"]["company"] == "ACME"
    assert result.predictions["X003"]["company"] == "ACME"


def test_batch_size_of_one_still_works() -> None:
    records = _records(3)

    result = run_benchmark_batched(records, lambda batch: [_answer(r) for r in batch], batch_size=1)

    assert len(result.predictions) == 3
    assert result.errors == {}
