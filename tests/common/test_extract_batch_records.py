"""Batched and per-image extraction must write the SAME record shape.

raw_extractions.jsonl is read by the clean stage and by resume
(read_completed_images). If the batch path wrote a different shape, a run
would clean inconsistently and resume would mis-skip — and the difference
would only show up downstream, far from its cause.
"""

from typing import Any

import pytest

from common.vllm_dp_workers import _extract_batch_records

_EXPECTED_KEYS = {
    "image_name",
    "image_path",
    "document_type",
    "raw_response",
    "processing_time",
    "prompt_used",
    "error",
}


def _batch(count: int = 3) -> list[dict[str, Any]]:
    return [
        {
            "image_name": f"CASE{i:03d}_receipt.png",
            "image_path": f"/data/CASE{i:03d}_receipt.png",
            "document_type": "RECEIPT",
        }
        for i in range(count)
    ]


class _Processor:
    """Stands in for DocumentOrchestrator."""

    def __init__(self, *, batch_fails: bool = False, bad_image: str | None = None) -> None:
        self.batch_fails = batch_fails
        self.bad_image = bad_image
        self.batch_calls = 0
        self.single_calls = 0

    def extract_batch(self, image_paths, classification_infos, verbose=False):
        self.batch_calls += 1
        if self.batch_fails:
            raise RuntimeError("CUDA out of memory")
        return [{"raw_response": f"RESPONSE {p}"} for p in image_paths]

    def process_document_aware(self, image_path, classification, verbose=False):
        self.single_calls += 1
        if self.bad_image and self.bad_image in image_path:
            raise RuntimeError("genuinely bad image")
        return {"raw_response": f"RESPONSE {image_path}"}


def test_batched_records_carry_exactly_the_per_image_keys() -> None:
    records = _extract_batch_records(_Processor(), _batch(), verbose=False)

    for record in records:
        assert set(record) == _EXPECTED_KEYS


def test_one_record_per_document_in_submission_order() -> None:
    """Responses are matched by position; an off-by-one would attach every
    extraction to the wrong image."""
    batch = _batch(3)

    records = _extract_batch_records(_Processor(), batch, verbose=False)

    assert [r["image_name"] for r in records] == [c["image_name"] for c in batch]
    for record in records:
        assert record["image_path"] in record["raw_response"]


def test_a_whole_batch_is_one_engine_call() -> None:
    processor = _Processor()

    _extract_batch_records(processor, _batch(3), verbose=False)

    assert processor.batch_calls == 1
    assert processor.single_calls == 0


def test_a_failed_batch_retries_one_at_a_time() -> None:
    """An out-of-memory must cost only the document that genuinely fails."""
    processor = _Processor(batch_fails=True, bad_image="CASE001")

    records = _extract_batch_records(processor, _batch(3), verbose=False)

    assert processor.single_calls == 3
    assert len(records) == 3
    failed = [r for r in records if r["error"]]
    assert [r["image_name"] for r in failed] == ["CASE001_receipt.png"]


def test_a_failed_document_still_produces_a_record() -> None:
    """Dropping it would shrink the denominator and silently improve the score."""
    processor = _Processor(batch_fails=True, bad_image="CASE001")

    records = _extract_batch_records(processor, _batch(3), verbose=False)

    bad = next(r for r in records if r["image_name"] == "CASE001_receipt.png")
    assert bad["raw_response"] == ""
    assert bad["prompt_used"] == "error"


def test_a_short_response_list_falls_back_rather_than_misaligning() -> None:
    """Fewer results than images means position matching is unsafe."""

    class _ShortProcessor(_Processor):
        def extract_batch(self, image_paths, classification_infos, verbose=False):
            self.batch_calls += 1
            return [{"raw_response": "only one"}]

    processor = _ShortProcessor()
    records = _extract_batch_records(processor, _batch(3), verbose=False)

    assert processor.single_calls == 3
    assert [r["image_name"] for r in records] == [c["image_name"] for c in _batch(3)]


@pytest.mark.parametrize("count", [1, 2, 5])
def test_every_document_appears_exactly_once(count: int) -> None:
    records = _extract_batch_records(_Processor(), _batch(count), verbose=False)

    assert len(records) == count
    assert len({r["image_name"] for r in records}) == count
