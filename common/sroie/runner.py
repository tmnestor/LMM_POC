"""The SROIE benchmark loop.

Model-independent on purpose: the caller supplies a function that turns a
record into a raw model response, so the loop is exercised in tests without
a GPU and the same code runs every model under test.

Inference failures are recorded rather than raised. Aborting at image 300
of 347 discards the work already done, and a per-image error is not a
dataset defect. The caller is expected to surface a non-empty ``errors``
map rather than publish the run as clean.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from common.sroie.ground_truth import SroieRecord
from common.sroie.parse import parse_sroie_response

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkRun:
    """Everything one pass over the split produced."""

    predictions: dict[str, dict[str, str]] = field(default_factory=dict)
    raw_responses: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


def run_benchmark_batched(
    records: list[SroieRecord],
    generate_batch: Callable[[list[SroieRecord]], list[str]],
    *,
    batch_size: int,
) -> BenchmarkRun:
    """Run the split in batches so vLLM can schedule requests concurrently.

    Submitting one request at a time keeps exactly one sequence in flight
    regardless of ``max_num_seqs``; batching is what lets the scheduler
    fill the GPU.

    A batch that raises is retried one image at a time, so an out-of-memory
    on a large batch costs only the receipt that genuinely fails rather
    than every receipt that happened to share it.

    Args:
        records: Ground-truth records for the split.
        generate_batch: Returns one raw response per record, in order.
        batch_size: Records per submission.

    Returns:
        The same shape ``run_benchmark`` produces.
    """
    run = BenchmarkRun()
    total = len(records)

    for start in range(0, total, batch_size):
        batch = records[start : start + batch_size]
        try:
            responses = generate_batch(batch)
            if len(responses) != len(batch):
                raise ValueError(
                    f"backend returned {len(responses)} responses for "
                    f"{len(batch)} images; matching by position is unsafe"
                )
        except Exception as err:  # noqa: BLE001 - fall back, do not lose the batch
            if len(batch) == 1:
                logger.error("%s FAILED: %s", batch[0].image_id, err)
                _record_failure(run, batch[0], str(err))
                continue

            logger.warning(
                "Batch of %d failed (%s) — retrying one at a time so only the offending image is lost",
                len(batch),
                err,
            )
            for record in batch:
                sub = run_benchmark_batched([record], generate_batch, batch_size=1)
                run.predictions.update(sub.predictions)
                run.raw_responses.update(sub.raw_responses)
                run.errors.update(sub.errors)
            continue

        for record, response in zip(batch, responses, strict=True):
            run.raw_responses[record.image_id] = response
            run.predictions[record.image_id] = parse_sroie_response(response)

        logger.info("[%d/%d] batch of %d complete", start + len(batch), total, len(batch))

    return run


def _record_failure(run: BenchmarkRun, record: SroieRecord, message: str) -> None:
    """Mark one record as failed without dropping it from the run."""
    run.errors[record.image_id] = message
    run.raw_responses[record.image_id] = ""
    run.predictions[record.image_id] = {}


def inference_seconds_from_responses(
    responses: list[dict[str, Any]],
    *,
    fallback: float,
) -> float:
    """Wall-clock inference time across data-parallel workers.

    Workers run concurrently, so the elapsed time is the SLOWEST worker's
    total, not the sum of all of them — summing would report total compute
    and understate throughput by roughly the number of GPUs.

    Engine startup is excluded because each worker times only its own
    generate calls. That keeps the number comparable with the
    single-engine path, which starts its timer after the model has loaded.

    Args:
        responses: Worker dicts carrying ``gpu_id`` and ``elapsed``.
        fallback: Value to use when workers reported no timings, e.g. the
            caller's own wall clock. Never falls back to 0.0, which would
            surface as an absurd throughput rather than as missing data.

    Returns:
        Seconds of inference on the slowest worker.
    """
    per_gpu: dict[Any, float] = {}
    for item in responses:
        if "gpu_id" not in item or "elapsed" not in item:
            continue
        per_gpu[item["gpu_id"]] = per_gpu.get(item["gpu_id"], 0.0) + float(item["elapsed"])

    return max(per_gpu.values()) if per_gpu else fallback


def benchmark_from_worker_responses(
    records: list[SroieRecord],
    responses: list[dict[str, Any]],
) -> BenchmarkRun:
    """Reassemble a run from data-parallel worker output.

    Parsing happens here rather than in the workers, so the serial and
    data-parallel paths share one parser and cannot drift apart.

    Args:
        records: Ground-truth records for the split, in scoring order.
        responses: Worker dicts carrying ``image_id``, ``raw_response``
            and ``error``.

    Returns:
        The same shape ``run_benchmark`` produces. A record absent from
        *responses* — a worker died part-way through its partition —
        becomes an explicit error, never a missing row.
    """
    by_id = {str(item["image_id"]): item for item in responses}
    run = BenchmarkRun()

    for record in records:
        item = by_id.get(record.image_id)
        if item is None:
            run.errors[record.image_id] = (
                "no response returned — the data-parallel worker holding this image did not report it"
            )
            run.raw_responses[record.image_id] = ""
            run.predictions[record.image_id] = {}
            continue

        error = item.get("error")
        response = item.get("raw_response") or ""
        run.raw_responses[record.image_id] = response
        if error:
            run.errors[record.image_id] = str(error)
            run.predictions[record.image_id] = {}
            continue
        run.predictions[record.image_id] = parse_sroie_response(response)

    return run


def run_benchmark(
    records: list[SroieRecord],
    generate: Callable[[SroieRecord], str],
) -> BenchmarkRun:
    """Run every record through the model and parse the replies.

    Args:
        records: Ground-truth records for the split.
        generate: Returns the raw model response for one record.

    Returns:
        Parsed predictions, raw responses, and any per-image errors. Every
        record appears in ``predictions``, so a failure scores as a miss
        rather than shrinking the denominator.
    """
    run = BenchmarkRun()

    for index, record in enumerate(records, start=1):
        try:
            response = generate(record)
        except Exception as err:  # noqa: BLE001 - one image must not end the run
            logger.error("[%d/%d] %s FAILED: %s", index, len(records), record.image_id, err)
            run.errors[record.image_id] = str(err)
            run.raw_responses[record.image_id] = ""
            run.predictions[record.image_id] = {}
            continue

        run.raw_responses[record.image_id] = response
        run.predictions[record.image_id] = parse_sroie_response(response)
        logger.info(
            "[%d/%d] %s -> %d/4 fields",
            index,
            len(records),
            record.image_id,
            len(run.predictions[record.image_id]),
        )

    return run
