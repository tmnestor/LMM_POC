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
