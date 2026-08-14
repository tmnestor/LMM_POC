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

from common.sroie.ground_truth import SroieRecord
from common.sroie.parse import parse_sroie_response

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkRun:
    """Everything one pass over the split produced."""

    predictions: dict[str, dict[str, str]] = field(default_factory=dict)
    raw_responses: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


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
