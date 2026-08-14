"""Per-stage timing: inference vs wall clock vs engine startup.

``entrypoint.sh`` times each stage from OUTSIDE the python process, so its
number bundles engine startup into "inference time" and knows nothing about
data parallelism. Two consequences:

* On a short run startup is a large share of wall clock — measured at 41%
  on one 347-image benchmark — so throughput computed from it understates
  the model badly.
* Under data parallelism the workers run concurrently, so elapsed inference
  is the SLOWEST worker's total, not the sum. Summing reports total compute
  and understates throughput by roughly the GPU count.

Each GPU stage writes one row here; ``evaluate`` aggregates them. Timing is
observability, so every read path degrades quietly: a missing or corrupt
sidecar falls back to the entrypoint's number rather than failing a run.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_FILENAME = ".stage_timing.jsonl"


@dataclass(frozen=True)
class StageTiming:
    """What one GPU stage spent."""

    stage: str
    wall_clock: float
    inference_seconds: float
    execution_mode: str
    images: int


@dataclass(frozen=True)
class TimingSummary:
    """Aggregate across a run's GPU stages."""

    wall_clock: float
    inference_seconds: float
    startup_seconds: float
    execution_mode: str


def write_stage_timing(output_dir: Path, timing: StageTiming) -> Path:
    """Record one stage's timing, replacing any earlier row for that stage.

    Replacing matters: re-running extract against an existing output dir
    would otherwise leave two rows summing to double the real time.

    Args:
        output_dir: The run's output directory.
        timing: What this stage spent.

    Returns:
        The sidecar path.
    """
    path = output_dir / _FILENAME
    existing = [t for t in read_stage_timings(output_dir) if t.stage != timing.stage]
    existing.append(timing)

    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(asdict(t)) + "\n" for t in existing), encoding="utf-8")
    return path


def read_stage_timings(output_dir: Path) -> list[StageTiming]:
    """Read every stage timing recorded for a run.

    Args:
        output_dir: The run's output directory.

    Returns:
        Timings in write order. Empty when the sidecar is absent — callers
        fall back rather than reporting 0.0s, which would surface as an
        absurd throughput instead of as missing data.
    """
    path = output_dir / _FILENAME
    if not path.is_file():
        return []

    timings = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            timings.append(StageTiming(**json.loads(line)))
        except (json.JSONDecodeError, TypeError) as err:
            # Observability must never fail an evaluation run.
            logger.warning("Skipping unreadable stage timing row: %s", err)
    return timings


def inference_seconds_from_records(records: list[dict], *, fallback: float) -> float:
    """Elapsed inference implied by per-image processing times.

    Records carrying ``gpu_id`` came from concurrent data-parallel workers,
    so elapsed is the SLOWEST worker's total — summing across workers gives
    total compute and understates throughput by roughly the GPU count.
    Without ``gpu_id`` the work was sequential and the sum IS elapsed.

    Args:
        records: Per-image result dicts with ``processing_time``.
        fallback: Value when nothing was timed, e.g. the caller's own wall
            clock. Never falls back to 0.0, which would surface as an
            absurd throughput rather than as missing data.

    Returns:
        Seconds of inference.
    """
    per_gpu: dict[object, float] = {}
    timed = False
    for record in records:
        elapsed = record.get("processing_time")
        if elapsed is None:
            continue
        timed = True
        per_gpu[record.get("gpu_id")] = per_gpu.get(record.get("gpu_id"), 0.0) + float(elapsed)

    if not timed:
        return fallback
    return max(per_gpu.values())


def summarise_timings(timings: list[StageTiming]) -> TimingSummary | None:
    """Aggregate across stages.

    Args:
        timings: Per-stage rows.

    Returns:
        The summary, or None when there is nothing recorded.
    """
    if not timings:
        return None

    wall_clock = sum(t.wall_clock for t in timings)
    inference = sum(t.inference_seconds for t in timings)

    modes = {t.execution_mode for t in timings}
    if len(modes) == 1:
        execution_mode = modes.pop()
    else:
        # Naming each stage beats reporting one mode as though it described
        # the whole run.
        execution_mode = ", ".join(f"{t.stage}={t.execution_mode}" for t in timings)

    return TimingSummary(
        wall_clock=wall_clock,
        inference_seconds=inference,
        # Clock skew must not print a negative startup.
        startup_seconds=max(0.0, wall_clock - inference),
        execution_mode=execution_mode,
    )
