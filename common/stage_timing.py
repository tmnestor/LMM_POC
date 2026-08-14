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


@dataclass(frozen=True)
class ResolvedTiming:
    """Which clock the summary is using, and where it came from.

    ``source`` is printed alongside the number. A run's timing can come
    from three places of very different quality, and an unlabelled figure
    invites the reader to trust the worst of them.
    """

    inference_seconds: float | None
    source: str
    summary: TimingSummary | None


def resolve_timing(
    timings: list[StageTiming],
    *,
    entrypoint_seconds: float | None,
    per_image_total: float,
    images: int,
) -> ResolvedTiming:
    """Choose the best available timing and name it.

    Preference order:

    1. **Stage sidecar** — written by the GPU stages themselves, excludes
       engine startup, DP-aware. Rejected as stale when it covers a
       different image count than the evaluation, which is what happens
       when ``evaluate`` is re-run against an older output dir.
    2. **Entrypoint elapsed** — measured outside the process, so it
       includes engine startup and knows nothing about parallelism. Also
       undetectably stale on a standalone re-run: one such run reported
       3.0s for 165 images, i.e. 3300 images/min. Labelled, not hidden.
    3. **Sum of per-image processing time** — under data parallelism this
       is total COMPUTE, not elapsed, so it understates throughput by
       roughly the GPU count.

    With none of the three, the answer is that timing was not recorded —
    NOT 0.0s, which a caller would turn into an absurd throughput.

    Args:
        timings: Stage sidecar rows for this output dir.
        entrypoint_seconds: The ``--inference-seconds`` value, if passed.
        per_image_total: Sum of per-image processing times.
        images: Number of documents being evaluated now.

    Returns:
        The chosen seconds (or None), its provenance, and the stage
        summary when one applies.
    """
    summary = summarise_timings(timings)
    if summary is not None and summary.inference_seconds > 0:
        recorded_images = sum(t.images for t in timings) // max(1, len(timings))
        if recorded_images == images:
            return ResolvedTiming(summary.inference_seconds, "stage timing (this run)", summary)
        logger.warning(
            "Stage timing covers %d images but %d are being evaluated — treating it as stale",
            recorded_images,
            images,
        )

    if entrypoint_seconds is not None and entrypoint_seconds > 0:
        return ResolvedTiming(entrypoint_seconds, "entrypoint elapsed (includes engine startup)", None)

    if per_image_total > 0:
        return ResolvedTiming(per_image_total, "per-image sum (unreliable under data parallelism)", None)

    return ResolvedTiming(None, "not recorded", None)


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
