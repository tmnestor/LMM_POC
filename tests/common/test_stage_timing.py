"""Tests for per-stage timing: inference vs wall clock vs startup.

`entrypoint.sh` times each stage from OUTSIDE the python process, so its
number bundles engine startup into "inference time" and knows nothing about
data parallelism. On a short run startup can be 40% of wall clock, and
under DP the sum of per-image times is total COMPUTE, not elapsed.
"""

import json
from pathlib import Path

from common.stage_timing import (
    StageTiming,
    inference_seconds_from_records,
    read_stage_timings,
    summarise_timings,
    write_stage_timing,
)


def test_a_timing_round_trips(tmp_path: Path) -> None:
    timing = StageTiming(
        stage="extract",
        wall_clock=300.0,
        inference_seconds=240.0,
        execution_mode="data-parallel (2 GPUs)",
        images=165,
    )

    write_stage_timing(tmp_path, timing)

    assert read_stage_timings(tmp_path) == [timing]


def test_stages_append_rather_than_overwrite(tmp_path: Path) -> None:
    """classify runs before extract; both must survive."""
    write_stage_timing(tmp_path, StageTiming("classify", 100.0, 60.0, "single-engine", 165))
    write_stage_timing(tmp_path, StageTiming("extract", 300.0, 240.0, "single-engine", 165))

    assert [t.stage for t in read_stage_timings(tmp_path)] == ["classify", "extract"]


def test_a_rerun_of_one_stage_replaces_only_that_stage(tmp_path: Path) -> None:
    """Re-running extract must not leave two extract rows summing to
    double the real time."""
    write_stage_timing(tmp_path, StageTiming("classify", 100.0, 60.0, "single-engine", 165))
    write_stage_timing(tmp_path, StageTiming("extract", 300.0, 240.0, "single-engine", 165))
    write_stage_timing(tmp_path, StageTiming("extract", 200.0, 150.0, "single-engine", 165))

    timings = read_stage_timings(tmp_path)
    assert [t.stage for t in timings] == ["classify", "extract"]
    assert next(t for t in timings if t.stage == "extract").wall_clock == 200.0


def test_summary_sums_across_stages(tmp_path: Path) -> None:
    write_stage_timing(tmp_path, StageTiming("classify", 100.0, 60.0, "single-engine", 165))
    write_stage_timing(tmp_path, StageTiming("extract", 300.0, 240.0, "single-engine", 165))

    summary = summarise_timings(read_stage_timings(tmp_path))

    assert summary.inference_seconds == 300.0
    assert summary.wall_clock == 400.0
    assert summary.startup_seconds == 100.0


def test_one_execution_mode_is_reported_plainly() -> None:
    summary = summarise_timings(
        [
            StageTiming("classify", 100.0, 60.0, "data-parallel (2 GPUs)", 165),
            StageTiming("extract", 300.0, 240.0, "data-parallel (2 GPUs)", 165),
        ]
    )

    assert summary.execution_mode == "data-parallel (2 GPUs)"


def test_mixed_execution_modes_are_named_not_hidden() -> None:
    """A run where classify was single-engine and extract data-parallel
    must not report either one as though it described the whole run."""
    summary = summarise_timings(
        [
            StageTiming("classify", 100.0, 60.0, "single-engine", 165),
            StageTiming("extract", 300.0, 240.0, "data-parallel (2 GPUs)", 165),
        ]
    )

    assert "classify=single-engine" in summary.execution_mode
    assert "extract=data-parallel (2 GPUs)" in summary.execution_mode


def test_no_timings_is_absent_not_zero(tmp_path: Path) -> None:
    """A missing sidecar must not report 0.0s, which would surface as an
    absurd throughput rather than as missing data."""
    assert read_stage_timings(tmp_path) == []
    assert summarise_timings([]) is None


def test_a_corrupt_line_is_skipped_not_fatal(tmp_path: Path) -> None:
    """Timing is observability; it must never fail an evaluation run."""
    write_stage_timing(tmp_path, StageTiming("extract", 300.0, 240.0, "single-engine", 165))
    path = tmp_path / ".stage_timing.jsonl"
    path.write_text(path.read_text() + "{not json\n")

    assert [t.stage for t in read_stage_timings(tmp_path)] == ["extract"]


def test_inference_never_exceeds_wall_clock_in_the_summary() -> None:
    """Startup cannot be negative; a clock skew must not print one."""
    summary = summarise_timings([StageTiming("extract", 100.0, 140.0, "single-engine", 165)])

    assert summary.startup_seconds == 0.0


def test_inference_from_records_is_the_slowest_worker() -> None:
    """Workers run concurrently, so elapsed is the slowest one's total.
    Summing them reports total COMPUTE and understates throughput by
    roughly the GPU count."""
    records = [
        {"gpu_id": 0, "processing_time": 10.0},
        {"gpu_id": 0, "processing_time": 12.0},
        {"gpu_id": 1, "processing_time": 5.0},
        {"gpu_id": 1, "processing_time": 6.0},
    ]

    assert inference_seconds_from_records(records, fallback=999.0) == 22.0


def test_inference_from_single_engine_records_is_the_sum() -> None:
    """Without a gpu_id the work was sequential, so the sum IS elapsed."""
    records = [{"processing_time": 10.0}, {"processing_time": 12.0}]

    assert inference_seconds_from_records(records, fallback=999.0) == 22.0


def test_inference_falls_back_when_nothing_was_timed() -> None:
    assert inference_seconds_from_records([{"image_name": "x"}], fallback=42.0) == 42.0


def test_written_file_is_readable_json(tmp_path: Path) -> None:
    write_stage_timing(tmp_path, StageTiming("extract", 300.0, 240.0, "single-engine", 165))

    line = (tmp_path / ".stage_timing.jsonl").read_text().strip()
    assert json.loads(line)["stage"] == "extract"
