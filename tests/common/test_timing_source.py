"""The summary must say where its timing came from, or say it has none.

A standalone `evaluate` re-run against an older output dir has no stage
sidecar and may find a stale `.inference_elapsed`. Presenting that as
"Inference Time" produced 3.0s and 3300 images/min for a 165-image VLM run
— a confident, fabricated number.
"""

from common.stage_timing import StageTiming, resolve_timing


def _sidecar(images: int = 165) -> list[StageTiming]:
    return [StageTiming("extract", 1500.0, 1200.0, "data-parallel (2 GPUs)", images)]


def test_stage_timing_is_preferred_and_labelled() -> None:
    resolved = resolve_timing(_sidecar(), entrypoint_seconds=9999.0, per_image_total=0.0, images=165)

    assert resolved.inference_seconds == 1200.0
    assert resolved.source == "stage timing (this run)"
    assert resolved.summary is not None


def test_a_sidecar_for_a_different_image_count_is_stale() -> None:
    """Re-running evaluate against a dir whose extract covered a different
    set must not borrow that run's clock."""
    resolved = resolve_timing(
        _sidecar(images=24), entrypoint_seconds=1443.0, per_image_total=0.0, images=165
    )

    assert resolved.inference_seconds == 1443.0
    assert "entrypoint" in resolved.source


def test_entrypoint_timing_is_labelled_as_including_startup() -> None:
    resolved = resolve_timing([], entrypoint_seconds=1443.0, per_image_total=0.0, images=165)

    assert resolved.inference_seconds == 1443.0
    assert "startup" in resolved.source


def test_per_image_sum_is_labelled_unreliable() -> None:
    """Under data parallelism the sum is total compute, not elapsed."""
    resolved = resolve_timing([], entrypoint_seconds=None, per_image_total=2500.0, images=165)

    assert resolved.inference_seconds == 2500.0
    assert "unreliable" in resolved.source


def test_no_timing_at_all_is_reported_as_missing() -> None:
    """Not 0.0s, and not a throughput computed from it."""
    resolved = resolve_timing([], entrypoint_seconds=None, per_image_total=0.0, images=165)

    assert resolved.inference_seconds is None
    assert resolved.source == "not recorded"


def test_a_stale_entrypoint_clock_is_still_reported_not_invented() -> None:
    """We cannot detect staleness in the entrypoint file, so the number is
    shown — but its label tells the reader what it is, which is the whole
    point. This pins the behaviour so nobody 'fixes' it into silence."""
    resolved = resolve_timing([], entrypoint_seconds=3.0, per_image_total=0.0, images=165)

    assert resolved.inference_seconds == 3.0
    assert "entrypoint" in resolved.source
