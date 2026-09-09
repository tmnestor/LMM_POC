"""Both tile bounds must reach the backend from the extraction dispatch.

`min_tiles` is the lever: the InternVL tiling algorithm picks its grid by
closest aspect-ratio match, so a receipt settles on 2-3 detail tiles and
never approaches max_tiles. Injecting only `_max_tiles` left the backend on
its `min_tiles=1` default, so `inference.tiling.budgets.<type>.min_tiles`
was inert for every document type EXCEPT bank statements — which reach the
backend through UnifiedBankExtractor, the one caller that passed both.

Symptom when it regresses: raising min_tiles in run_config produces a
bit-for-bit identical run.
"""

from models.orchestrator import DocumentOrchestrator


def test_orchestrator_forwards_both_bounds() -> None:
    classification = {"document_type": "RECEIPT", "_max_tiles": 6, "_min_tiles": 6}

    extra = DocumentOrchestrator.tile_extra_from_classification(classification)

    assert extra == {"max_tiles": 6, "min_tiles": 6}


def test_missing_floor_defaults_to_one_not_to_the_ceiling() -> None:
    """Older classification records carry only _max_tiles. Falling back to
    1 reproduces the previous behaviour rather than silently forcing a
    dense grid on a caller that never asked for one."""
    extra = DocumentOrchestrator.tile_extra_from_classification(
        {"document_type": "RECEIPT", "_max_tiles": 6}
    )

    assert extra == {"max_tiles": 6, "min_tiles": 1}


def test_no_budget_means_no_extra() -> None:
    """Pre-tiling off: the backend must take its single-image path."""
    assert DocumentOrchestrator.tile_extra_from_classification({"document_type": "RECEIPT"}) is None
