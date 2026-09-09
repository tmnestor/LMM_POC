"""extract_batch must carry the same tile budget as the per-image path.

It built GenerationParams with no `extra`, so routing documents through it
would have run them with pre-tiling DISABLED while the per-image path tiled
normally — two pipelines producing the same artefact from different inputs.
"""

from models.orchestrator import DocumentOrchestrator


def test_a_batch_inherits_its_types_tile_budget() -> None:
    """A batch holds one document type, so the first record speaks for all."""
    extra = DocumentOrchestrator.tile_extra_from_classification(
        {"document_type": "RECEIPT", "_max_tiles": 6, "_min_tiles": 6}
    )

    assert extra == {"max_tiles": 6, "min_tiles": 6}


def test_no_budget_means_the_backend_takes_its_single_image_path() -> None:
    """Pre-tiling off must behave identically batched or not."""
    assert DocumentOrchestrator.tile_extra_from_classification({"document_type": "RECEIPT"}) is None
