"""Forwarding of the receipt tile budget to the backend.

One definition, used by both the single-engine stage and the data-parallel
worker, so the two cannot disagree about how images are cropped — the runs
they produce are meant to be interchangeable.
"""


def tile_extra(budget: dict[str, int]) -> dict[str, int]:
    """Build the GenerationParams.extra carrying a full tile budget.

    BOTH bounds must travel. ``min_tiles`` is the actual lever: the
    InternVL tiling algorithm picks its grid by closest aspect-ratio
    match, so a tall receipt settles on 2-3 detail tiles and never
    approaches ``max_tiles``. Forwarding only the ceiling leaves the
    backend on its ``min_tiles=1`` default, which pins every image to the
    aspect-matched grid — a raised floor in run_config then changes
    nothing, and the experiment reads as "tiling does not help".

    Args:
        budget: A tile budget holding ``min_tiles`` and ``max_tiles``.

    Returns:
        The extra dict for GenerationParams.

    Raises:
        KeyError: If either bound is absent. Defaulting silently is the
            defect this function exists to prevent.
    """
    return {"min_tiles": budget["min_tiles"], "max_tiles": budget["max_tiles"]}
