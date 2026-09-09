"""The tile budget must reach the backend in full.

Both bounds matter and only one of them is the lever. The InternVL tiling
algorithm picks its grid by closest aspect-ratio match, so a tall receipt
settles well below max_tiles and only min_tiles forces a denser crop.

Forwarding max_tiles alone let the backend fall back to its min_tiles=1
default, which silently pinned every receipt to the aspect-matched grid —
a config change to min_tiles then produced a bit-for-bit identical run.
"""

import pytest

from common.sroie.tiling import tile_extra


def test_both_bounds_are_forwarded() -> None:
    assert tile_extra({"min_tiles": 6, "max_tiles": 6}) == {"min_tiles": 6, "max_tiles": 6}


def test_min_tiles_is_never_dropped() -> None:
    """Dropping it is the exact defect this function exists to prevent:
    the backend's default of 1 would override the configured floor."""
    assert "min_tiles" in tile_extra({"min_tiles": 12, "max_tiles": 18})


def test_a_budget_missing_a_bound_is_an_error() -> None:
    """Silently defaulting is what made the last experiment a no-op."""
    with pytest.raises(KeyError):
        tile_extra({"max_tiles": 6})
