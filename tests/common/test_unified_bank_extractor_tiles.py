"""Tests that UnifiedBankExtractor forwards the pre-tiling tile budget.

tests/ is gitignored — local-only. The dense-table bank misses are recovered by
threading max_tiles into the backend as extra["max_tiles"]; this locks that the
bank extractor forwards it (and omits it when unset). See
plans/2026-06-04-adaptive-pre-tiling.md.
"""

from common.unified_bank_extractor import UnifiedBankExtractor


class _Recorder:
    """Stand-in for processor.generate that records the kwargs it receives."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, image, prompt, *, max_tokens, extra=None) -> str:
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens, "extra": extra})
        return "STORE: x\nDATE: 01/01\nAMOUNT: 1.00"


class TestTileForwarding:
    def test_forwards_extra_tile_budget_when_set(self) -> None:
        rec = _Recorder()
        ube = UnifiedBankExtractor(generate_fn=rec, verbose=False, max_tiles=18, min_tiles=12)
        ube._gen("IMG", "PROMPT", 500)
        assert rec.calls[0]["extra"] == {"max_tiles": 18, "min_tiles": 12}
        assert rec.calls[0]["max_tokens"] == 500

    def test_min_tiles_defaults_to_one_when_unset(self) -> None:
        rec = _Recorder()
        ube = UnifiedBankExtractor(generate_fn=rec, verbose=False, max_tiles=18)  # min unset
        ube._gen("IMG", "PROMPT", 500)
        assert rec.calls[0]["extra"] == {"max_tiles": 18, "min_tiles": 1}

    def test_omits_extra_when_max_tiles_unset(self) -> None:
        rec = _Recorder()
        ube = UnifiedBankExtractor(generate_fn=rec, verbose=False)  # max_tiles defaults None
        ube._gen("IMG", "PROMPT", 4096)
        assert rec.calls[0]["extra"] is None

    def test_omits_extra_when_max_tiles_zero(self) -> None:
        rec = _Recorder()
        ube = UnifiedBankExtractor(generate_fn=rec, verbose=False, max_tiles=0)
        ube._gen("IMG", "PROMPT", 4096)
        assert rec.calls[0]["extra"] is None
