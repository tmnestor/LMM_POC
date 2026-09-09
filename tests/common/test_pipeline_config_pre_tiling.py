"""Tests for the pre_tiling config block resolver (_resolve_pre_tiling).

tests/ is gitignored — local-only. Absent block -> off; present block must carry
enabled(bool) + image_size(positive int) + use_thumbnail(bool), else fail fast
with full diagnostics. See plans/2026-06-04-adaptive-pre-tiling.md.
"""

from pathlib import Path

import pytest

from common.pipeline_config import _resolve_pre_tiling

_CFG = Path("config/run_config.yml")


class TestResolvePreTiling:
    def test_absent_block_is_off_with_defaults(self) -> None:
        assert _resolve_pre_tiling({}, _CFG) == (False, 448, True)

    def test_enabled_block(self) -> None:
        block = {
            "inference": {
                "tiling": {"pre_tiling": {"enabled": True, "image_size": 448, "use_thumbnail": True}}
            }
        }
        assert _resolve_pre_tiling(block, _CFG) == (True, 448, True)

    def test_disabled_explicit(self) -> None:
        block = {
            "inference": {
                "tiling": {"pre_tiling": {"enabled": False, "image_size": 448, "use_thumbnail": False}}
            }
        }
        assert _resolve_pre_tiling(block, _CFG) == (False, 448, False)

    def test_custom_image_size(self) -> None:
        block = {
            "inference": {
                "tiling": {"pre_tiling": {"enabled": True, "image_size": 224, "use_thumbnail": True}}
            }
        }
        assert _resolve_pre_tiling(block, _CFG) == (True, 224, True)


def _wrap(pre_tiling: dict) -> dict:
    return {"inference": {"tiling": {"pre_tiling": pre_tiling}}}


class TestFailFast:
    def test_missing_key_fails(self, assert_diagnostic_error) -> None:
        with pytest.raises(ValueError) as exc:
            _resolve_pre_tiling(_wrap({"enabled": True, "image_size": 448}), _CFG)
        assert_diagnostic_error(str(exc.value))

    def test_non_bool_enabled_fails(self, assert_diagnostic_error) -> None:
        block = _wrap({"enabled": "yes", "image_size": 448, "use_thumbnail": True})
        with pytest.raises(ValueError) as exc:
            _resolve_pre_tiling(block, _CFG)
        assert_diagnostic_error(str(exc.value))

    def test_non_int_image_size_fails(self, assert_diagnostic_error) -> None:
        block = _wrap({"enabled": True, "image_size": "big", "use_thumbnail": True})
        with pytest.raises(ValueError) as exc:
            _resolve_pre_tiling(block, _CFG)
        assert_diagnostic_error(str(exc.value))
        assert "image_size" in str(exc.value)

    def test_zero_image_size_fails(self, assert_diagnostic_error) -> None:
        block = _wrap({"enabled": True, "image_size": 0, "use_thumbnail": True})
        with pytest.raises(ValueError) as exc:
            _resolve_pre_tiling(block, _CFG)
        assert_diagnostic_error(str(exc.value))
