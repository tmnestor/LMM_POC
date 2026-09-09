"""Tests for the tracing config block resolver (_resolve_tracing).

tests/ is gitignored — local-only. Absent block -> off; present block must carry
raw_prompts(bool) + path, else fail fast with full diagnostics.
"""

from pathlib import Path

import pytest

from common.pipeline_config import _resolve_tracing

_CFG = Path("config/run_config.yml")


class TestResolveTracing:
    def test_absent_block_is_off(self) -> None:
        assert _resolve_tracing({}, _CFG) == (False, None)

    def test_enabled_with_none_path(self) -> None:
        cfg = {"inference": {"tracing": {"raw_prompts": True, "path": "none"}}}
        assert _resolve_tracing(cfg, _CFG) == (True, None)

    def test_enabled_with_explicit_path(self) -> None:
        cfg = {"inference": {"tracing": {"raw_prompts": True, "path": "/x/t.jsonl"}}}
        out = _resolve_tracing(cfg, _CFG)
        assert out == (True, "/x/t.jsonl")

    def test_disabled_explicit(self) -> None:
        cfg = {"inference": {"tracing": {"raw_prompts": False, "path": "none"}}}
        assert _resolve_tracing(cfg, _CFG) == (False, None)


class TestFailFast:
    def test_missing_path_key_fails(self, assert_diagnostic_error) -> None:
        with pytest.raises(ValueError) as exc:
            _resolve_tracing({"inference": {"tracing": {"raw_prompts": True}}}, _CFG)
        assert_diagnostic_error(str(exc.value))

    def test_non_bool_raw_prompts_fails(self, assert_diagnostic_error) -> None:
        cfg = {"inference": {"tracing": {"raw_prompts": "yes", "path": "none"}}}
        with pytest.raises(ValueError) as exc:
            _resolve_tracing(cfg, _CFG)
        assert_diagnostic_error(str(exc.value))
        assert "raw_prompts" in str(exc.value)
