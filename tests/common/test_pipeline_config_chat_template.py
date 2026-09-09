"""Tests for the model.chat_template config knob (vLLM template override).

tests/ is gitignored — local-only. Covers _resolve_chat_template: the explicit
no-op values resolve to None, a real path passes through, and missing-key /
bad-path both fail fast with full diagnostics (validated at config load).
"""

from pathlib import Path

import pytest

from common.pipeline_config import _resolve_chat_template

_CFG = Path("config/run_config.yml")


class TestNoOpValues:
    @pytest.mark.parametrize("value", ["none", "None", "NULL", "", "  ", None])
    def test_noop_values_resolve_to_none(self, value: object) -> None:
        assert _resolve_chat_template({"chat_template": value}, _CFG) is None


class TestValidPath:
    def test_existing_path_passes_through(self, tmp_path: Path) -> None:
        template = tmp_path / "no_think.jinja"
        template.write_text("{{ messages }}")
        resolved = _resolve_chat_template({"chat_template": str(template)}, _CFG)
        assert resolved == str(template)


class TestFailFast:
    def test_missing_key_fails_fast(self, assert_diagnostic_error) -> None:
        with pytest.raises(ValueError) as exc:
            _resolve_chat_template({}, _CFG)
        assert_diagnostic_error(str(exc.value))
        assert "model.chat_template" in str(exc.value)

    def test_nonexistent_path_fails_fast(self, assert_diagnostic_error) -> None:
        with pytest.raises(ValueError) as exc:
            _resolve_chat_template({"chat_template": "/no/such/template.jinja"}, _CFG)
        assert_diagnostic_error(str(exc.value))
        assert "does not exist" in str(exc.value)
