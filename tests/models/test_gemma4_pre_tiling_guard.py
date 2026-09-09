"""Fail-fast guard: pre-tiling must not be enabled for a non-tiling model.

tests/ is gitignored — local-only. App-side pre-tiling crops InternVL 448-px
tiles; Gemma 4 sizes images via its own soft-token budget and is configured with
limit_mm_per_prompt=1, so feeding it crops would make vLLM silently drop all but
the first — a run that looks healthy while reading one corner of the page.

See plans/2026-07-27-reintegrate-gemma4-vllm.md (gap G3).
"""

from dataclasses import dataclass

import pytest

from models.model_loader import VllmSpec, build_vllm_loader


@dataclass
class _Config:
    """Minimal PipelineConfig stand-in for the loader's pre-flight checks."""

    pre_tiling_enabled: bool
    model_path: str = "/models/gemma-4-31B-it-qat-w4a16-ct"
    num_gpus: int = 1
    enforce_eager: bool = True


def _load(spec: VllmSpec, *, pre_tiling: bool) -> None:
    """Enter the loader context — raises before touching vLLM if misconfigured."""
    loader = build_vllm_loader(spec)
    with loader(_Config(pre_tiling_enabled=pre_tiling), app_config=None):
        pass  # pragma: no cover - never reached in these tests


class TestPreTilingGuard:
    def test_pre_tiling_with_non_tiling_model_fails_fast(self, assert_diagnostic_error) -> None:
        spec = VllmSpec(model_type="gemma4-31b-w4a16-vllm", supports_pre_tiling=False)
        with pytest.raises(ValueError) as excinfo:
            _load(spec, pre_tiling=True)

        message = str(excinfo.value)
        assert_diagnostic_error(message)
        # Must name the offending key and the selected model, or the operator
        # has to read the source to know what to change.
        assert "inference.tiling.pre_tiling.enabled" in message
        assert "gemma4-31b-w4a16-vllm" in message
        assert "enabled: false" in message

    def test_non_tiling_model_without_pre_tiling_gets_past_the_guard(self) -> None:
        # Should fail LATER (no AppConfig for engine tuning), not on the guard.
        spec = VllmSpec(model_type="gemma4-31b-w4a16-vllm", supports_pre_tiling=False)
        with pytest.raises(ValueError) as excinfo:
            _load(spec, pre_tiling=False)
        assert "pre_tiling" not in str(excinfo.value)
        assert "AppConfig" in str(excinfo.value)

    def test_tiling_model_with_pre_tiling_is_not_blocked_by_this_guard(self) -> None:
        # InternVL keeps its own limit_mm_per_prompt check; regression guard that
        # the new gate doesn't reject the model it was never meant to touch.
        spec = VllmSpec(model_type="internvl3-vllm")
        with pytest.raises(ValueError) as excinfo:
            _load(spec, pre_tiling=True)
        assert "does not support it" not in str(excinfo.value)
