"""Tests for the Gemma 4 W4A16 vLLM registration and its capability flags.

tests/ is gitignored — local-only. Covers the registry entry, the capability
accessor the stages use to gate the data-parallel fast path, and the assertion
that InternVL's behaviour is unchanged by the new fields.

See plans/2026-07-27-reintegrate-gemma4-vllm.md.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from models.model_loader import VllmSpec, build_vllm_processor_creator
from models.registry import get_model, list_models, supports_data_parallel

_GEMMA = "gemma4-31b-w4a16-vllm"
_GEMMA_12B = "gemma4-12b-unified-vllm"
_GEMMA_12B_W4A16 = "gemma4-12b-unified-w4a16-vllm"
_GEMMA_TYPES = (_GEMMA, _GEMMA_12B, _GEMMA_12B_W4A16)

# Types that must NOT take the DP fast path: one whole engine per GPU would OOM.
_DP_INCAPABLE = (_GEMMA, _GEMMA_12B)
# 9.56 GiB of weights fits one whole engine per L4, so DP is the point of it.
_DP_CAPABLE = (_GEMMA_12B_W4A16,)

# Normalised keys the YAML must use. NOT derivable by eye — normalisation is not
# uniform across suffixes, and a wrong key falls back to defaults SILENTLY.
# The W4A16 12B deliberately SHARES the BF16 12B's key: -vllm strips, "w4a16"
# fails the size-suffix test, then -w4a16 strips — landing on gemma4-12b-unified.
_NORMALISED = {
    _GEMMA: "gemma4-31b",
    _GEMMA_12B: "gemma4-12b-unified",
    _GEMMA_12B_W4A16: "gemma4-12b-unified",
}

# Soft-token budget override paths inside hf_overrides, per architecture. Verified
# 2026-07-27 (31B, BF16 12B) and 2026-08-11 (W4A16 12B) against each checkpoint's
# own config.json on the Hub:
#   gemma4          (dense 31B): vision_soft_tokens_per_image + vision_config.default_output_length
#   gemma4_unified  (both 12Bs): vision_config.num_soft_tokens  (the other two do NOT exist)
_BUDGET_FIELDS: dict[str, tuple[tuple[str, ...], ...]] = {
    _GEMMA: (("vision_soft_tokens_per_image",), ("vision_config", "default_output_length")),
    _GEMMA_12B: (("vision_config", "num_soft_tokens"),),
    # Same architecture as _GEMMA_12B, so the same single field.
    _GEMMA_12B_W4A16: (("vision_config", "num_soft_tokens"),),
}

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class _FakeConfig:
    """Minimal stand-in for PipelineConfig — only what the creator reads."""

    chat_template: str | None = None
    pre_tiling_enabled: bool = False
    pre_tiling_image_size: int = 448
    pre_tiling_use_thumbnail: bool = True
    debug: bool = False
    verbose: bool = False
    device_map: str = "auto"
    batch_size: int | None = 1
    trace_raw_prompts: bool = False
    trace_path: str | None = None


def _backend_from_creator(creator: Any) -> Any:
    """Return the VllmBackend a processor_creator wires up.

    DocumentOrchestrator is stubbed out: the unit under test is the spec ->
    backend wiring, and the real orchestrator's __init__ needs a full AppConfig
    that has nothing to do with it.
    """
    captured: dict[str, Any] = {}

    def _capture(*, backend: Any, **_kwargs: Any) -> object:
        captured["backend"] = backend
        return object()

    with patch("models.model_loader.DocumentOrchestrator", _capture):
        creator(
            object(),  # engine — never called, message construction is pure
            None,
            _FakeConfig(),
            {"detection_file": "", "detection_key": "detection", "extraction_files": {}},
            [],
            {},
            app_config=None,
        )
    return captured["backend"]


class TestRegistration:
    @pytest.mark.parametrize("model_type", _GEMMA_TYPES)
    def test_gemma4_is_registered(self, model_type: str) -> None:
        assert model_type in list_models()

    @pytest.mark.parametrize("model_type", _GEMMA_TYPES)
    def test_reuses_the_internvl_prompt_file(self, model_type: str) -> None:
        # Prompt resolution is registry-driven, so Gemma needs no prompt work.
        assert get_model(model_type).prompt_file == "internvl3_prompts.yaml"

    @pytest.mark.parametrize("model_type", _GEMMA_TYPES)
    def test_is_a_vllm_model(self, model_type: str) -> None:
        assert get_model(model_type).is_vllm is True

    def test_internvl_remains_registered_and_default_capable(self) -> None:
        # Regression guard: adding capability fields must not change InternVL.
        reg = get_model("internvl3-vllm")
        assert reg.is_vllm is True
        assert reg.supports_data_parallel is True


class TestDataParallelCapability:
    @pytest.mark.parametrize("model_type", _DP_INCAPABLE)
    def test_large_gemma4_cannot_data_parallel(self, model_type: str) -> None:
        # Weights plus KV and vision activations want the whole card (31B W4A16
        # ~23.3 GB; 12B BF16 ~23.9 GB, which alone exceeds an L4).
        assert supports_data_parallel(model_type) is False

    @pytest.mark.parametrize("model_type", _DP_CAPABLE)
    def test_quantised_12b_can_data_parallel(self, model_type: str) -> None:
        # 9.56 GiB of ~22.5 GiB per L4 leaves ~10.7 GiB for KV + activations, so
        # run_dp spawns one engine per card instead of falling back to one shared.
        assert supports_data_parallel(model_type) is True

    def test_quantised_12b_is_registered(self) -> None:
        assert _GEMMA_12B_W4A16 in list_models()

    def test_internvl_can_data_parallel(self) -> None:
        assert supports_data_parallel("internvl3-vllm") is True

    def test_unknown_model_defaults_to_true(self) -> None:
        # get_model() is the gate that rejects unknown types, not this helper.
        assert supports_data_parallel("not-a-real-model") is True


class TestSpecDefaults:
    """The new fields must default to today's InternVL behaviour."""

    def test_capabilities_default_permissive(self) -> None:
        spec = VllmSpec(model_type="x")
        assert spec.supports_pre_tiling is True
        assert spec.supports_data_parallel is True
        assert spec.default_image_first is False
        assert spec.chat_template_kwargs == {}


class TestBackendWiring:
    """``chat_template_kwargs`` was declared but never read (gap G1).

    These assert the spec value actually reaches the constructed backend — the
    specific defect that went unnoticed until a second model needed it.
    """

    def test_chat_template_kwargs_reach_the_backend(self) -> None:
        spec = VllmSpec(model_type="g", chat_template_kwargs={"enable_thinking": False})
        backend = _backend_from_creator(build_vllm_processor_creator(spec))
        assert backend._chat_template_kwargs == {"enable_thinking": False}

    def test_default_image_first_reaches_the_backend(self) -> None:
        spec = VllmSpec(model_type="g", default_image_first=True)
        backend = _backend_from_creator(build_vllm_processor_creator(spec))
        assert backend._default_image_first is True

    def test_internvl_style_spec_leaves_both_at_defaults(self) -> None:
        backend = _backend_from_creator(build_vllm_processor_creator(VllmSpec(model_type="internvl3-vllm")))
        assert backend._chat_template_kwargs == {}
        assert backend._default_image_first is False

    @pytest.mark.parametrize("model_type", _GEMMA_TYPES)
    def test_registered_gemma_backend_suppresses_thinking_and_leads_with_image(
        self, model_type: str
    ) -> None:
        # Exercises the REAL registrations, not hand-built specs.
        backend = _backend_from_creator(get_model(model_type).processor_creator)
        assert backend._chat_template_kwargs == {"enable_thinking": False}
        assert backend._default_image_first is True


class TestYamlWiring:
    """run_config.yml must describe Gemma fully without becoming the default."""

    @staticmethod
    def _raw() -> dict[str, Any]:
        return yaml.safe_load((_REPO_ROOT / "config" / "run_config.yml").read_text())

    def test_selected_model_is_registered_and_self_consistent(self) -> None:
        """Whatever model is selected, its config must be coherent.

        This deliberately does NOT assert a particular model: evaluation runs
        switch bootstrap.model.type between InternVL and the two Gemma 4 models,
        so pinning one would fail on every legitimate switch. What must always
        hold is that the selection is registered and its path matches the entry
        in default_paths.
        """
        model = self._raw()["bootstrap"]["model"]
        selected = model["type"]
        assert selected in list_models(), f"{selected!r} is not registered"
        assert model["path"] == model["default_paths"][selected], (
            f"bootstrap.model.path disagrees with default_paths[{selected!r}]"
        )

    def test_gemma_selection_requires_pre_tiling_off(self) -> None:
        """The coupled edit people forget, caught locally instead of on the GPU.

        Gemma sizes images via its own soft-token budget, so app-side 448-px
        pre-tiling must be off. The loader fails fast on this, but only at engine
        load — which on a GPU-queue workflow can be hours after the mistake.
        """
        raw = self._raw()
        selected = raw["bootstrap"]["model"]["type"]
        pre_tiling = raw["inference"]["tiling"]["pre_tiling"]["enabled"]
        if selected.startswith("gemma4"):
            assert pre_tiling is False, (
                f"{selected} selected but inference.tiling.pre_tiling.enabled is "
                f"{pre_tiling} — Gemma must run with pre-tiling off"
            )

    def test_gemma_has_a_model_path(self) -> None:
        paths = self._raw()["bootstrap"]["model"]["default_paths"]
        assert "w4a16" in paths[_GEMMA]
        assert "12B" in paths[_GEMMA_12B]
        assert "w4a16" in paths[_GEMMA_12B_W4A16]

    @pytest.mark.parametrize("model_type", _GEMMA_TYPES)
    def test_gemma_vllm_block_states_every_required_key(self, model_type: str) -> None:
        block = self._raw()["inference"]["vllm"]["models"][model_type]
        for key in (
            "max_model_len",
            "gpu_memory_utilization",
            "limit_mm_per_prompt",
            "max_num_seqs",
            "enable_prefix_caching",
            "mm_processor_kwargs",
            "hf_overrides",
        ):
            assert key in block, f"missing required key {key!r}"

    @pytest.mark.parametrize("model_type", _GEMMA_TYPES)
    def test_soft_token_budget_is_a_legal_gemma_value(self, model_type: str) -> None:
        # Gemma 4 accepts a discrete set only; anything else is silently wrong.
        legal = {70, 140, 280, 560, 1120}
        block = self._raw()["inference"]["vllm"]["models"][model_type]
        assert block["mm_processor_kwargs"]["max_soft_tokens"] in legal

    @pytest.mark.parametrize("model_type", _GEMMA_TYPES)
    def test_every_budget_field_agrees_with_the_processor_kwarg(self, model_type: str) -> None:
        """All budget fields must carry the same number, or the override no-ops.

        The field names are ARCHITECTURE-SPECIFIC (see _BUDGET_FIELDS): the dense
        31B uses vision_soft_tokens_per_image + vision_config.default_output_length,
        while the encoder-free 12B uses vision_config.num_soft_tokens. Setting the
        wrong one leaves the engine at the checkpoint default (280) while the YAML
        claims otherwise.
        """
        block = self._raw()["inference"]["vllm"]["models"][model_type]
        budget = block["mm_processor_kwargs"]["max_soft_tokens"]
        overrides = block["hf_overrides"]

        for path in _BUDGET_FIELDS[model_type]:
            node: Any = overrides
            for part in path:
                assert part in node, f"{model_type}: hf_overrides missing {'.'.join(path)}"
                node = node[part]
            assert node == budget, f"{model_type}: {'.'.join(path)} = {node}, expected {budget}"

    def test_no_cross_architecture_budget_fields(self) -> None:
        # The 12B must NOT carry the 31B's field names (they read as configured
        # while doing nothing), and vice versa.
        models = self._raw()["inference"]["vllm"]["models"]
        for unified in (_GEMMA_12B, _GEMMA_12B_W4A16):
            block = models[unified]["hf_overrides"]
            assert "vision_soft_tokens_per_image" not in block, unified
            assert "default_output_length" not in block.get("vision_config", {}), unified
        assert "num_soft_tokens" not in models[_GEMMA]["hf_overrides"].get("vision_config", {})

    @pytest.mark.parametrize("model_type", _GEMMA_TYPES)
    def test_limit_mm_is_one_whole_image(self, model_type: str) -> None:
        # Gemma pan-and-scans internally; we never send crops.
        block = self._raw()["inference"]["vllm"]["models"][model_type]
        assert block["limit_mm_per_prompt"] == 1

    def test_generation_and_batch_use_the_normalised_key(self) -> None:
        raw = self._raw()
        batch = raw["pipeline"]["batch"]
        for model_type in _GEMMA_TYPES:
            key = _NORMALISED[model_type]
            assert key in raw["inference"]["generation"]["models"], (
                f"generation.models is missing {key!r} (for {model_type})"
            )
            for section in ("default_sizes", "max_sizes", "conservative_sizes"):
                assert batch[section][key] == 1


class TestNormalisedKeysMatchYaml:
    """The YAML keys must equal what _normalize_model_type actually produces.

    This is the guard for a defect that shipped once already: the keys were
    written as ``gemma4``, but ``gemma4-31b-w4a16-vllm`` normalises to
    ``gemma4-31b`` (the size-suffix check runs BEFORE ``-w4a16`` is stripped).
    Nothing failed — generation fell back to ``generation.defaults`` and batch to
    ``batch.min_size``, and both fallbacks happened to equal the intended values.
    An exact-match test is the only thing that catches this.
    """

    @staticmethod
    def _raw() -> dict[str, Any]:
        return yaml.safe_load((_REPO_ROOT / "config" / "run_config.yml").read_text())

    def test_normalisation_matches_the_expected_key(self) -> None:
        from common.app_config import AppConfig

        for model_type, expected in _NORMALISED.items():
            assert AppConfig._normalize_model_type(model_type) == expected

    def test_no_orphan_gemma_keys_in_yaml(self) -> None:
        # A leftover key no model normalises to is dead config — it reads as
        # configured behaviour while doing nothing.
        raw = self._raw()
        live = set(_NORMALISED.values())
        gen_keys = {k for k in raw["inference"]["generation"]["models"] if k.startswith("gemma4")}
        assert gen_keys == live
        for section in ("default_sizes", "max_sizes", "conservative_sizes"):
            batch_keys = {k for k in raw["pipeline"]["batch"][section] if k.startswith("gemma4")}
            assert batch_keys == live

    def test_vllm_block_keys_are_full_model_types_not_normalised(self) -> None:
        # inference.vllm.models is looked up by the FULL type (get_vllm_config
        # does no normalisation) — the opposite convention to the two above.
        models = self._raw()["inference"]["vllm"]["models"]
        for model_type in _GEMMA_TYPES:
            assert model_type in models
            assert _NORMALISED[model_type] not in models
