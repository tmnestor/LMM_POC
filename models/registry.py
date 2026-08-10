"""Model registry with lazy loading for document extraction processors.

Central registry mapping model type strings to loader/creator callables.
All heavy imports (torch, transformers) are deferred to function bodies
so that importing this module has zero GPU/ML overhead.
"""

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable

type ModelLoader = Callable[..., AbstractContextManager[tuple[Any, Any]]]
type ProcessorCreator = Callable[..., Any]


@dataclass
class ModelRegistration:
    """Registration entry for a vision-language model.

    Attributes:
        model_type: Short identifier, e.g. "internvl3".
        loader: Callable(PipelineConfig) -> ContextManager[(model, tokenizer)].
        processor_creator: Callable(model, tokenizer, config, prompt_config,
                           universal_fields, field_definitions) -> DocumentProcessor.
        prompt_file: Extraction prompt YAML filename, e.g. "internvl3_prompts.yaml".
        description: Human-readable description for help text.
    """

    model_type: str
    loader: ModelLoader
    processor_creator: ProcessorCreator
    prompt_file: str
    description: str = ""
    requires_sharding: bool = False  # True = model must shard across GPUs (keep device_map="auto")
    is_vllm: bool = False  # True = vLLM backend (eligible for data-parallel)
    # False = one whole engine per GPU won't fit; the DP fast path must not fire.
    supports_data_parallel: bool = True


_REGISTRY: dict[str, ModelRegistration] = {}


def register_model(registration: ModelRegistration) -> None:
    """Register a model type. Overwrites any existing registration."""
    _REGISTRY[registration.model_type] = registration


def get_model(model_type: str) -> ModelRegistration:
    """Look up a registered model by type string.

    Raises:
        ValueError: If the model_type is not registered, with a list of available types.
    """
    if model_type not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(
            f"Unknown model type: {model_type!r}.\n"
            f"  What: model type {model_type!r} is not registered in models/registry.py.\n"
            "  Where: config/run_config.yml -> bootstrap.model.type (the single source that "
            "selects the model); a typo or a retired model name here is the usual cause.\n"
            f"  Expected: one of the registered types: {available}\n"
            "  How to fix: set bootstrap.model.type in config/run_config.yml to a registered "
            "value (e.g. bootstrap:\\n  model:\\n    type: internvl3-vllm)."
        )
    return _REGISTRY[model_type]


def list_models() -> list[str]:
    """Return sorted list of registered model type strings."""
    return sorted(_REGISTRY)


def is_vllm_model(model_type: str) -> bool:
    """Check if a registered model uses the vLLM backend."""
    reg = _REGISTRY.get(model_type)
    return reg is not None and reg.is_vllm


def supports_data_parallel(model_type: str) -> bool:
    """Check whether *model_type* can run one independent engine per GPU.

    False for models too large to replicate per GPU (a quantised 31B still wants
    the whole card once KV cache and vision activations are counted). Unknown
    types answer True — ``get_model()`` is the place that rejects them.
    """
    reg = _REGISTRY.get(model_type)
    return reg.supports_data_parallel if reg else True


def _get_requires_sharding(model_type: str) -> bool:
    """Check if a registered model requires cross-GPU sharding."""
    reg = _REGISTRY.get(model_type)
    return reg.requires_sharding if reg else False


# ============================================================================
# Declarative registrations via VllmSpec
#
# Each VllmSpec replaces ~120 lines of hand-written vLLM loader + creator.
# ============================================================================

from models.model_loader import VllmSpec, register_vllm_model  # noqa: E402

# -- vLLM models -----------------------------------------------------------------

register_vllm_model(
    VllmSpec(
        model_type="internvl3-vllm",
        prompt_file="internvl3_prompts.yaml",
        description="InternVL3.5-8B via vLLM (PagedAttention, no flash-attn required)",
    )
)

register_vllm_model(
    VllmSpec(
        model_type="internvl3-14b-vllm",
        prompt_file="internvl3_prompts.yaml",
        description="InternVL3.5-14B via vLLM (~30 GB BF16)",
    )
)

register_vllm_model(
    VllmSpec(
        model_type="internvl3-38b-vllm",
        prompt_file="internvl3_prompts.yaml",
        description="InternVL3.5-38B via vLLM (~77 GB BF16)",
    )
)

# -- Gemma 4 ---------------------------------------------------------------------
# Google's own QAT W4A16 checkpoint (compressed-tensors), which vLLM loads
# natively — 23.3 GB on disk (NOT the ~18 GB a naive 4-bit estimate suggests;
# the vision embedder is excluded from quantisation). Fits a single L40S, but
# at gpu_memory_utilization 0.85 that leaves ~16 GB for KV cache AND vision
# activations — and activations are what OOM'd the BF16 31B at
# max_soft_tokens=1120, so raising the budget still needs care.
# Engine tuning (soft-token budget, max_model_len, gpu_memory_utilization) lives
# in run_config.yml under inference.vllm.models — only capabilities are here.
# Registered as an ALTERNATIVE for A/B against InternVL3.5; not the default.
register_vllm_model(
    VllmSpec(
        model_type="gemma4-31b-w4a16-vllm",
        prompt_file="internvl3_prompts.yaml",
        description="Gemma 4 31B-it QAT W4A16 via vLLM (~18 GB, 1xL40S tp=1)",
        # Gemma's chat template reasons by default; unlike InternVL3.5 it honours
        # enable_thinking, so suppress it here rather than via a template file.
        chat_template_kwargs={"enable_thinking": False},
        # Sizes images via its own soft-token budget + pan-and-scan, NOT InternVL
        # 448-px dynamic tiling.
        supports_pre_tiling=False,
        # ~18 GB of weights plus KV and vision activations wants the whole card;
        # one engine per GPU would OOM on anything smaller than the L40S.
        supports_data_parallel=False,
        # Model card recommends image content before the text.
        default_image_first=True,
    )
)

# Gemma 4 12B "Unified" — a DIFFERENT architecture from the 31B above, not just a
# smaller size. model_type is `gemma4_unified` /
# Gemma4UnifiedForConditionalGeneration: encoder-free, projecting raw image
# patches (and audio waveforms) straight into the LLM embedding space instead of
# running a vision tower.
#
# REQUIRES vLLM >= 0.23.0. Support (vllm-project/vllm#44429) shipped in STABLE
# v0.23.0 on 2026-06-15 — no nightly needed; use a pinned stable release.
# conda_envs/vllm_env2.yaml pins vllm==0.19.0, so this model CANNOT load on the
# standard env: it needs a separate env on >= 0.23.0. Verified present at v0.25.1.
# Anything older fails at engine load with an unknown-architecture error that is
# unrelated to this repo. Setup runbook: docs/gemma4-12b-sandbox-setup.md.
# NOTE: the 31B W4A16 above needs none of this — compressed-tensors is already a
# vllm 0.19.0 dependency, so it runs on the standard env today.
#
# ~12 B params -> ~24 GB BF16, so it fits the single L40S unquantised (a QAT
# w4a16-ct variant also exists if the headroom is ever wanted).
#
# Thinking works differently from the 31B: it is OPT-IN, enabled by a `<|think|>`
# token at the start of the system prompt (or vLLM's --reasoning-parser gemma4).
# This pipeline sends a bare user message with NO system prompt and no reasoning
# parser, so thinking should already be off; enable_thinking=False below is
# belt-and-braces, and the response parser still strips the empty <think> block
# the model may emit regardless.
register_vllm_model(
    VllmSpec(
        model_type="gemma4-12b-unified-vllm",
        prompt_file="internvl3_prompts.yaml",
        description="Gemma 4 12B-it Unified (encoder-free) via vLLM — needs vLLM >= 0.23.0",
        chat_template_kwargs={"enable_thinking": False},
        supports_pre_tiling=False,
        supports_data_parallel=False,
        default_image_first=True,
    )
)

# Gemma 4 12B Unified, Google's QAT W4A16 (compressed-tensors) — the SAME
# architecture as the BF16 12B above (Gemma4UnifiedForConditionalGeneration /
# gemma4_unified), verified 2026-08-11 against this checkpoint's own config.json,
# so the spec and the hf_overrides field name are identical. Only weights differ.
#
# 10,264,229,896 bytes (9.56 GiB) — NOT the ~6 GB a naive 4-bit estimate gives:
# the quantisation ignore list excludes the vision embedder, the patch dense
# layers and lm_head, the same effect that made the 31B 23.3 GB, not ~18 GB.
#
# THIS is the registration that unlocks 2xL4. At 9.56 GiB of ~22.5 GiB per card
# one WHOLE engine fits per GPU, so supports_data_parallel is True and run_dp
# spawns one engine per L4 (tp=1 each) rather than sharding a single engine tp=2
# across PCIe (L4 has no NVLink). The DP path also uses no NCCL and no /dev/shm
# (see common/vllm_dp.py), so it sidesteps the tensor-parallel SHM deadlock
# recorded at the top of entrypoint.sh entirely.
#
# Same vLLM >= 0.23.0 floor as the BF16 12B — run it in vllm_env3.
register_vllm_model(
    VllmSpec(
        model_type="gemma4-12b-unified-w4a16-vllm",
        prompt_file="internvl3_prompts.yaml",
        description="Gemma 4 12B-it Unified QAT W4A16 via vLLM (9.56 GiB, 2xL4 DP=2/tp=1)",
        chat_template_kwargs={"enable_thinking": False},
        supports_pre_tiling=False,
        # The one capability that differs from the BF16 12B above.
        supports_data_parallel=True,
        default_image_first=True,
    )
)
