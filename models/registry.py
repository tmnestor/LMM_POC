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
