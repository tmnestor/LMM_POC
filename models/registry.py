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
    requires_sharding: bool = (
        False  # True = model must shard across GPUs (keep device_map="auto")
    )
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
        raise ValueError(f"Unknown model type: '{model_type}'. Available: {available}")
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
# Backend factory functions (lazy imports — only run at model-load time)
# ============================================================================


def _internvl3_backend(model, tokenizer, debug):
    """Build InternVL3 backend with .chat() / .batch_chat() API."""
    from models.backends.internvl3 import InternVL3Backend

    return InternVL3Backend(model=model, tokenizer=tokenizer, debug=debug)


def _internvl3_post_load(model, processor, cfg):
    """Set pad_token_id to suppress generation warnings."""
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = processor.eos_token_id


def _llama_backend(model, processor, debug):
    """Build Llama 3.2 backend."""
    from models.backends.llama import LlamaBackend

    return LlamaBackend(model=model, processor=processor, debug=debug)


def _llama_post_load(model, processor, cfg):
    """Tie weights and fix generation_config for Llama 3.2."""
    try:
        model.tie_weights()
    except Exception:
        pass
    if hasattr(model, "generation_config"):
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.pad_token_id = processor.tokenizer.eos_token_id


def _llama4scout_quantization(cfg):
    """Build NF4 quantization config at load time (lazy transformers import)."""
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=cfg.torch_dtype,
        bnb_4bit_use_double_quant=True,
    )


# ============================================================================
# Declarative registrations via ModelSpec / VllmSpec
#
# Each ModelSpec replaces ~200 lines of hand-written loader + processor_creator.
# Each VllmSpec replaces ~120 lines of hand-written vLLM loader + creator.
# ============================================================================

from models.model_loader import (  # noqa: E402
    ModelSpec,
    VllmSpec,
    register_hf_model,
    register_vllm_model,
)

# -- InternVL3 family (custom backend: model.chat() / model.batch_chat() API) --

register_hf_model(
    ModelSpec(
        model_type="internvl3",
        model_class="AutoModel",
        processor_class="AutoTokenizer",
        prompt_file="internvl3_prompts.yaml",
        description="InternVL3.5-8B vision-language model",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        load_kwargs={"low_cpu_mem_usage": True, "use_flash_attn": True},
        processor_kwargs={"use_fast": True, "fix_mistral_regex": True},
        backend_factory=_internvl3_backend,
        post_load=_internvl3_post_load,
    )
)

register_hf_model(
    ModelSpec(
        model_type="internvl3-14b",
        model_class="AutoModel",
        processor_class="AutoTokenizer",
        prompt_file="internvl3_prompts.yaml",
        description="InternVL3.5-14B vision-language model (~30 GB BF16)",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        load_kwargs={"low_cpu_mem_usage": True, "use_flash_attn": True},
        processor_kwargs={"use_fast": True, "fix_mistral_regex": True},
        backend_factory=_internvl3_backend,
        post_load=_internvl3_post_load,
    )
)

register_hf_model(
    ModelSpec(
        model_type="internvl3-38b",
        model_class="AutoModel",
        processor_class="AutoTokenizer",
        prompt_file="internvl3_prompts.yaml",
        description="InternVL3.5-38B vision-language model (2x L40S)",
        requires_sharding=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        load_kwargs={"low_cpu_mem_usage": True, "use_flash_attn": True},
        processor_kwargs={"use_fast": True, "fix_mistral_regex": True},
        backend_factory=_internvl3_backend,
        post_load=_internvl3_post_load,
    )
)

# -- Llama 3.2 (custom backend: processor.apply_chat_template + model.generate) --

register_hf_model(
    ModelSpec(
        model_type="llama",
        model_class="MllamaForConditionalGeneration",
        prompt_file="llama_prompts.yaml",
        description="Llama 3.2 11B Vision Instruct",
        backend_factory=_llama_backend,
        post_load=_llama_post_load,
    )
)

# -- Llama 4 Scout (standard HFChatTemplateBackend, NF4 quantization) -----------

register_hf_model(
    ModelSpec(
        model_type="llama4scout",
        model_class="Llama4ForConditionalGeneration",
        prompt_file="llama4scout_prompts.yaml",
        description="Llama 4 Scout 17B-16E (109B MoE) vision-language model",
        load_kwargs={
            "attn_implementation": "sdpa",
            "quantization_config": _llama4scout_quantization,
        },
        suppress_gen_warnings=("temperature", "top_p"),
        message_style="one_step",
        image_content_key="url",
    )
)

# -- Granite 4.0 3B Vision (standard backend, force single GPU, LoRA merge) -----

register_hf_model(
    ModelSpec(
        model_type="granite4",
        model_class="AutoModelForImageTextToText",
        prompt_file="internvl3_prompts.yaml",
        description="IBM Granite 4.0 3B Vision (~8 GB BF16)",
        trust_remote_code=True,
        force_single_gpu=True,
        merge_lora=True,
        message_style="two_step",
    )
)

# -- Standard HuggingFace models ------------------------------------------------

register_hf_model(
    ModelSpec(
        model_type="qwen3vl",
        model_class="Qwen3VLForConditionalGeneration",
        prompt_file="qwen3vl_prompts.yaml",
        description="Qwen3-VL-8B-Instruct vision-language model",
        attn_implementation="flash_attention_2",
        suppress_gen_warnings=("temperature", "top_p", "top_k"),
        message_style="two_step",
    )
)

register_hf_model(
    ModelSpec(
        model_type="nemotron",
        model_class="AutoModelForCausalLM",
        prompt_file="internvl3_prompts.yaml",
        description="NVIDIA Nemotron Nano 12B v2 VL (hybrid Transformer-Mamba)",
        trust_remote_code=True,
        message_style="two_step",
        system_message="/no_think",
        tokenizer_attr="tokenizer",
    )
)

register_hf_model(
    ModelSpec(
        model_type="qwen35",
        model_class="Qwen3_5ForConditionalGeneration",
        prompt_file="internvl3_prompts.yaml",
        description="Qwen3.5-27B early-fusion VLM (~54 GB BF16)",
        requires_sharding=True,
        suppress_gen_warnings=("temperature", "top_p", "top_k"),
        message_style="one_step",
        image_content_key="image",
        chat_template_kwargs={"enable_thinking": False},
    )
)

# -- Gemma 4 (Mac MPS, HFChatTemplateBackend two_step) ---------------------------


def _gemma4_post_load(model: Any, processor: Any, cfg: Any) -> None:
    """Suppress generation_config warnings for Gemma4."""
    if hasattr(model, "generation_config"):
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        pad_id = getattr(processor, "eos_token_id", None)
        if pad_id is not None:
            model.generation_config.pad_token_id = pad_id


register_hf_model(
    ModelSpec(
        model_type="gemma4-e4b-mps",
        model_class="AutoModelForImageTextToText",
        processor_class="AutoProcessor",
        prompt_file="gemma4_prompts.yaml",
        description="Gemma 4 E4B (4.5B eff) — Mac MPS, float16, ~11 GB",
        message_style="two_step",
        chat_template_kwargs={"enable_thinking": False},
        suppress_gen_warnings=("temperature", "top_p"),
        post_load=_gemma4_post_load,
    )
)


def _gemma4_27b_quantization(cfg: Any) -> Any:
    """Build INT4 QuantoConfig for Gemma4 27B on MPS (bitsandbytes unsupported)."""
    from transformers import QuantoConfig

    return QuantoConfig(weights="int4")


register_hf_model(
    ModelSpec(
        model_type="gemma4-27b-mps",
        model_class="AutoModelForImageTextToText",
        processor_class="AutoProcessor",
        prompt_file="gemma4_prompts.yaml",
        description="Gemma 4 27B MoE — Mac MPS, INT4 (~14 GB, needs 16 GB+ RAM)",
        message_style="two_step",
        chat_template_kwargs={"enable_thinking": False},
        load_kwargs={"quantization_config": _gemma4_27b_quantization},
        suppress_gen_warnings=("temperature", "top_p"),
        post_load=_gemma4_post_load,
    )
)


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

register_vllm_model(
    VllmSpec(
        model_type="llama4scout-w4a16",
        prompt_file="llama4scout_prompts.yaml",
        description="Llama 4 Scout W4A16 via vLLM (tensor parallel, ~55 GB)",
        gpu_memory_utilization=0.92,
    )
)

register_vllm_model(
    VllmSpec(
        model_type="qwen3vl-vllm",
        prompt_file="qwen3vl_prompts.yaml",
        description="Qwen3-VL-8B via vLLM (PagedAttention, tensor parallelism)",
        gpu_memory_utilization=0.92,
    )
)

register_vllm_model(
    VllmSpec(
        model_type="qwen35-vllm",
        prompt_file="internvl3_prompts.yaml",
        description="Qwen3.5-27B via vLLM (~54 GB BF16)",
        max_model_len=16384,
        gpu_memory_utilization=0.92,
    )
)

register_vllm_model(
    VllmSpec(
        model_type="gemma4",
        prompt_file="internvl3_prompts.yaml",
        description="Gemma 4 31B-it via vLLM (~58 GB BF16)",
        mm_processor_kwargs={"max_soft_tokens": 560},
        hf_overrides={
            "vision_config": {"default_output_length": 560},
            "vision_soft_tokens_per_image": 560,
        },
    )
)
