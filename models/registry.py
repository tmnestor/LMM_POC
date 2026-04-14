"""Model registry with lazy loading for document extraction processors.

Central registry mapping model type strings to loader/creator callables.
All heavy imports (torch, transformers) are deferred to function bodies
so that importing this module has zero GPU/ML overhead.
"""

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable

# Re-export extracted infrastructure for backward compatibility
from models.attention import (
    is_sdpa_patched as _is_sdpa_patched,
)
from models.attention import (
    mark_sdpa_patched as _mark_sdpa_patched,
)
from models.attention import (
    patch_eager_attention_to_sdpa as _patch_eager_attention_to_sdpa,
)
from models.gpu_utils import print_gpu_status as _print_gpu_status
from models.gpu_utils import quiet_loading as _quiet_loading
from models.sharding import split_internvl_model as _split_internvl_model

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


def _get_requires_sharding(model_type: str) -> bool:
    """Check if a registered model requires cross-GPU sharding."""
    reg = _REGISTRY.get(model_type)
    return reg.requires_sharding if reg else False


# ============================================================================
# InternVL3 Registration (custom — model.chat() API + InternVL3 sharding)
# ============================================================================


def _internvl3_loader(config):
    """Context manager for loading InternVL3 model and tokenizer.

    Load InternVL3 model and tokenizer into GPU memory.
    """
    from contextlib import contextmanager

    import torch
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from transformers import AutoModel, AutoTokenizer

    console = Console()

    @contextmanager
    def _loader(cfg):
        model = None
        tokenizer = None

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            console.print(f"\n[bold]Loading model from: {cfg.model_path}[/bold]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Loading tokenizer...", total=None)

                tokenizer = AutoTokenizer.from_pretrained(
                    str(cfg.model_path),
                    trust_remote_code=cfg.trust_remote_code,
                    use_fast=cfg.use_fast_tokenizer,
                )

                progress.update(task, description="Loading model weights...")

                # For large models that need cross-GPU sharding, use the
                # official InternVL split_model function instead of
                # device_map="auto" — it keeps vision encoder + first/last
                # LLM layers on GPU 0 to avoid cross-device tensor errors.
                effective_device_map = cfg.device_map
                if (
                    cfg.device_map == "auto"
                    and torch.cuda.device_count() > 1
                    and _get_requires_sharding(cfg.model_type)
                ):
                    effective_device_map = _split_internvl_model(str(cfg.model_path))
                    console.print(
                        f"[bold cyan]Sharding across {torch.cuda.device_count()} GPUs "
                        f"(split_model)[/bold cyan]"
                    )

                with _quiet_loading():
                    model = AutoModel.from_pretrained(
                        str(cfg.model_path),
                        dtype=cfg.torch_dtype,
                        low_cpu_mem_usage=cfg.low_cpu_mem_usage,
                        use_flash_attn=cfg.flash_attn,
                        trust_remote_code=cfg.trust_remote_code,
                        device_map=effective_device_map,
                    ).eval()

                # Set pad_token_id on generation_config to suppress
                # "Setting pad_token_id to eos_token_id" warnings.
                if hasattr(model, "generation_config"):
                    model.generation_config.pad_token_id = tokenizer.eos_token_id

                progress.update(task, description="Model loaded!")

            # Check if native flash-attn package is installed
            _has_native_flash = False
            if cfg.flash_attn:
                try:
                    import flash_attn  # noqa: F401

                    _has_native_flash = True
                except ImportError:
                    pass

            if _has_native_flash:
                console.print("Flash Attention 2: native (flash-attn package)")
            elif cfg.flash_attn:
                console.print("Flash Attention 2: via SDPA patch")
            else:
                console.print("Flash Attention 2: disabled")

            # Only apply SDPA monkey-patch when flash-attn is NOT installed.
            # With native flash-attn, the model uses flash_attention_2 backend
            # directly — the eager->SDPA patch is unnecessary.
            if cfg.flash_attn and not _has_native_flash and not _is_sdpa_patched():
                if _patch_eager_attention_to_sdpa():
                    console.print("[bold]Patched eager attention -> SDPA[/bold]")
                    _mark_sdpa_patched()
                else:
                    console.print(
                        "[yellow]Warning: could not patch attention "
                        "to SDPA -- high tile counts may OOM[/yellow]"
                    )

            # Skip per-loader GPU status in multi-GPU mode (orchestrator prints once)
            if not getattr(cfg, "_multi_gpu", False):
                _print_gpu_status(console)

            yield model, tokenizer

        finally:
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return _loader(config)


def _internvl3_processor_creator(
    model,
    tokenizer,
    config,
    prompt_config,
    universal_fields,
    field_definitions,
    *,
    app_config=None,
):
    """Create a DocumentAwareInternVL3HybridProcessor from loaded components."""
    from models.document_aware_internvl3_processor import (
        DocumentAwareInternVL3HybridProcessor,
    )

    return DocumentAwareInternVL3HybridProcessor(
        field_list=universal_fields,
        model_path=str(config.model_path),
        debug=config.verbose,
        batch_size=config.batch_size,
        pre_loaded_model=model,
        pre_loaded_tokenizer=tokenizer,
        prompt_config=prompt_config,
        max_tiles=config.max_tiles,
        min_tiles=config.min_tiles,
        field_definitions=field_definitions,
        app_config=app_config,
    )


register_model(
    ModelRegistration(
        model_type="internvl3",
        loader=_internvl3_loader,
        processor_creator=_internvl3_processor_creator,
        prompt_file="internvl3_prompts.yaml",
        description="InternVL3.5-8B vision-language model",
    )
)

# Same architecture, same API — just larger models
register_model(
    ModelRegistration(
        model_type="internvl3-14b",
        loader=_internvl3_loader,
        processor_creator=_internvl3_processor_creator,
        prompt_file="internvl3_prompts.yaml",
        description="InternVL3.5-14B vision-language model (~30 GB BF16)",
    )
)

register_model(
    ModelRegistration(
        model_type="internvl3-38b",
        loader=_internvl3_loader,
        processor_creator=_internvl3_processor_creator,
        prompt_file="internvl3_prompts.yaml",
        description="InternVL3.5-38B vision-language model (2x L40S)",
        requires_sharding=True,  # ~77 GB BF16, must shard across 2x L40S
    )
)


# ============================================================================
# Llama Registration (custom — MllamaForConditionalGeneration + tie_weights)
# ============================================================================


def _llama_loader(config):
    """Context manager for loading Llama Vision model and processor.

    Uses MllamaForConditionalGeneration + AutoProcessor (not AutoModel/AutoTokenizer).
    Supports 8-bit quantization for memory efficiency.
    """
    from contextlib import contextmanager

    import torch
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from transformers import AutoProcessor, MllamaForConditionalGeneration

    console = Console()

    @contextmanager
    def _loader(cfg):
        model = None
        processor = None

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            console.print(f"\n[bold]Loading Llama model from: {cfg.model_path}[/bold]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Loading processor...", total=None)

                processor = AutoProcessor.from_pretrained(str(cfg.model_path))

                progress.update(task, description="Loading model weights...")

                # Build load kwargs from config
                load_kwargs = {
                    "dtype": cfg.torch_dtype,
                    "device_map": cfg.device_map,
                }

                with _quiet_loading():
                    model = MllamaForConditionalGeneration.from_pretrained(
                        str(cfg.model_path),
                        **load_kwargs,
                    )

                # Tie weights for Llama models
                try:
                    model.tie_weights()
                except Exception:
                    pass

                # Clean the model's built-in GenerationConfig to suppress
                # warnings from transformers during generation.
                if hasattr(model, "generation_config"):
                    model.generation_config.temperature = None
                    model.generation_config.top_p = None
                    # Set pad_token_id once at load time so per-call
                    # passing is unnecessary (suppresses open-end warning).
                    model.generation_config.pad_token_id = (
                        processor.tokenizer.eos_token_id
                    )

                progress.update(task, description="Model loaded!")

            # Llama uses PyTorch SDPA natively; flash_attention_2 attn_implementation
            # is incompatible (MllamaVisionAttention lacks is_causal attribute).
            console.print("Flash Attention 2: not applicable (Llama uses SDPA)")

            # Skip per-loader GPU status in multi-GPU mode (orchestrator prints once)
            if not getattr(cfg, "_multi_gpu", False):
                _print_gpu_status(console)

            # Yield (model, processor) — processor is Llama's equivalent of tokenizer
            yield model, processor

        finally:
            del model
            del processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return _loader(config)


def _llama_processor_creator(
    model,
    tokenizer_or_processor,
    config,
    prompt_config,
    universal_fields,
    field_definitions,
    *,
    app_config=None,
):
    """Create a DocumentAwareLlamaProcessor from loaded components.

    Note: tokenizer_or_processor is an AutoProcessor for Llama (not a tokenizer).
    """
    from models.document_aware_llama_processor import DocumentAwareLlamaProcessor

    return DocumentAwareLlamaProcessor(
        field_list=universal_fields,
        model_path=str(config.model_path),
        debug=config.verbose,
        batch_size=config.batch_size,
        pre_loaded_model=model,
        pre_loaded_processor=tokenizer_or_processor,
        prompt_config=prompt_config,
        field_definitions=field_definitions,
        app_config=app_config,
    )


register_model(
    ModelRegistration(
        model_type="llama",
        loader=_llama_loader,
        processor_creator=_llama_processor_creator,
        prompt_file="llama_prompts.yaml",
        description="Llama 3.2 11B Vision Instruct",
    )
)


# ============================================================================
# Llama 4 Scout Registration (custom — BitsAndBytesConfig NF4 quantization)
# ============================================================================


def _llama4scout_loader(config):
    """Context manager for loading Llama 4 Scout model and processor.

    Uses Llama4ForConditionalGeneration + AutoProcessor.
    NF4 quantization (~55 GB) to fit 109B MoE model on 2x L40S (96 GiB).
    """
    from contextlib import contextmanager

    import torch
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Llama4ForConditionalGeneration,
    )

    console = Console()

    @contextmanager
    def _loader(cfg):
        model = None
        processor = None

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            console.print(
                f"\n[bold]Loading Llama 4 Scout from: {cfg.model_path}[/bold]"
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Loading processor...", total=None)

                processor = AutoProcessor.from_pretrained(str(cfg.model_path))

                progress.update(task, description="Loading model weights...")

                load_kwargs = {
                    "dtype": cfg.torch_dtype,
                    "device_map": cfg.device_map,
                    "attn_implementation": "sdpa",
                }

                # NF4 quantization: 109B MoE -> ~55 GB, fits 2x L40S (96 GiB)
                # FP8 (~109 GB) does NOT fit; bf16 (~218 GB) definitely not.
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=cfg.torch_dtype,
                    bnb_4bit_use_double_quant=True,
                )
                load_kwargs["quantization_config"] = quantization_config
                console.print(
                    "[bold]Using NF4 quantization (~55 GB for 109B MoE)[/bold]"
                )

                with _quiet_loading():
                    model = Llama4ForConditionalGeneration.from_pretrained(
                        str(cfg.model_path),
                        **load_kwargs,
                    )

                # Suppress spurious generation_config warnings
                if hasattr(model, "generation_config"):
                    model.generation_config.temperature = None
                    model.generation_config.top_p = None

                progress.update(task, description="Model loaded!")

            console.print("Flash Attention 2: not applicable (Llama 4 uses SDPA)")

            if not getattr(cfg, "_multi_gpu", False):
                _print_gpu_status(console)

            yield model, processor

        finally:
            del model
            del processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return _loader(config)


def _llama4scout_processor_creator(
    model,
    tokenizer_or_processor,
    config,
    prompt_config,
    universal_fields,
    field_definitions,
    *,
    app_config=None,
):
    """Create a DocumentAwareLlama4Processor from loaded components.

    Note: tokenizer_or_processor is an AutoProcessor for Llama 4 (not a tokenizer).
    """
    from models.document_aware_llama4_processor import (
        DocumentAwareLlama4Processor,
    )

    return DocumentAwareLlama4Processor(
        field_list=universal_fields,
        model_path=str(config.model_path),
        debug=config.verbose,
        batch_size=config.batch_size,
        pre_loaded_model=model,
        pre_loaded_processor=tokenizer_or_processor,
        prompt_config=prompt_config,
        field_definitions=field_definitions,
        app_config=app_config,
    )


register_model(
    ModelRegistration(
        model_type="llama4scout",
        loader=_llama4scout_loader,
        processor_creator=_llama4scout_processor_creator,
        prompt_file="llama4scout_prompts.yaml",
        description="Llama 4 Scout 17B-16E (109B MoE) vision-language model",
    )
)


# ============================================================================
# Granite 4.0 3B Vision (custom — force single GPU, LoRA merge)
# ============================================================================


def _granite4_vision_loader(config):
    """Context manager for loading Granite 4.0 3B Vision.

    SigLIP2 + WindowQFormer + GraniteMoeHybrid LLM with LoRA adapters.
    ~8 GB BF16, fits single GPU.  Adapters are merged at load time for
    faster inference.
    """
    from contextlib import contextmanager

    import torch
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from transformers import AutoModelForImageTextToText, AutoProcessor

    console = Console()

    @contextmanager
    def _loader(cfg):
        model = None
        processor = None

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            console.print(
                f"\n[bold]Loading Granite 4.0 3B Vision from: {cfg.model_path}[/bold]"
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Loading processor...", total=None)

                processor = AutoProcessor.from_pretrained(
                    str(cfg.model_path), trust_remote_code=True
                )

                progress.update(task, description="Loading model weights...")

                with _quiet_loading():
                    # Force single GPU — custom code has device mismatch
                    # bugs when accelerate splits across multiple GPUs.
                    # Only ~8 GB BF16, fits easily on one device.
                    model = AutoModelForImageTextToText.from_pretrained(
                        str(cfg.model_path),
                        trust_remote_code=True,
                        device_map="cuda:0",
                        dtype=torch.bfloat16,
                    ).eval()

                # Merge LoRA adapters for faster inference
                progress.update(task, description="Merging LoRA adapters...")
                if hasattr(model, "merge_lora_adapters"):
                    model.merge_lora_adapters()

                progress.update(task, description="Model loaded!")

            _print_gpu_status(console)
            yield model, processor

        finally:
            del model
            del processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return _loader(config)


def _granite4_vision_processor_creator(
    model,
    tokenizer_or_processor,
    config,
    prompt_config,
    universal_fields,
    field_definitions,
    *,
    app_config=None,
):
    """Create a placeholder processor for Granite 4.0 3B Vision.

    Benchmark scripts use run_inference() directly, so this is only needed
    for the cli.py document extraction pipeline (not yet wired up).
    """
    return None


register_model(
    ModelRegistration(
        model_type="granite4",
        loader=_granite4_vision_loader,
        processor_creator=_granite4_vision_processor_creator,
        prompt_file="internvl3_prompts.yaml",
        description="IBM Granite 4.0 3B Vision (~8 GB BF16)",
        requires_sharding=False,
    )
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

# -- Standard HuggingFace models ---------------------------------------------

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

# -- vLLM models --------------------------------------------------------------

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
