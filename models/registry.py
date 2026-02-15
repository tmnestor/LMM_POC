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


_REGISTRY: dict[str, ModelRegistration] = {}


def _print_gpu_status(console) -> None:
    """Print GPU memory usage table after model loading."""
    import torch

    if not torch.cuda.is_available():
        return

    from rich.table import Table

    gpu_table = Table(
        title="GPU Status",
        show_header=True,
        header_style="bold cyan",
    )
    gpu_table.add_column("GPU", style="white")
    gpu_table.add_column("Total", justify="right", style="dim")
    gpu_table.add_column("Allocated", justify="right")
    gpu_table.add_column("Reserved", justify="right")
    gpu_table.add_column("Utilization", justify="right")

    for gpu_id in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(gpu_id)
        vram = props.total_memory / (1024**3)
        alloc = torch.cuda.memory_allocated(gpu_id) / (1024**3)
        resv = torch.cuda.memory_reserved(gpu_id) / (1024**3)
        util = (resv / vram) * 100 if vram > 0 else 0
        color = "green" if util < 50 else ("yellow" if util < 80 else "red")
        gpu_table.add_row(
            f"{gpu_id}: {props.name}",
            f"{vram:.1f} GB",
            f"{alloc:.2f} GB",
            f"{resv:.2f} GB",
            f"[{color}]{util:.1f}%[/{color}]",
        )
    console.print(gpu_table)


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


# ============================================================================
# InternVL3 Registration (lazy imports — no torch/transformers at module level)
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

                model = AutoModel.from_pretrained(
                    str(cfg.model_path),
                    dtype=cfg.torch_dtype,
                    low_cpu_mem_usage=cfg.low_cpu_mem_usage,
                    use_flash_attn=cfg.flash_attn,
                    trust_remote_code=cfg.trust_remote_code,
                    device_map=cfg.device_map,
                ).eval()

                # Set pad_token_id on generation_config to suppress
                # "Setting pad_token_id to eos_token_id" warnings.
                if hasattr(model, "generation_config"):
                    model.generation_config.pad_token_id = tokenizer.eos_token_id

                progress.update(task, description="Model loaded!")

            flash_status = "✅ enabled" if cfg.flash_attn else "❌ disabled"
            console.print(f"⚡ Flash Attention 2: {flash_status}")

            # Display GPU memory status after loading
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
):
    """Create a DocumentAwareInternVL3HybridProcessor from loaded components.

    Create a DocumentAwareInternVL3HybridProcessor from loaded components.
    """
    from models.document_aware_internvl3_processor import (
        DocumentAwareInternVL3HybridProcessor,
    )

    return DocumentAwareInternVL3HybridProcessor(
        field_list=universal_fields,
        model_path=str(config.model_path),
        debug=config.verbose,
        pre_loaded_model=model,
        pre_loaded_tokenizer=tokenizer,
        prompt_config=prompt_config,
        max_tiles=config.max_tiles,
        field_definitions=field_definitions,
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


# ============================================================================
# Llama Registration (lazy imports — no torch/transformers at module level)
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

            # Llama does not use flash attention via config flag
            console.print("⚡ Flash Attention 2: ❌ not supported (Llama uses SDPA)")

            # Display GPU memory status after loading
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
):
    """Create a DocumentAwareLlamaProcessor from loaded components.

    Note: tokenizer_or_processor is an AutoProcessor for Llama (not a tokenizer).
    """
    from models.document_aware_llama_processor import DocumentAwareLlamaProcessor

    return DocumentAwareLlamaProcessor(
        field_list=universal_fields,
        model_path=str(config.model_path),
        debug=config.verbose,
        pre_loaded_model=model,
        pre_loaded_processor=tokenizer_or_processor,
        prompt_config=prompt_config,
        field_definitions=field_definitions,
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
# Qwen3-VL Registration (lazy imports — no torch/transformers at module level)
# ============================================================================


def _qwen3vl_loader(config):
    """Context manager for loading Qwen3-VL-8B-Instruct model and processor.

    Uses Qwen3VLForConditionalGeneration + AutoProcessor.
    """
    from contextlib import contextmanager

    import torch
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    console = Console()

    @contextmanager
    def _loader(cfg):
        model = None
        processor = None

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            console.print(f"\n[bold]Loading Qwen3-VL from: {cfg.model_path}[/bold]")

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
                }
                if cfg.flash_attn:
                    load_kwargs["attn_implementation"] = "flash_attention_2"

                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    str(cfg.model_path),
                    **load_kwargs,
                )

                # Suppress spurious generation_config warnings
                if hasattr(model, "generation_config"):
                    model.generation_config.temperature = None
                    model.generation_config.top_p = None
                    model.generation_config.top_k = None

                progress.update(task, description="Model loaded!")

            flash_status = "✅ enabled" if cfg.flash_attn else "❌ disabled"
            console.print(f"⚡ Flash Attention 2: {flash_status}")

            _print_gpu_status(console)

            yield model, processor

        finally:
            del model
            del processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return _loader(config)


def _qwen3vl_processor_creator(
    model,
    tokenizer_or_processor,
    config,
    prompt_config,
    universal_fields,
    field_definitions,
):
    """Create a DocumentAwareQwen3VLProcessor from loaded components.

    Note: tokenizer_or_processor is an AutoProcessor for Qwen3-VL.
    """
    from models.document_aware_qwen3vl_processor import (
        DocumentAwareQwen3VLProcessor,
    )

    return DocumentAwareQwen3VLProcessor(
        field_list=universal_fields,
        model_path=str(config.model_path),
        debug=config.verbose,
        pre_loaded_model=model,
        pre_loaded_processor=tokenizer_or_processor,
        prompt_config=prompt_config,
        field_definitions=field_definitions,
    )


register_model(
    ModelRegistration(
        model_type="qwen3vl",
        loader=_qwen3vl_loader,
        processor_creator=_qwen3vl_processor_creator,
        prompt_file="qwen3vl_prompts.yaml",
        description="Qwen3-VL-8B-Instruct vision-language model",
    )
)
