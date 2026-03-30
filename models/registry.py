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
_sdpa_patched: bool = False


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


def _patch_eager_attention_to_sdpa() -> bool:
    """Monkey-patch eager attention to use PyTorch SDPA globally.

    Replaces the eager_attention_forward entry in transformers'
    ALL_ATTENTION_FUNCTIONS registry with F.scaled_dot_product_attention.
    This avoids materializing the full O(N²) attention matrix, which
    OOMs on multi-GPU with high tile counts.

    Returns:
        True if the patch was applied, False otherwise.
    """
    import torch.nn.functional as F

    def _sdpa_attention(
        module,
        query,
        key,
        value,
        attention_mask=None,
        scaling=None,
        dropout=0.0,
        **kwargs,
    ):
        dp = dropout if module.training else 0.0

        # Expand KV heads to match Q heads so all SDPA backends are eligible.
        # enable_gqa=True dispatches to flash on PyTorch 2.9+ but produces
        # degraded accuracy and throughput in practice (64.4% vs 67.9%,
        # 3.48 vs 4.31 img/min) — likely due to attention mask interaction.
        # Manual expansion is proven reliable. See notebooks/sdpa_gqa_diagnostic.ipynb.
        num_kv_heads = key.shape[1]
        num_q_heads = query.shape[1]
        if num_kv_heads != num_q_heads:
            repeat_factor = num_q_heads // num_kv_heads
            key = key.repeat_interleave(repeat_factor, dim=1)
            value = value.repeat_interleave(repeat_factor, dim=1)

        # Prepare causal mask: truncate to KV length and ensure
        # head dim is broadcastable.
        causal_mask = attention_mask
        if causal_mask is not None:
            causal_mask = causal_mask[:, :, :, : key.shape[-2]]
            if causal_mask.shape[1] > 1 and causal_mask.shape[1] != num_q_heads:
                causal_mask = causal_mask[:, :1, :, :]

        is_causal = causal_mask is None and query.shape[2] > 1
        if is_causal:
            causal_mask = None

        attn_output = F.scaled_dot_product_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            attn_mask=causal_mask,
            dropout_p=dp,
            scale=scaling,
            is_causal=is_causal,
        )
        # Transpose to [batch, seq, heads, dim] to match eager_attention_forward's
        # output layout — the caller does .reshape(*input_shape, -1) which
        # requires seq before heads.
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, None

    patched = False

    # Patch the global attention function registry (transformers 4.46+)
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        ALL_ATTENTION_FUNCTIONS["eager"] = _sdpa_attention
        patched = True
    except (ImportError, AttributeError):
        pass

    # Also patch the module-level function reference as a fallback
    try:
        from transformers.models.qwen3 import modeling_qwen3

        modeling_qwen3.eager_attention_forward = _sdpa_attention
        patched = True
    except (ImportError, AttributeError):
        pass

    return patched


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

            # Check if native flash-attn package is installed
            _has_native_flash = False
            if cfg.flash_attn:
                try:
                    import flash_attn  # noqa: F401

                    _has_native_flash = True
                except ImportError:
                    pass

            if _has_native_flash:
                console.print("⚡ Flash Attention 2: ✅ native (flash-attn package)")
            elif cfg.flash_attn:
                console.print("⚡ Flash Attention 2: ✅ via SDPA patch")
            else:
                console.print("⚡ Flash Attention 2: ❌ disabled")

            # Only apply SDPA monkey-patch when flash-attn is NOT installed.
            # With native flash-attn, the model uses flash_attention_2 backend
            # directly — the eager->SDPA patch is unnecessary.
            global _sdpa_patched  # noqa: PLW0603
            if cfg.flash_attn and not _has_native_flash and not _sdpa_patched:
                if _patch_eager_attention_to_sdpa():
                    console.print("[bold]Patched eager attention -> SDPA[/bold]")
                    _sdpa_patched = True
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
        batch_size=config.batch_size,
        pre_loaded_model=model,
        pre_loaded_tokenizer=tokenizer,
        prompt_config=prompt_config,
        max_tiles=config.max_tiles,
        min_tiles=config.min_tiles,
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

# Same architecture, same API — just a larger model
register_model(
    ModelRegistration(
        model_type="internvl3-38b",
        loader=_internvl3_loader,
        processor_creator=_internvl3_processor_creator,
        prompt_file="internvl3_prompts.yaml",
        description="InternVL3.5-38B vision-language model (2x L40S)",
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

            # Llama uses PyTorch SDPA natively; flash_attention_2 attn_implementation
            # is incompatible (MllamaVisionAttention lacks is_causal attribute).
            console.print("⚡ Flash Attention 2: ❌ not applicable (Llama uses SDPA)")

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

            if not getattr(cfg, "_multi_gpu", False):
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
        batch_size=config.batch_size,
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


# ============================================================================
# Llama 4 Scout Registration (lazy imports — no torch/transformers at module level)
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

                model = Llama4ForConditionalGeneration.from_pretrained(
                    str(cfg.model_path),
                    **load_kwargs,
                )

                # Suppress spurious generation_config warnings
                if hasattr(model, "generation_config"):
                    model.generation_config.temperature = None
                    model.generation_config.top_p = None

                progress.update(task, description="Model loaded!")

            console.print("⚡ Flash Attention 2: ❌ not applicable (Llama 4 uses SDPA)")

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
# Nemotron Nano 2 VL Registration (lazy imports)
# ============================================================================


def _nemotron_loader(config):
    """Context manager for loading Nemotron Nano 2 VL model and processor.

    Uses AutoModelForCausalLM + AutoProcessor with trust_remote_code=True.
    Hybrid Transformer-Mamba architecture (CRadioV2-H vision encoder + Mamba SSM).
    BF16 (~24 GB) fits single L4; FP8 (~12 GB) leaves more headroom.
    """
    from contextlib import contextmanager

    import torch
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from transformers import AutoModelForCausalLM, AutoProcessor

    console = Console()

    @contextmanager
    def _loader(cfg):
        model = None
        processor = None

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            console.print(
                f"\n[bold]Loading Nemotron Nano 2 VL from: {cfg.model_path}[/bold]"
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

                model = AutoModelForCausalLM.from_pretrained(
                    str(cfg.model_path),
                    trust_remote_code=True,
                    device_map=cfg.device_map,
                    torch_dtype=cfg.torch_dtype,
                ).eval()

                # Suppress spurious generation_config warnings
                if hasattr(model, "generation_config"):
                    model.generation_config.temperature = None
                    model.generation_config.top_p = None

                progress.update(task, description="Model loaded!")

            console.print(
                "⚡ Architecture: hybrid Transformer-Mamba "
                "(flash-attn N/A — Mamba uses linear recurrence)"
            )

            if not getattr(cfg, "_multi_gpu", False):
                _print_gpu_status(console)

            yield model, processor

        finally:
            del model
            del processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return _loader(config)


def _nemotron_processor_creator(
    model,
    tokenizer_or_processor,
    config,
    prompt_config,
    universal_fields,
    field_definitions,
):
    """Nemotron full pipeline processor — not yet implemented.

    Use benchmark_sroie.py for SROIE evaluation.
    """
    raise NotImplementedError(
        "Nemotron Nano 2 VL full pipeline processor not yet implemented. "
        "Use benchmark_sroie.py for SROIE evaluation."
    )


register_model(
    ModelRegistration(
        model_type="nemotron",
        loader=_nemotron_loader,
        processor_creator=_nemotron_processor_creator,
        prompt_file="sroie_prompts.yaml",
        description="NVIDIA Nemotron Nano 12B v2 VL (hybrid Transformer-Mamba)",
    )
)
