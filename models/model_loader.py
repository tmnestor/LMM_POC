"""Generic model loader factories.

Provides ModelSpec and VllmSpec dataclasses and factory functions
that replace the ~600 lines of near-identical loader/processor_creator
functions in registry.py with ~8-line declarative registrations.

Usage:
    register_hf_model(ModelSpec(
        model_type="qwen3vl",
        model_class="Qwen3VLForConditionalGeneration",
        prompt_file="qwen3vl_prompts.yaml",
        message_style="two_step",
    ))
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

from models.backends.hf_chat_template import ChatTemplateConfig, HFChatTemplateBackend
from models.backends.vllm_backend import VllmBackend
from models.orchestrator import DocumentOrchestrator

type BackendFactory = Callable[[Any, Any, bool], Any]
type PostLoadHook = Callable[[Any, Any, Any], None]


@dataclass(frozen=True)
class ModelSpec:
    """Declarative specification for any HuggingFace model.

    Provides all the information needed to:
    1. Load the model and processor from disk (build_hf_loader)
    2. Create a backend (HFChatTemplateBackend or custom via backend_factory)
    3. Wire into a DocumentOrchestrator

    For standard HF models using apply_chat_template + generate, set
    message_style and related fields — HFChatTemplateBackend is used.

    For models with non-standard APIs (InternVL3's .chat(), Llama's
    template handling), provide a backend_factory callable that receives
    (model, processor, debug) and returns a ModelBackend.
    """

    model_type: str
    model_class: str  # e.g. "Qwen3VLForConditionalGeneration"
    processor_class: str = "AutoProcessor"
    prompt_file: str = "internvl3_prompts.yaml"
    description: str = ""
    requires_sharding: bool = False
    trust_remote_code: bool = False
    attn_implementation: str | None = None
    load_kwargs: dict[str, Any] = field(default_factory=dict)
    processor_kwargs: dict[str, Any] = field(default_factory=dict)
    suppress_gen_warnings: tuple[str, ...] = ("temperature", "top_p")
    # Backend selection — None => HFChatTemplateBackend (default)
    backend_factory: BackendFactory | None = None
    # Post-load hook — called as post_load(model, processor, cfg)
    post_load: PostLoadHook | None = None
    # HFChatTemplateBackend config (ignored when backend_factory is set)
    message_style: str = "two_step"
    system_message: str | None = None
    image_content_type: str = "image"
    image_content_key: str | None = None
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    generate_kwargs: dict[str, Any] = field(default_factory=dict)
    tokenizer_attr: str = "tokenizer"
    # Post-load model transforms
    merge_lora: bool = False
    force_single_gpu: bool = False
    # OOM recovery in orchestrator
    has_oom_recovery: bool = True


def build_hf_loader(spec: ModelSpec):
    """Build a context-manager loader from a ModelSpec.

    Returns a callable(config) -> ContextManager[(model, processor)].

    The loader accepts ``app_config`` for a uniform signature with the vLLM
    loader (so ``load_model`` can always pass it); HF models ignore it.
    """

    def _loader_fn(config, *, app_config: Any | None = None):
        import torch
        from rich.console import Console
        from rich.progress import Progress, SpinnerColumn, TextColumn

        from models.gpu_utils import print_gpu_status, quiet_loading

        console = Console()

        @contextmanager
        def _loader(cfg):
            model = None
            processor = None

            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                console.print(f"\n[bold]Loading {spec.model_type} from: {cfg.model_path}[/bold]")

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Loading processor...", total=None)

                    # Import processor class dynamically
                    import transformers

                    proc_cls = getattr(transformers, spec.processor_class)
                    proc_kwargs = dict(spec.processor_kwargs)
                    if spec.trust_remote_code:
                        proc_kwargs["trust_remote_code"] = True
                    processor = proc_cls.from_pretrained(str(cfg.model_path), **proc_kwargs)

                    progress.update(task, description="Loading model weights...")

                    # Import model class dynamically
                    model_cls = getattr(transformers, spec.model_class)

                    # For sharded models on multi-GPU, use InternVL3's
                    # split_model for better layer placement than auto.
                    effective_device_map = cfg.device_map
                    if (
                        spec.requires_sharding
                        and cfg.device_map == "auto"
                        and torch.cuda.device_count() > 1
                    ):
                        from models.sharding import split_internvl_model

                        effective_device_map = split_internvl_model(str(cfg.model_path))
                        console.print(
                            f"[bold cyan]Sharding across "
                            f"{torch.cuda.device_count()} GPUs "
                            f"(split_model)[/bold cyan]"
                        )

                    load_kwargs: dict[str, Any] = {
                        "device_map": effective_device_map,
                    }

                    # Use dtype= for newer models, torch_dtype for legacy
                    if spec.model_class in (
                        "AutoModelForCausalLM",
                        "AutoModelForImageTextToText",
                    ):
                        load_kwargs["torch_dtype"] = cfg.torch_dtype
                    else:
                        load_kwargs["dtype"] = cfg.torch_dtype

                    if spec.trust_remote_code:
                        load_kwargs["trust_remote_code"] = True
                    if spec.attn_implementation and cfg.flash_attn:
                        # Only pass attn_implementation=flash_attention_2 to
                        # from_pretrained when the flash_attn package is
                        # importable. HF validates FA2 strictly and raises
                        # ImportError before the SDPA patch below can run.
                        # When flash_attn is absent (e.g. prod), let HF use
                        # its default (eager) -- the SDPA monkey-patch at
                        # line ~202 redirects eager through
                        # F.scaled_dot_product_attention globally. This
                        # matches feature/multi-gpu behavior on prod where
                        # flash_attn has never been installed.
                        if spec.attn_implementation == "flash_attention_2":
                            try:
                                import flash_attn  # noqa: F401

                                load_kwargs["attn_implementation"] = spec.attn_implementation
                            except ImportError:
                                pass
                        else:
                            load_kwargs["attn_implementation"] = spec.attn_implementation

                    # Override device_map for single-GPU models
                    if spec.force_single_gpu:
                        load_kwargs["device_map"] = "cuda:0"
                        load_kwargs["dtype"] = torch.bfloat16

                    # Merge any model-specific load kwargs
                    load_kwargs.update(spec.load_kwargs)

                    # Expand callable values (e.g. lazy BitsAndBytesConfig)
                    for k, v in list(load_kwargs.items()):
                        if callable(v):
                            load_kwargs[k] = v(cfg)

                    with quiet_loading():
                        model = model_cls.from_pretrained(str(cfg.model_path), **load_kwargs)

                    # Optional post-load steps
                    if hasattr(model, "eval"):
                        model.eval()

                    if spec.merge_lora and hasattr(model, "merge_lora_adapters"):
                        progress.update(task, description="Merging LoRA adapters...")
                        model.merge_lora_adapters()

                    # Suppress generation_config warnings
                    if hasattr(model, "generation_config"):
                        for attr in spec.suppress_gen_warnings:
                            if hasattr(model.generation_config, attr):
                                setattr(model.generation_config, attr, None)

                    # Call model-specific post-load hook
                    if spec.post_load is not None:
                        spec.post_load(model, processor, cfg)

                    progress.update(task, description="Model loaded!")

                # SDPA patching for models that need it
                if spec.attn_implementation == "flash_attention_2" and cfg.flash_attn:
                    _apply_sdpa_if_needed(console, cfg)

                if not getattr(cfg, "_multi_gpu", False):
                    print_gpu_status(console)

                yield model, processor

            finally:
                del model
                del processor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return _loader(config)

    return _loader_fn


def _apply_sdpa_if_needed(console, cfg) -> None:
    """Apply SDPA patch if flash-attn is not natively installed."""
    from models.attention import (
        is_sdpa_patched,
        mark_sdpa_patched,
        patch_eager_attention_to_sdpa,
    )

    _has_native_flash = False
    try:
        import flash_attn  # noqa: F401

        _has_native_flash = True
    except ImportError:
        pass

    if _has_native_flash:
        console.print("Flash Attention 2: native (flash-attn package)")
    elif not is_sdpa_patched():
        if patch_eager_attention_to_sdpa():
            console.print("[bold]Patched eager attention -> SDPA[/bold]")
            mark_sdpa_patched()
        else:
            console.print("[yellow]Warning: could not patch attention to SDPA[/yellow]")


def build_hf_processor_creator(spec: ModelSpec):
    """Build a processor_creator callable from a ModelSpec.

    When spec.backend_factory is None, builds an HFChatTemplateBackend.
    When set, calls the factory to build a custom backend.
    Both are wrapped in a DocumentOrchestrator.
    """

    def _creator(
        model,
        tokenizer_or_processor,
        config,
        prompt_config,
        universal_fields,
        field_definitions,
        *,
        app_config=None,
    ):
        # debug=config.debug → Tier C (PARSING DEBUG, TENSOR_DTYPE, prompt/response dumps).
        # verbose=config.verbose → Tier B (init details, auto-batch, gen-config).
        if spec.backend_factory is not None:
            backend = spec.backend_factory(model, tokenizer_or_processor, config.debug)
        else:
            chat_config = ChatTemplateConfig(
                message_style=spec.message_style,
                system_message=spec.system_message,
                image_content_type=spec.image_content_type,
                image_content_key=spec.image_content_key,
                chat_template_kwargs=dict(spec.chat_template_kwargs),
                generate_kwargs=dict(spec.generate_kwargs),
                suppress_gen_warnings=spec.suppress_gen_warnings,
                tokenizer_attr=spec.tokenizer_attr,
            )

            backend = HFChatTemplateBackend(
                model=model,
                processor=tokenizer_or_processor,
                config=chat_config,
                debug=config.debug,
            )

        return DocumentOrchestrator(
            backend=backend,
            field_list=universal_fields,
            prompt_config=prompt_config,
            field_definitions=field_definitions,
            debug=config.debug,
            verbose=config.verbose,
            device=str(config.device_map),
            batch_size=config.batch_size,
            model_type_key=spec.model_type,
            app_config=app_config,
            has_oom_recovery=spec.has_oom_recovery,
        )

    return _creator


def register_hf_model(spec: ModelSpec) -> None:
    """Register a standard HF model with the registry.

    Builds loader + processor_creator from the spec and registers them.
    """
    from models.registry import ModelRegistration, register_model

    register_model(
        ModelRegistration(
            model_type=spec.model_type,
            loader=build_hf_loader(spec),
            processor_creator=build_hf_processor_creator(spec),
            prompt_file=spec.prompt_file,
            description=spec.description,
            requires_sharding=spec.requires_sharding,
        )
    )


# ============================================================================
# vLLM Loader Factory
# ============================================================================


@dataclass(frozen=True)
class VllmSpec:
    """Declarative specification for a vLLM model.

    Holds only *structural* fields. Engine tuning (gpu_memory_utilization,
    max_model_len, max_num_seqs, limit_mm_per_prompt, enable_prefix_caching)
    lives in ``config/run_config.yml`` under ``vllm:`` — the YAML is the
    single source of truth and is resolved at load time via
    ``app_config.get_vllm_config(model_type)``.
    """

    model_type: str
    prompt_file: str = "internvl3_prompts.yaml"
    description: str = ""
    attention_backend: str | None = None  # None = vLLM auto-selects
    mm_processor_kwargs: dict[str, Any] = field(default_factory=dict)
    hf_overrides: dict[str, Any] = field(default_factory=dict)
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)


def _resolve_vllm_overrides(app_config: Any | None, model_type: str) -> dict[str, Any]:
    """Resolve per-model vLLM engine tuning from the runtime AppConfig.

    The YAML is the single source of truth — there is no spec-level or Python
    fallback, so an AppConfig is mandatory for vLLM loads.

    Raises:
        ValueError: If *app_config* is None, with a four-element diagnostic.
    """
    if app_config is None:
        raise ValueError(
            f"vLLM model load requires an AppConfig, but none was provided for "
            f"model_type={model_type!r}.\n"
            "  What:  load_model() was called without app_config= for a vLLM backend; "
            "per-model engine tuning has no Python fallback.\n"
            "  Where: the caller of common.pipeline_ops.load_model (a stage, worker, or "
            "benchmark script), plus the `vllm:` section of config/run_config.yml.\n"
            "  Example: load_model(config, app_config=app_cfg)  # app_cfg = AppConfig.load(...)\n"
            "  Fix:   thread the AppConfig already in scope into load_model; scripts that "
            "call registration.loader() directly must pass loader(cfg, app_config=app_cfg)."
        ) from None
    return app_config.get_vllm_config(model_type)


def build_vllm_loader(spec: VllmSpec):
    """Build a context-manager vLLM loader from a VllmSpec.

    The returned loader accepts ``app_config`` at call time (mirroring
    ``create_processor``); per-model engine tuning is sourced from
    ``app_config.get_vllm_config(spec.model_type)``.
    """

    def _loader_fn(config, *, app_config: Any | None = None):
        from rich.console import Console

        console = Console()

        @contextmanager
        def _loader(cfg):
            import os

            os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

            from vllm import LLM

            llm = None

            try:
                # Detect GPU count: config.num_gpus > CUDA_VISIBLE_DEVICES > nvidia-smi
                if cfg.num_gpus and cfg.num_gpus > 0:
                    tp_size = cfg.num_gpus
                else:
                    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                    if cuda_visible is not None:
                        tp_size = len(cuda_visible.split(","))
                    else:
                        import subprocess

                        result = subprocess.run(
                            ["nvidia-smi", "-L"],
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        tp_size = len(result.stdout.strip().splitlines()) if result.returncode == 0 else 1
                tp_size = max(1, tp_size)

                # All engine tuning comes from app_config (YAML single source of
                # truth). No spec-level fallback — a vLLM model MUST be loaded
                # with an AppConfig. Keys are required, not optional, per the
                # "every config key is required" rule.
                vllm_overrides = _resolve_vllm_overrides(app_config, spec.model_type)

                # gpu_memory_utilization precedence: env var > YAML.
                gpu_mem_util = float(
                    os.environ.get(
                        "LMM_GPU_MEMORY_UTILIZATION",
                        vllm_overrides["gpu_memory_utilization"],
                    )
                )
                effective_max_model_len = vllm_overrides["max_model_len"]
                effective_max_num_seqs = vllm_overrides["max_num_seqs"]
                effective_enable_prefix_caching = vllm_overrides["enable_prefix_caching"]
                effective_limit_mm = vllm_overrides["limit_mm_per_prompt"]

                console.print(
                    f"\n[bold]Loading {spec.model_type} via vLLM "
                    f"(tp={tp_size}, max_model_len={effective_max_model_len})[/bold]"
                )
                console.print(f"[dim]Model path: {cfg.model_path}[/dim]")
                console.print(
                    f"[dim]enforce_eager: {cfg.enforce_eager} "
                    f"({'skip CUDA graphs' if cfg.enforce_eager else 'CUDA graphs enabled'})[/dim]"
                )
                if os.environ.get("LMM_GPU_MEMORY_UTILIZATION"):
                    console.print(
                        f"[dim]gpu_memory_utilization: {gpu_mem_util} "
                        f"(from LMM_GPU_MEMORY_UTILIZATION env var)[/dim]"
                    )
                elif vllm_overrides.get("gpu_memory_utilization") is not None:
                    console.print(f"[dim]gpu_memory_utilization: {gpu_mem_util} (from app_config)[/dim]")

                llm_kwargs: dict[str, Any] = {
                    "model": str(cfg.model_path),
                    "tensor_parallel_size": tp_size,
                    "max_model_len": effective_max_model_len,
                    "gpu_memory_utilization": gpu_mem_util,
                    "max_num_seqs": effective_max_num_seqs,
                    "limit_mm_per_prompt": {"image": effective_limit_mm},
                    "trust_remote_code": True,
                    "disable_log_stats": True,
                    "enforce_eager": cfg.enforce_eager,
                    "enable_prefix_caching": effective_enable_prefix_caching,
                }

                if spec.attention_backend is not None:
                    llm_kwargs["attention_backend"] = spec.attention_backend

                # mm_processor_kwargs: spec defaults, then app_config overlay
                effective_mm_proc = dict(spec.mm_processor_kwargs) if spec.mm_processor_kwargs else {}
                if vllm_overrides.get("mm_processor_kwargs"):
                    effective_mm_proc.update(vllm_overrides["mm_processor_kwargs"])
                if effective_mm_proc:
                    llm_kwargs["mm_processor_kwargs"] = effective_mm_proc

                if spec.hf_overrides:
                    llm_kwargs["hf_overrides"] = dict(spec.hf_overrides)

                llm = LLM(**llm_kwargs)

                console.print("[bold green]vLLM engine ready![/bold green]")

                yield llm, None

            finally:
                del llm

        return _loader(config)

    return _loader_fn


def build_vllm_processor_creator(spec: VllmSpec):
    """Build a processor_creator for a vLLM model."""

    def _creator(
        model,
        tokenizer_or_processor,
        config,
        prompt_config,
        universal_fields,
        field_definitions,
        *,
        app_config=None,
    ):
        # debug=config.debug → Tier C; verbose=config.verbose → Tier B. See
        # `plans/quiet-pipeline-output.md` for the split.
        backend = VllmBackend(
            engine=model,
            model_type_key=spec.model_type,
            debug=config.debug,
        )

        return DocumentOrchestrator(
            backend=backend,
            field_list=universal_fields,
            prompt_config=prompt_config,
            field_definitions=field_definitions,
            debug=config.debug,
            verbose=config.verbose,
            device=str(config.device_map),
            batch_size=config.batch_size,
            model_type_key=spec.model_type,
            app_config=app_config,
            has_oom_recovery=False,
        )

    return _creator


def register_vllm_model(spec: VllmSpec) -> None:
    """Register a vLLM model with the registry.

    Per-model engine tuning is resolved from the runtime AppConfig at load
    time (see ``build_vllm_loader``), so no config is captured at registration.

    Args:
        spec: Declarative vLLM model specification.
    """
    from models.registry import ModelRegistration, register_model

    register_model(
        ModelRegistration(
            model_type=spec.model_type,
            loader=build_vllm_loader(spec),
            processor_creator=build_vllm_processor_creator(spec),
            prompt_file=spec.prompt_file,
            description=spec.description,
            requires_sharding=True,
            is_vllm=True,
        )
    )
