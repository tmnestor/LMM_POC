"""vLLM model loader factory.

Provides the ``VllmSpec`` dataclass and factory functions that turn an ~8-line
declarative registration into a vLLM engine loader + processor creator wired
into a ``DocumentOrchestrator``.

Usage:
    register_vllm_model(VllmSpec(
        model_type="internvl3-vllm",
        prompt_file="internvl3_prompts.yaml",
    ))
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from common.prompt_trace import effective_trace_path
from models.backends.vllm_backend import VllmBackend
from models.orchestrator import DocumentOrchestrator


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


def ensure_corrected_tokenizer(model_path: str) -> str:
    """Return a path to a ``fix_mistral_regex``-corrected copy of the tokenizer.

    InternVL3.5 ships a Mistral-based tokenizer whose default pre-tokenizer regex
    mis-splits runs of whitespace and digits (transformers warns at load and
    advises ``fix_mistral_regex=True``). On dense bank statements — whitespace-
    aligned, digit-heavy amount columns — that corrupts amount tokenization.

    vLLM loads its tokenizer internally in BOTH the front-end and every spawned
    EngineCore child, and exposes no kwarg to set the flag, so a load-time
    monkeypatch cannot reach the child. Instead we bake the corrected regex into a
    tokenizer saved to disk once and hand vLLM that directory via
    ``LLM(tokenizer=...)``; every process then loads the already-fixed files.

    Idempotent and concurrency-safe: the corrected copy is written to a private
    temp dir and atomically renamed into place, so the parallel DP workers that
    each call this can't corrupt a shared cache. Cached under
    ``$LMM_TOKENIZER_CACHE`` (default ``~/.cache/lmm_poc_tokenizers/<model-name>``)
    and reused across loads — the source model dir is typically read-only NFS, so
    we never write next to it. ``entrypoint.sh`` pre-warms this once before the
    workers spawn; this function is also safe to call directly.
    """
    import os
    import shutil
    from pathlib import Path

    cache_root = os.environ.get("LMM_TOKENIZER_CACHE") or str(Path.home() / ".cache" / "lmm_poc_tokenizers")
    out_dir = Path(cache_root) / Path(model_path).name
    if (out_dir / "tokenizer_config.json").exists():
        return str(out_dir)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, fix_mistral_regex=True, trust_remote_code=True
    )
    tmp_dir = out_dir.parent / f".{out_dir.name}.tmp.{os.getpid()}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(str(tmp_dir))
    try:
        os.replace(tmp_dir, out_dir)  # atomic; loses harmlessly if a racer won
    except OSError:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return str(out_dir)


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

                # Pre-tiling hands vLLM one image PER tile (plus a thumbnail), so
                # limit_mm_per_prompt must admit the largest per-doc-type tile
                # budget + 1. Fail fast here — otherwise vLLM silently drops the
                # extra sub-images and the dense bank statements stay under-tiled.
                if cfg.pre_tiling_enabled and "internvl" in spec.model_type:
                    max_budget = app_config.max_image_budget_tiles()
                    needed = max_budget + 1  # tiles + thumbnail
                    if effective_limit_mm < needed:
                        raise ValueError(
                            "What: pre-tiling is enabled but "
                            f"vllm.models.{spec.model_type}.limit_mm_per_prompt="
                            f"{effective_limit_mm} is too low — pre-tiling sends one "
                            f"image per tile, and the largest image budget is "
                            f"{max_budget} tiles (+1 thumbnail = {needed}).\n"
                            f"  Where: config/run_config.yml -> vllm.models."
                            f"{spec.model_type}.limit_mm_per_prompt (and image_budgets.*.max_tiles)\n"
                            f"  Expected: limit_mm_per_prompt >= {needed}, e.g.:\n"
                            f"    vllm:\n      models:\n        {spec.model_type}:\n"
                            f"          limit_mm_per_prompt: {needed}\n"
                            f"  How to fix: raise limit_mm_per_prompt to at least "
                            f"{needed} for {spec.model_type}, or set pre_tiling.enabled: "
                            f"false to run the single-image baseline."
                        ) from None

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

                # InternVL3.5's Mistral tokenizer ships a buggy whitespace/digit
                # regex; hand vLLM a fix_mistral_regex-corrected copy on disk so
                # BOTH the front-end and every EngineCore child load the fixed
                # tokenizer (a load-time patch can't reach the spawned child).
                # entrypoint.sh pre-warms this cache before the workers spawn;
                # the call here is the idempotent fallback for un-warmed runs.
                # Best-effort: a degraded (buggy-regex) tokenizer still runs, so a
                # build hiccup must not brick a prod engine load — log and proceed.
                if "internvl" in spec.model_type:
                    try:
                        llm_kwargs["tokenizer"] = ensure_corrected_tokenizer(str(cfg.model_path))
                    except Exception as exc:  # noqa: BLE001 - degraded tokenizer beats a dead engine
                        console.print(
                            f"[yellow]WARNING: could not build fix_mistral_regex tokenizer "
                            f"({exc}); loading the model's own tokenizer — the whitespace/"
                            f"digit regex bug remains.[/yellow]"
                        )

                if spec.attention_backend is not None:
                    llm_kwargs["attention_backend"] = spec.attention_backend

                # mm_processor_kwargs: spec defaults, then app_config overlay
                effective_mm_proc = dict(spec.mm_processor_kwargs) if spec.mm_processor_kwargs else {}
                if vllm_overrides.get("mm_processor_kwargs"):
                    effective_mm_proc.update(vllm_overrides["mm_processor_kwargs"])
                if effective_mm_proc:
                    llm_kwargs["mm_processor_kwargs"] = effective_mm_proc

                # hf_overrides: spec defaults, then app_config (run_config) overlay.
                # InternVL tile count (max_dynamic_patch) must be set HERE, by
                # overriding the HF config field — NOT via mm_processor_kwargs,
                # which vLLM forwards to the video processor constructor (which
                # rejects it: InternVLVideoProcessor has no max_dynamic_patch).
                effective_hf = dict(spec.hf_overrides) if spec.hf_overrides else {}
                if vllm_overrides.get("hf_overrides"):
                    effective_hf.update(vllm_overrides["hf_overrides"])
                if effective_hf:
                    llm_kwargs["hf_overrides"] = effective_hf

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
            chat_template=config.chat_template,
            trace_path=effective_trace_path(config),
            pre_tiling_enabled=config.pre_tiling_enabled,
            tile_image_size=config.pre_tiling_image_size,
            tile_use_thumbnail=config.pre_tiling_use_thumbnail,
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
