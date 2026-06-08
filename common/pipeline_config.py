"""Pipeline configuration system for InternVL3.5-8B document extraction.

Provides configuration loading, merging, and validation without
framework dependencies (no typer, rich, or torch at module level).
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# ============================================================================
# Constants
# ============================================================================

DEFAULT_MODEL_PATHS = [
    "/home/jovyan/nfs_share/models/InternVL3_5-8B",
    "/models/InternVL3_5-8B",
    "./models/InternVL3_5-8B",
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}

ENV_PREFIX = "IVL_"

# Default structure suffixes (overridden by extraction YAML settings.structure_suffixes)
_DEFAULT_STRUCTURE_SUFFIXES = ("_flat", "_date_grouped")


def strip_structure_suffixes(key: str, suffixes: tuple[str, ...] = _DEFAULT_STRUCTURE_SUFFIXES) -> str:
    """Strip layout variant suffixes to get the base document type.

    e.g. "BANK_STATEMENT_FLAT" → "BANK_STATEMENT", "invoice" → "invoice"
    """
    result = key
    for suffix in suffixes:
        if result.upper().endswith(suffix.upper()):
            result = result[: -len(suffix)]
    return result


def load_structure_suffixes(
    extraction_yaml_path: Path | None = None,
) -> tuple[str, ...]:
    """Load structure suffixes from extraction YAML settings.

    Falls back to _DEFAULT_STRUCTURE_SUFFIXES if not specified.
    """
    if extraction_yaml_path is None:
        return _DEFAULT_STRUCTURE_SUFFIXES

    try:
        with extraction_yaml_path.open() as f:
            data = yaml.safe_load(f)
        suffixes = data.get("settings", {}).get("structure_suffixes")
        if suffixes:
            return tuple(suffixes)
    except Exception:
        pass

    return _DEFAULT_STRUCTURE_SUFFIXES


# ============================================================================
# Configuration Dataclass
# ============================================================================


@dataclass
class PipelineConfig:
    """Configuration for the extraction pipeline."""

    # Data paths
    data_dir: Path
    output_dir: Path
    model_path: Path | None = None
    ground_truth: Path | None = None

    # Processing options
    max_images: int | None = None
    document_types: list[str] | None = None
    batch_size: int | None = None  # None = auto-detect from VRAM
    bank_v2: bool = True
    balance_correction: bool = True

    # Model options
    model_type: str = "internvl3-vllm"
    max_tiles: int = 11
    min_tiles: int | None = None  # Set to enable adaptive quality-based tiling
    flash_attn: bool = True
    enforce_eager: bool = True  # vLLM only: True = skip CUDA graph compilation
    dtype: str = "bfloat16"
    max_new_tokens: int = 2000
    chat_template: str | None = None  # vLLM only: path to a chat-template override, or None
    trace_raw_prompts: bool = False  # debug: persist every VLM prompt/response to JSONL
    trace_path: str | None = None  # explicit trace JSONL path, or None for <output_dir> default
    # vLLM pre-tiling: crop images into per-doc-type tiles ourselves and hand
    # vLLM the crops as separate images (bypasses the checkpoint's max_dynamic_patch
    # cap for dense bank statements). See plans/2026-06-04-adaptive-pre-tiling.md.
    pre_tiling_enabled: bool = False
    pre_tiling_image_size: int = 448
    pre_tiling_use_thumbnail: bool = True

    # Model loading options
    trust_remote_code: bool = True
    device_map: str = "auto"

    # Multi-GPU options
    num_gpus: int = 0  # 0 = auto-detect all GPUs, 1 = single GPU, N = use N GPUs
    data_parallel_size: int | None = None  # None = auto (num_gpus for vLLM, ignored for HF)

    # Output options
    skip_visualizations: bool = False
    skip_reports: bool = False
    # verbose: Tier B output (init details, batch auto-detect, generation config,
    # per-image field counts). Default False — per-image progress is emitted as
    # logger.info at the stage level regardless of this flag.
    verbose: bool = False
    # debug: Tier C output (PARSING DEBUG, CONFIG DEBUG, TENSOR_DTYPE, prompt/
    # response dumps, error tracebacks). Strictly opt-in for developers.
    debug: bool = False

    # Runtime state (set during execution)
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    def __post_init__(self) -> None:
        """Convert string paths to Path objects."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.model_path, str):
            self.model_path = Path(self.model_path)
        if isinstance(self.ground_truth, str):
            self.ground_truth = Path(self.ground_truth)


# ============================================================================
# Configuration Loading
# ============================================================================


def _resolve_chat_template(model_cfg: dict[str, Any], config_path: Path) -> str | None:
    """Resolve and validate ``model.chat_template`` (vLLM chat-template override).

    Returns None for the explicit no-op values (``none`` / ``null`` / empty), or
    the path string for a real template. Fails fast — at config load, before any
    model work — if the key is missing or names a file that does not exist.

    Args:
        model_cfg: The ``model:`` mapping from run_config.yml.
        config_path: Path to the config file (for diagnostics).

    Returns:
        The validated template path, or None to use the model's own template.
    """
    if "chat_template" not in model_cfg:
        raise ValueError(
            "What: required key 'model.chat_template' is missing.\n"
            f"Where: {config_path} -> model.chat_template\n"
            "Expected: 'none' to use the model's own template, or a path to a "
            "*.jinja file, e.g.:\n  model:\n    chat_template: none\n"
            "How to fix: add 'chat_template: none' under the 'model:' section."
        )

    value = model_cfg["chat_template"]
    if value is None or str(value).strip().lower() in {"none", "null", ""}:
        return None

    template_path = Path(str(value)).expanduser()
    if not template_path.is_file():
        raise ValueError(
            f"What: 'model.chat_template' points to a file that does not exist: {value}\n"
            f"Where: {config_path} -> model.chat_template\n"
            "Expected: 'none' (use the model's template) or a path to an existing "
            "*.jinja chat-template file.\n"
            f"How to fix: correct the path, or set 'chat_template: none' in {config_path}."
        )
    return str(template_path)


def _resolve_tracing(raw_config: dict[str, Any], config_path: Path) -> tuple[bool, str | None]:
    """Resolve the optional ``tracing`` block (raw-prompt trace).

    An absent ``tracing`` block means tracing is off (it is a debug feature). If
    the block is present it must carry both ``raw_prompts`` (bool) and ``path``;
    fails fast on a malformed block, at config load.

    Returns:
        ``(enabled, path)`` where ``path`` is None for the no-op values
        (``none`` / ``null`` / empty) or an explicit string.
    """
    tracing = raw_config.get("tracing")
    if tracing is None:
        return False, None
    if not isinstance(tracing, dict) or "raw_prompts" not in tracing or "path" not in tracing:
        raise ValueError(
            "What: the 'tracing' block is malformed (needs 'raw_prompts' and 'path').\n"
            f"Where: {config_path} -> tracing\n"
            "Expected:\n  tracing:\n    raw_prompts: false\n    path: none\n"
            "How to fix: add both 'raw_prompts' (true/false) and 'path' under 'tracing'."
        )
    enabled = tracing["raw_prompts"]
    if not isinstance(enabled, bool):
        raise ValueError(
            f"What: 'tracing.raw_prompts' must be a boolean (got {enabled!r}).\n"
            f"Where: {config_path} -> tracing.raw_prompts\n"
            "Expected: true or false, e.g.: raw_prompts: false\n"
            "How to fix: set 'tracing.raw_prompts' to true or false."
        )
    path_raw = tracing["path"]
    path = (
        None if path_raw is None or str(path_raw).strip().lower() in {"none", "null", ""} else str(path_raw)
    )
    return enabled, path


def _resolve_pre_tiling(raw_config: dict[str, Any], config_path: Path) -> tuple[bool, int, bool]:
    """Resolve the optional ``pre_tiling`` block (vLLM app-side tiling).

    An absent ``pre_tiling`` block means pre-tiling is off (it is opt-in, like
    ``tracing``). If the block is present it must carry ``enabled`` (bool),
    ``image_size`` (positive int) and ``use_thumbnail`` (bool); fails fast on a
    malformed block, at config load.

    Returns:
        ``(enabled, image_size, use_thumbnail)``.
    """
    pre_tiling = raw_config.get("pre_tiling")
    if pre_tiling is None:
        return False, 448, True

    required = ("enabled", "image_size", "use_thumbnail")
    if not isinstance(pre_tiling, dict) or any(k not in pre_tiling for k in required):
        raise ValueError(
            "What: the 'pre_tiling' block is malformed (needs 'enabled', "
            "'image_size' and 'use_thumbnail').\n"
            f"Where: {config_path} -> pre_tiling\n"
            "Expected:\n  pre_tiling:\n    enabled: false\n    image_size: 448\n"
            "    use_thumbnail: true\n"
            "How to fix: add all three keys under 'pre_tiling', or remove the "
            "block entirely to disable pre-tiling."
        )

    enabled = pre_tiling["enabled"]
    use_thumbnail = pre_tiling["use_thumbnail"]
    image_size = pre_tiling["image_size"]
    if not isinstance(enabled, bool) or not isinstance(use_thumbnail, bool):
        raise ValueError(
            "What: 'pre_tiling.enabled' and 'pre_tiling.use_thumbnail' must be "
            f"booleans (got {enabled!r} and {use_thumbnail!r}).\n"
            f"Where: {config_path} -> pre_tiling.enabled / pre_tiling.use_thumbnail\n"
            "Expected: true or false, e.g.: enabled: false\n"
            "How to fix: set both to true or false."
        )
    if not isinstance(image_size, int) or image_size < 1:
        raise ValueError(
            f"What: 'pre_tiling.image_size' must be a positive integer (got {image_size!r}).\n"
            f"Where: {config_path} -> pre_tiling.image_size\n"
            "Expected: the InternVL tile size in pixels, e.g.: image_size: 448\n"
            "How to fix: set 'pre_tiling.image_size' to a positive integer (448 for InternVL3)."
        )
    return enabled, image_size, use_thumbnail


def load_yaml_config(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load configuration from YAML file.

    Returns:
        Tuple of (flat_config for PipelineConfig, raw_config for apply_yaml_overrides).

    Raises:
        FileNotFoundError: If config_path does not exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open() as f:
        raw_config = yaml.safe_load(f)

    # Flatten nested structure for PipelineConfig
    flat_config: dict[str, Any] = {}
    model_cfg = raw_config.get("bootstrap", {}).get("model", {})
    if model_cfg:
        flat_config["model_type"] = model_cfg.get("type")
        flat_config["model_path"] = model_cfg.get("path")
        flat_config["max_tiles"] = model_cfg.get("max_tiles")
        flat_config["min_tiles"] = model_cfg.get("min_tiles")
        flat_config["flash_attn"] = model_cfg.get("flash_attn")
        flat_config["enforce_eager"] = model_cfg.get("enforce_eager")
        flat_config["dtype"] = model_cfg.get("dtype")
        flat_config["max_new_tokens"] = model_cfg.get("max_new_tokens")
        flat_config["chat_template"] = _resolve_chat_template(model_cfg, config_path)

    if "data" in raw_config:
        flat_config["data_dir"] = raw_config["data"].get("dir")
        flat_config["ground_truth"] = raw_config["data"].get("ground_truth")
        flat_config["max_images"] = raw_config["data"].get("max_images")
        flat_config["document_types"] = raw_config["data"].get("document_types")

    if "output" in raw_config:
        flat_config["output_dir"] = raw_config["output"].get("dir")
        flat_config["skip_visualizations"] = raw_config["output"].get("skip_visualizations")
        flat_config["skip_reports"] = raw_config["output"].get("skip_reports")

    if "processing" in raw_config:
        flat_config["batch_size"] = raw_config["processing"].get("batch_size")
        flat_config["bank_v2"] = raw_config["processing"].get("bank_v2")
        flat_config["balance_correction"] = raw_config["processing"].get("balance_correction")
        flat_config["verbose"] = raw_config["processing"].get("verbose")
        flat_config["debug"] = raw_config["processing"].get("debug")

    gpus_cfg = raw_config.get("bootstrap", {}).get("gpus", {})
    flat_config["num_gpus"] = gpus_cfg.get("num_gpus")
    flat_config["data_parallel_size"] = gpus_cfg.get("data_parallel_size")

    # Flatten model loading options (device_map / trust_remote_code now live
    # under bootstrap.model) into PipelineConfig fields.
    if "trust_remote_code" in model_cfg:
        flat_config["trust_remote_code"] = model_cfg["trust_remote_code"]
    if "device_map" in model_cfg:
        flat_config["device_map"] = str(model_cfg["device_map"])

    # Raw-prompt trace (debug observability; absent block -> off)
    trace_enabled, trace_path = _resolve_tracing(raw_config, config_path)
    flat_config["trace_raw_prompts"] = trace_enabled
    if trace_path is not None:
        flat_config["trace_path"] = trace_path

    # vLLM pre-tiling (opt-in; absent block -> off)
    pre_tiling_enabled, pre_tiling_image_size, pre_tiling_use_thumbnail = _resolve_pre_tiling(
        raw_config, config_path
    )
    flat_config["pre_tiling_enabled"] = pre_tiling_enabled
    flat_config["pre_tiling_image_size"] = pre_tiling_image_size
    flat_config["pre_tiling_use_thumbnail"] = pre_tiling_use_thumbnail

    # Remove None values from flat config
    flat_config = {k: v for k, v in flat_config.items() if v is not None}

    return flat_config, raw_config


def load_env_config() -> dict[str, Any]:
    """Load configuration from environment variables."""
    env_config: dict[str, Any] = {}

    env_mappings: dict[str, tuple[str, Any]] = {
        f"{ENV_PREFIX}DATA_DIR": ("data_dir", str),
        f"{ENV_PREFIX}OUTPUT_DIR": ("output_dir", str),
        f"{ENV_PREFIX}MODEL_TYPE": ("model_type", str),
        f"{ENV_PREFIX}MODEL_PATH": ("model_path", str),
        f"{ENV_PREFIX}GROUND_TRUTH": ("ground_truth", str),
        f"{ENV_PREFIX}MAX_IMAGES": ("max_images", int),
        f"{ENV_PREFIX}BATCH_SIZE": ("batch_size", int),
        f"{ENV_PREFIX}NUM_GPUS": ("num_gpus", int),
        f"{ENV_PREFIX}MAX_TILES": ("max_tiles", int),
        f"{ENV_PREFIX}MIN_TILES": ("min_tiles", int),
        f"{ENV_PREFIX}FLASH_ATTN": ("flash_attn", lambda x: x.lower() == "true"),
        f"{ENV_PREFIX}ENFORCE_EAGER": ("enforce_eager", lambda x: x.lower() == "true"),
        f"{ENV_PREFIX}DTYPE": ("dtype", str),
        f"{ENV_PREFIX}BANK_V2": ("bank_v2", lambda x: x.lower() == "true"),
        f"{ENV_PREFIX}VERBOSE": ("verbose", lambda x: x.lower() == "true"),
        f"{ENV_PREFIX}DEBUG": ("debug", lambda x: x.lower() == "true"),
        f"{ENV_PREFIX}DATA_PARALLEL_SIZE": ("data_parallel_size", int),
    }

    for env_var, (config_key, converter) in env_mappings.items():
        value = os.environ.get(env_var)
        if value is not None:
            env_config[config_key] = converter(value)

    return env_config


def auto_detect_model_path(
    search_paths: list[str] | None = None,
) -> Path | None:
    """Auto-detect model path from common locations.

    Args:
        search_paths: Custom list of paths to search. Falls back to DEFAULT_MODEL_PATHS.
    """
    paths = search_paths or DEFAULT_MODEL_PATHS
    for path_str in paths:
        path = Path(path_str)
        if path.exists() and (path / "config.json").exists():
            return path
    return None


def merge_configs(
    cli_args: dict[str, Any],
    yaml_config: dict[str, Any],
    env_config: dict[str, Any],
    raw_config: dict[str, Any] | None = None,
) -> PipelineConfig:
    """Merge configs with CLI > YAML > ENV > defaults priority."""
    # Start with defaults (handled by dataclass)
    merged: dict[str, Any] = {}

    # Layer in env config (lowest priority)
    merged.update({k: v for k, v in env_config.items() if v is not None})

    # Layer in YAML config
    merged.update({k: v for k, v in yaml_config.items() if v is not None})

    # Layer in CLI args (highest priority)
    merged.update({k: v for k, v in cli_args.items() if v is not None})

    # If CLI overrides model_type but not model_path, the YAML model_path
    # belongs to the YAML's model_type — clear it so auto-detect can resolve
    # the correct path for the CLI-requested model.
    if "model_type" in cli_args and "model_path" not in cli_args:
        yaml_model_type = yaml_config.get("model_type")
        if yaml_model_type and yaml_model_type != cli_args["model_type"]:
            merged.pop("model_path", None)

    # Auto-detect model path if not specified
    if not merged.get("model_path"):
        model_type = merged.get("model_type", "internvl3-vllm")
        search_paths = _resolve_default_paths(raw_config, model_type)
        detected = auto_detect_model_path(search_paths)
        if detected:
            merged["model_path"] = detected

    return PipelineConfig(**merged)


def _resolve_default_paths(
    raw_config: dict[str, Any] | None,
    model_type: str,
) -> list[str] | None:
    """Resolve model search paths from config, preferring the requested model_type.

    Supports both dict form ({model_type: path}) and legacy list form ([path, ...]).
    """
    if not raw_config or not raw_config.get("bootstrap", {}).get("model", {}).get("default_paths"):
        return None

    default_paths = raw_config["bootstrap"]["model"]["default_paths"]

    if isinstance(default_paths, dict):
        # Dict form: try the requested model_type first, then all paths
        type_path = default_paths.get(model_type)
        if type_path:
            return [type_path]
        return list(default_paths.values())

    if isinstance(default_paths, list):
        return default_paths

    return None


# ============================================================================
# Image Discovery
# ============================================================================


def discover_images(
    data_dir: Path,
    document_types: list[str] | None = None,
) -> list[Path]:
    """Discover images in data directory."""
    images: list[Path] = []

    for ext in IMAGE_EXTENSIONS:
        images.extend(data_dir.glob(f"*{ext}"))
        images.extend(data_dir.glob(f"*{ext.upper()}"))

    # Sort by filename for reproducibility
    images = sorted(images, key=lambda p: p.name.lower())

    # Filter by document type if specified (based on filename patterns)
    if document_types:
        filtered = []
        type_patterns = [t.lower() for t in document_types]
        for img in images:
            name_lower = img.name.lower()
            if any(pattern in name_lower for pattern in type_patterns):
                filtered.append(img)
        images = filtered

    return images


# ============================================================================
# Validation
# ============================================================================


def validate_config(config: PipelineConfig) -> list[str]:
    """Validate configuration and return list of error messages.

    Returns an empty list if configuration is valid.
    """
    errors: list[str] = []

    # Validate data directory
    if not config.data_dir.exists():
        errors.append(f"Data directory not found: {config.data_dir}")
        return errors

    # Validate model path
    if not config.model_path:
        searched = ", ".join(DEFAULT_MODEL_PATHS)
        errors.append(
            f"Model path not specified and could not be auto-detected. "
            f"Searched: {searched}. "
            f"Specify with --model-path or IVL_MODEL_PATH environment variable."
        )
        return errors

    if not config.model_path.exists():
        errors.append(f"Model path not found: {config.model_path}")
        return errors

    # Ground truth is only needed by the evaluate stage, which validates it
    # at its own entry point — skip here so classify/extract/clean don't fail.

    # Validate num_gpus
    if config.num_gpus < 0:
        errors.append(f"Invalid num_gpus: {config.num_gpus}. Must be >= 0 (0 = auto).")

    # Validate batch_size
    if config.batch_size is not None and config.batch_size < 1:
        errors.append(f"Invalid batch_size: {config.batch_size}. Must be >= 1.")

    # Validate dtype
    valid_dtypes = {"bfloat16", "float16", "float32"}
    if config.dtype not in valid_dtypes:
        errors.append(f"Invalid dtype: {config.dtype}. Valid options: {', '.join(valid_dtypes)}")

    # Note: image discovery check is done by the caller after discover_images()
    # to avoid scanning the data directory twice.

    return errors
