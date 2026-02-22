"""Pipeline configuration system for InternVL3.5-8B document extraction.

Provides configuration loading, merging, and validation without
framework dependencies (no typer, rich, or torch at module level).
"""

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


# Default structure suffixes (overridden by extraction YAML settings.structure_suffixes)
_DEFAULT_STRUCTURE_SUFFIXES = ("_flat", "_date_grouped")


def strip_structure_suffixes(
    key: str, suffixes: tuple[str, ...] = _DEFAULT_STRUCTURE_SUFFIXES
) -> str:
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
    model_type: str = "internvl3"
    max_tiles: int = 11
    flash_attn: bool = True
    dtype: str = "bfloat16"
    max_new_tokens: int = 2000

    # Model loading options
    trust_remote_code: bool = True
    use_fast_tokenizer: bool = False
    low_cpu_mem_usage: bool = True
    device_map: str = "auto"

    # Multi-GPU options
    num_gpus: int = 0  # 0 = auto-detect all GPUs, 1 = single GPU, N = use N GPUs

    # Output options
    skip_visualizations: bool = False
    skip_reports: bool = False
    verbose: bool = True

    # Runtime state (set during execution)
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )

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

    @property
    def torch_dtype(self):
        """Convert dtype string to torch.dtype (lazy import)."""
        import torch

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.dtype, torch.bfloat16)


# ============================================================================
# Configuration Loading
# ============================================================================


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
    if "model" in raw_config:
        flat_config["model_type"] = raw_config["model"].get("type")
        flat_config["model_path"] = raw_config["model"].get("path")
        flat_config["max_tiles"] = raw_config["model"].get("max_tiles")
        flat_config["flash_attn"] = raw_config["model"].get("flash_attn")
        flat_config["dtype"] = raw_config["model"].get("dtype")
        flat_config["max_new_tokens"] = raw_config["model"].get("max_new_tokens")

    if "data" in raw_config:
        flat_config["data_dir"] = raw_config["data"].get("dir")
        flat_config["ground_truth"] = raw_config["data"].get("ground_truth")
        flat_config["max_images"] = raw_config["data"].get("max_images")
        flat_config["document_types"] = raw_config["data"].get("document_types")

    if "output" in raw_config:
        flat_config["output_dir"] = raw_config["output"].get("dir")
        flat_config["skip_visualizations"] = raw_config["output"].get(
            "skip_visualizations"
        )
        flat_config["skip_reports"] = raw_config["output"].get("skip_reports")

    if "processing" in raw_config:
        flat_config["batch_size"] = raw_config["processing"].get("batch_size")
        flat_config["bank_v2"] = raw_config["processing"].get("bank_v2")
        flat_config["balance_correction"] = raw_config["processing"].get(
            "balance_correction"
        )
        flat_config["verbose"] = raw_config["processing"].get("verbose")
        flat_config["num_gpus"] = raw_config["processing"].get("num_gpus")

    # Flatten model_loading options into PipelineConfig fields
    if "model_loading" in raw_config:
        ml = raw_config["model_loading"]
        if "trust_remote_code" in ml:
            flat_config["trust_remote_code"] = ml["trust_remote_code"]
        if "use_fast_tokenizer" in ml:
            flat_config["use_fast_tokenizer"] = ml["use_fast_tokenizer"]
        if "low_cpu_mem_usage" in ml:
            flat_config["low_cpu_mem_usage"] = ml["low_cpu_mem_usage"]
        if "device_map" in ml:
            flat_config["device_map"] = str(ml["device_map"])

    # Remove None values from flat config
    flat_config = {k: v for k, v in flat_config.items() if v is not None}

    return flat_config, raw_config


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
    raw_config: dict[str, Any] | None = None,
) -> PipelineConfig:
    """Merge configs with CLI > YAML > defaults priority."""
    # Start with defaults (handled by dataclass)
    merged: dict[str, Any] = {}

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
        model_type = merged.get("model_type", "internvl3")
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
    if not raw_config or "model_loading" not in raw_config:
        return None

    default_paths = raw_config["model_loading"].get("default_paths")
    if default_paths is None:
        return None

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
    images = []

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
            f"Specify with --model-path or model.path in run_config.yml."
        )
        return errors

    if not config.model_path.exists():
        errors.append(f"Model path not found: {config.model_path}")
        return errors

    # Ground truth is optional — clear it if the path doesn't exist.
    # Stages that require it (evaluate) validate independently.
    if config.ground_truth and not config.ground_truth.exists():
        config.ground_truth = None

    # Validate num_gpus
    if config.num_gpus < 0:
        errors.append(f"Invalid num_gpus: {config.num_gpus}. Must be >= 0 (0 = auto).")

    # Validate batch_size
    if config.batch_size is not None and config.batch_size < 1:
        errors.append(f"Invalid batch_size: {config.batch_size}. Must be >= 1.")

    # Validate dtype
    valid_dtypes = {"bfloat16", "float16", "float32"}
    if config.dtype not in valid_dtypes:
        errors.append(
            f"Invalid dtype: {config.dtype}. Valid options: {', '.join(valid_dtypes)}"
        )

    # Note: image discovery check is done by the caller after discover_images()
    # to avoid scanning the data directory twice.

    return errors
