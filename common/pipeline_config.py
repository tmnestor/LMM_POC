"""Pipeline configuration system for InternVL3.5-8B document extraction.

Provides configuration loading, merging, and validation without
framework dependencies (no typer, rich, or torch at module level).
"""

from __future__ import annotations

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
    bank_v2: bool = True
    balance_correction: bool = True

    # Model options
    max_tiles: int = 11
    flash_attn: bool = True
    dtype: str = "bfloat16"
    max_new_tokens: int = 2000

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


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load configuration from YAML file.

    Raises:
        FileNotFoundError: If config_path does not exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open() as f:
        config = yaml.safe_load(f)

    # Flatten nested structure
    flat_config: dict[str, Any] = {}
    if "model" in config:
        flat_config["model_path"] = config["model"].get("path")
        flat_config["max_tiles"] = config["model"].get("max_tiles")
        flat_config["flash_attn"] = config["model"].get("flash_attn")
        flat_config["dtype"] = config["model"].get("dtype")
        flat_config["max_new_tokens"] = config["model"].get("max_new_tokens")

    if "data" in config:
        flat_config["data_dir"] = config["data"].get("dir")
        flat_config["ground_truth"] = config["data"].get("ground_truth")
        flat_config["max_images"] = config["data"].get("max_images")
        flat_config["document_types"] = config["data"].get("document_types")

    if "output" in config:
        flat_config["output_dir"] = config["output"].get("dir")
        flat_config["skip_visualizations"] = config["output"].get("skip_visualizations")
        flat_config["skip_reports"] = config["output"].get("skip_reports")

    if "processing" in config:
        flat_config["bank_v2"] = config["processing"].get("bank_v2")
        flat_config["balance_correction"] = config["processing"].get(
            "balance_correction"
        )
        flat_config["verbose"] = config["processing"].get("verbose")

    # Remove None values
    return {k: v for k, v in flat_config.items() if v is not None}


def load_env_config() -> dict[str, Any]:
    """Load configuration from environment variables."""
    env_config: dict[str, Any] = {}

    env_mappings: dict[str, tuple[str, Any]] = {
        f"{ENV_PREFIX}DATA_DIR": ("data_dir", str),
        f"{ENV_PREFIX}OUTPUT_DIR": ("output_dir", str),
        f"{ENV_PREFIX}MODEL_PATH": ("model_path", str),
        f"{ENV_PREFIX}GROUND_TRUTH": ("ground_truth", str),
        f"{ENV_PREFIX}MAX_IMAGES": ("max_images", int),
        f"{ENV_PREFIX}MAX_TILES": ("max_tiles", int),
        f"{ENV_PREFIX}FLASH_ATTN": ("flash_attn", lambda x: x.lower() == "true"),
        f"{ENV_PREFIX}DTYPE": ("dtype", str),
        f"{ENV_PREFIX}BANK_V2": ("bank_v2", lambda x: x.lower() == "true"),
        f"{ENV_PREFIX}VERBOSE": ("verbose", lambda x: x.lower() == "true"),
    }

    for env_var, (config_key, converter) in env_mappings.items():
        value = os.environ.get(env_var)
        if value is not None:
            env_config[config_key] = converter(value)

    return env_config


def auto_detect_model_path() -> Path | None:
    """Auto-detect model path from common locations."""
    for path_str in DEFAULT_MODEL_PATHS:
        path = Path(path_str)
        if path.exists() and (path / "config.json").exists():
            return path
    return None


def merge_configs(
    cli_args: dict[str, Any],
    yaml_config: dict[str, Any],
    env_config: dict[str, Any],
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

    # Auto-detect model path if not specified
    if not merged.get("model_path"):
        detected = auto_detect_model_path()
        if detected:
            merged["model_path"] = detected

    return PipelineConfig(**merged)


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
            f"Specify with --model-path or IVL_MODEL_PATH environment variable."
        )
        return errors

    if not config.model_path.exists():
        errors.append(f"Model path not found: {config.model_path}")
        return errors

    # Validate ground truth if specified
    if config.ground_truth and not config.ground_truth.exists():
        errors.append(f"Ground truth file not found: {config.ground_truth}")

    # Validate dtype
    valid_dtypes = {"bfloat16", "float16", "float32"}
    if config.dtype not in valid_dtypes:
        errors.append(
            f"Invalid dtype: {config.dtype}. Valid options: {', '.join(valid_dtypes)}"
        )

    # Check for images in data directory
    images = list(discover_images(config.data_dir, config.document_types))
    if not images:
        exts = ", ".join(IMAGE_EXTENSIONS)
        errors.append(
            f"No images found in: {config.data_dir}. Supported formats: {exts}"
        )

    return errors
