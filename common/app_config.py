"""Unified application configuration.

Single entry point (AppConfig.load) replaces the 7-step config dance
in cli.py and eliminates mutable module globals.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from common.field_schema import FieldSchema, get_field_schema

if TYPE_CHECKING:
    from common.pipeline_config import PipelineConfig


class ConfigError(Exception):
    """Raised by AppConfig.load() when validation fails."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


@dataclass(frozen=True)
class BatchSettings:
    """Typed replacement for the 13 mutable batch/GPU globals in model_config.py."""

    default_sizes: dict[str, int]
    max_sizes: dict[str, int]
    conservative_sizes: dict[str, int]
    min_size: int = 1
    strategy: str = "balanced"
    auto_detect: bool = True
    memory_safety_margin: float = 0.8
    clear_cache_after_batch: bool = True
    timeout_seconds: int = 300
    fallback_enabled: bool = True
    fallback_steps: tuple[int, ...] = (8, 4, 2, 1)
    gpu_memory_thresholds: dict[str, int] = field(
        default_factory=lambda: {"low": 8, "medium": 16, "high": 24, "very_high": 64}
    )

    # Batch size defaults (formerly mutable globals in model_config.py)
    _DEFAULT_SIZES: ClassVar[dict[str, int]] = {
        "internvl3": 4,
        "internvl3-2b": 4,
        "internvl3-8b": 4,
        "qwen3vl": 4,
    }
    _MAX_SIZES: ClassVar[dict[str, int]] = {
        "internvl3": 8,
        "internvl3-2b": 8,
        "internvl3-8b": 16,
        "qwen3vl": 8,
    }
    _CONSERVATIVE_SIZES: ClassVar[dict[str, int]] = {
        "internvl3": 1,
        "internvl3-2b": 2,
        "internvl3-8b": 1,
        "qwen3vl": 2,
    }

    @classmethod
    def from_raw(cls, raw_config: dict) -> BatchSettings:
        """Build from raw YAML config (replaces apply_yaml_overrides batch/gpu sections)."""
        default_sizes = dict(cls._DEFAULT_SIZES)
        max_sizes = dict(cls._MAX_SIZES)
        conservative_sizes = dict(cls._CONSERVATIVE_SIZES)
        kwargs: dict[str, Any] = {}

        batch = raw_config.get("batch", {})
        if batch:
            if "default_sizes" in batch:
                default_sizes.update(batch["default_sizes"])
            if "max_sizes" in batch:
                max_sizes.update(batch["max_sizes"])
            if "conservative_sizes" in batch:
                conservative_sizes.update(batch["conservative_sizes"])
            for key in (
                "min_size",
                "strategy",
                "auto_detect",
                "memory_safety_margin",
                "clear_cache_after_batch",
                "timeout_seconds",
                "fallback_enabled",
            ):
                if key in batch:
                    kwargs[key] = batch[key]
            if "fallback_steps" in batch:
                kwargs["fallback_steps"] = tuple(batch["fallback_steps"])

        # GPU memory thresholds
        thresholds = {"low": 8, "medium": 16, "high": 24, "very_high": 64}
        gpu = raw_config.get("gpu", {})
        if gpu:
            mem_thresholds = gpu.get("memory_thresholds", {})
            if mem_thresholds:
                thresholds.update(mem_thresholds)
        kwargs["gpu_memory_thresholds"] = thresholds

        return cls(
            default_sizes=default_sizes,
            max_sizes=max_sizes,
            conservative_sizes=conservative_sizes,
            **kwargs,
        )


def _build_generation_registry(raw_config: dict) -> dict[str, dict]:
    """Build generation config registry with YAML overrides applied.

    Deep-copies the base registry from model_config, then applies
    YAML overrides. Returns the result without mutating the originals.
    """
    from common.model_config import _GENERATION_CONFIG_REGISTRY

    registry = copy.deepcopy(_GENERATION_CONFIG_REGISTRY)

    gen = raw_config.get("generation", {})
    if gen and "internvl3" in registry:
        ivl_config = registry["internvl3"]
        for key in (
            "max_new_tokens_base",
            "max_new_tokens_per_field",
            "do_sample",
            "use_cache",
        ):
            if key in gen:
                ivl_config[key] = gen[key]

    return registry


class AppConfig:
    """Unified, immutable configuration surface.

    Constructed once at startup, threaded to all consumers.
    Replaces the 7-step config dance in cli.py and eliminates
    all mutable module globals in model_config and field_config.
    """

    __slots__ = (
        "pipeline",
        "batch",
        "fields",
        "_generation_registry",
        "_token_limits",
        "_min_tokens_by_type",
    )

    def __init__(
        self,
        pipeline: "PipelineConfig",
        batch: BatchSettings,
        fields: FieldSchema,
        generation_registry: dict[str, dict],
        token_limits: dict[str, int | None] | None = None,
        min_tokens_by_type: dict[str, int] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.batch = batch
        self.fields = fields
        self._generation_registry = generation_registry
        self._token_limits = token_limits or {}
        self._min_tokens_by_type = min_tokens_by_type or {}

    @classmethod
    def load(
        cls,
        cli_args: dict[str, Any],
        *,
        config_path: Path | None = None,
    ) -> AppConfig:
        """Single entry point. No mutable globals touched.

        Handles: YAML loading, ENV loading, merge (CLI > YAML > ENV > defaults),
        validation, batch settings, generation registry, field schema.

        Raises:
            ConfigError: On validation failure.
            FileNotFoundError: If an explicit config_path does not exist.
        """
        from common.pipeline_config import (
            load_env_config,
            load_yaml_config,
            merge_configs,
            validate_config,
        )

        # 1. Resolve config path
        default = Path(__file__).parent.parent / "config" / "run_config.yml"
        resolved = config_path or (default if default.exists() else None)

        # 2. Load YAML
        yaml_config: dict[str, Any] = {}
        raw_config: dict[str, Any] = {}
        if resolved:
            yaml_config, raw_config = load_yaml_config(resolved)

        # 3. Load ENV
        env_config = load_env_config()

        # 4. Pre-validate required fields (before PipelineConfig construction)
        merged_preview = {**env_config, **yaml_config, **cli_args}
        errors: list[str] = []
        if not merged_preview.get("data_dir"):
            errors.append(
                "--data-dir is required "
                "(via CLI, config file, or IVL_DATA_DIR environment variable)"
            )
        if not merged_preview.get("output_dir"):
            errors.append(
                "--output-dir is required "
                "(via CLI, config file, or IVL_OUTPUT_DIR environment variable)"
            )
        if errors:
            raise ConfigError(errors)

        # 5. Merge with precedence: CLI > YAML > ENV > defaults
        pipeline = merge_configs(cli_args, yaml_config, env_config, raw_config)

        # 6. Validate
        val_errors = validate_config(pipeline)
        if val_errors:
            raise ConfigError(val_errors)

        # 7. Build BatchSettings (immutable copy of batch/gpu config)
        batch = BatchSettings.from_raw(raw_config)

        # 8. Build generation registry (immutable copy with YAML overrides)
        generation_registry = _build_generation_registry(raw_config)

        # 9. Build token limits from YAML overrides
        token_limits: dict[str, int | None] = {"2b": None, "8b": 800}
        gen = raw_config.get("generation", {})
        yaml_limits = gen.get("token_limits", {})
        for size_key, value in yaml_limits.items():
            token_limits[str(size_key)] = value

        # 10-11. Load field schema (single YAML read) and extract min_tokens
        fields = get_field_schema()

        return cls(
            pipeline=pipeline,
            batch=batch,
            fields=fields,
            generation_registry=generation_registry,
            token_limits=token_limits,
            min_tokens_by_type=fields.min_tokens_by_type,
        )

    # -- Drop-in replacements for model_config functions -----------------------

    # Fallback generation config for unknown model types
    _FALLBACK_GENERATION_CONFIG: dict[str, Any] = {
        "max_new_tokens_base": 512,
        "max_new_tokens_per_field": 64,
        "temperature": 0.0,
        "do_sample": False,
        "top_p": 0.95,
        "use_cache": True,
    }

    @staticmethod
    def _normalize_model_type(model_type: str) -> str:
        """Strip deployment suffixes to find base model config.

        ``"internvl3-vllm"`` -> ``"internvl3"``,
        ``"internvl3-14b-vllm"`` -> ``"internvl3"``.
        """
        key = model_type.lower()
        # Strip -vllm suffix
        if key.endswith("-vllm"):
            key = key[: -len("-vllm")]
        # Strip size suffixes (-8b, -14b, -38b, etc.)
        parts = key.rsplit("-", 1)
        if len(parts) == 2 and parts[1].endswith("b") and parts[1][:-1].isdigit():
            key = parts[0]
        # Strip -w4a16 quantization suffix
        if key.endswith("-w4a16"):
            key = key[: -len("-w4a16")]
        return key

    def get_generation_config(self, model_type: str) -> dict[str, Any]:
        """Same signature as model_config.get_generation_config().

        Returns a copy so callers can mutate freely.
        Strips deployment suffixes (``-vllm``, ``-14b``) for config lookup.
        """
        key = self._normalize_model_type(model_type)
        return dict(
            self._generation_registry.get(key, self._FALLBACK_GENERATION_CONFIG)
        )

    def get_batch_size_for_model(
        self, model_name: str, strategy: str | None = None
    ) -> int:
        """Get recommended batch size for a model based on strategy."""
        strategy = strategy or self.batch.strategy
        model_name = self._normalize_model_type(model_name)

        if strategy == "conservative":
            return self.batch.conservative_sizes.get(model_name, self.batch.min_size)
        elif strategy == "aggressive":
            return self.batch.max_sizes.get(model_name, self.batch.min_size)
        else:
            return self.batch.default_sizes.get(model_name, self.batch.min_size)

    def get_auto_batch_size(
        self, model_name: str, available_memory_gb: float | None = None
    ) -> int:
        """Same signature as model_config.get_auto_batch_size()."""
        if not self.batch.auto_detect or available_memory_gb is None:
            return self.get_batch_size_for_model(model_name, self.batch.strategy)

        thresholds = self.batch.gpu_memory_thresholds
        if available_memory_gb >= thresholds["very_high"]:
            strategy = "aggressive"
        elif available_memory_gb >= thresholds["high"]:
            strategy = "aggressive"
        elif available_memory_gb >= thresholds["medium"]:
            strategy = "balanced"
        else:
            strategy = "conservative"

        return self.get_batch_size_for_model(model_name, strategy)

    def get_max_new_tokens(
        self,
        field_count: int | None = None,
        document_type: str | None = None,
    ) -> int:
        """Same signature as model_config.get_max_new_tokens()."""
        effective_count = field_count or self.fields.field_count or 15

        config = self._generation_registry.get("internvl3", {})
        base = int(config.get("max_new_tokens_base", 2000))
        per_field = int(config.get("max_new_tokens_per_field", 50))
        base_tokens = max(base, effective_count * per_field)

        if document_type:
            min_tokens = self._min_tokens_by_type.get(document_type)
            if min_tokens:
                return max(base_tokens, min_tokens)

        return base_tokens
