"""Unified application configuration.

Single entry point (AppConfig.load) replaces the 7-step config dance
in cli.py and eliminates mutable module globals.

Phase 1: AppConfig coexists with model_config/field_config globals.
Phase 2: Globals removed, all consumers use AppConfig.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

    @classmethod
    def from_raw(cls, raw_config: dict) -> BatchSettings:
        """Build from raw YAML config (replaces apply_yaml_overrides batch/gpu sections)."""
        from common.model_config import (
            CONSERVATIVE_BATCH_SIZES,
            DEFAULT_BATCH_SIZES,
            MAX_BATCH_SIZES,
        )

        # Start with Python-level defaults
        default_sizes = dict(DEFAULT_BATCH_SIZES)
        max_sizes = dict(MAX_BATCH_SIZES)
        conservative_sizes = dict(CONSERVATIVE_BATCH_SIZES)
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


@dataclass(frozen=True)
class FieldSchema:
    """Typed replacement for the 13 lazy globals in field_config.py.

    Loaded once from field_definitions.yaml via SimpleFieldLoader.
    """

    extraction_fields: tuple[str, ...]
    field_count: int
    field_types: dict[str, str]
    monetary_fields: tuple[str, ...]
    date_fields: tuple[str, ...]
    list_fields: tuple[str, ...]
    boolean_fields: tuple[str, ...]
    calculated_fields: tuple[str, ...]
    transaction_list_fields: tuple[str, ...]
    text_fields: tuple[str, ...]
    phone_fields: tuple[str, ...]
    numeric_id_fields: tuple[str, ...]
    validation_only_fields: tuple[str, ...] = (
        "TRANSACTION_AMOUNTS_RECEIVED",
        "ACCOUNT_BALANCE",
    )

    @classmethod
    def from_yaml(cls) -> FieldSchema:
        """Load from field_definitions.yaml (single read, single cache)."""
        from common.field_definitions_loader import SimpleFieldLoader

        loader = SimpleFieldLoader()

        # Get universal extraction fields (same logic as field_config._get_config)
        extraction_fields = loader.get_document_fields("universal")
        _exclude = ["TRANSACTION_AMOUNTS_RECEIVED"]
        extraction_fields = [f for f in extraction_fields if f not in _exclude]

        field_types_from_yaml = loader.get_field_types()

        return cls(
            extraction_fields=tuple(extraction_fields),
            field_count=len(extraction_fields),
            field_types={f: "text" for f in extraction_fields},
            monetary_fields=tuple(field_types_from_yaml.get("monetary", [])),
            date_fields=tuple(field_types_from_yaml.get("date", [])),
            list_fields=tuple(field_types_from_yaml.get("list", [])),
            boolean_fields=tuple(field_types_from_yaml.get("boolean", [])),
            calculated_fields=tuple(field_types_from_yaml.get("calculated", [])),
            transaction_list_fields=tuple(
                field_types_from_yaml.get("transaction_list", [])
            ),
            text_fields=tuple(field_types_from_yaml.get("text", extraction_fields)),
            phone_fields=(),
            numeric_id_fields=(),
        )

    def is_evaluation_field(self, field_name: str) -> bool:
        """Check if a field should be included in evaluation metrics."""
        return field_name not in self.validation_only_fields

    def filter_evaluation_fields(self, fields: list[str]) -> list[str]:
        """Filter a list to exclude validation-only fields."""
        return [f for f in fields if self.is_evaluation_field(f)]

    def get_document_type_fields(self, document_type: str) -> list[str]:
        """Get fields specific to a document type, filtered for evaluation."""
        from common.field_definitions_loader import SimpleFieldLoader

        loader = SimpleFieldLoader()
        doc_type_mapping = {
            "invoice": "invoice",
            "tax_invoice": "invoice",
            "bill": "invoice",
            "receipt": "receipt",
            "purchase_receipt": "receipt",
            "bank_statement": "bank_statement",
            "statement": "bank_statement",
            "transaction_link": "transaction_link",
        }
        mapped_type = doc_type_mapping.get(document_type.lower(), document_type.lower())

        try:
            field_names = loader.get_document_fields(mapped_type)
        except Exception:
            return self.filter_evaluation_fields(list(self.extraction_fields))

        return self.filter_evaluation_fields(field_names)


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
    )

    def __init__(
        self,
        pipeline: "PipelineConfig",
        batch: BatchSettings,
        fields: FieldSchema,
        generation_registry: dict[str, dict],
        token_limits: dict[str, int | None] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.batch = batch
        self.fields = fields
        self._generation_registry = generation_registry
        self._token_limits = token_limits or {}

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

        # 7. Phase 1 backward compat: mutate module globals so InternVL3/Llama
        #    processors that read them directly still get YAML overrides.
        #    Removed in Phase 2 when all processors use AppConfig.
        if raw_config:
            from common.model_config import apply_yaml_overrides

            apply_yaml_overrides(raw_config)

        # 8. Build BatchSettings (immutable copy of batch/gpu config)
        batch = BatchSettings.from_raw(raw_config)

        # 9. Build generation registry (immutable copy with YAML overrides)
        generation_registry = _build_generation_registry(raw_config)

        # 10. Build token limits from YAML overrides
        token_limits: dict[str, int | None] = {"2b": None, "8b": 800}
        gen = raw_config.get("generation", {})
        yaml_limits = gen.get("token_limits", {})
        for size_key, value in yaml_limits.items():
            token_limits[str(size_key)] = value

        # 11. Load field schema (single YAML read)
        fields = FieldSchema.from_yaml()

        return cls(
            pipeline=pipeline,
            batch=batch,
            fields=fields,
            generation_registry=generation_registry,
            token_limits=token_limits,
        )

    # -- Drop-in replacements for model_config functions -----------------------

    def get_generation_config(self, model_type: str) -> dict[str, Any]:
        """Same signature as model_config.get_generation_config().

        Returns a copy so callers can mutate freely.
        """
        from common.model_config import QWEN3VL_GENERATION_CONFIG

        return dict(
            self._generation_registry.get(model_type, QWEN3VL_GENERATION_CONFIG)
        )

    def get_batch_size_for_model(
        self, model_name: str, strategy: str | None = None
    ) -> int:
        """Get recommended batch size for a model based on strategy."""
        strategy = strategy or self.batch.strategy
        model_name = model_name.lower()

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
            from common.model_config import _get_min_tokens_for_type

            min_tokens = _get_min_tokens_for_type(document_type)
            if min_tokens:
                return max(base_tokens, min_tokens)

        return base_tokens
