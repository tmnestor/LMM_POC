"""Unified application configuration.

Single entry point (AppConfig.load) replaces the 7-step config dance
in cli.py and eliminates mutable module globals.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from common.field_schema import FieldSchema, get_field_schema

if TYPE_CHECKING:
    from common.pipeline_config import PipelineConfig

_VALID_SECONDARY_SORTS = ("none", "image_area_asc", "image_area_desc")


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

    # Schema documentation only. Runtime values come from run_config.yml.
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

    Supports two YAML formats:
    - **New structured format** (``generation.defaults`` + ``generation.models``):
      builds per-model configs by merging ``defaults | per_model_overrides``.
    - **Legacy flat format** (``generation.max_new_tokens_base``, etc.):
      deep-copies the base registry from model_config and applies flat overrides
      to the InternVL3 entry only (backwards compat).

    Returns the result without mutating the originals.
    """
    gen = raw_config.get("generation", {})

    # ── New structured format ──────────────────────────────────────────
    if "defaults" in gen:
        defaults = dict(gen["defaults"])
        models_section = gen.get("models", {})
        registry: dict[str, dict] = {}
        for model_name, overrides in models_section.items():
            registry[model_name] = {**defaults, **overrides}
        # Ensure a "__defaults__" sentinel so callers can get generic config
        registry["__defaults__"] = dict(defaults)
        return registry

    # ── Legacy flat format (backwards compat) ──────────────────────────
    from common.model_config import _GENERATION_CONFIG_REGISTRY

    registry = copy.deepcopy(_GENERATION_CONFIG_REGISTRY)

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
        "_token_budgets",
        "_vllm_config",
        "_infrastructure",
        "_classification",
        "_extraction_order",
        "_secondary_sort",
        "_extraction_skip_labels",
        "_image_budgets",
        "_bank_header_cache",
        "_band_split",
    )

    _DEFAULT_VLLM_CONFIG: ClassVar[dict[str, Any]] = {
        "gpu_memory_utilization": 0.90,
        "max_model_len": 8192,
        "max_num_seqs": 1,
        "limit_mm_per_prompt": 1,
        "enable_prefix_caching": True,
    }

    _DEFAULT_INFRASTRUCTURE: ClassVar[dict[str, int | float]] = {
        "dp_join_timeout": 60,
        "gpu_memory_threshold_gb": 1.0,
        "gpu_memory_fallback_gb": 24.0,
    }

    def __init__(
        self,
        pipeline: "PipelineConfig",
        batch: BatchSettings,
        fields: FieldSchema,
        generation_registry: dict[str, dict],
        token_limits: dict[str, int | None] | None = None,
        min_tokens_by_type: dict[str, int] | None = None,
        token_budgets: dict[str, int] | None = None,
        vllm_config: dict[str, dict] | None = None,
        infrastructure: dict[str, int | float] | None = None,
        classification: dict[str, str] | None = None,
        extraction_order: list[str] | None = None,
        secondary_sort: str = "none",
        extraction_skip_labels: list[str] | None = None,
        image_budgets: dict[str, dict[str, int]] | None = None,
        bank_header_cache: dict[str, Any] | None = None,
        band_split: dict[str, Any] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.batch = batch
        self.fields = fields
        self._generation_registry = generation_registry
        self._token_limits = token_limits or {}
        self._min_tokens_by_type = min_tokens_by_type or {}
        self._token_budgets = token_budgets or {}
        self._vllm_config = vllm_config or {}
        self._infrastructure = {**self._DEFAULT_INFRASTRUCTURE, **(infrastructure or {})}
        self._classification = classification or {}
        self._extraction_order = extraction_order or []
        self._secondary_sort = secondary_sort
        self._extraction_skip_labels = extraction_skip_labels or []
        self._image_budgets = image_budgets or {}
        self._bank_header_cache = bank_header_cache or {"enabled": False, "key_pattern": ""}
        self._band_split = band_split or {
            "enabled": False,
            "target_band_height": 900,
            "overlap_frac": 0.08,
            "max_bands": 6,
        }

    @classmethod
    def load(
        cls,
        cli_args: dict[str, Any],
        *,
        config_path: Path | None = None,
    ) -> AppConfig:
        """Single entry point. No mutable globals touched.

        Handles: YAML loading, ENV loading, merge (CLI > YAML > ENV > defaults),
        validation, batch settings, generation registry, field schema,
        token budgets, vLLM config, infrastructure settings.

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
                "--data-dir is required (via CLI, config file, or IVL_DATA_DIR environment variable)"
            )
        if not merged_preview.get("output_dir"):
            errors.append(
                "--output-dir is required (via CLI, config file, or IVL_OUTPUT_DIR environment variable)"
            )
        if errors:
            raise ConfigError(errors)

        # 4b. Extract vLLM-specific CLI overrides before PipelineConfig merge
        #     (PipelineConfig doesn't know about max_num_seqs)
        cli_max_num_seqs = cli_args.pop("max_num_seqs", None)

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

        # 12. Token budgets — YAML is the single source of truth
        yaml_budgets = raw_config.get("token_budgets", {})

        # 13. Build vLLM config (defaults + per-model overrides)
        vllm_section = raw_config.get("vllm", {})
        vllm_defaults = vllm_section.get("defaults", dict(cls._DEFAULT_VLLM_CONFIG))
        vllm_models = vllm_section.get("models", {})
        vllm_config: dict[str, dict] = {"__defaults__": dict(vllm_defaults)}
        for model_name, overrides in vllm_models.items():
            vllm_config[model_name] = {**vllm_defaults, **overrides}

        # 13b. Apply CLI max_num_seqs override (beats YAML per-model and defaults)
        if cli_max_num_seqs is not None:
            for key in vllm_config:
                vllm_config[key]["max_num_seqs"] = cli_max_num_seqs

        # 14. Build infrastructure settings
        infra_section = raw_config.get("infrastructure", {})

        # 15. Build classification settings
        classification_section = raw_config.get("classification", {})

        # 16. Validate and build extraction_order
        config_file = str(resolved) if resolved else "config/run_config.yml"
        extraction_order = cls._validate_extraction_order(raw_config, config_file)

        # 17. Validate and build secondary_sort
        secondary_sort = cls._validate_secondary_sort(raw_config, config_file)

        # 18. Validate and build extraction_skip_labels
        skip_labels = cls._validate_extraction_skip_labels(raw_config, config_file)

        # 19. Validate and build image_budgets
        image_budgets = cls._validate_image_budgets(raw_config, config_file)

        # 20. Validate and build bank_header_cache
        bank_header_cache = cls._validate_bank_header_cache(raw_config, config_file)
        band_split = cls._validate_band_split(raw_config, config_file)

        return cls(
            pipeline=pipeline,
            batch=batch,
            fields=fields,
            generation_registry=generation_registry,
            token_limits=token_limits,
            min_tokens_by_type=fields.min_tokens_by_type,
            token_budgets=yaml_budgets,
            vllm_config=vllm_config,
            infrastructure=infra_section,
            classification=classification_section,
            extraction_order=extraction_order,
            secondary_sort=secondary_sort,
            extraction_skip_labels=skip_labels,
            image_budgets=image_budgets,
            bank_header_cache=bank_header_cache,
            band_split=band_split,
        )

    # -- Token budgets (Phase 1) -----------------------------------------------

    @property
    def token_budgets(self) -> dict[str, int]:
        """Read-only view of all resolved token budgets."""
        return dict(self._token_budgets)

    def get_token_budget(self, name: str) -> int:
        """Resolve a named token budget.

        Args:
            name: Budget name (e.g. "classify", "extract_bank").

        Returns:
            Token count for the named budget.

        Raises:
            KeyError: If *name* is not a known budget, with available names listed.
        """
        try:
            return self._token_budgets[name]
        except KeyError:
            available = ", ".join(sorted(self._token_budgets))
            msg = f"Unknown token budget {name!r}. Available budgets: {available}"
            raise KeyError(msg) from None

    # -- vLLM config (Phase 3) -------------------------------------------------

    def get_vllm_config(self, model_type: str) -> dict[str, Any]:
        """Return vLLM engine parameters for *model_type*.

        Falls back to ``vllm.defaults`` (or class-level defaults) if
        no per-model override exists.
        """
        defaults = self._vllm_config.get("__defaults__", dict(self._DEFAULT_VLLM_CONFIG))
        per_model = self._vllm_config.get(model_type, {})
        if per_model:
            return {**defaults, **per_model}
        return dict(defaults)

    # -- Infrastructure (Phase 5) -----------------------------------------------

    def get_infra(self, name: str) -> int | float:
        """Resolve an infrastructure setting by name.

        Args:
            name: Setting name (e.g. "dp_join_timeout", "gpu_memory_threshold_gb").

        Returns:
            The setting value.

        Raises:
            KeyError: If *name* is not a known infrastructure setting.
        """
        try:
            return self._infrastructure[name]
        except KeyError:
            available = ", ".join(sorted(self._infrastructure))
            msg = f"Unknown infrastructure setting {name!r}. Available settings: {available}"
            raise KeyError(msg) from None

    # -- Classification config --------------------------------------------------

    @property
    def classification_fallback_type(self) -> str:
        """Default document type when classification response can't be parsed."""
        return self._classification["fallback_type"]

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
        base = self._generation_registry.get(key)
        if base is not None:
            return dict(base)
        # Try __defaults__ sentinel from structured format
        defaults = self._generation_registry.get("__defaults__")
        if defaults is not None:
            return dict(defaults)
        return dict(self._FALLBACK_GENERATION_CONFIG)

    def get_batch_size_for_model(self, model_name: str, strategy: str | None = None) -> int:
        """Get recommended batch size for a model based on strategy."""
        strategy = strategy or self.batch.strategy
        model_name = self._normalize_model_type(model_name)

        if strategy == "conservative":
            return self.batch.conservative_sizes.get(model_name, self.batch.min_size)
        elif strategy == "aggressive":
            return self.batch.max_sizes.get(model_name, self.batch.min_size)
        else:
            return self.batch.default_sizes.get(model_name, self.batch.min_size)

    def get_auto_batch_size(self, model_name: str, available_memory_gb: float | None = None) -> int:
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

    # -- Extraction ordering (Phase 2) -----------------------------------------

    @property
    def extraction_order(self) -> list[str]:
        """Ordered list of doc_type values for batch submission sorting."""
        return list(self._extraction_order)

    @property
    def secondary_sort(self) -> str:
        """Secondary sort key within each doc_type group."""
        return self._secondary_sort

    # -- Extraction skip labels (Phase 4) --------------------------------------

    @property
    def extraction_skip_labels(self) -> list[str]:
        """Classification labels that bypass extraction entirely."""
        return list(self._extraction_skip_labels)

    # -- Image budgets (Phase 3) -----------------------------------------------

    def get_image_budget(self, doc_type: str) -> dict[str, int]:
        """Return tile budget for *doc_type*, falling back to ``default``.

        Returns:
            Dict with at least ``max_tiles`` key.
        """
        budget = self._image_budgets.get(doc_type.lower())
        if budget is not None:
            return dict(budget)
        return dict(self._image_budgets["default"])

    def max_image_budget_tiles(self) -> int:
        """Return the largest ``max_tiles`` across all configured image budgets.

        Used to size ``limit_mm_per_prompt`` when pre-tiling is enabled (the
        backend sends one image per tile, so the engine must admit the ceiling).
        """
        return max(b["max_tiles"] for b in self._image_budgets.values())

    # -- Bank header cache (Phase 7) -------------------------------------------

    @property
    def bank_header_cache_config(self) -> dict[str, Any]:
        """Bank header cache configuration dict."""
        return dict(self._bank_header_cache)

    # -- Validation classmethods -----------------------------------------------

    @classmethod
    def _validate_extraction_order(cls, raw_config: dict, config_file: str) -> list[str]:
        """Validate ``extraction_order`` key in YAML."""
        if "extraction_order" not in raw_config:
            raise ConfigError(
                [
                    f"Missing required key 'extraction_order' in {config_file}. "
                    f"What: the key 'extraction_order' is absent. "
                    f"Where: {config_file} → extraction_order. "
                    f"Expected: a YAML list of doc_type strings, e.g.:\n"
                    f"  extraction_order:\n"
                    f"    - bank_statement\n"
                    f"    - invoice\n"
                    f"    - receipt\n"
                    f"How to fix: add an 'extraction_order:' list to {config_file}."
                ]
            )
        value = raw_config["extraction_order"]
        if not isinstance(value, list):
            raise ConfigError(
                [
                    f"Invalid type for 'extraction_order' in {config_file}: "
                    f"expected a list, got {type(value).__name__}. "
                    f"Where: {config_file} → extraction_order. "
                    f"Expected: a YAML list of doc_type strings, e.g.:\n"
                    f"  extraction_order:\n"
                    f"    - bank_statement\n"
                    f"    - invoice\n"
                    f"    - receipt\n"
                    f"How to fix: change 'extraction_order' to a YAML list in {config_file}."
                ]
            )
        return list(value)

    @classmethod
    def _validate_secondary_sort(cls, raw_config: dict, config_file: str) -> str:
        """Validate ``secondary_sort`` key in YAML."""
        if "secondary_sort" not in raw_config:
            raise ConfigError(
                [
                    f"Missing required key 'secondary_sort' in {config_file}. "
                    f"What: the key 'secondary_sort' is absent. "
                    f"Where: {config_file} → extraction_order.secondary_sort. "
                    f"Expected: one of {list(_VALID_SECONDARY_SORTS)}, e.g.:\n"
                    f"  secondary_sort: none\n"
                    f"How to fix: add 'secondary_sort: none' to {config_file}."
                ]
            )
        value = raw_config["secondary_sort"]
        if value not in _VALID_SECONDARY_SORTS:
            raise ConfigError(
                [
                    f"Invalid value for 'secondary_sort' in {config_file}: "
                    f"got {value!r}, expected one of {list(_VALID_SECONDARY_SORTS)}. "
                    f"Where: {config_file} → extraction_order.secondary_sort. "
                    f"How to fix: set 'secondary_sort' to one of "
                    f"{list(_VALID_SECONDARY_SORTS)} in {config_file}."
                ]
            )
        return str(value)

    @classmethod
    def _validate_extraction_skip_labels(cls, raw_config: dict, config_file: str) -> list[str]:
        """Validate ``extraction_skip_labels`` key in YAML."""
        if "extraction_skip_labels" not in raw_config:
            raise ConfigError(
                [
                    f"Missing required key 'extraction_skip_labels' in {config_file}. "
                    f"What: the key 'extraction_skip_labels' is absent. "
                    f"Where: {config_file} → extraction_skip_labels. "
                    f"Expected: a YAML list of label strings (may be empty), e.g.:\n"
                    f"  extraction_skip_labels: []\n"
                    f"How to fix: add 'extraction_skip_labels: []' to {config_file}."
                ]
            )
        value = raw_config["extraction_skip_labels"]
        # YAML parses `extraction_skip_labels:` (no value) as None
        if value is None:
            value = []
        if not isinstance(value, list):
            raise ConfigError(
                [
                    f"Invalid type for 'extraction_skip_labels' in {config_file}: "
                    f"expected a list, got {type(value).__name__}. "
                    f"Where: {config_file} → extraction_skip_labels. "
                    f"Expected: a YAML list of label strings, e.g.:\n"
                    f"  extraction_skip_labels:\n"
                    f"    - junk\n"
                    f"    - blank\n"
                    f"How to fix: change 'extraction_skip_labels' to a YAML list "
                    f"in {config_file}."
                ]
            )
        for i, entry in enumerate(value):
            if not isinstance(entry, str):
                raise ConfigError(
                    [
                        f"Invalid entry at index {i} in 'extraction_skip_labels' "
                        f"in {config_file}: expected a string, got "
                        f"{type(entry).__name__} ({entry!r}). "
                        f"Where: {config_file} → extraction_skip_labels[{i}]. "
                        f"Expected: a string label, e.g. 'junk'. "
                        f"How to fix: ensure all entries in 'extraction_skip_labels' "
                        f"are strings in {config_file}."
                    ]
                )
        return list(value)

    @classmethod
    def _validate_image_budgets(cls, raw_config: dict, config_file: str) -> dict[str, dict[str, int]]:
        """Validate ``image_budgets`` section in YAML."""
        if "image_budgets" not in raw_config:
            raise ConfigError(
                [
                    f"Missing required key 'image_budgets' in {config_file}. "
                    f"What: the key 'image_budgets' is absent. "
                    f"Where: {config_file} → image_budgets. "
                    f"Expected: a mapping with at least a 'default' entry, e.g.:\n"
                    f"  image_budgets:\n"
                    f"    default:\n"
                    f"      max_tiles: 18\n"
                    f"How to fix: add an 'image_budgets:' section with a 'default' "
                    f"entry to {config_file}."
                ]
            )
        budgets = raw_config["image_budgets"]
        if not isinstance(budgets, dict):
            raise ConfigError(
                [
                    f"Invalid type for 'image_budgets' in {config_file}: "
                    f"expected a mapping, got {type(budgets).__name__}. "
                    f"Where: {config_file} → image_budgets. "
                    f"Expected: a YAML mapping, e.g.:\n"
                    f"  image_budgets:\n"
                    f"    default:\n"
                    f"      max_tiles: 18\n"
                    f"How to fix: change 'image_budgets' to a YAML mapping "
                    f"in {config_file}."
                ]
            )
        if "default" not in budgets:
            raise ConfigError(
                [
                    f"Missing required key 'image_budgets.default' in {config_file}. "
                    f"What: the 'default' entry is absent from 'image_budgets'. "
                    f"Where: {config_file} → image_budgets.default. "
                    f"Expected: a mapping with 'max_tiles', e.g.:\n"
                    f"  image_budgets:\n"
                    f"    default:\n"
                    f"      max_tiles: 18\n"
                    f"How to fix: add a 'default:' entry under 'image_budgets' "
                    f"in {config_file}."
                ]
            )
        for doc_type, entry in budgets.items():
            if not isinstance(entry, dict) or "min_tiles" not in entry or "max_tiles" not in entry:
                raise ConfigError(
                    [
                        f"Invalid entry for 'image_budgets.{doc_type}' in {config_file}. "
                        f"What: each entry must have both 'min_tiles' and 'max_tiles' "
                        f"keys; one or both are missing. "
                        f"Where: {config_file} → image_budgets.{doc_type}. "
                        f"Expected: a mapping with 'min_tiles' and 'max_tiles', e.g.:\n"
                        f"  image_budgets:\n"
                        f"    {doc_type}:\n"
                        f"      min_tiles: 1\n"
                        f"      max_tiles: 18\n"
                        f"How to fix: add 'min_tiles: <int>' and 'max_tiles: <int>' "
                        f"under 'image_budgets.{doc_type}' in {config_file}."
                    ]
                )
            for key in ("min_tiles", "max_tiles"):
                if not isinstance(entry[key], int) or entry[key] < 1:
                    raise ConfigError(
                        [
                            f"Invalid '{key}' for 'image_budgets.{doc_type}' in {config_file}. "
                            f"What: '{key}' must be a positive integer, got {entry[key]!r}. "
                            f"Where: {config_file} → image_budgets.{doc_type}.{key}. "
                            f"Expected: a positive integer, e.g. 18. "
                            f"How to fix: set '{key}' to a positive integer under "
                            f"'image_budgets.{doc_type}' in {config_file}."
                        ]
                    )
            if entry["min_tiles"] > entry["max_tiles"]:
                raise ConfigError(
                    [
                        f"Invalid tile budget for 'image_budgets.{doc_type}' in {config_file}. "
                        f"What: min_tiles ({entry['min_tiles']}) exceeds max_tiles "
                        f"({entry['max_tiles']}). "
                        f"Where: {config_file} → image_budgets.{doc_type}. "
                        f"Expected: min_tiles <= max_tiles, e.g. min_tiles: 12, "
                        f"max_tiles: 18. "
                        f"How to fix: lower 'min_tiles' or raise 'max_tiles' under "
                        f"'image_budgets.{doc_type}' in {config_file}."
                    ]
                )
        return dict(budgets)

    @classmethod
    def _validate_bank_header_cache(cls, raw_config: dict, config_file: str) -> dict[str, Any]:
        """Validate ``bank_header_cache`` section in YAML."""
        if "bank_header_cache" not in raw_config:
            raise ConfigError(
                [
                    f"Missing required key 'bank_header_cache' in {config_file}. "
                    f"What: the key 'bank_header_cache' is absent. "
                    f"Where: {config_file} → bank_header_cache. "
                    f"Expected: a mapping with 'enabled' and 'key_pattern', e.g.:\n"
                    f"  bank_header_cache:\n"
                    f"    enabled: false\n"
                    f'    key_pattern: "^(?P<institution>[A-Za-z_]+)_"\n'
                    f"How to fix: add a 'bank_header_cache:' section to {config_file}."
                ]
            )
        cache = raw_config["bank_header_cache"]
        if not isinstance(cache, dict):
            raise ConfigError(
                [
                    f"Invalid type for 'bank_header_cache' in {config_file}: "
                    f"expected a mapping, got {type(cache).__name__}. "
                    f"Where: {config_file} → bank_header_cache. "
                    f"Expected: a mapping with 'enabled' and 'key_pattern'. "
                    f"How to fix: change 'bank_header_cache' to a YAML mapping "
                    f"in {config_file}."
                ]
            )
        if "enabled" not in cache:
            raise ConfigError(
                [
                    f"Missing required key 'bank_header_cache.enabled' in "
                    f"{config_file}. "
                    f"What: the 'enabled' key is absent from 'bank_header_cache'. "
                    f"Where: {config_file} → bank_header_cache.enabled. "
                    f"Expected: a boolean (true/false), e.g.:\n"
                    f"  bank_header_cache:\n"
                    f"    enabled: false\n"
                    f"How to fix: add 'enabled: false' under 'bank_header_cache' "
                    f"in {config_file}."
                ]
            )
        if "key_pattern" not in cache:
            raise ConfigError(
                [
                    f"Missing required key 'bank_header_cache.key_pattern' in "
                    f"{config_file}. "
                    f"What: the 'key_pattern' key is absent from 'bank_header_cache'. "
                    f"Where: {config_file} → bank_header_cache.key_pattern. "
                    f"Expected: a regex string, e.g.:\n"
                    f"  bank_header_cache:\n"
                    f'    key_pattern: "^(?P<institution>[A-Za-z_]+)_"\n'
                    f"How to fix: add 'key_pattern: <regex>' under "
                    f"'bank_header_cache' in {config_file}."
                ]
            )
        # Validate regex compiles
        pattern = cache["key_pattern"]
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ConfigError(
                [
                    f"Invalid regex in 'bank_header_cache.key_pattern' in "
                    f"{config_file}: {exc}. "
                    f"Pattern: {pattern!r}. "
                    f"Where: {config_file} → bank_header_cache.key_pattern. "
                    f"Expected: a valid Python regex string. "
                    f"How to fix: correct the regex pattern in "
                    f"'bank_header_cache.key_pattern' in {config_file}."
                ]
            ) from None
        return dict(cache)

    @classmethod
    def _validate_band_split(cls, raw_config: dict, config_file: str) -> dict[str, Any]:
        """Validate ``bank_extraction.band_split`` (optional section).

        Absent → disabled default (a new opt-in feature must not break existing
        configs). Present → all four keys required and type-checked, fail-fast.
        """
        default = {"enabled": False, "target_band_height": 900, "overlap_frac": 0.08, "max_bands": 6}
        bs = (raw_config.get("bank_extraction") or {}).get("band_split")
        if bs is None:
            return default
        if not isinstance(bs, dict):
            raise ConfigError(
                [
                    f"Invalid type for 'bank_extraction.band_split' in {config_file}. "
                    f"What: expected a mapping, got {type(bs).__name__}. "
                    f"Where: {config_file} → bank_extraction.band_split. "
                    f"Expected: a mapping with enabled/target_band_height/overlap_frac/max_bands. "
                    f"How to fix: make 'band_split' a YAML mapping in {config_file}."
                ]
            )
        missing = {"enabled", "target_band_height", "overlap_frac", "max_bands"} - set(bs)
        if missing:
            raise ConfigError(
                [
                    f"Missing keys {sorted(missing)} in 'bank_extraction.band_split' in {config_file}. "
                    f"What: every band_split key is required when the section is present. "
                    f"Where: {config_file} → bank_extraction.band_split. "
                    f"Expected: enabled (bool), target_band_height (int>0), overlap_frac "
                    f"(0<=f<0.5), max_bands (int>=1), e.g.:\n"
                    f"  bank_extraction:\n    band_split:\n      enabled: false\n"
                    f"      target_band_height: 900\n      overlap_frac: 0.08\n      max_bands: 6\n"
                    f"How to fix: add the missing key(s) under 'band_split' in {config_file}."
                ]
            )
        if not isinstance(bs["enabled"], bool):
            raise ConfigError(
                [
                    f"Invalid 'bank_extraction.band_split.enabled' in {config_file}. "
                    f"What: expected a boolean, got {bs['enabled']!r}. "
                    f"Where: {config_file} → bank_extraction.band_split.enabled. "
                    f"Expected: true or false. "
                    f"How to fix: set 'enabled' to a boolean in {config_file}."
                ]
            )
        if not isinstance(bs["target_band_height"], int) or bs["target_band_height"] <= 0:
            raise ConfigError(
                [
                    f"Invalid 'bank_extraction.band_split.target_band_height' in {config_file}. "
                    f"What: expected a positive integer (pixels), got {bs['target_band_height']!r}. "
                    f"Where: {config_file} → bank_extraction.band_split.target_band_height. "
                    f"Expected: a positive int, e.g. 900. "
                    f"How to fix: set 'target_band_height' to a positive int in {config_file}."
                ]
            )
        if not isinstance(bs["overlap_frac"], (int, float)) or not 0.0 <= bs["overlap_frac"] < 0.5:
            raise ConfigError(
                [
                    f"Invalid 'bank_extraction.band_split.overlap_frac' in {config_file}. "
                    f"What: expected a number in [0, 0.5), got {bs['overlap_frac']!r}. "
                    f"Where: {config_file} → bank_extraction.band_split.overlap_frac. "
                    f"Expected: e.g. 0.08. "
                    f"How to fix: set 'overlap_frac' to a value in [0, 0.5) in {config_file}."
                ]
            )
        if not isinstance(bs["max_bands"], int) or bs["max_bands"] < 1:
            raise ConfigError(
                [
                    f"Invalid 'bank_extraction.band_split.max_bands' in {config_file}. "
                    f"What: expected an integer >= 1, got {bs['max_bands']!r}. "
                    f"Where: {config_file} → bank_extraction.band_split.max_bands. "
                    f"Expected: e.g. 6. "
                    f"How to fix: set 'max_bands' to an int >= 1 in {config_file}."
                ]
            )
        return dict(bs)

    def band_split_config(self) -> dict[str, Any]:
        """Band-split bank-extraction config (enabled/target_band_height/overlap_frac/max_bands)."""
        return dict(self._band_split)
