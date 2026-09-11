"""Unified application configuration.

`AppConfig.load` is the single entry point: it reads run_config.yml, merges
CLI overrides over it, and validates every section before any work begins.
There are no mutable module globals and no Python-side defaults -- a missing
key fails at startup rather than resolving to a constant.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar


if TYPE_CHECKING:
    from common.pipeline_config import PipelineConfig


class ConfigError(Exception):
    """Raised by AppConfig.load() when validation fails."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


def _validate_model_override_keys(
    *,
    vllm_models: dict,
    generation_models: dict,
) -> None:
    """Reject per-model override keys that no registered model resolves to.

    Both ``inference.vllm.models`` and ``inference.generation.models`` fall back
    to their ``defaults`` block when a key is absent. That fallback is
    deliberate -- it is how a model inherits shared tuning -- but it makes a
    TYPO indistinguishable from inheritance: the block reads as configured
    while doing nothing, and the run completes on default tuning.

    The two sections use DIFFERENT key conventions, which is the trap this
    guards. ``inference.vllm.models`` is keyed by the FULL registered type
    (``gemma4-12b-unified-w4a16-vllm``); ``inference.generation.models`` is
    keyed by the NORMALISED form (``gemma4-12b-unified``), and the
    normalisation is not uniform across suffixes.

    Raises:
        ValueError: If any key matches no registered model, with a
            four-element diagnostic naming the offending keys.
    """
    from models.registry import list_models

    registered = list_models()
    normalised = {AppConfig._normalize_model_type(name) for name in registered}

    unknown_vllm = sorted(set(vllm_models) - set(registered))
    unknown_generation = sorted(set(generation_models) - normalised)
    if not unknown_vllm and not unknown_generation:
        return

    problems = []
    if unknown_vllm:
        problems.append(f"inference.vllm.models: {unknown_vllm}")
    if unknown_generation:
        problems.append(f"inference.generation.models: {unknown_generation}")

    msg = (
        f"What:  per-model override key(s) match no registered model, so the "
        f"block is dead config -- the lookup silently falls through to the "
        f"`defaults` block and the tuning is never applied. "
        f"Offending: {'; '.join(problems)}\n"
        f"  Where: config/run_config.yml -> {' and '.join(p.split(':')[0] for p in problems)}\n"
        f"  Expected: inference.vllm.models is keyed by the FULL registered "
        f"type, one of: {', '.join(sorted(registered))}. "
        f"inference.generation.models is keyed by the NORMALISED type, one of: "
        f"{', '.join(sorted(normalised))}. The two conventions differ and the "
        f"normalisation is not uniform -- check it rather than inferring it.\n"
        f"  How to fix: correct the key to a listed value, or delete the block "
        f"if the model should inherit the `defaults` above it."
    )
    raise ValueError(msg)


def _build_generation_registry(raw_config: dict) -> dict[str, dict]:
    """Build the per-model generation config from ``inference.generation``.

    One format: a ``defaults`` block, merged with any per-model overrides under
    ``models``. There used to be a second, flat, legacy format handled by
    falling back to a hardcoded registry in ``common.model_config`` -- which is
    the silent-fallback shape: a YAML that had drifted out of the supported
    layout produced a working run on Python constants rather than an error, and
    nothing said which one it had used.

    Args:
        raw_config: The parsed YAML.

    Returns:
        `{model_name: config}` plus a `__defaults__` entry, so a caller with an
        unregistered model gets the shared tuning rather than nothing.

    Raises:
        ConfigError: The ``defaults`` block is missing.
    """
    gen = raw_config.get("inference", {}).get("generation", {})

    if "defaults" not in gen:
        raise ConfigError(
            [
                "Missing required key 'inference.generation.defaults'.\n"
                "  What:        the generation config has no `defaults` block, so there "
                "is no baseline for per-model tuning to merge with.\n"
                "  Where:       config/run_config.yml → inference.generation.defaults\n"
                "  Expected:    a mapping of generation hyper-parameters, e.g.\n"
                "                 inference:\n"
                "                   generation:\n"
                "                     defaults:\n"
                "                       max_new_tokens_base: 512\n"
                "                       temperature: 0.0\n"
                "                       do_sample: false\n"
                "  How to fix:  add the defaults block. Per-model overrides go under "
                "`inference.generation.models.<type>` and inherit from it."
            ]
        )

    defaults = dict(gen["defaults"])
    registry: dict[str, dict] = {
        model_name: {**defaults, **overrides} for model_name, overrides in gen.get("models", {}).items()
    }
    # A sentinel so callers can ask for generic config without naming a model.
    registry["__defaults__"] = dict(defaults)
    return registry


class AppConfig:
    """Unified, immutable configuration surface.

    Constructed once at startup, threaded to all consumers.
    Replaces the 7-step config dance in cli.py and eliminates
    all mutable module globals in model_config and field_config.
    """

    __slots__ = (
        "pipeline",
        "_generation_registry",
        "_token_limits",
        "_token_budgets",
        "_vllm_config",
        "_infrastructure",
        "_classification",
        "_image_budgets",
        "_quality_screen",
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
        generation_registry: dict[str, dict],
        token_limits: dict[str, int | None] | None = None,
        token_budgets: dict[str, int] | None = None,
        vllm_config: dict[str, dict] | None = None,
        infrastructure: dict[str, int | float] | None = None,
        classification: dict[str, str] | None = None,
        image_budgets: dict[str, dict[str, int]] | None = None,
        quality_screen: dict[str, Any] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self._generation_registry = generation_registry
        self._token_limits = token_limits or {}
        self._token_budgets = token_budgets or {}
        self._vllm_config = vllm_config or {}
        self._infrastructure = {**self._DEFAULT_INFRASTRUCTURE, **(infrastructure or {})}
        self._classification = classification or {}
        self._image_budgets = image_budgets or {}
        self._quality_screen = quality_screen or {}

    @classmethod
    def load(
        cls,
        cli_args: dict[str, Any],
        *,
        config_path: Path | None = None,
    ) -> AppConfig:
        """Single entry point. No mutable globals touched.

        Handles: YAML loading, merge (CLI > YAML > defaults),
        validation, batch settings, generation registry, field schema,
        token budgets, vLLM config, infrastructure settings.

        Raises:
            ConfigError: On validation failure.
            FileNotFoundError: If an explicit config_path does not exist.
        """
        from common.pipeline_config import (
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

        # 3. Pre-validate required fields (before PipelineConfig construction)
        merged_preview = {**yaml_config, **cli_args}
        errors: list[str] = []
        if not merged_preview.get("data_dir"):
            errors.append(
                "--data-dir is required (via CLI, or pipeline.information_extraction.input.dir "
                "in config/run_config.yml)"
            )
        if not merged_preview.get("output_dir"):
            errors.append(
                "--output-dir is required (via CLI, or pipeline.information_extraction.output.dir "
                "in config/run_config.yml)"
            )
        if errors:
            raise ConfigError(errors)

        # 3b. Extract vLLM-specific CLI overrides before PipelineConfig merge
        #     (PipelineConfig doesn't know about max_num_seqs)
        cli_max_num_seqs = cli_args.pop("max_num_seqs", None)

        # 4. Merge with precedence: CLI > YAML > defaults
        pipeline = merge_configs(cli_args, yaml_config, raw_config)

        # 5. Validate
        val_errors = validate_config(pipeline)
        if val_errors:
            raise ConfigError(val_errors)

        # 7. Build generation registry (immutable copy with YAML overrides)
        generation_registry = _build_generation_registry(raw_config)

        # 8. Build token limits from YAML overrides
        token_limits: dict[str, int | None] = {"2b": None, "8b": 800}
        gen = raw_config.get("inference", {}).get("generation", {})
        yaml_limits = gen.get("token_limits", {})
        for size_key, value in yaml_limits.items():
            token_limits[str(size_key)] = value

        # 11. Token budgets — YAML is the single source of truth
        yaml_budgets = raw_config.get("pipeline", {}).get("token_budgets", {})

        # 12. Build vLLM config (defaults + per-model overrides)
        vllm_section = raw_config.get("inference", {}).get("vllm", {})
        vllm_defaults = vllm_section.get("defaults", dict(cls._DEFAULT_VLLM_CONFIG))
        vllm_models = vllm_section.get("models", {})

        # A per-model key that matches no registered model is dead config: the
        # lookup falls through to `defaults` and the block reads as configured
        # while doing nothing. Catch it at startup rather than at the end of a
        # GPU run whose tuning was silently ignored.
        _validate_model_override_keys(
            vllm_models=vllm_models,
            generation_models=gen.get("models", {}),
        )
        vllm_config: dict[str, dict] = {"__defaults__": dict(vllm_defaults)}
        for model_name, overrides in vllm_models.items():
            vllm_config[model_name] = {**vllm_defaults, **overrides}

        # 12b. Apply CLI max_num_seqs override (beats YAML per-model and defaults)
        if cli_max_num_seqs is not None:
            for key in vllm_config:
                vllm_config[key]["max_num_seqs"] = cli_max_num_seqs

        # 13. Build infrastructure settings
        infra_section = raw_config.get("resources", {}).get("infrastructure", {})

        # 14. Build classification settings
        classification_section = raw_config.get("pipeline", {}).get("classification", {})

        # 15. Validate and build image_budgets
        config_file = str(resolved) if resolved else "config/run_config.yml"
        image_budgets = cls._validate_image_budgets(raw_config, config_file)

        # 16. Validate and build quality_screen
        quality_screen = cls._validate_quality_screen(raw_config, config_file)

        return cls(
            pipeline=pipeline,
            generation_registry=generation_registry,
            token_limits=token_limits,
            token_budgets=yaml_budgets,
            vllm_config=vllm_config,
            infrastructure=infra_section,
            classification=classification_section,
            image_budgets=image_budgets,
            quality_screen=quality_screen,
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

    # -- Image-quality screen --------------------------------------------------

    @property
    def quality_screen_config(self) -> dict[str, Any]:
        """Image-quality screen configuration dict."""
        return dict(self._quality_screen)

    # -- Validation classmethods -----------------------------------------------

    @classmethod
    def _validate_image_budgets(cls, raw_config: dict, config_file: str) -> dict[str, dict[str, int]]:
        """Validate ``inference.tiling.budgets`` section in YAML."""
        budgets = raw_config.get("inference", {}).get("tiling", {}).get("budgets")
        if not budgets:
            raise ConfigError(
                [
                    f"Missing required key 'inference.tiling.budgets' in {config_file}. "
                    f"What: the key 'inference.tiling.budgets' is absent or empty. "
                    f"Where: {config_file} → inference.tiling.budgets. "
                    f"Expected: a mapping with at least a 'default' entry, e.g.:\n"
                    f"  inference:\n"
                    f"    tiling:\n"
                    f"      budgets:\n"
                    f"        default:\n"
                    f"          max_tiles: 18\n"
                    f"How to fix: add an 'inference.tiling.budgets:' section with a "
                    f"'default' entry to {config_file}."
                ]
            )
        if not isinstance(budgets, dict):
            raise ConfigError(
                [
                    f"Invalid type for 'inference.tiling.budgets' in {config_file}: "
                    f"expected a mapping, got {type(budgets).__name__}. "
                    f"Where: {config_file} → inference.tiling.budgets. "
                    f"Expected: a YAML mapping, e.g.:\n"
                    f"  inference:\n"
                    f"    tiling:\n"
                    f"      budgets:\n"
                    f"        default:\n"
                    f"          max_tiles: 18\n"
                    f"How to fix: change 'inference.tiling.budgets' to a YAML mapping "
                    f"in {config_file}."
                ]
            )
        if "default" not in budgets:
            raise ConfigError(
                [
                    f"Missing required key 'inference.tiling.budgets.default' in {config_file}. "
                    f"What: the 'default' entry is absent from 'inference.tiling.budgets'. "
                    f"Where: {config_file} → inference.tiling.budgets.default. "
                    f"Expected: a mapping with 'max_tiles', e.g.:\n"
                    f"  inference:\n"
                    f"    tiling:\n"
                    f"      budgets:\n"
                    f"        default:\n"
                    f"          max_tiles: 18\n"
                    f"How to fix: add a 'default:' entry under 'inference.tiling.budgets' "
                    f"in {config_file}."
                ]
            )
        for doc_type, entry in budgets.items():
            if not isinstance(entry, dict) or "min_tiles" not in entry or "max_tiles" not in entry:
                raise ConfigError(
                    [
                        f"Invalid entry for 'inference.tiling.budgets.{doc_type}' in {config_file}. "
                        f"What: each entry must have both 'min_tiles' and 'max_tiles' "
                        f"keys; one or both are missing. "
                        f"Where: {config_file} → inference.tiling.budgets.{doc_type}. "
                        f"Expected: a mapping with 'min_tiles' and 'max_tiles', e.g.:\n"
                        f"  inference:\n"
                        f"    tiling:\n"
                        f"      budgets:\n"
                        f"        {doc_type}:\n"
                        f"          min_tiles: 1\n"
                        f"          max_tiles: 18\n"
                        f"How to fix: add 'min_tiles: <int>' and 'max_tiles: <int>' "
                        f"under 'inference.tiling.budgets.{doc_type}' in {config_file}."
                    ]
                )
            for key in ("min_tiles", "max_tiles"):
                if not isinstance(entry[key], int) or entry[key] < 1:
                    raise ConfigError(
                        [
                            f"Invalid '{key}' for 'inference.tiling.budgets.{doc_type}' in {config_file}. "
                            f"What: '{key}' must be a positive integer, got {entry[key]!r}. "
                            f"Where: {config_file} → inference.tiling.budgets.{doc_type}.{key}. "
                            f"Expected: a positive integer, e.g. 18. "
                            f"How to fix: set '{key}' to a positive integer under "
                            f"'inference.tiling.budgets.{doc_type}' in {config_file}."
                        ]
                    )
            if entry["min_tiles"] > entry["max_tiles"]:
                raise ConfigError(
                    [
                        f"Invalid tile budget for 'inference.tiling.budgets.{doc_type}' in {config_file}. "
                        f"What: min_tiles ({entry['min_tiles']}) exceeds max_tiles "
                        f"({entry['max_tiles']}). "
                        f"Where: {config_file} → inference.tiling.budgets.{doc_type}. "
                        f"Expected: min_tiles <= max_tiles, e.g. min_tiles: 12, "
                        f"max_tiles: 18. "
                        f"How to fix: lower 'min_tiles' or raise 'max_tiles' under "
                        f"'inference.tiling.budgets.{doc_type}' in {config_file}."
                    ]
                )
        return dict(budgets)

    @classmethod
    def _validate_quality_screen(cls, raw_config: dict, config_file: str) -> dict[str, Any]:
        """Validate ``pipeline.quality_screen`` section in YAML.

        Every key is required. The screen's whole purpose is a trustworthy
        answer key, and each of these silently defaulted would produce a run
        that looks fine and measures the wrong thing: the wrong prompt variant,
        results written where the scorer will not find them, or a condition
        mapping that quietly drops a severity from the report.
        """
        screen = raw_config.get("pipeline", {}).get("quality_screen")
        example = (
            "  pipeline:\n"
            "    quality_screen:\n"
            "      prompt_file: prompts/quality_screen.yaml\n"
            "      variant: quality_screen_v5\n"
            "      output_name: quality_screen.jsonl\n"
            "      condition_to_level:\n"
            "        clean: NONE\n"
            "        moderate: MODERATE\n"
            "        heavy: HEAVY\n"
            "      tiling:\n"
            "        min_tiles: 6\n"
            "        max_tiles: 12\n"
            "      routing:\n"
            "        pass_levels: [GOOD]\n"
            "        multiple_documents: reject"
        )
        allowed_multiple = ("reject", "allow")
        if screen is None:
            raise ConfigError(
                [
                    f"Missing required key 'pipeline.quality_screen' in {config_file}. "
                    f"What: the key 'pipeline.quality_screen' is absent. "
                    f"Where: {config_file} → pipeline.quality_screen. "
                    f"Expected: a mapping, e.g.:\n{example}\n"
                    f"How to fix: add a 'pipeline.quality_screen:' section to {config_file}."
                ]
            )
        if not isinstance(screen, dict):
            raise ConfigError(
                [
                    f"Invalid type for 'pipeline.quality_screen' in {config_file}: "
                    f"expected a mapping, got {type(screen).__name__}. "
                    f"Where: {config_file} → pipeline.quality_screen. "
                    f"Expected: a mapping, e.g.:\n{example}\n"
                    f"How to fix: change 'pipeline.quality_screen' to a YAML mapping."
                ]
            )
        for key in ("prompt_file", "variant", "output_name", "condition_to_level", "tiling", "routing"):
            if key not in screen:
                raise ConfigError(
                    [
                        f"Missing required key 'pipeline.quality_screen.{key}' in {config_file}. "
                        f"What: the '{key}' key is absent from 'pipeline.quality_screen'. "
                        f"Where: {config_file} → pipeline.quality_screen.{key}. "
                        f"Expected: all of prompt_file, variant, output_name and "
                        f"condition_to_level, e.g.:\n{example}\n"
                        f"How to fix: add '{key}:' under 'pipeline.quality_screen'."
                    ]
                )
        tiling = screen["tiling"]
        if not isinstance(tiling, dict) or not {"min_tiles", "max_tiles"} <= set(tiling):
            raise ConfigError(
                [
                    f"Invalid 'pipeline.quality_screen.tiling' in {config_file}. "
                    f"What: it must declare both 'min_tiles' and 'max_tiles'. Without a budget "
                    f"the backend skips pre-tiling and a small receipt is seen at roughly one "
                    f"tile, at which resolution the model reports heavy damage as good condition. "
                    f"Where: {config_file} → pipeline.quality_screen.tiling. "
                    f"Expected: a mapping with both keys, e.g.:\n{example}\n"
                    f"How to fix: add 'min_tiles:' and 'max_tiles:' under "
                    f"'pipeline.quality_screen.tiling'."
                ]
            )
        if tiling["min_tiles"] > tiling["max_tiles"]:
            raise ConfigError(
                [
                    f"Inverted tile budget in {config_file}. "
                    f"What: min_tiles ({tiling['min_tiles']}) exceeds max_tiles "
                    f"({tiling['max_tiles']}), so the floor cannot be satisfied. "
                    f"Where: {config_file} → pipeline.quality_screen.tiling. "
                    f"Expected: min_tiles <= max_tiles, e.g.:\n{example}\n"
                    f"How to fix: lower min_tiles or raise max_tiles."
                ]
            )
        if not isinstance(screen["condition_to_level"], dict) or not screen["condition_to_level"]:
            raise ConfigError(
                [
                    f"Invalid 'pipeline.quality_screen.condition_to_level' in {config_file}. "
                    f"What: it is not a non-empty mapping, so no corpus condition can be "
                    f"scored against a prompt severity level. "
                    f"Where: {config_file} → pipeline.quality_screen.condition_to_level. "
                    f"Expected: one entry per condition the corpus uses, e.g.:\n{example}\n"
                    f"How to fix: map every corpus condition to an OVERALL level."
                ]
            )
        routing = screen["routing"]
        if not isinstance(routing, dict) or not {"pass_levels", "multiple_documents"} <= set(routing):
            raise ConfigError(
                [
                    f"Invalid 'pipeline.quality_screen.routing' in {config_file}. "
                    f"What: it must declare both 'pass_levels' and 'multiple_documents'. This is "
                    f"the decision the pipeline makes -- send the image on, or send it back -- "
                    f"and it is a policy choice rather than something the scorer can derive from "
                    f"the severity ladder. "
                    f"Where: {config_file} → pipeline.quality_screen.routing. "
                    f"Expected: a mapping with both keys, e.g.:\n{example}\n"
                    f"How to fix: add 'pass_levels:' and 'multiple_documents:' under "
                    f"'pipeline.quality_screen.routing'."
                ]
            )
        if not isinstance(routing["pass_levels"], list) or not routing["pass_levels"]:
            raise ConfigError(
                [
                    f"Invalid 'pipeline.quality_screen.routing.pass_levels' in {config_file}. "
                    f"What: it is not a non-empty list, so no image could ever pass the screen "
                    f"and every document would be returned to the taxpayer. "
                    f"Where: {config_file} → pipeline.quality_screen.routing.pass_levels. "
                    f"Expected: a list of the variant's own OVERALL levels, e.g.:\n{example}\n"
                    f"How to fix: list the severity levels that should pass, e.g. '[GOOD]'."
                ]
            )
        if routing["multiple_documents"] not in allowed_multiple:
            raise ConfigError(
                [
                    f"Invalid 'pipeline.quality_screen.routing.multiple_documents' in "
                    f"{config_file}: {routing['multiple_documents']!r}. "
                    f"What: it must be one of {', '.join(allowed_multiple)}. It decides whether a "
                    f"photograph of several documents is sent on regardless of how good the "
                    f"photograph is. "
                    f"Where: {config_file} → pipeline.quality_screen.routing.multiple_documents. "
                    f"Expected: one of {allowed_multiple}, e.g.:\n{example}\n"
                    f"How to fix: set it to 'reject' when downstream extraction cannot split a "
                    f"collage, or 'allow' when it can."
                ]
            )
        return dict(screen)
