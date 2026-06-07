"""Model generation configuration constants and registry.

Read-only generation config schema and a registry for looking them up by
model type key.  All mutable batch/GPU globals and YAML override logic
have been absorbed into ``common.app_config.AppConfig``.

Runtime values come from ``config/run_config.yml`` (generation.defaults +
generation.models).  This module retains the schema for legacy flat-format
YAML fallback and the ``get_model_name_with_size()`` utility.
"""

# ============================================================================
# GENERATION CONFIGURATION SCHEMA (validation / legacy fallback only)
# ============================================================================
# Runtime values come from run_config.yml generation.defaults + generation.models.
# This registry documents the historical code-level defaults and serves as
# fallback when run_config.yml uses the legacy flat format.

_GENERATION_CONFIG_SCHEMA: dict[str, dict] = {
    "internvl3": {
        "max_new_tokens_base": 2000,
        "max_new_tokens_per_field": 50,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": None,  # Set dynamically from tokenizer
    },
}

# Backwards-compat alias: _build_generation_registry() in app_config.py
# imports this name for the legacy flat YAML format fallback path.
_GENERATION_CONFIG_REGISTRY = _GENERATION_CONFIG_SCHEMA


def get_generation_config(model_type: str) -> dict:
    """Look up generation config by model type key.

    Returns a *copy* so callers can mutate without affecting the originals.
    Falls back to InternVL3 config for unknown models.
    """
    return dict(_GENERATION_CONFIG_SCHEMA.get(model_type, _GENERATION_CONFIG_SCHEMA["internvl3"]))


# ============================================================================
# UTILITY
# ============================================================================


def get_model_name_with_size(
    base_model_name: str,
    model_path: str | None = None,
    is_8b_model: bool | None = None,
) -> str:
    """Generate size-aware model name for batch size configuration lookup.

    Args:
        base_model_name: Base model name ('internvl3', etc.)
        model_path: Path to model (used for size detection if is_8b_model not provided)
        is_8b_model: Whether model is 8B variant (overrides path detection)

    Returns:
        Size-aware model name ('internvl3-2b', 'internvl3-8b', or original name)
    """
    base_name = base_model_name.lower()

    if base_name != "internvl3":
        return base_name

    if is_8b_model is None and model_path:
        is_8b_model = "8B" in str(model_path)

    if is_8b_model:
        return "internvl3-8b"
    return "internvl3-2b"
