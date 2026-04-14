"""Model generation configuration constants and registry.

Read-only generation config dicts and a registry for looking them up by
model type key.  All mutable batch/GPU globals and YAML override logic
have been absorbed into ``common.app_config.AppConfig``.
"""

# ============================================================================
# GENERATION CONFIGURATION (read-only defaults)
# ============================================================================

INTERNVL3_GENERATION_CONFIG: dict = {
    "max_new_tokens_base": 2000,
    "max_new_tokens_per_field": 50,
    "do_sample": False,
    "use_cache": True,
    "pad_token_id": None,  # Set dynamically from tokenizer
}

LLAMA_GENERATION_CONFIG: dict = {
    "max_new_tokens_base": 400,
    "max_new_tokens_per_field": 50,
    "temperature": 0.0,
    "do_sample": False,
    "top_p": 0.95,
    "use_cache": True,
}

LLAMA4SCOUT_GENERATION_CONFIG: dict = {
    "max_new_tokens_base": 512,
    "max_new_tokens_per_field": 64,
    "temperature": 0.0,
    "do_sample": False,
    "top_p": 0.95,
    "use_cache": True,
}

QWEN3VL_GENERATION_CONFIG: dict = {
    "max_new_tokens_base": 512,
    "max_new_tokens_per_field": 64,
    "temperature": 0.0,
    "do_sample": False,
    "top_p": 0.95,
    "use_cache": True,
}

GEMMA4_GENERATION_CONFIG: dict = {
    "max_new_tokens_base": 512,
    "max_new_tokens_per_field": 64,
    "do_sample": False,
}

GRANITE4_GENERATION_CONFIG: dict = {
    "max_new_tokens_base": 1024,
    "max_new_tokens_per_field": 64,
    "do_sample": False,
}

# -- Generation config registry (used by AppConfig._build_generation_registry) --
_GENERATION_CONFIG_REGISTRY: dict[str, dict] = {
    "internvl3": INTERNVL3_GENERATION_CONFIG,
    "llama": LLAMA_GENERATION_CONFIG,
    "llama4scout": LLAMA4SCOUT_GENERATION_CONFIG,
    "qwen3vl": QWEN3VL_GENERATION_CONFIG,
    "qwen35": QWEN3VL_GENERATION_CONFIG,
    "nemotron": QWEN3VL_GENERATION_CONFIG,
    "gemma4": GEMMA4_GENERATION_CONFIG,
    "granite4": GRANITE4_GENERATION_CONFIG,
}


def get_generation_config(model_type: str) -> dict:
    """Look up generation config by model type key.

    Returns a *copy* so callers can mutate without affecting the originals.
    Falls back to QWEN3VL config for unknown models.
    """
    return dict(_GENERATION_CONFIG_REGISTRY.get(model_type, QWEN3VL_GENERATION_CONFIG))


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
