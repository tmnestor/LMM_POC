"""SDPA attention patching for eager attention backends.

Extracted from registry.py — purely mechanical move.

Monkey-patches eager attention to use PyTorch's scaled_dot_product_attention,
which uses the flash backend on Ampere+ GPUs natively. This avoids
materializing the full O(N^2) attention matrix.
"""

_sdpa_patched: bool = False


def is_sdpa_patched() -> bool:
    """Return whether SDPA patching has been applied."""
    return _sdpa_patched


def mark_sdpa_patched() -> None:
    """Mark SDPA as patched (called after successful patch)."""
    global _sdpa_patched  # noqa: PLW0603
    _sdpa_patched = True


def patch_eager_attention_to_sdpa() -> bool:
    """Monkey-patch eager attention to use PyTorch SDPA globally.

    Replaces the eager_attention_forward entry in transformers'
    ALL_ATTENTION_FUNCTIONS registry with F.scaled_dot_product_attention.
    This avoids materializing the full O(N^2) attention matrix, which
    OOMs on multi-GPU with high tile counts.

    Returns:
        True if the patch was applied, False otherwise.
    """
    import torch.nn.functional as F

    def _sdpa_attention(
        module,
        query,
        key,
        value,
        attention_mask=None,
        scaling=None,
        dropout=0.0,
        **kwargs,
    ):
        dp = dropout if module.training else 0.0

        # Expand KV heads to match Q heads so all SDPA backends are eligible.
        # enable_gqa=True dispatches to flash on PyTorch 2.9+ but produces
        # degraded accuracy and throughput in practice (64.4% vs 67.9%,
        # 3.48 vs 4.31 img/min) — likely due to attention mask interaction.
        # Manual expansion is proven reliable. See notebooks/sdpa_gqa_diagnostic.ipynb.
        num_kv_heads = key.shape[1]
        num_q_heads = query.shape[1]
        if num_kv_heads != num_q_heads:
            repeat_factor = num_q_heads // num_kv_heads
            key = key.repeat_interleave(repeat_factor, dim=1)
            value = value.repeat_interleave(repeat_factor, dim=1)

        # Prepare causal mask: truncate to KV length and ensure
        # head dim is broadcastable.
        causal_mask = attention_mask
        if causal_mask is not None:
            causal_mask = causal_mask[:, :, :, : key.shape[-2]]
            if causal_mask.shape[1] > 1 and causal_mask.shape[1] != num_q_heads:
                causal_mask = causal_mask[:, :1, :, :]

        is_causal = causal_mask is None and query.shape[2] > 1
        if is_causal:
            causal_mask = None

        attn_output = F.scaled_dot_product_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            attn_mask=causal_mask,
            dropout_p=dp,
            scale=scaling,
            is_causal=is_causal,
        )
        # Transpose to [batch, seq, heads, dim] to match eager_attention_forward's
        # output layout — the caller does .reshape(*input_shape, -1) which
        # requires seq before heads.
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, None

    patched = False

    # Patch the global attention function registry (transformers 4.46+)
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        ALL_ATTENTION_FUNCTIONS["eager"] = _sdpa_attention
        patched = True
    except (ImportError, AttributeError):
        pass

    # Also patch module-level function references as a fallback.
    # The global registry patch is primary; these catch models that
    # reference eager_attention_forward directly in their module.
    for module_path in (
        "transformers.models.qwen3.modeling_qwen3",
        "transformers.models.qwen3_5.modeling_qwen3_5",
        "transformers.models.qwen3_vl.modeling_qwen3_vl",
    ):
        try:
            import importlib

            mod = importlib.import_module(module_path)
            mod.eager_attention_forward = _sdpa_attention  # type: ignore[attr-defined]
            patched = True
        except (ImportError, AttributeError):
            pass

    return patched
