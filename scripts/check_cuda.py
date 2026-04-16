#!/usr/bin/env python3
"""Check CUDA, PyTorch, and GPU environment."""

import shutil
import subprocess


def run_cmd(cmd: str) -> str:
    """Run a shell command and return output."""
    try:
        result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "not found"


def main() -> None:
    print("=" * 60)
    print("CUDA / PyTorch Environment")
    print("=" * 60)

    # nvidia-smi
    if shutil.which("nvidia-smi"):
        print(f"\n{'nvidia-smi':.<30s} found")
        print(run_cmd("nvidia-smi"))
    else:
        print(f"\n{'nvidia-smi':.<30s} NOT FOUND")

    # nvcc
    nvcc = run_cmd("nvcc --version")
    print(f"\n{'CUDA Toolkit (nvcc)':.<30s}")
    print(nvcc)

    # PyTorch
    print(f"\n{'=' * 60}")
    print("PyTorch Details")
    print("=" * 60)

    try:
        import torch

        print(f"{'PyTorch version':.<30s} {torch.__version__}")
        print(f"{'CUDA available':.<30s} {torch.cuda.is_available()}")
        print(f"{'CUDA version':.<30s} {torch.version.cuda or 'N/A'}")
        print(f"{'cuDNN version':.<30s} {torch.backends.cudnn.version()}")
        print(f"{'cuDNN enabled':.<30s} {torch.backends.cudnn.enabled}")
        print(f"{'GPU count':.<30s} {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_gb = (
                getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1024**3
            )
            print(f"\n  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    {'Total memory':.<26s} {total_gb:.1f} GiB")
            print(f"    {'Compute capability':.<26s} {props.major}.{props.minor}")
            print(f"    {'Multi-processors':.<26s} {props.multi_processor_count}")

        # Build config
        print(f"\n{'=' * 60}")
        print("PyTorch Build Config")
        print("=" * 60)
        print(torch.__config__.show())

    except ImportError:
        print("PyTorch: NOT INSTALLED")

    # torchvision
    print(f"{'=' * 60}")
    print("Related Packages")
    print("=" * 60)

    try:
        import torchvision

        print(f"{'torchvision':.<30s} {torchvision.__version__}")
    except ImportError:
        print(f"{'torchvision':.<30s} NOT INSTALLED")

    try:
        import flash_attn

        print(f"{'flash_attn':.<30s} {flash_attn.__version__}")
    except ImportError:
        print(f"{'flash_attn':.<30s} NOT INSTALLED")

    try:
        import transformers

        print(f"{'transformers':.<30s} {transformers.__version__}")
    except ImportError:
        print(f"{'transformers':.<30s} NOT INSTALLED")

    try:
        import accelerate

        print(f"{'accelerate':.<30s} {accelerate.__version__}")
    except ImportError:
        print(f"{'accelerate':.<30s} NOT INSTALLED")

    # Attention backend diagnostic
    print(f"\n{'=' * 60}")
    print("Attention Backend")
    print("=" * 60)
    _check_attention_backend()


def _check_attention_backend() -> None:
    """Inspect what 'eager' attention resolves to in the HF registry.

    After models.attention.patch_eager_attention_to_sdpa() runs, the
    'eager' entry in transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS
    should point to our _sdpa_attention wrapper (which calls
    F.scaled_dot_product_attention). If it still points to transformers'
    eager_attention_forward, the SDPA patch did NOT fire and the model
    will materialize the full O(N^2) attention matrix -> OOM risk.

    This runs the patch (idempotent, safe outside a model load), then
    prints the resolved function for each attention key. Useful on prod
    to confirm SDPA is active before a full eval.
    """
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except ImportError as exc:
        print(f"ALL_ATTENTION_FUNCTIONS: unavailable ({exc})")
        return

    # Pre-patch state
    pre = ALL_ATTENTION_FUNCTIONS.get("eager")
    pre_repr = f"{pre.__module__}.{pre.__name__}" if pre is not None else "None"
    print(f"{'eager (pre-patch)':.<30s} {pre_repr}")

    # Apply the patch (idempotent) and show post-patch state
    try:
        from models.attention import (
            is_sdpa_patched,
            mark_sdpa_patched,
            patch_eager_attention_to_sdpa,
        )
    except ImportError as exc:
        print(f"models.attention: unavailable ({exc})")
        return

    if not is_sdpa_patched():
        if patch_eager_attention_to_sdpa():
            mark_sdpa_patched()

    post = ALL_ATTENTION_FUNCTIONS.get("eager")
    post_repr = f"{post.__module__}.{post.__name__}" if post is not None else "None"
    print(f"{'eager (post-patch)':.<30s} {post_repr}")
    print(f"{'is_sdpa_patched()':.<30s} {is_sdpa_patched()}")

    # Native flash availability
    try:
        import flash_attn  # noqa: F401

        print(f"{'native flash-attn':.<30s} available (preferred)")
    except ImportError:
        print(f"{'native flash-attn':.<30s} NOT installed (SDPA patch required)")

    # Expected post-patch: models.attention._sdpa_attention
    # If you see transformers.*.eager_attention_forward, the patch did
    # NOT apply and attention will materialize the full NxN matrix.
    expected = "models.attention._sdpa_attention"
    if post_repr == expected:
        print(f"\n[OK] eager is patched to SDPA ({expected})")
    else:
        print("\n[WARN] eager is NOT patched to SDPA")
        print(f"       expected: {expected}")
        print(f"       actual:   {post_repr}")


if __name__ == "__main__":
    main()
