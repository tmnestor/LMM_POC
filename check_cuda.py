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
            total_gb = props.total_mem / 1024**3
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


if __name__ == "__main__":
    main()
