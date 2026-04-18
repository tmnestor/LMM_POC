"""Smoke-test for vLLM environment at /efs/shared/.conda/envs/vllm."""

import sys


def check(label: str, fn):
    """Run a check and print pass/fail."""
    try:
        result = fn()
        print(f"  PASS  {label}: {result}")
    except Exception as e:
        print(f"  FAIL  {label}: {e}")


def main():
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")
    print()

    # Core packages
    check("torch import + version", lambda: __import__("torch").__version__)
    check("CUDA available", lambda: __import__("torch").cuda.is_available())
    check("GPU count", lambda: __import__("torch").cuda.device_count())
    check(
        "GPU names",
        lambda: [
            __import__("torch").cuda.get_device_name(i)
            for i in range(__import__("torch").cuda.device_count())
        ],
    )
    check("torchvision", lambda: __import__("torchvision").__version__)
    check("torchaudio", lambda: __import__("torchaudio").__version__)

    # vLLM
    check("vllm import + version", lambda: __import__("vllm").__version__)

    # vLLM dependencies
    check("flashinfer", lambda: __import__("flashinfer").__version__)

    # Quick functional test — can vLLM see GPUs?
    check(
        "vllm GPU check",
        lambda: (
            (
                __import__("torch").cuda.is_available()
                and __import__("torch").cuda.device_count() > 0
                and "GPUs visible to vLLM"
            )
            or "No GPUs detected"
        ),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
