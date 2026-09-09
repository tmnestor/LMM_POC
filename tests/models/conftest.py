"""Conftest for models tests — stubs vllm so tests run without a GPU install."""

import sys
import types


def _ensure_vllm_stub() -> None:
    """Insert a minimal vllm stub when the real package is not installed."""
    if "vllm" in sys.modules:
        return

    vllm_mod = types.ModuleType("vllm")

    class SamplingParams:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    class LLM:
        """Engine stub — constructing it in a test means a guard failed to fire."""

        def __init__(self, **kwargs: object) -> None:
            raise AssertionError(
                "vllm.LLM was constructed in a test; loader tests must fail before "
                f"engine construction (kwargs={sorted(kwargs)})"
            )

    vllm_mod.SamplingParams = SamplingParams  # type: ignore[attr-defined]
    vllm_mod.LLM = LLM  # type: ignore[attr-defined]
    sys.modules["vllm"] = vllm_mod


_ensure_vllm_stub()
