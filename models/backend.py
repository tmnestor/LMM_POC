"""ModelBackend protocol and generation parameters.

Defines the minimal interface every model backend must implement.
The DocumentOrchestrator (orchestrator.py) handles all shared logic
(detection, prompt loading, parsing, cleaning, OOM recovery) and
delegates only raw inference to backends.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from PIL import Image


@dataclass(frozen=True)
class GenerationParams:
    """Parameters controlling model generation.

    Backends receive these from the orchestrator and translate them
    into model-specific generation kwargs.
    """

    max_tokens: int = 1024  # protocol last resort; callers should set explicitly
    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ModelBackend(Protocol):
    """Minimal interface for model inference.

    Every model backend must expose:
      - model: the underlying model object (for Protocol/multi-GPU compat)
      - processor: tokenizer or processor (for Protocol compat)
      - generate(): single-image inference
    """

    model: Any
    processor: Any

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        params: GenerationParams,
    ) -> str:
        """Run model inference on a single image with a text prompt.

        Args:
            image: PIL Image to process.
            prompt: Text prompt for the model.
            params: Generation hyper-parameters.

        Returns:
            Raw model response string.
        """
        ...

    def cache_hit_summary(self) -> dict:
        """Return cumulative prefix-cache hit statistics.

        Returns:
            Dict with at least ``{"available": bool}``.  When available,
            also includes ``cached_prompt_tokens``, ``total_prompt_tokens``,
            and ``hit_ratio``.
        """
        ...
