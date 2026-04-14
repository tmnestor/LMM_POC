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

    max_tokens: int = 1024
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


@runtime_checkable
class BatchInference(Protocol):
    """Optional interface for backends that support batched inference.

    Implementing this enables the orchestrator to batch detection
    and extraction calls for higher GPU utilization.
    """

    def generate_batch(
        self,
        images: list[Image.Image],
        prompts: list[str],
        params: GenerationParams,
    ) -> list[str]:
        """Run batched inference on multiple images.

        Args:
            images: List of PIL Images.
            prompts: List of text prompts (one per image).
            params: Generation hyper-parameters (same for all).

        Returns:
            List of raw model response strings (one per image).
        """
        ...
