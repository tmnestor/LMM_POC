"""DocumentOrchestrator — one prompt, one image, the text that comes back.

Composition rather than inheritance: it has-a ModelBackend and adds what is
the same whatever the prompt asks -- image loading, OOM recovery, and per-image
trace attribution.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from common import prompt_trace
from models.backend import GenerationParams, ModelBackend

if TYPE_CHECKING:
    from common.app_config import AppConfig


class DocumentOrchestrator:
    """Sends one prompt at one image and returns the text that comes back.

    Composition rather than inheritance: it has-a ModelBackend and adds the
    things that are the same whatever the prompt asks -- image loading, OOM
    recovery, and per-image trace attribution.

    It used to own detection, classification, prompt resolution, extraction,
    parsing and cleaning as well, and required a prompt-routing config, a
    universal field list and per-type field definitions to be constructed. The
    screen used none of that: it sends one prompt and reads seven answers. The
    scaffolding is gone rather than passed in empty, because while it was still
    required it kept `prompts/internvl3_prompts.yaml`, the field schema, the
    prompt catalogue and the response handler alive -- files nothing on this
    branch reads, held up by a constructor argument.

    Attributes:
        model: Underlying model object (delegates to backend).
        tokenizer: Tokenizer / processor (delegates to backend).
    """

    def __init__(
        self,
        backend: ModelBackend,
        *,
        debug: bool = False,
        verbose: bool = False,
        device: str = "cuda",
        model_type_key: str = "internvl3",
        app_config: AppConfig | None = None,
        has_oom_recovery: bool = True,
    ) -> None:
        self._backend = backend
        # debug → Tier C (dev-noise: prompt/response dumps, tracebacks).
        # verbose → Tier B (init/config details).
        self.debug = debug
        self._verbose = verbose
        self.device = device
        self._model_type_key = model_type_key
        self._has_oom_recovery = has_oom_recovery
        # Retained for callers that pass it; nothing on the screen path reads
        # it, since the token budget arrives with each call.
        self.app_config = app_config

        if self._verbose:
            print(f"DocumentOrchestrator initialized: model_type={model_type_key}")

    # -- Protocol-required attributes ------------------------------------------

    @property
    def model(self) -> Any:
        """Underlying model object (for DocumentProcessor protocol)."""
        return self._backend.model

    @property
    def tokenizer(self) -> Any:
        """Tokenizer / processor (for DocumentProcessor protocol)."""
        return self._backend.processor

    def load_document_image(self, image_path: str) -> Image.Image:
        """Load document image with error handling."""
        try:
            return Image.open(image_path)
        except Exception as e:
            if self.debug:
                print(f"Error loading image {image_path}: {e}")
            raise

    # -- Core generate (delegates to backend with OOM recovery) ----------------

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_tokens: int = 1024,
        extra: dict | None = None,
    ) -> str:
        """Run model inference, with optional OOM recovery.

        This is the Protocol-required generate() method.

        Args:
            extra: Optional dict passed through to GenerationParams.extra
                (e.g. ``{"max_tiles": 6}`` for per-type tile budgets).
        """
        params = GenerationParams(max_tokens=max_tokens, extra=extra or {})
        if self._has_oom_recovery:
            return self._resilient_generate(image, prompt, params)
        return self._backend.generate(image, prompt, params)

    def cache_hit_summary(self) -> dict:
        """Proxy the backend's cumulative prefix-cache hit summary."""
        return self._backend.cache_hit_summary()

    def _resilient_generate(self, image: Image.Image, prompt: str, params: GenerationParams) -> str:
        """Generate with OOM recovery (halve tokens and retry).

        Cleanup happens OUTSIDE the except block -- see MEMORY.md for why.
        """
        import torch

        oom = False
        try:
            return self._backend.generate(image, prompt, params)
        except torch.cuda.OutOfMemoryError:
            oom = True

        assert oom  # noqa: S101 -- always True; satisfies mypy reachability
        gc.collect()
        torch.cuda.empty_cache()
        if self.debug:
            print(f"OOM at {params.max_tokens} tokens, retrying at {params.max_tokens // 2}")
        retry_params = GenerationParams(
            max_tokens=params.max_tokens // 2,
            do_sample=params.do_sample,
            temperature=params.temperature,
            top_p=params.top_p,
            extra=params.extra,
        )
        return self._backend.generate(image, prompt, retry_params)

    def screen_batch(
        self,
        image_paths: list[str],
        prompt: str,
        max_tokens: int,
        verbose: bool = False,
        tile_extra: dict | None = None,
    ) -> list[str]:
        """Run one prompt over each image and return the raw responses.

        Deliberately returns text and nothing else -- no parsing. The
        image-quality screen's reader lives in `common.quality_screen_parser`
        because reading its answers is a testable unit in its own right, and
        the distinction between "the model said NO" and "the model said
        something unreadable" is the whole point of it.

        "batch" here means one call per batch of images, not one engine call:
        the images are sent one at a time. Throughput comes from sharding
        across GPUs in `common.vllm_dp`, not from batching within a process.
        This method briefly had a batched branch guarded by a `supports_batch`
        check; no backend has ever implemented `generate_batch`, so that branch
        never ran, and the assertion inside it -- a precondition copied from a
        caller that does not exist here -- crashed the first GPU run. Going one
        at a time also keeps `generate`'s OOM recovery, which a raw
        `generate_batch` call would bypass.

        Args:
            image_paths: Images to send.
            prompt: The prompt to ask about every image.
            max_tokens: Generation budget.
            verbose: Whether to log progress.
            tile_extra: Optional `{"min_tiles": n, "max_tiles": m}` forwarded to
                GenerationParams.extra. Without it the backend skips app-side
                pre-tiling and lets vLLM tile internally, where the grid is
                chosen by closest aspect-ratio match -- which settles a small
                receipt on roughly ONE tile. `min_tiles` is the floor that
                forces a denser grid; `max_tiles` alone does nothing.

        Returns:
            One raw response per image, in the order given.
        """
        if not image_paths:
            return []

        if verbose:
            sys.stdout.write(f"Screening {len(image_paths)} images (tiles={tile_extra})\n")
            sys.stdout.flush()

        images = [self.load_document_image(path) for path in image_paths]

        # Each call is wrapped in its own trace context so the raw-prompt trace
        # can be read back per image. Without this every line carries
        # image_name: null, which is close to useless past the first image --
        # the 2026-08-11 finding, which regressed here when the DP workers that
        # were the only trace_context callers were deleted. The context is
        # scoped to one call, so it cannot leak into a later unrelated one.
        responses = []
        for path, image in zip(image_paths, images, strict=True):
            with prompt_trace.trace_context(
                image_name=Path(path).name, label="quality_screen", pipeline="quality_screen"
            ):
                responses.append(self.generate(image, prompt, max_tokens, extra=tile_extra))
        return responses
