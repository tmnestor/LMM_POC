"""vLLM offline engine backend.

Extracted from document_aware_vllm_processor.py.
Uses vLLM's chat API with OpenAI-compatible message format.
"""

import base64
import io
import logging
from typing import Any

from PIL import Image

from common import image_tiling, prompt_trace
from common.extraction_types import GenerateResult, NodeGenParams
from models.backend import GenerationParams, ModelBackend

logger = logging.getLogger(__name__)


class VllmBackend:
    """Backend for vLLM offline engine inference.

    Implements ModelBackend protocol only (vLLM handles batching internally).
    No OOM recovery needed -- vLLM manages memory via PagedAttention.

    Also provides ``generate_for_graph()`` for the GraphExecutor interface,
    ensuring a single source of truth for vLLM message construction.
    """

    def __init__(
        self,
        engine: Any,
        *,
        model_type_key: str = "internvl3",
        chat_template: str | None = None,
        trace_path: str | None = None,
        pre_tiling_enabled: bool = False,
        tile_image_size: int = 448,
        tile_use_thumbnail: bool = True,
        debug: bool = False,
    ) -> None:
        self.model = engine  # vLLM LLM engine
        self.processor = None  # vLLM handles tokenization internally
        self._model_type_key = model_type_key
        # Optional chat-template override (path validated at config load); None
        # uses the model's own template. Forwarded to every engine.chat() call.
        self._chat_template = chat_template
        self._debug = debug
        # Pre-tiling: when enabled, the backend crops each image into
        # ``max_tiles`` 448-square sub-images itself and hands vLLM the crops as
        # separate images, so OUR tile count is authoritative (vLLM's per-image
        # max_dynamic_patch cap no longer bounds dense bank statements). The
        # per-call tile count arrives via params.extra["max_tiles"] (generate) or
        # params.max_tiles (generate_for_graph); None -> the legacy single-image
        # path. See plans/2026-06-04-adaptive-pre-tiling.md.
        self._pre_tiling_enabled = pre_tiling_enabled
        self._tile_image_size = tile_image_size
        self._tile_use_thumbnail = tile_use_thumbnail
        # Raw-prompt trace: when a path is given, every generate() call is
        # appended to that JSONL via the shared prompt_trace sink (debug only).
        if trace_path:
            prompt_trace.enable(trace_path)

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        """PNG-encode a PIL image as a base64 data URI."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
        buf.close()
        return data_uri

    def _image_parts(
        self, image: Image.Image, max_tiles: int | None, min_tiles: int = 1
    ) -> list[dict[str, Any]]:
        """Build the image content parts for one source image.

        When pre-tiling is enabled and *max_tiles* is given, crop the image into
        ``min_tiles``..``max_tiles`` 448-square tiles (plus a thumbnail) and
        return one ``image_url`` part per tile — each 448-square crop is a single
        block inside vLLM, so the crop count is authoritative. Otherwise return a
        single part with the full image (vLLM tiles it internally, capped at the
        checkpoint default).

        ``min_tiles`` is the FLOOR: the InternVL tiling algorithm picks the grid
        by closest aspect-ratio match, so a clean A4-portrait statement settles
        on a 6-tile (2x3) grid and never reaches ``max_tiles`` on its own.
        Raising the floor forces the denser grid dense tables need. The floor is
        clamped to ``[1, max_tiles]`` so a misconfigured budget can't invert.
        """
        if not (self._pre_tiling_enabled and max_tiles):
            logger.info(
                "pre-tiling OFF for %dx%d image (enabled=%s, max_tiles=%s) -- "
                "sending single image; vLLM tiles internally at the checkpoint default",
                image.width,
                image.height,
                self._pre_tiling_enabled,
                max_tiles,
            )
            return [{"type": "image_url", "image_url": {"url": self._encode_image(image)}}]

        min_num = max(1, min(min_tiles, max_tiles))
        tiles = image_tiling.dynamic_preprocess(
            image,
            min_num=min_num,
            max_num=max_tiles,
            image_size=self._tile_image_size,
            use_thumbnail=self._tile_use_thumbnail,
        )
        logger.info(
            "pre-tiling ON: cropped %dx%d image into %d sub-images "
            "(min_tiles=%d, max_tiles=%d, thumbnail=%s)",
            image.width,
            image.height,
            len(tiles),
            min_num,
            max_tiles,
            self._tile_use_thumbnail,
        )
        return [{"type": "image_url", "image_url": {"url": self._encode_image(tile)}} for tile in tiles]

    def _emit_trace(
        self,
        prompt: str,
        text: str,
        prompt_token_ids: Any,
        completion_token_ids: Any,
    ) -> None:
        """Append this VLM call to the raw-prompt trace (no-op when disabled)."""
        if not prompt_trace.is_enabled():
            return
        prompt_trace.record(
            prompt=prompt,
            response=text,
            model=self._model_type_key,
            prompt_tokens=len(prompt_token_ids) if prompt_token_ids else None,
            completion_tokens=len(completion_token_ids) if completion_token_ids else None,
        )

    def _build_messages(
        self,
        image: Image.Image,
        prompt: str,
        *,
        image_first: bool = False,
        max_tiles: int | None = None,
        min_tiles: int = 1,
    ) -> list[dict[str, Any]]:
        """Build OpenAI-compatible messages.

        Args:
            image: PIL image to encode.
            prompt: Text prompt.
            image_first: If True, place image before text in the content
                array.  Extraction tasks use image-first for lower FP
                rates; classification uses text-first (the default).
            max_tiles: When pre-tiling is enabled, crop into at most this many
                tiles and send them as separate images; None uses the
                single-image path (vLLM tiles internally).
            min_tiles: Pre-tiling floor — forces at least this many tiles so
                clean-aspect documents don't under-tile (see ``_image_parts``).

        Returns:
            The messages list.
        """
        text_part: dict[str, Any] = {"type": "text", "text": prompt}
        image_parts = self._image_parts(image, max_tiles, min_tiles)
        content = [*image_parts, text_part] if image_first else [text_part, *image_parts]

        return [{"role": "user", "content": content}]

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        params: GenerationParams,
    ) -> str:
        """Run inference via vLLM engine.chat()."""
        from vllm import SamplingParams

        messages = self._build_messages(
            image,
            prompt,
            max_tiles=params.extra.get("max_tiles"),
            min_tiles=params.extra.get("min_tiles", 1),
        )

        sampling = SamplingParams(
            max_tokens=params.max_tokens,
            temperature=0,
        )

        outputs = self.model.chat(
            messages=messages,
            sampling_params=sampling,
            chat_template=self._chat_template,
            use_tqdm=False,
        )

        text = outputs[0].outputs[0].text.strip()
        self._emit_trace(
            prompt, text, getattr(outputs[0], "prompt_token_ids", None), outputs[0].outputs[0].token_ids
        )
        # Free vLLM output objects to release shared memory buffer slots.
        del outputs, messages
        return text

    def generate_for_graph(
        self,
        image: Image.Image,
        prompt: str,
        params: NodeGenParams,
    ) -> GenerateResult:
        """Run inference for GraphExecutor.

        Accepts ``NodeGenParams`` and returns ``GenerateResult``, matching
        the ``(Image, str, NodeGenParams) -> GenerateResult`` signature
        expected by ``GraphExecutor``.

        Uses image-first ordering — empirically produces fewer false
        positives on extraction/compliance tasks than text-first.
        """
        from vllm import SamplingParams

        messages = self._build_messages(
            image,
            prompt,
            image_first=True,
            max_tiles=getattr(params, "max_tiles", None),
            min_tiles=getattr(params, "min_tiles", 1),
        )

        # Build SamplingParams from NodeGenParams fields
        sampling_kwargs: dict[str, Any] = {
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
        }

        if params.output_schema is not None:
            from vllm.sampling_params import StructuredOutputsParams

            sampling_kwargs["structured_output"] = StructuredOutputsParams(json=params.output_schema)
        elif params.stop:
            sampling_kwargs["stop"] = params.stop

        if params.logprobs is not None:
            sampling_kwargs["logprobs"] = params.logprobs

        sampling = SamplingParams(**sampling_kwargs)

        outputs = self.model.chat(
            messages=messages,
            sampling_params=sampling,
            chat_template=self._chat_template,
            use_tqdm=False,
        )

        text = outputs[0].outputs[0].text.strip()
        self._emit_trace(
            prompt, text, getattr(outputs[0], "prompt_token_ids", None), outputs[0].outputs[0].token_ids
        )

        # Extract logprobs if requested
        token_logprobs = None
        if params.logprobs and outputs[0].outputs[0].logprobs:
            token_logprobs = [
                {tok.decoded_token: tok.logprob for tok in step.values()}
                for step in outputs[0].outputs[0].logprobs
            ]

        del outputs, messages
        return GenerateResult(text=text, logprobs=token_logprobs)


# Verify protocol compliance at import time
_dummy_check: type[ModelBackend] = VllmBackend  # type: ignore[assignment]
