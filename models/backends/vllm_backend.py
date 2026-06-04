"""vLLM offline engine backend.

Extracted from document_aware_vllm_processor.py.
Uses vLLM's chat API with OpenAI-compatible message format.
"""

import base64
import io
from typing import Any

from PIL import Image

from common import prompt_trace
from common.extraction_types import GenerateResult, NodeGenParams
from models.backend import GenerationParams, ModelBackend


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
        debug: bool = False,
    ) -> None:
        self.model = engine  # vLLM LLM engine
        self.processor = None  # vLLM handles tokenization internally
        self._model_type_key = model_type_key
        # Optional chat-template override (path validated at config load); None
        # uses the model's own template. Forwarded to every engine.chat() call.
        self._chat_template = chat_template
        self._debug = debug
        # Raw-prompt trace: when a path is given, every generate() call is
        # appended to that JSONL via the shared prompt_trace sink (debug only).
        if trace_path:
            prompt_trace.enable(trace_path)

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
        self, image: Image.Image, prompt: str, *, image_first: bool = False
    ) -> tuple[list[dict[str, Any]], str]:
        """Build OpenAI-compatible messages.

        Args:
            image: PIL image to encode.
            prompt: Text prompt.
            image_first: If True, place image before text in the content
                array.  Extraction tasks use image-first for lower FP
                rates; classification uses text-first (the default).

        Returns:
            Tuple of (messages list, data_uri string). The data_uri is
            returned so callers can ``del`` it after use.
        """
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
        buf.close()

        text_part: dict[str, Any] = {"type": "text", "text": prompt}
        image_part: dict[str, Any] = {"type": "image_url", "image_url": {"url": data_uri}}
        content = [image_part, text_part] if image_first else [text_part, image_part]

        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        return messages, data_uri

    def _chat_template_kwargs(self) -> dict[str, Any]:
        """Build chat_template_kwargs for thinking suppression.

        Only affects Qwen3/Gemma-style templates that honour the
        ``enable_thinking`` variable. InternVL3.5 ignores it — and on that family
        sending ANY system prompt turns thinking ON, so detection sends none: a
        bare user message keeps the model in direct-answer mode.
        """
        if self._model_type_key.startswith(("qwen35", "gemma4")):
            return {"chat_template_kwargs": {"enable_thinking": False}}
        return {}

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        params: GenerationParams,
    ) -> str:
        """Run inference via vLLM engine.chat()."""
        from vllm import SamplingParams

        messages, data_uri = self._build_messages(image, prompt)

        sampling = SamplingParams(
            max_tokens=params.max_tokens,
            temperature=0,
        )

        outputs = self.model.chat(
            messages=messages,
            sampling_params=sampling,
            chat_template=self._chat_template,
            use_tqdm=False,
            **self._chat_template_kwargs(),
        )

        text = outputs[0].outputs[0].text.strip()
        self._emit_trace(
            prompt, text, getattr(outputs[0], "prompt_token_ids", None), outputs[0].outputs[0].token_ids
        )
        # Free vLLM output objects to release shared memory buffer slots.
        del outputs, messages, data_uri
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

        messages, data_uri = self._build_messages(image, prompt, image_first=True)

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
            **self._chat_template_kwargs(),
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

        del outputs, messages, data_uri
        return GenerateResult(text=text, logprobs=token_logprobs)


# Verify protocol compliance at import time
_dummy_check: type[ModelBackend] = VllmBackend  # type: ignore[assignment]
