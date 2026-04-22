"""generate_fn wrappers for the agentic extraction engine.

Adapts existing model backends to the ``(image, prompt, NodeGenParams)
-> GenerateResult`` interface consumed by ``GraphExecutor``.

Does NOT modify any existing backend code.
"""

import base64
import io
from collections.abc import Callable
from typing import Any

from PIL import Image

from common.extraction_types import GenerateResult, NodeGenParams


def make_vllm_generate_fn(
    engine: Any,
    *,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> Callable[[Image.Image, str, NodeGenParams], GenerateResult]:
    """Wrap a vLLM ``LLM`` engine into the generate_fn interface.

    Builds ``SamplingParams`` via ``match`` on ``NodeGenParams`` fields.
    Accesses the engine directly -- does not modify ``VllmBackend``.

    Args:
        engine: vLLM ``LLM`` instance (``backend.model``).
        chat_template_kwargs: Extra kwargs for ``engine.chat()``
            (e.g. ``{"enable_thinking": False}`` for Qwen3-VL).
    """
    extra_chat_kwargs = chat_template_kwargs or {}

    def generate(
        image: Image.Image, prompt: str, params: NodeGenParams
    ) -> GenerateResult:
        from vllm import SamplingParams

        # Encode image to base64 data URI (same pattern as vllm_backend.py)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
        buf.close()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Build SamplingParams via match on NodeGenParams
        sampling_kwargs: dict[str, Any] = {
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
        }

        match params:
            case NodeGenParams(output_schema=dict() as schema):
                from vllm.sampling_params import StructuredOutputsParams

                sampling_kwargs["structured_output"] = StructuredOutputsParams(
                    json=schema
                )
            case NodeGenParams(stop=list() as stops) if stops:
                sampling_kwargs["stop"] = stops

        if params.logprobs is not None:
            sampling_kwargs["logprobs"] = params.logprobs

        sampling = SamplingParams(**sampling_kwargs)

        chat_kwargs: dict[str, Any] = {}
        if extra_chat_kwargs:
            chat_kwargs["chat_template_kwargs"] = extra_chat_kwargs

        outputs = engine.chat(
            messages=messages,
            sampling_params=sampling,
            use_tqdm=False,
            **chat_kwargs,
        )

        text = outputs[0].outputs[0].text.strip()

        # Extract logprobs if requested
        token_logprobs = None
        if params.logprobs and outputs[0].outputs[0].logprobs:
            token_logprobs = [
                {tok.decoded_token: tok.logprob for tok in step.values()}
                for step in outputs[0].outputs[0].logprobs
            ]

        del outputs, messages, data_uri
        return GenerateResult(text=text, logprobs=token_logprobs)

    return generate


def make_simple_generate_fn(
    processor: Any,
) -> Callable[[Image.Image, str, NodeGenParams], GenerateResult]:
    """Wrap a processor with ``generate(image, prompt, max_tokens)`` API.

    For non-vLLM backends (InternVL3 .chat(), Llama .generate(), etc.).
    No structured output or logprobs support.

    Args:
        processor: Any object with a ``generate(image, prompt, max_tokens)``
            method returning a string.
    """

    def generate(
        image: Image.Image, prompt: str, params: NodeGenParams
    ) -> GenerateResult:
        text = processor.generate(image, prompt, max_tokens=params.max_tokens)
        return GenerateResult(text=text)

    return generate
