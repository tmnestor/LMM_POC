"""vLLM offline engine backend.

Extracted from document_aware_vllm_processor.py.
Uses vLLM's chat API with OpenAI-compatible message format.
"""

import base64
import io
from typing import Any

from PIL import Image

from models.backend import GenerationParams, ModelBackend


class VllmBackend:
    """Backend for vLLM offline engine inference.

    Implements ModelBackend protocol only (vLLM handles batching internally).
    No OOM recovery needed -- vLLM manages memory via PagedAttention.
    """

    def __init__(
        self,
        engine: Any,
        *,
        model_type_key: str = "internvl3",
        debug: bool = False,
    ) -> None:
        self.model = engine  # vLLM LLM engine
        self.processor = None  # vLLM handles tokenization internally
        self._model_type_key = model_type_key
        self._debug = debug

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        params: GenerationParams,
    ) -> str:
        """Run inference via vLLM engine.chat()."""
        from vllm import SamplingParams

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

        sampling = SamplingParams(
            max_tokens=params.max_tokens,
            temperature=0,
        )

        chat_kwargs: dict[str, Any] = {}
        if self._model_type_key.startswith(("qwen35", "gemma4")):
            chat_kwargs["chat_template_kwargs"] = {"enable_thinking": False}

        outputs = self.model.chat(
            messages=messages,
            sampling_params=sampling,
            use_tqdm=False,
            **chat_kwargs,
        )

        text = outputs[0].outputs[0].text.strip()
        # Free vLLM output objects to release shared memory buffer slots.
        del outputs, messages, data_uri
        return text


# Verify protocol compliance at import time
_dummy_check: type[ModelBackend] = VllmBackend  # type: ignore[assignment]
