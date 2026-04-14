"""Llama 3.2 Vision model backend.

Extracted from document_aware_llama_processor.py.
Uses processor.apply_chat_template() + model.generate() API.
"""

from typing import Any

import torch
from PIL import Image

from models.backend import GenerationParams, ModelBackend


class LlamaBackend:
    """Backend for Llama 3.2 Vision (MllamaForConditionalGeneration).

    Implements ModelBackend protocol only (no batch support).
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        debug: bool = False,
    ) -> None:
        self.model = model
        self.processor = processor
        self._debug = debug

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        params: GenerationParams,
    ) -> str:
        """Run Llama Vision inference on a single image + prompt."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_input = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            images=[image], text=text_input, return_tensors="pt"
        ).to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=params.max_tokens,
            do_sample=params.do_sample,
            temperature=params.temperature,
            top_p=params.top_p,
        )

        generate_ids = output[:, inputs["input_ids"].shape[1] : -1]
        response = self.processor.decode(
            generate_ids[0], clean_up_tokenization_spaces=False
        )

        del inputs, output, generate_ids
        torch.cuda.empty_cache()
        return response


# Verify protocol compliance at import time
_dummy_check: type[ModelBackend] = LlamaBackend  # type: ignore[assignment]
