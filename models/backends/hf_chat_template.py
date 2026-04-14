"""Parametric HuggingFace chat template backend.

Covers multiple chat template styles used by different HF models:

| Style      | Models              | API                                                     |
|------------|---------------------|---------------------------------------------------------|
| two_step   | Qwen3VL, Nemotron   | apply_chat_template(tokenize=False) then processor()    |
| one_step   | Qwen35, Llama4      | apply_chat_template(tokenize=True, return_dict=True)    |

This replaces 5 near-identical stub processors with one parametric class.
"""

from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image

from models.backend import GenerationParams, ModelBackend


@dataclass
class ChatTemplateConfig:
    """Configuration for a chat template backend.

    Passed at construction time to parameterize the backend for a specific model.
    """

    message_style: str = "two_step"
    system_message: str | None = None
    image_content_type: str = "image"
    image_content_key: str | None = None
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    generate_kwargs: dict[str, Any] = field(default_factory=dict)
    suppress_gen_warnings: tuple[str, ...] = ("temperature", "top_p")
    tokenizer_attr: str = "tokenizer"


class HFChatTemplateBackend:
    """Parametric backend for standard HuggingFace VLMs.

    Implements ModelBackend protocol.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        config: ChatTemplateConfig | None = None,
        *,
        debug: bool = False,
    ) -> None:
        self.model = model
        self.processor = processor
        self._config = config or ChatTemplateConfig()
        self._debug = debug

    def _build_messages(self, image: Image.Image, prompt: str) -> list[dict[str, Any]]:
        """Build chat messages in the expected format for this model."""
        image_content: dict[str, Any] = {"type": self._config.image_content_type}
        if self._config.image_content_key is not None:
            image_content[self._config.image_content_key] = image
        elif self._config.image_content_type == "image":
            # Some models want {"type": "image"} with no value
            # Others want {"type": "image", "image": <PIL>}
            pass

        messages: list[dict[str, Any]] = []

        if self._config.system_message is not None:
            messages.append({"role": "system", "content": self._config.system_message})

        messages.append(
            {
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text", "text": prompt},
                ],
            }
        )

        return messages

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        params: GenerationParams,
    ) -> str:
        """Run inference using the configured chat template style."""
        messages = self._build_messages(image, prompt)

        if self._config.message_style == "two_step":
            return self._generate_two_step(messages, image, params)
        elif self._config.message_style == "one_step":
            return self._generate_one_step(messages, image, params)
        else:
            raise ValueError(f"Unknown message_style: {self._config.message_style}")

    def _generate_two_step(
        self,
        messages: list[dict],
        image: Image.Image,
        params: GenerationParams,
    ) -> str:
        """Two-step: apply_chat_template(tokenize=False) then processor()."""
        # Determine which object has apply_chat_template
        template_obj = self.processor
        if hasattr(self.processor, self._config.tokenizer_attr):
            tokenizer = getattr(self.processor, self._config.tokenizer_attr)
            if hasattr(tokenizer, "apply_chat_template") and not hasattr(
                self.processor, "apply_chat_template"
            ):
                template_obj = tokenizer

        text = template_obj.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self._config.chat_template_kwargs,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        ).to(self.model.device)

        gen_kwargs = self._build_generate_kwargs(params, inputs)

        output_ids = self.model.generate(**inputs, **gen_kwargs)

        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        del inputs, output_ids, generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return response.strip()

    def _generate_one_step(
        self,
        messages: list[dict],
        image: Image.Image,
        params: GenerationParams,
    ) -> str:
        """One-step: apply_chat_template(tokenize=True, return_dict=True)."""
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **self._config.chat_template_kwargs,
        )
        inputs = inputs.to(self.model.device)

        gen_kwargs = self._build_generate_kwargs(params, inputs)

        output_ids = self.model.generate(**inputs, **gen_kwargs)

        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        del inputs, output_ids, generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return response.strip()

    def _build_generate_kwargs(
        self, params: GenerationParams, inputs: Any
    ) -> dict[str, Any]:
        """Build model.generate() kwargs from GenerationParams."""
        kwargs: dict[str, Any] = {
            "max_new_tokens": params.max_tokens,
            "do_sample": params.do_sample,
        }

        # Only pass temperature/top_p when sampling
        if params.do_sample:
            if params.temperature is not None:
                kwargs["temperature"] = params.temperature
            if params.top_p is not None:
                kwargs["top_p"] = params.top_p
        else:
            # Suppress warnings by setting to None
            for key in self._config.suppress_gen_warnings:
                kwargs[key] = None

        # Merge model-specific generate kwargs
        kwargs.update(self._config.generate_kwargs)

        return kwargs


# Verify protocol compliance at import time
_dummy_check: type[ModelBackend] = HFChatTemplateBackend  # type: ignore[assignment]
