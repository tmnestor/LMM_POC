"""Qwen3.5-27B document extraction processor.

Inherits shared orchestration from SimpleDocumentProcessor.
Only model loading and inference are implemented here.

Uses AutoModelForCausalLM + AutoProcessor (early-fusion VLM architecture).
One-step API: processor.apply_chat_template(tokenize=True) handles both
chat formatting and image tokenization.  Thinking mode disabled via
enable_thinking=False.
"""

from typing import override

import torch
from PIL import Image

from models.simple_processor import SimpleDocumentProcessor


class DocumentAwareQwen35Processor(SimpleDocumentProcessor):
    """Document extraction processor for Qwen3.5-27B."""

    model_type_key = "qwen35"

    @override
    def _load_model(self) -> None:
        """Load Qwen3.5-27B model and processor from disk."""
        from transformers import (  # type: ignore[attr-defined]
            AutoProcessor,
            Qwen3_5ForConditionalGeneration,
        )

        if self.debug:
            print(f"Loading Qwen3.5-27B from {self.model_path}")

        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            device_map="auto",
        )

        if hasattr(self.model, "generation_config"):
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None
            self.model.generation_config.top_k = None

        if self.debug:
            print(f"Device: {self.model.device}")
            param_count = sum(p.numel() for p in self.model.parameters())
            print(f"Model parameters: {param_count:,}")

    @override
    def generate(self, image: Image.Image, prompt: str, max_tokens: int = 1024) -> str:
        """Run Qwen3.5-27B inference on a single image + prompt.

        One-step API: apply_chat_template(tokenize=True) handles both chat
        formatting and image tokenization.  Thinking mode disabled.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        inputs = inputs.to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        del inputs, output_ids, generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return response.strip()
