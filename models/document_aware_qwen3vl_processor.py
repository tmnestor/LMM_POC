"""Qwen3-VL-8B document extraction processor.

Inherits shared orchestration from SimpleDocumentProcessor.
Only model loading and inference are implemented here.

Uses Qwen3VLForConditionalGeneration + AutoProcessor with
processor.apply_chat_template() + model.generate() API.
"""

from typing import override

import torch
from PIL import Image

from models.simple_processor import SimpleDocumentProcessor


class DocumentAwareQwen3VLProcessor(SimpleDocumentProcessor):
    """Document extraction processor for Qwen3-VL-8B-Instruct."""

    model_type_key = "qwen3vl"

    @override
    def _load_model(self) -> None:
        """Load Qwen3-VL model and processor from disk."""
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if self.debug:
            print(f"Loading Qwen3-VL from {self.model_path}")

        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
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
        """Run Qwen3-VL inference on a single image + prompt.

        Two-step approach: apply_chat_template for text formatting,
        then processor() for combined text+image tokenization.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        ).to(self.model.device)

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
