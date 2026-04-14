"""Llama 4 Scout document extraction processor.

Inherits shared orchestration from SimpleDocumentProcessor.
Only model loading and inference are implemented here.

Uses Llama4ForConditionalGeneration + AutoProcessor with
processor.apply_chat_template() all-in-one API.

Key differences from Llama 3.2 (MllamaForConditionalGeneration):
- Uses Llama4ForConditionalGeneration (MoE architecture, 109B total / 17B active)
- apply_chat_template returns ready-to-use inputs (no separate processor() call)
- Falls back to BitsAndBytes NF4 if fbgemm-gpu is unavailable
"""

from typing import Any, override

import torch
from PIL import Image

from models.simple_processor import SimpleDocumentProcessor


class DocumentAwareLlama4Processor(SimpleDocumentProcessor):
    """Document extraction processor for Llama 4 Scout 17B-16E."""

    model_type_key = "llama4scout"

    @override
    def _load_model(self) -> None:
        """Load Llama 4 Scout model and processor from disk."""
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Llama4ForConditionalGeneration,
        )

        if self.debug:
            print(f"Loading Llama 4 Scout from {self.model_path}")

        self.processor = AutoProcessor.from_pretrained(self.model_path)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        load_kwargs: dict[str, Any] = {
            "dtype": torch.bfloat16,
            "device_map": "auto",
            "attn_implementation": "sdpa",
            "quantization_config": quantization_config,
        }

        if self.debug:
            print("Using NF4 quantization (~55 GB for 109B MoE)")

        self.model = Llama4ForConditionalGeneration.from_pretrained(
            self.model_path,
            **load_kwargs,
        )

        if hasattr(self.model, "generation_config"):
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None

        if self.debug:
            param_count = sum(p.numel() for p in self.model.parameters())
            print(f"Model parameters: {param_count:,}")

    @override
    def generate(self, image: Image.Image, prompt: str, max_tokens: int = 1024) -> str:
        """Run Llama 4 Scout inference on a single image + prompt.

        Uses the all-in-one apply_chat_template API that handles both
        text formatting and image tokenization in one step.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
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
