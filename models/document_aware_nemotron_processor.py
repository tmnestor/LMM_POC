"""Nemotron Nano 2 VL document extraction processor.

Inherits shared orchestration from SimpleDocumentProcessor.
Only model loading and inference are implemented here.

Uses AutoModelForCausalLM + AutoProcessor with trust_remote_code=True.
Two-step API: tokenizer.apply_chat_template() for chat formatting,
processor() for image tokenization.  /no_think system message disables
chain-of-thought reasoning.
"""

from typing import override

import torch
from PIL import Image

from models.simple_processor import SimpleDocumentProcessor


class DocumentAwareNemotronProcessor(SimpleDocumentProcessor):
    """Document extraction processor for NVIDIA Nemotron Nano 2 VL."""

    model_type_key = "nemotron"

    @override
    def _load_model(self) -> None:
        """Load Nemotron Nano 2 VL model and processor from disk."""
        from transformers import AutoModelForCausalLM, AutoProcessor

        if self.debug:
            print(f"Loading Nemotron Nano 2 VL from {self.model_path}")

        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()

        if hasattr(self.model, "generation_config"):
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None

        if self.debug:
            print(f"Device: {self.model.device}")
            param_count = sum(p.numel() for p in self.model.parameters())
            print(f"Model parameters: {param_count:,}")

    @override
    def generate(self, image: Image.Image, prompt: str, max_tokens: int = 1024) -> str:
        """Run Nemotron inference on a single image + prompt.

        Two-step API: tokenizer.apply_chat_template() for chat formatting,
        processor() for image tokenization.  /no_think disables CoT.
        """
        tokenizer = self.processor.tokenizer

        messages = [
            {"role": "system", "content": "/no_think"},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": ""},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(
            self.model.device
        )

        output_ids = self.model.generate(
            pixel_values=inputs.pixel_values,
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

        generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        del inputs, output_ids, generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return response.strip()
