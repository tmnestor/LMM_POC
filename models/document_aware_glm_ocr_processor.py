"""GLM-OCR document extraction processor.

Inherits shared detection, classification, prompt resolution, and extraction
orchestration from BaseDocumentProcessor.  Only model-specific inference
(generate, token calculation, single-image processing) is implemented here.

Uses AutoModelForImageTextToText + AutoProcessor with
processor.apply_chat_template() + model.generate() API.

GLM-OCR is a lightweight 0.9B-parameter OCR model optimised for document
understanding.  It does NOT require trust_remote_code.
"""

import gc
import time
from pathlib import Path
from typing import Any, override

import torch
from PIL import Image

from common.extraction_parser import parse_extraction_response
from common.gpu_optimization import (
    configure_cuda_memory_allocation,
    handle_memory_fragmentation,
)
from common.model_config import GLM_OCR_GENERATION_CONFIG
from models.base_processor import BaseDocumentProcessor


class DocumentAwareGLMOCRProcessor(BaseDocumentProcessor):
    """Document extraction processor for GLM-OCR (0.9B).

    Satisfies the DocumentProcessor Protocol.  Inherits from
    BaseDocumentProcessor for shared logic (detection, classification,
    prompt resolution, extraction orchestration).
    """

    def __init__(
        self,
        field_list: list[str],
        model_path: str,
        device: str = "cuda",
        debug: bool = False,
        batch_size: int | None = None,
        pre_loaded_model=None,
        pre_loaded_processor=None,
        prompt_config: dict[str, Any] | None = None,
        field_definitions: dict[str, list[str]] | None = None,
    ):
        self.model_path = model_path
        self.model = pre_loaded_model
        self.processor = pre_loaded_processor

        configure_cuda_memory_allocation()

        if self.model is None:
            self._load_model()

        # Shared init: validates config, loads field defs, sets batch size
        self._init_shared(
            field_list=field_list,
            prompt_config=prompt_config,
            field_definitions=field_definitions,
            debug=debug,
            device=device,
            batch_size=batch_size,
            model_type_key="glm_ocr",
        )

        self._configure_generation()

        if self.debug:
            print(f"GLM-OCR processor initialized for {self.field_count} fields")

    # -- Protocol compatibility ------------------------------------------------

    @property
    def tokenizer(self):
        """Return tokenizer for Protocol / BankStatementAdapter compatibility.

        GLM-OCR uses AutoProcessor which wraps a tokenizer.
        """
        if self.processor is not None:
            return self.processor.tokenizer
        return None

    # -- Model loading ---------------------------------------------------------

    def _load_model(self) -> None:
        """Load GLM-OCR model and processor from disk."""
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if self.debug:
            print(f"Loading GLM-OCR from {self.model_path}")

        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
        )

        if self.debug:
            print(f"Device: {self.model.device}")
            param_count = sum(p.numel() for p in self.model.parameters())
            print(f"Model parameters: {param_count:,}")

    def _configure_generation(self) -> None:
        """Load generation hyper-parameters from model_config."""
        self.gen_config = dict(GLM_OCR_GENERATION_CONFIG)

        self.fallback_max_tokens = max(
            self.gen_config["max_new_tokens_base"],
            self.field_count * self.gen_config["max_new_tokens_per_field"],
        )

        if self.debug:
            print(
                f"Generation config: max_new_tokens={self.fallback_max_tokens}, "
                f"do_sample={self.gen_config['do_sample']}"
            )

    # -- Abstract method implementations ---------------------------------------

    @override
    def generate(self, image: Image.Image, prompt: str, max_tokens: int = 1024) -> str:
        """Run GLM-OCR inference on a single image + prompt.

        Uses processor.apply_chat_template(tokenize=True) which handles both
        text formatting and image encoding in a single pass.  GLM-OCR expects
        image content as a URL/path-style reference in the message; the
        processor resolves PIL images automatically when passed via the
        ``images`` kwarg of __call__.
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

        # GLM-OCR's apply_chat_template can tokenize in one step
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Remove token_type_ids if present (GLM-OCR doesn't use them)
        inputs.pop("token_type_ids", None)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=self.gen_config.get("do_sample", False),
        )

        # Trim input tokens from output
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        response = self.processor.decode(generated_ids[0], skip_special_tokens=False)

        del inputs, output_ids, generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return response.strip()

    @override
    def _calculate_max_tokens(self, field_count: int, document_type: str) -> int:
        """Calculate token budget based on field count and document type."""
        base = self.gen_config.get("max_new_tokens_base", 512)
        per_field = self.gen_config.get("max_new_tokens_per_field", 64)
        tokens = base + (field_count * per_field)

        # Bank statements need more tokens for many transactions
        if document_type == "bank_statement":
            tokens = max(tokens, 1500)
        return tokens

    # -- Single image processing -----------------------------------------------

    def process_single_image(
        self,
        image_path: str,
        custom_prompt: str | None = None,
        custom_max_tokens: int | None = None,
        field_list: list[str] | None = None,
    ) -> dict:
        """Process one document image end-to-end.

        Called by the inherited process_document_aware() method.
        """
        active_fields = field_list or self.field_list
        active_count = len(active_fields)
        start_time = time.time()
        image_name = Path(image_path).name

        try:
            handle_memory_fragmentation(threshold_gb=1.0, aggressive=True)

            image = self.load_document_image(image_path)

            prompt = custom_prompt or self.get_extraction_prompt()
            max_tokens = custom_max_tokens or self._calculate_max_tokens(
                active_count, "universal"
            )

            if self.debug:
                import sys

                sys.stdout.write(f"Processing {image_name} ({active_count} fields)\n")
                sys.stdout.write(
                    f"Prompt: {len(prompt)} chars, max_tokens: {max_tokens}\n"
                )
                sys.stdout.flush()

            raw_response = self._resilient_generate(image, prompt, max_tokens)
            processing_time = time.time() - start_time

            if self.debug:
                import sys

                sys.stdout.write(f"Response ({len(raw_response)} chars):\n")
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.write(raw_response + "\n")
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.flush()

            # Parse structured fields from response
            extracted_data = parse_extraction_response(
                raw_response, expected_fields=active_fields
            )

            # Clean values
            for field_name, value in extracted_data.items():
                extracted_data[field_name] = self.cleaner.clean_field_value(
                    field_name, value
                )

            found = sum(1 for v in extracted_data.values() if v != "NOT_FOUND")

            if self.debug:
                print(f"Extracted {found}/{active_count} fields")

            # Cleanup
            del image
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {
                "image_name": image_name,
                "extracted_data": extracted_data,
                "raw_response": raw_response,
                "processing_time": processing_time,
                "response_completeness": found / max(active_count, 1),
                "content_coverage": found / max(active_count, 1),
                "extracted_fields_count": found,
                "field_count": active_count,
            }

        except Exception as e:
            processing_time = time.time() - start_time
            if self.debug:
                import traceback

                print(f"Error processing {image_name}: {e}")
                traceback.print_exc()
            return {
                "image_name": image_name,
                "extracted_data": {f: "NOT_FOUND" for f in active_fields},
                "raw_response": f"Error: {e}",
                "processing_time": processing_time,
                "response_completeness": 0.0,
                "content_coverage": 0.0,
                "extracted_fields_count": 0,
                "field_count": active_count,
            }

    def _resilient_generate(
        self, image: Image.Image, prompt: str, max_tokens: int
    ) -> str:
        """Generate with OOM recovery (halve tokens and retry)."""
        oom = False
        try:
            return self.generate(image, prompt, max_tokens)
        except torch.cuda.OutOfMemoryError:
            oom = True

        # Cleanup OUTSIDE except block (see MEMORY.md)
        if oom:
            gc.collect()
            torch.cuda.empty_cache()
            if self.debug:
                print(f"OOM at {max_tokens} tokens, retrying at {max_tokens // 2}")
            return self.generate(image, prompt, max_tokens // 2)

    def get_model_info(self) -> dict:
        """Return model metadata for reporting."""
        return {
            "model_type": "glm_ocr",
            "model_path": self.model_path,
            "batch_size": self.batch_size,
        }
