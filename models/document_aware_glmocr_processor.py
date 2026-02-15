"""GLM-OCR document extraction processor.

Inherits shared detection from BaseDocumentProcessor.  Overrides extraction
to use GLM-OCR's native capabilities:

  - Standard documents (invoices, receipts): "OCR:" prompt → regex field extraction
  - Bank statements: "Table Recognition:" prompt → markdown table → field extraction

GLM-OCR is a 0.9B OCR-specialized model.  It excels at text/table recognition
but cannot follow complex structured extraction instructions like 8B+ VLMs.
Instead of expecting FIELD_NAME: value output, we get raw text and parse it.

Uses GlmOcrForConditionalGeneration + AutoProcessor with
processor.apply_chat_template() + model.generate() API.
"""

import gc
import time
from pathlib import Path
from typing import Any, override

import torch
from PIL import Image

from common.gpu_optimization import (
    configure_cuda_memory_allocation,
    handle_memory_fragmentation,
)
from common.model_config import GLMOCR_GENERATION_CONFIG
from common.ocr_field_extractor import (
    extract_bank_fields_from_table,
    extract_fields_from_ocr,
)
from models.base_processor import BaseDocumentProcessor


class DocumentAwareGlmOcrProcessor(BaseDocumentProcessor):
    """Document extraction processor for GLM-OCR.

    Satisfies the DocumentProcessor Protocol.  Inherits from
    BaseDocumentProcessor for shared detection/classification logic.

    Uses an OCR-first pipeline instead of structured prompt extraction:
    - "OCR:" prompt returns raw text → parsed with regex into fields
    - "Table Recognition:" prompt returns markdown tables → parsed into bank fields
    """

    # GLM-OCR cannot follow multi-turn bank prompts — skip BankStatementAdapter
    supports_multi_turn = False

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
            model_type_key="glmocr",
        )

        self._configure_generation()

        if self.debug:
            print(f"GLM-OCR processor initialized for {self.field_count} fields")
            print("  Pipeline: OCR-native (raw text + regex extraction)")

    # -- Protocol compatibility ------------------------------------------------

    @property
    def tokenizer(self):
        """Return tokenizer for Protocol / BankStatementAdapter compatibility."""
        if self.processor is not None:
            return self.processor.tokenizer
        return None

    # -- Model loading ---------------------------------------------------------

    def _load_model(self) -> None:
        """Load GLM-OCR model and processor from disk."""
        from transformers import AutoProcessor, GlmOcrForConditionalGeneration

        if self.debug:
            print(f"Loading GLM-OCR from {self.model_path}")

        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = GlmOcrForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        if self.debug:
            print(f"Device: {self.model.device}")
            param_count = sum(p.numel() for p in self.model.parameters())
            print(f"Model parameters: {param_count:,}")

    def _configure_generation(self) -> None:
        """Load generation hyper-parameters from model_config."""
        self.gen_config = dict(GLMOCR_GENERATION_CONFIG)

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
    def generate(self, image: Image.Image, prompt: str, max_tokens: int = 8192) -> str:
        """Run GLM-OCR inference on a single image + prompt.

        Uses the two-step approach: apply_chat_template for text formatting,
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

        # Step 1: Build text template (no tokenization, no image bytes)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Step 2: Tokenize text + encode image together
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        ).to(self.model.device)

        # GLM-OCR may include token_type_ids which aren't needed for generation
        inputs.pop("token_type_ids", None)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )

        # Trim input tokens from output
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        del inputs, output_ids, generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return response.strip()

    @override
    def _calculate_max_tokens(self, field_count: int, document_type: str) -> int:
        """Calculate token budget based on field count and document type."""
        base = self.gen_config.get("max_new_tokens_base", 2000)
        per_field = self.gen_config.get("max_new_tokens_per_field", 64)
        tokens = base + (field_count * per_field)

        # Bank statements need more tokens for many transactions
        if document_type == "bank_statement":
            tokens = max(tokens, 4000)
        return tokens

    # -- OCR-native extraction pipeline ----------------------------------------

    @override
    def process_document_aware(
        self, image_path: str, classification_info: dict, verbose: bool = False
    ) -> dict:
        """Process document using OCR-native extraction.

        Overrides BaseDocumentProcessor to use GLM-OCR's native capabilities
        instead of structured extraction prompts:
        - Bank statements: "Table Recognition:" → markdown table → fields
        - Other documents: "OCR:" → raw text → regex field extraction
        """
        try:
            document_type = classification_info["document_type"].upper()
            doc_type_lower = document_type.lower()

            if verbose:
                print(f"📊 GLM-OCR processing {document_type} (OCR-native pipeline)")

            # Resolve document-specific field list
            doc_fields = dict(self.document_field_lists)
            active_fields = doc_fields.get(doc_type_lower, self.field_list)

            if document_type == "BANK_STATEMENT":
                return self._process_bank_table(image_path, active_fields, verbose)
            else:
                return self._process_with_ocr(
                    image_path, document_type, active_fields, verbose
                )

        except Exception as e:
            if verbose:
                print(f"❌ Error in GLM-OCR document-aware processing: {e}")
            return self.process_single_image(image_path)

    def _process_bank_table(
        self,
        image_path: str,
        active_fields: list[str],
        verbose: bool,
    ) -> dict:
        """Extract bank statement fields using Table Recognition mode."""
        start_time = time.time()
        image_name = Path(image_path).name

        try:
            handle_memory_fragmentation(threshold_gb=1.0, aggressive=True)
            image = self.load_document_image(image_path)

            prompt = "Table Recognition:"
            max_tokens = self._calculate_max_tokens(
                len(active_fields), "bank_statement"
            )

            if verbose:
                import sys

                sys.stdout.write(
                    f"  Table Recognition: {image_name} ({len(active_fields)} fields)\n"
                )
                sys.stdout.flush()

            raw_response = self._resilient_generate(image, prompt, max_tokens)
            processing_time = time.time() - start_time

            if verbose:
                import sys

                sys.stdout.write(f"  Table response ({len(raw_response)} chars):\n")
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.write(raw_response[:500] + "\n")
                if len(raw_response) > 500:
                    sys.stdout.write(f"  ... ({len(raw_response) - 500} more chars)\n")
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.flush()

            # Parse markdown table into bank statement fields
            extracted_data = extract_bank_fields_from_table(raw_response, active_fields)

            # Clean values
            for field_name, value in extracted_data.items():
                extracted_data[field_name] = self.cleaner.clean_field_value(
                    field_name, value
                )

            found = sum(1 for v in extracted_data.values() if v != "NOT_FOUND")

            if verbose:
                print(f"  Extracted {found}/{len(active_fields)} bank fields")

            del image
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {
                "image_name": image_name,
                "extracted_data": extracted_data,
                "raw_response": raw_response,
                "processing_time": processing_time,
                "response_completeness": found / max(len(active_fields), 1),
                "content_coverage": found / max(len(active_fields), 1),
                "extracted_fields_count": found,
                "field_count": len(active_fields),
                "skip_math_enhancement": True,
            }

        except Exception as e:
            processing_time = time.time() - start_time
            if verbose:
                import traceback

                print(f"  Error in bank table extraction: {e}")
                traceback.print_exc()
            return {
                "image_name": image_name,
                "extracted_data": {f: "NOT_FOUND" for f in active_fields},
                "raw_response": f"Error: {e}",
                "processing_time": processing_time,
                "response_completeness": 0.0,
                "content_coverage": 0.0,
                "extracted_fields_count": 0,
                "field_count": len(active_fields),
            }

    def _process_with_ocr(
        self,
        image_path: str,
        document_type: str,
        active_fields: list[str],
        verbose: bool,
    ) -> dict:
        """Extract fields from standard documents using OCR mode."""
        start_time = time.time()
        image_name = Path(image_path).name

        try:
            handle_memory_fragmentation(threshold_gb=1.0, aggressive=True)
            image = self.load_document_image(image_path)

            prompt = "OCR:"
            max_tokens = self._calculate_max_tokens(len(active_fields), "universal")

            if verbose:
                import sys

                sys.stdout.write(
                    f"  OCR extraction: {image_name} "
                    f"({document_type}, {len(active_fields)} fields)\n"
                )
                sys.stdout.flush()

            raw_response = self._resilient_generate(image, prompt, max_tokens)
            processing_time = time.time() - start_time

            if verbose:
                import sys

                sys.stdout.write(f"  OCR response ({len(raw_response)} chars):\n")
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.write(raw_response[:500] + "\n")
                if len(raw_response) > 500:
                    sys.stdout.write(f"  ... ({len(raw_response) - 500} more chars)\n")
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.flush()

            # Parse raw OCR text into structured fields
            extracted_data = extract_fields_from_ocr(
                raw_response, document_type, active_fields
            )

            # Clean values
            for field_name, value in extracted_data.items():
                extracted_data[field_name] = self.cleaner.clean_field_value(
                    field_name, value
                )

            found = sum(1 for v in extracted_data.values() if v != "NOT_FOUND")

            if verbose:
                print(
                    f"  Extracted {found}/{len(active_fields)} fields via OCR parsing"
                )

            del image
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {
                "image_name": image_name,
                "extracted_data": extracted_data,
                "raw_response": raw_response,
                "processing_time": processing_time,
                "response_completeness": found / max(len(active_fields), 1),
                "content_coverage": found / max(len(active_fields), 1),
                "extracted_fields_count": found,
                "field_count": len(active_fields),
            }

        except Exception as e:
            processing_time = time.time() - start_time
            if verbose:
                import traceback

                print(f"  Error in OCR extraction: {e}")
                traceback.print_exc()
            return {
                "image_name": image_name,
                "extracted_data": {f: "NOT_FOUND" for f in active_fields},
                "raw_response": f"Error: {e}",
                "processing_time": processing_time,
                "response_completeness": 0.0,
                "content_coverage": 0.0,
                "extracted_fields_count": 0,
                "field_count": len(active_fields),
            }

    # -- Single image processing (fallback) ------------------------------------

    def process_single_image(
        self,
        image_path: str,
        custom_prompt: str | None = None,
        custom_max_tokens: int | None = None,
        field_list: list[str] | None = None,
    ) -> dict:
        """Process one document image end-to-end using OCR mode.

        Called as fallback from process_document_aware() error handling,
        and for any direct single-image invocation.  Always uses "OCR:"
        prompt regardless of custom_prompt (GLM-OCR can't follow structured
        extraction prompts).
        """
        active_fields = field_list or self.field_list
        active_count = len(active_fields)
        start_time = time.time()
        image_name = Path(image_path).name

        try:
            handle_memory_fragmentation(threshold_gb=1.0, aggressive=True)
            image = self.load_document_image(image_path)

            # Always use OCR mode — GLM-OCR can't follow structured prompts
            prompt = "OCR:"
            max_tokens = custom_max_tokens or self._calculate_max_tokens(
                active_count, "universal"
            )

            if self.debug:
                import sys

                sys.stdout.write(f"Processing {image_name} ({active_count} fields)\n")
                sys.stdout.write(f"Prompt: OCR: (max_tokens: {max_tokens})\n")
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

            # Parse OCR text into fields
            extracted_data = extract_fields_from_ocr(
                raw_response, "UNIVERSAL", active_fields
            )

            # Clean values
            for field_name, value in extracted_data.items():
                extracted_data[field_name] = self.cleaner.clean_field_value(
                    field_name, value
                )

            found = sum(1 for v in extracted_data.values() if v != "NOT_FOUND")

            if self.debug:
                print(f"Extracted {found}/{active_count} fields")

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
            "model_type": "glmocr",
            "model_path": self.model_path,
            "batch_size": self.batch_size,
            "pipeline": "ocr_native",
        }
