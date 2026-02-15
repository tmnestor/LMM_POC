#!/usr/bin/env python3
"""
Document-Aware Llama Processor - inherits BaseDocumentProcessor.

Llama-specific implementation for document-aware extraction with dynamic
field lists.  Shared logic (detection, classification, prompt resolution,
extraction orchestration) lives in BaseDocumentProcessor.

DOCUMENT AWARE REDUCTION OPTIMIZED:
- Invoice/Receipt: 11 fields (62-67% reduction) -> ~50% faster processing
- Bank Statement: 5 fields (75% reduction) -> ~75% faster processing
- Dynamic max_new_tokens automatically scales with reduced field counts
"""

import gc
import time
import warnings
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    MllamaForConditionalGeneration,
)

from common.gpu_optimization import (
    comprehensive_memory_cleanup,
    configure_cuda_memory_allocation,
    handle_memory_fragmentation,
    optimize_model_for_gpu,
)
from common.model_config import (
    LLAMA_GENERATION_CONFIG,
)
from models.base_processor import BaseDocumentProcessor

warnings.filterwarnings("ignore")

LLAMA_MODEL_PATH = "/efs/shared/PTM/Llama-3.2-11B-Vision-Instruct"


class DocumentAwareLlamaProcessor(BaseDocumentProcessor):
    """Document-aware Llama processor with dynamic field support."""

    def __init__(
        self,
        field_list: list[str],
        model_path: str | None = None,
        device: str = "cuda",
        debug: bool = False,
        batch_size: int | None = None,
        skip_model_loading: bool = False,
        pre_loaded_model=None,
        pre_loaded_processor=None,
        prompt_config: dict | None = None,
        field_definitions: dict | None = None,
    ):
        """
        Initialize document-aware processor with specific field list.

        Args:
            field_list (list[str]): Fields to extract
            model_path (str): Path to Llama model
            device (str): Device to run on
            debug (bool): Enable debug output
            batch_size (int): Batch size (auto-detected if None)
            skip_model_loading (bool): Skip loading model (for reusing existing model)
            pre_loaded_model: Pre-loaded model instance
            pre_loaded_processor: Pre-loaded AutoProcessor instance
            prompt_config (dict): Prompt routing config (detection_file, extraction_files)
            field_definitions (dict): Pre-loaded field definitions dict
        """
        self.model_path = model_path or LLAMA_MODEL_PATH

        # Initialize components
        self.model = pre_loaded_model
        self.processor = pre_loaded_processor
        self.generation_config = None

        # Configure CUDA memory allocation
        configure_cuda_memory_allocation()

        # Load model and processor if not pre-loaded
        if self.model is None and not skip_model_loading:
            self._load_model()

        # Shared init: field_list, prompt_config, cleaner, field_definitions, batch
        self._init_shared(
            field_list=field_list,
            prompt_config=prompt_config,
            field_definitions=field_definitions,
            debug=debug,
            device=device,
            batch_size=batch_size,
            model_type_key="llama",
        )

        if self.debug:
            print(
                f"Document-aware Llama processor initialized for {self.field_count} fields"
            )

        # Configure generation parameters for dynamic field count
        self._configure_generation()

    # -- abstract method implementations ---------------------------------------

    def generate(self, image: Image.Image, prompt: str, max_tokens: int = 1024) -> str:
        """Run model inference on an image with a text prompt."""
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
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

        generate_ids = output[:, inputs["input_ids"].shape[1] : -1]
        response = self.processor.decode(
            generate_ids[0], clean_up_tokenization_spaces=False
        )

        del inputs, output, generate_ids
        torch.cuda.empty_cache()
        return response

    def _calculate_max_tokens(self, field_count: int, document_type: str) -> int:
        """Calculate max_new_tokens for generation based on field count."""
        base = LLAMA_GENERATION_CONFIG.get("max_new_tokens_base", 512)
        per_field = LLAMA_GENERATION_CONFIG.get("max_new_tokens_per_field", 64)
        tokens = base + (field_count * per_field)
        # Bank statements need more tokens for many transactions
        if document_type == "bank_statement":
            tokens = max(tokens, 1500)
        return tokens

    # -- Llama-specific methods ------------------------------------------------

    def _configure_generation(self):
        """
        Configure generation parameters for dynamic field count.

        DOCUMENT AWARE REDUCTION: Automatically calculates optimal max_new_tokens
        for reduced field counts, providing significant performance gains.
        """
        # Initialize generation config
        self.generation_config = LLAMA_GENERATION_CONFIG.copy()

        # Use configured max_new_tokens from model if available, otherwise calculate
        self.configured_max_tokens = None  # Will be set when model is assigned

        # Calculate dynamic max_new_tokens based on actual field count
        config = LLAMA_GENERATION_CONFIG
        self.fallback_max_tokens = max(
            config["max_new_tokens_base"],
            self.field_count * config["max_new_tokens_per_field"],
        )
        self.generation_config["max_new_tokens"] = self.fallback_max_tokens

        if self.debug:
            print(
                f"🎯 Generation config: max_new_tokens={self.generation_config['max_new_tokens']}, "
                f"temperature={self.generation_config['temperature']}, "
                f"do_sample={self.generation_config['do_sample']}"
            )

    def _load_model(self):
        """Load Llama Vision model and processor with optimal configuration."""
        if self.debug:
            print(f"🔄 Loading Llama Vision model from: {self.model_path}")

        try:
            # Configure 8-bit quantization for V100 compatibility
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
                llm_int8_threshold=6.0,
            )

            # Load model
            self.model = MllamaForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=quantization_config,
            )

            # Load processor for multimodal inputs
            self.processor = AutoProcessor.from_pretrained(self.model_path)

            # Call tie_weights() after loading
            try:
                self.model.tie_weights()
                if self.debug:
                    print(
                        "✅ Llama Vision model loaded successfully (tie_weights called)"
                    )
            except Exception as e:
                if self.debug:
                    print(
                        f"⚠️ Llama Vision model loaded (tie_weights warning ignored): {e}"
                    )

            if self.debug:
                print(f"🔧 Device: {self.model.device}")
                print(
                    f"💾 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
                )

            # Apply V100 optimizations
            optimize_model_for_gpu(self.model)

        except Exception as e:
            if self.debug:
                print(f"❌ Error loading Llama model: {e}")
            raise

    @property
    def tokenizer(self):
        """Return the tokenizer for Protocol/BankStatementAdapter compatibility.

        Llama uses AutoProcessor which wraps a tokenizer, unlike InternVL3
        which has a separate AutoTokenizer.
        """
        if self.processor is not None:
            return self.processor.tokenizer
        return None

    def detect_document_type(self, field_list: list[str] | None = None) -> str:
        """
        Detect document type based on field composition.
        Simple heuristic for backward compatibility.

        Args:
            field_list: Optional field list to analyze (uses self.field_list if None)

        Returns:
            Detected document type (invoice, receipt, bank_statement)
        """
        fields_to_check = field_list or self.field_list
        field_set = set(fields_to_check)

        # Simple heuristic based on unique fields
        if "STATEMENT_DATE_RANGE" in field_set or "TRANSACTION_DATES" in field_set:
            return "bank_statement"
        elif "INVOICE_DATE" in field_set and "GST_AMOUNT" in field_set:
            # Both invoice and receipt have these, default to invoice
            return "invoice"
        else:
            # Fallback to invoice for unknown patterns
            return "invoice"

    def _resilient_generate(self, inputs, **generation_kwargs):
        """Resilient generation with OOM fallback."""

        # Build clean generation parameters
        do_sample = generation_kwargs.get(
            "do_sample", self.generation_config["do_sample"]
        )
        clean_generation_kwargs = {
            "max_new_tokens": generation_kwargs.get(
                "max_new_tokens", self.generation_config["max_new_tokens"]
            ),
            "do_sample": do_sample,
            "use_cache": generation_kwargs.get(
                "use_cache", self.generation_config["use_cache"]
            ),
            "pad_token_id": self.processor.tokenizer.eos_token_id,
        }
        # Only include temperature/top_p when sampling is enabled
        if do_sample:
            clean_generation_kwargs["temperature"] = generation_kwargs.get(
                "temperature", self.generation_config["temperature"]
            )
            clean_generation_kwargs["top_p"] = generation_kwargs.get(
                "top_p", self.generation_config["top_p"]
            )

        # First attempt
        oom_first = False
        try:
            return self.model.generate(**inputs, **clean_generation_kwargs)
        except torch.cuda.OutOfMemoryError:
            oom_first = True
            if self.debug:
                print("⚠️ CUDA OOM during generation, attempting recovery...")

        # Cleanup OUTSIDE except block so traceback refs are released
        if oom_first:
            gc.collect()
            torch.cuda.empty_cache()
            handle_memory_fragmentation(threshold_gb=1.0, aggressive=True)

        # Second attempt after cleanup
        oom_second = False
        try:
            return self.model.generate(**inputs, **clean_generation_kwargs)
        except torch.cuda.OutOfMemoryError:
            oom_second = True
            if self.debug:
                print("❌ Still OOM after cleanup, falling back to CPU")

        # CPU fallback — cleanup OUTSIDE except block
        if oom_second:
            gc.collect()
            torch.cuda.empty_cache()

            inputs_cpu = {
                k: v.cpu() if hasattr(v, "cpu") else v for k, v in inputs.items()
            }
            self.model = self.model.cpu()

            output = self.model.generate(**inputs_cpu, **clean_generation_kwargs)

            # Move back to GPU
            self.model = self.model.to(self.device)
            return output

    def process_single_image(
        self,
        image_path: str,
        custom_prompt: str | None = None,
        custom_max_tokens: int | None = None,
        field_list: list[str] | None = None,
    ) -> dict:
        """Process single image with document-aware extraction.

        Args:
            image_path: Path to document image.
            custom_prompt: Optional prompt override.
            custom_max_tokens: Optional max token override.
            field_list: Optional field list override (uses self.field_list if None).
        """
        # Resolve effective field list for this invocation
        active_field_list = field_list if field_list is not None else self.field_list
        active_field_count = len(active_field_list)

        try:
            start_time = time.time()

            # Memory cleanup
            handle_memory_fragmentation(threshold_gb=1.0, aggressive=True)

            # Load image
            image = self.load_document_image(image_path)

            # Use custom prompt if provided, otherwise generate from schema
            if custom_prompt:
                prompt = custom_prompt
                document_type = "CUSTOM"  # Indicate custom prompt usage
            else:
                # Get document-aware prompt
                document_type = self.detect_document_type()

                # For bank statements, use vision-based structure classification
                if document_type == "bank_statement":
                    try:
                        # Try different import methods to handle various execution contexts
                        try:
                            from common.vision_bank_statement_classifier import (
                                classify_bank_statement_structure_vision,
                            )
                        except ImportError:
                            import sys
                            from pathlib import Path

                            sys.path.insert(
                                0, str(Path(__file__).resolve().parent.parent)
                            )
                            from common.vision_bank_statement_classifier import (
                                classify_bank_statement_structure_vision,
                            )

                        if self.debug:
                            print(
                                "🔍 Running vision-based structure classification for bank statement"
                            )

                        # Use the model and processor for classification
                        structure_type = classify_bank_statement_structure_vision(
                            image_path,
                            model=self.model,
                            processor=self.processor,
                            verbose=self.debug,
                        )

                        # Map structure to specific prompt
                        document_type = f"bank_statement_{structure_type}"

                        if self.debug:
                            print(f"🏗️ Bank statement structure: {structure_type}")
                            print(f"📝 Using prompt key: {document_type}")

                    except Exception as e:
                        if self.debug:
                            print(f"⚠️ Vision classification failed: {e}")
                            print("📝 Falling back to universal prompt")
                        document_type = "universal"

                prompt = self.get_extraction_prompt(document_type=document_type)

            if self.debug:
                # Use direct stdout to bypass Rich console completely
                import sys

                if custom_prompt:
                    sys.stdout.write("📝 Using custom YAML prompt\n")
                    sys.stdout.write(f"🔍 CUSTOM YAML PROMPT ({len(prompt)} chars):\n")
                else:
                    sys.stdout.write(
                        f"📝 Generated prompt for {active_field_count} fields\n"
                    )
                    sys.stdout.write(
                        f"   Fields: {active_field_list[:3]}{'...' if len(active_field_list) > 3 else ''}\n"
                    )
                    sys.stdout.write(
                        f"🔍 DOCUMENT-AWARE PROMPT ({len(prompt)} chars):\n"
                    )
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.write(prompt + "\n")
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.flush()

            # Create multimodal conversation
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Apply chat template
            input_text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )

            # Process inputs
            inputs = self.processor(image, input_text, return_tensors="pt").to(
                self.device
            )

            # Use custom max_tokens if provided (for YAML prompts)
            generation_config = self.generation_config.copy()
            if custom_max_tokens:
                generation_config["max_new_tokens"] = custom_max_tokens

            if self.debug:
                print(f"🖼️  Input tensor shape: {inputs['input_ids'].shape}")
                print(
                    f"💭 Generating with max_new_tokens={generation_config['max_new_tokens']}"
                )

            # Generate response
            output = self._resilient_generate(inputs, **generation_config)

            # Decode response
            full_response = self.processor.decode(output[0], skip_special_tokens=True)

            # Extract assistant's response - handle multiple chat template formats
            if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
                # Llama 3.2 official format
                response = full_response.split(
                    "<|start_header_id|>assistant<|end_header_id|>"
                )[1]
                response = response.split("<|eot_id|>")[0].strip()
            elif "\nassistant\n\n" in full_response:
                # Common format: user\n[prompt]\nassistant\n\n[response]
                response = full_response.split("\nassistant\n\n")[1].strip()
            elif "\nassistant\n" in full_response:
                # Variant format: user\n[prompt]\nassistant\n[response]
                response = full_response.split("\nassistant\n")[1].strip()
            elif "assistant" in full_response:
                # Fallback: split on assistant keyword and take the last part
                parts = full_response.split("assistant")
                response = parts[-1].strip()
            else:
                # No chat template detected, use full response
                response = full_response.strip()

            if self.debug:
                # Use direct stdout to bypass Rich console completely
                import sys

                sys.stdout.write(f"📄 RAW MODEL RESPONSE ({len(response)} chars):\n")
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.write(response + "\n")
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.flush()

            # Parse response using robust extraction parser with field filtering
            from common.extraction_parser import parse_extraction_response

            extracted_data = parse_extraction_response(
                response, expected_fields=active_field_list
            )

            # Apply ExtractionCleaner for value normalization
            cleaned_data = {}
            for field in active_field_list:
                raw_value = extracted_data.get(field, "NOT_FOUND")
                if raw_value != "NOT_FOUND":
                    cleaned_value = self.cleaner.clean_field_value(field, raw_value)
                    cleaned_data[field] = cleaned_value
                else:
                    cleaned_data[field] = "NOT_FOUND"

            # Replace extracted_data with cleaned version
            extracted_data = cleaned_data

            if self.debug:
                # Use direct stdout to bypass Rich console completely
                import sys

                sys.stdout.write("📊 PARSED EXTRACTION RESULTS:\n")
                sys.stdout.write("=" * 80 + "\n")
                for field in active_field_list:
                    value = extracted_data.get(field, "NOT_FOUND")
                    status = "✅" if value != "NOT_FOUND" else "❌"
                    sys.stdout.write(f'  {status} {field}: "{value}"\n')
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.flush()

                found_fields = [
                    k for k, v in extracted_data.items() if v != "NOT_FOUND"
                ]
                print(f"✅ Extracted {len(found_fields)}/{active_field_count} fields")
                if found_fields:
                    print(
                        f"   Found: {found_fields[:3]}{'...' if len(found_fields) > 3 else ''}"
                    )

            # Calculate metrics
            extracted_fields_count = len(
                [k for k in extracted_data.keys() if k in active_field_list]
            )
            response_completeness = extracted_fields_count / active_field_count
            content_coverage = extracted_fields_count / active_field_count

            # Cleanup with V100 optimizations
            del inputs, output, image
            comprehensive_memory_cleanup(self.model, self.processor, verbose=self.debug)

            return {
                "image_name": Path(image_path).name,
                "extracted_data": extracted_data,
                "raw_response": response,
                "processing_time": time.time() - start_time,
                "response_completeness": response_completeness,
                "content_coverage": content_coverage,
                "extracted_fields_count": extracted_fields_count,
                "field_count": active_field_count,
            }

        except Exception as e:
            if self.debug:
                print(f"❌ Error processing {image_path}: {e}")
                import traceback

                traceback.print_exc()

            # Return error result with dynamic fields
            return {
                "image_name": Path(image_path).name,
                "extracted_data": {field: "NOT_FOUND" for field in active_field_list},
                "raw_response": f"Error: {str(e)}",
                "processing_time": 0,
                "response_completeness": 0,
                "content_coverage": 0,
                "extracted_fields_count": 0,
                "field_count": active_field_count,
            }

    def _extract_with_custom_prompt(
        self, image_path: str, prompt: str, **generation_kwargs
    ) -> str:
        """
        Extract fields using a custom prompt with specific generation parameters.

        Required method for DocumentTypeDetector compatibility.

        Args:
            image_path (str): Path to image file
            prompt (str): Custom extraction prompt
            **generation_kwargs: Additional generation parameters

        Returns:
            str: Raw model response
        """
        try:
            # Load image
            image = self.load_document_image(image_path)

            # Create multimodal conversation with custom prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Apply chat template
            input_text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )

            # Process inputs
            inputs = self.processor(image, input_text, return_tensors="pt").to(
                self.device
            )

            # Merge generation kwargs with defaults
            final_generation_kwargs = {
                "do_sample": self.generation_config["do_sample"],
                "top_p": self.generation_config["top_p"],
                "use_cache": self.generation_config["use_cache"],
                "pad_token_id": self.processor.tokenizer.eos_token_id,
            }
            final_generation_kwargs.update(generation_kwargs)

            # Debug: Show actual token configuration being used
            if self.debug:
                passed_tokens = generation_kwargs.get("max_new_tokens", "not specified")
                config_tokens = self.generation_config.get("max_new_tokens", "not set")
                final_tokens = final_generation_kwargs.get("max_new_tokens", "not set")
                print(
                    f"🎯 Token debug - Passed: {passed_tokens}, Config: {config_tokens}, Final: {final_tokens}"
                )

            # Clean up temperature if do_sample is False to avoid warnings
            if not final_generation_kwargs.get("do_sample", False):
                final_generation_kwargs.pop("temperature", None)
                final_generation_kwargs.pop("top_p", None)

            # Generate response with resilient fallback
            with torch.no_grad():
                output = self._resilient_generate(inputs, **final_generation_kwargs)

            # Decode response
            response = self.processor.decode(output[0], skip_special_tokens=True)

            # Extract only the assistant's response
            if "assistant\\n\\n" in response:
                response = response.split("assistant\\n\\n")[-1].strip()
            elif "assistant" in response:
                response = response.split("assistant")[-1].strip()

            # Cleanup with V100 optimizations
            del inputs, output, image
            comprehensive_memory_cleanup(self.model, self.processor, verbose=self.debug)

            return response

        except Exception as e:
            if self.debug:
                print(f"❌ Error in custom prompt extraction: {e}")
            return f"Error: {str(e)}"

    def process_universal_single_pass(self, image_path: str) -> dict:
        """
        Process image using universal single-pass extraction with all 17 fields.

        This method eliminates the document detection stage by extracting all possible
        fields in a single call, then inferring document type from the results.
        Provides significant performance improvement (~50% faster) by eliminating
        double image loading.

        Args:
            image_path (str): Path to document image

        Returns:
            dict: Extraction results with same format as document-aware processing
        """
        try:
            start_time = time.time()

            if self.debug:
                print(
                    f"🌍 Starting universal single-pass extraction for: {Path(image_path).name}"
                )
                print(
                    "   Eliminating document detection stage - processing all 17 fields"
                )

            # Memory cleanup before processing
            handle_memory_fragmentation(threshold_gb=1.0, aggressive=True)

            # Load image once (vs twice in document-aware mode)
            image = self.load_document_image(image_path)

            # Get universal prompt for single-pass extraction
            universal_prompt = self.get_extraction_prompt("universal")

            # Universal field list - 18 fields (excludes validation-only fields)
            # NOTE: TRANSACTION_AMOUNTS_RECEIVED excluded (validation-only)
            # ACCOUNT_BALANCE required for mathematical enhancement when balance column exists
            universal_fields = [
                "DOCUMENT_TYPE",
                "BUSINESS_ABN",
                "SUPPLIER_NAME",
                "BUSINESS_ADDRESS",
                "PAYER_NAME",
                "PAYER_ADDRESS",
                "INVOICE_DATE",
                "STATEMENT_DATE_RANGE",
                "LINE_ITEM_DESCRIPTIONS",
                "LINE_ITEM_QUANTITIES",
                "LINE_ITEM_PRICES",
                "LINE_ITEM_TOTAL_PRICES",
                "IS_GST_INCLUDED",
                "GST_AMOUNT",
                "TOTAL_AMOUNT",
                "TRANSACTION_DATES",
                "TRANSACTION_AMOUNTS_PAID",
                "ACCOUNT_BALANCE",
            ]

            if self.debug:
                # Use direct stdout to bypass Rich console completely
                import sys

                sys.stdout.write(
                    f"🌍 Universal field list: {len(universal_fields)} fields\n"
                )
                sys.stdout.write(
                    f"   Fields: {universal_fields[:3]}...{universal_fields[-2:]}\n"
                )
                sys.stdout.write(
                    f"🔍 UNIVERSAL PROMPT ({len(universal_prompt)} chars):\n"
                )
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.write(universal_prompt + "\n")
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.flush()

            # Create multimodal conversation with universal prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": universal_prompt},
                    ],
                }
            ]

            # Apply chat template
            input_text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )

            # Process inputs
            inputs = self.processor(image, input_text, return_tensors="pt").to(
                self.device
            )

            # Configure generation for universal extraction (all 17 fields)
            universal_generation_config = self.generation_config.copy()
            _cfg = LLAMA_GENERATION_CONFIG
            universal_generation_config["max_new_tokens"] = max(
                _cfg["max_new_tokens_base"],
                len(universal_fields) * _cfg["max_new_tokens_per_field"],
            )

            if self.debug:
                print(f"🖼️  Input tensor shape: {inputs['input_ids'].shape}")
                print(
                    f"💭 Generating with max_new_tokens={universal_generation_config['max_new_tokens']}"
                )

            # Generate response using resilient generation
            output = self._resilient_generate(inputs, **universal_generation_config)

            # Decode response
            full_response = self.processor.decode(output[0], skip_special_tokens=True)

            # Extract assistant's response (handle Llama chat template)
            if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
                response = full_response.split(
                    "<|start_header_id|>assistant<|end_header_id|>"
                )[1]
                response = response.split("<|eot_id|>")[0].strip()
            elif "\nassistant\n\n" in full_response:
                response = full_response.split("\nassistant\n\n")[1].strip()
            elif "\nassistant\n" in full_response:
                response = full_response.split("\nassistant\n")[1].strip()
            elif "assistant" in full_response:
                parts = full_response.split("assistant")
                response = parts[-1].strip()
            else:
                response = full_response.strip()

            if self.debug:
                # Use direct stdout to bypass Rich console completely
                import sys

                sys.stdout.write(
                    f"📄 RAW UNIVERSAL RESPONSE ({len(response)} chars):\n"
                )
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.write(response + "\n")
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.flush()

            # Parse response using robust extraction parser
            from common.extraction_parser import parse_extraction_response

            extracted_data = parse_extraction_response(
                response, expected_fields=universal_fields
            )

            # Apply ExtractionCleaner for value normalization (matching InternVL3 behavior)
            cleaned_data = {}
            for field in universal_fields:
                raw_value = extracted_data.get(field, "NOT_FOUND")
                if raw_value != "NOT_FOUND":
                    cleaned_value = self.cleaner.clean_field_value(field, raw_value)
                    cleaned_data[field] = cleaned_value
                else:
                    cleaned_data[field] = "NOT_FOUND"

            # Infer document type from extracted DOCUMENT_TYPE field
            detected_doc_type = cleaned_data.get("DOCUMENT_TYPE", "unknown").lower()

            # Normalize document type using simple heuristics
            if "statement" in detected_doc_type or "bank" in detected_doc_type:
                inferred_doc_type = "bank_statement"
            elif "receipt" in detected_doc_type:
                inferred_doc_type = "receipt"
            elif "invoice" in detected_doc_type or "tax" in detected_doc_type:
                inferred_doc_type = "invoice"
            else:
                # Fallback: infer from field presence
                if cleaned_data.get("STATEMENT_DATE_RANGE", "NOT_FOUND") != "NOT_FOUND":
                    inferred_doc_type = "bank_statement"
                elif cleaned_data.get("GST_AMOUNT", "NOT_FOUND") != "NOT_FOUND":
                    inferred_doc_type = "invoice"
                else:
                    inferred_doc_type = "receipt"  # Default fallback

            if self.debug:
                # Use direct stdout to bypass Rich console completely
                import sys

                sys.stdout.write("📊 UNIVERSAL EXTRACTION RESULTS:\n")
                sys.stdout.write("=" * 80 + "\n")
                for field in universal_fields:
                    value = cleaned_data.get(field, "NOT_FOUND")
                    status = "✅" if value != "NOT_FOUND" else "❌"
                    sys.stdout.write(f'  {status} {field}: "{value}"\n')
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.flush()

                found_fields = [k for k, v in cleaned_data.items() if v != "NOT_FOUND"]
                print(
                    f"✅ Universal extraction: {len(found_fields)}/{len(universal_fields)} fields found"
                )
                print(f"📄 Inferred document type: {inferred_doc_type}")
                if found_fields:
                    print(
                        f"   Found: {found_fields[:3]}{'...' if len(found_fields) > 3 else ''}"
                    )

            # Calculate metrics
            extracted_fields_count = len(
                [k for k, v in cleaned_data.items() if v != "NOT_FOUND"]
            )
            response_completeness = extracted_fields_count / len(universal_fields)
            content_coverage = extracted_fields_count / len(universal_fields)

            # Memory cleanup with V100 optimizations
            del inputs, output, image
            comprehensive_memory_cleanup(self.model, self.processor, verbose=self.debug)

            processing_time = time.time() - start_time

            # Return results in same format as document-aware processing for compatibility
            return {
                "image_name": Path(image_path).name,
                "extracted_data": cleaned_data,
                "raw_response": response,
                "processing_time": processing_time,
                "response_completeness": response_completeness,
                "content_coverage": content_coverage,
                "extracted_fields_count": extracted_fields_count,
                "field_count": len(universal_fields),
                "document_type": inferred_doc_type,  # Inferred from extraction
                "extraction_mode": "universal_single_pass",  # Mark as universal
            }

        except Exception as e:
            if self.debug:
                print(f"❌ Error in universal single-pass extraction: {e}")
                import traceback

                traceback.print_exc()

            # Return error result with default fields list (excludes validation-only fields)
            # ACCOUNT_BALANCE required for mathematical enhancement when balance column exists
            default_fields = [
                "DOCUMENT_TYPE",
                "BUSINESS_ABN",
                "SUPPLIER_NAME",
                "BUSINESS_ADDRESS",
                "PAYER_NAME",
                "PAYER_ADDRESS",
                "INVOICE_DATE",
                "STATEMENT_DATE_RANGE",
                "LINE_ITEM_DESCRIPTIONS",
                "LINE_ITEM_QUANTITIES",
                "LINE_ITEM_PRICES",
                "LINE_ITEM_TOTAL_PRICES",
                "IS_GST_INCLUDED",
                "GST_AMOUNT",
                "TOTAL_AMOUNT",
                "TRANSACTION_DATES",
                "TRANSACTION_AMOUNTS_PAID",
                "ACCOUNT_BALANCE",
            ]
            return {
                "image_name": Path(image_path).name,
                "extracted_data": {field: "NOT_FOUND" for field in default_fields},
                "raw_response": f"Error: {str(e)}",
                "processing_time": 0,
                "response_completeness": 0,
                "content_coverage": 0,
                "extracted_fields_count": 0,
                "field_count": len(default_fields),
                "document_type": "unknown",
                "extraction_mode": "universal_single_pass",
            }

    def _parse_document_aware_response(self, response_text: str) -> dict:
        """Parse extraction response for document-specific field list."""
        import re

        if not response_text:
            return {field: "NOT_FOUND" for field in self.field_list}

        # Initialize with NOT_FOUND for all document-specific fields
        extracted_data = {field: "NOT_FOUND" for field in self.field_list}

        # Clean Llama-specific conversation artifacts and extract only assistant response
        if "assistant" in response_text:
            # Split on 'assistant' and take the last part (the actual response)
            parts = response_text.split("assistant")
            response_text = parts[-1].strip()

        # Additional cleaning patterns
        clean_patterns = [
            r"I'll extract.*?\n",
            r"I can extract.*?\n",
            r"Here (?:is|are) the.*?\n",
            r"Based on.*?\n",
            r"Looking at.*?\n",
            r"<\|start_header_id\|>.*?<\|end_header_id\|>",
            r"<image>",
            r"^\s*Extract.*?below\.\s*\n",
        ]

        for pattern in clean_patterns:
            response_text = re.sub(
                pattern, "", response_text, flags=re.IGNORECASE | re.MULTILINE
            )

        # Process each line looking for key-value pairs
        lines = response_text.strip().split("\n")

        i = 0
        while i < len(lines):
            line = lines[i]

            # Skip empty lines and non-key-value lines
            if not line.strip() or ":" not in line:
                i += 1
                continue

            # Clean the line from various formatting issues
            clean_line = line
            # Remove all markdown formatting - simple and effective approach
            clean_line = clean_line.replace("**", "").replace("*", "")
            # Fix various prefix issues
            clean_line = re.sub(r"^KEY:\s*([A-Z_]+):", r"\1:", clean_line)
            clean_line = re.sub(r"^KEY\s+([A-Z_]+):", r"\1:", clean_line)
            # Fix GST field name variations (after markdown removal)
            clean_line = re.sub(r"^GST[_\s]*\d*%?:", "GST_AMOUNT:", clean_line)

            # Extract key and value
            parts = clean_line.split(":", 1)
            if len(parts) == 2:
                key = parts[0].strip().upper()
                raw_value = parts[1].strip()

                # Debug logging for markdown cleaning
                if self.debug and ("*" in raw_value):
                    print(f"🔍 DEBUG: Raw value before cleaning: '{raw_value}'")

                # Additional markdown cleanup for values that start with markdown
                # Handle cases like " ** STATEMENT", "**STATEMENT", " **STATEMENT" etc.
                value = re.sub(
                    r"^\s*\*+\s*", "", raw_value
                )  # Remove leading whitespace + asterisks + trailing spaces
                value = value.strip()  # Clean up any remaining whitespace

                # Debug logging for markdown cleaning
                if self.debug and raw_value != value:
                    print(f"🧹 DEBUG: Cleaned value: '{raw_value}' → '{value}'")

                # Normalize field name: convert spaces to underscores for matching
                normalized_key = key.replace(" ", "_")

                # Handle multi-line fields (like bullet point lists)
                if not value or value == "":
                    # Look ahead for bullet points or continuation lines
                    bullet_values = []
                    j = i + 1
                    while j < len(lines):
                        next_line = lines[j].strip()
                        if not next_line:
                            j += 1
                            continue

                        # Check if this is the start of a new field (BEFORE cleaning markdown)
                        # Look for patterns like "**FIELD_NAME:" or "FIELD_NAME:"
                        is_next_field = False
                        if ":" in next_line:
                            # Remove ** to check if it's a field name
                            clean_check = next_line.replace("**", "").strip().upper()
                            field_name = (
                                clean_check.split(":")[0].strip().replace(" ", "_")
                            )
                            if field_name in self.field_list:
                                is_next_field = True

                        if is_next_field:
                            # Next field found, stop collecting
                            break
                        elif (
                            next_line.startswith("*")
                            or next_line.startswith("-")
                            or next_line.startswith("•")
                        ):
                            # Extract bullet point content
                            bullet_content = next_line.lstrip("*-•").strip()
                            if bullet_content:
                                bullet_values.append(bullet_content)
                            j += 1
                        else:
                            # Skip section headers or other non-field content
                            j += 1

                    if bullet_values:
                        value = " | ".join(bullet_values)
                        i = j - 1  # Skip the processed lines

                # Store if it's in our document-specific field list
                if normalized_key in self.field_list:
                    # Normalize common "not found" variations before cleaning
                    if value.lower() in [
                        "not found",
                        "not_found",
                        "notfound",
                        "n/a",
                        "na",
                    ]:
                        cleaned_value = "NOT_FOUND"
                    else:
                        # Clean the extracted value using the centralized cleaner
                        cleaned_value = (
                            self.cleaner.clean_field_value(normalized_key, value)
                            if value
                            else "NOT_FOUND"
                        )
                    extracted_data[normalized_key] = cleaned_value

            i += 1

        return extracted_data
