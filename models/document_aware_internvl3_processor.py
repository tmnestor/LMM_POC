#!/usr/bin/env python3
"""
Document-Aware InternVL3 Hybrid Processor - SAFE COPY WITH INTERNVL3 MODEL

A hybrid processor that combines:
- InternVL3 model (for better accuracy potential)
- Llama's proven processing pipeline (for reliable processing)
- ExtractionCleaner integration (for 🧹 CLEANER CALLED output)

ZERO RISK to existing Llama processor - this is a completely independent copy.

DOCUMENT AWARE REDUCTION OPTIMIZED:
- Invoice/Receipt: 11 fields (62-67% reduction) → ~50% faster processing
- Bank Statement: 5 fields (75% reduction) → ~75% faster processing
- Dynamic max_new_tokens automatically scales with reduced field counts
"""

import gc
import time
import warnings

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from common.batch_processor import load_document_field_definitions
from common.extraction_cleaner import ExtractionCleaner
from common.gpu_optimization import (
    configure_cuda_memory_allocation,
    emergency_cleanup,
    get_available_gpu_memory,
    optimize_model_for_gpu,
)
from common.model_config import (
    get_auto_batch_size,
    get_max_new_tokens,
)
from common.pipeline_config import strip_structure_suffixes
from common.simple_prompt_loader import SimplePromptLoader, load_internvl3_prompt
from models.internvl3_image_preprocessor import InternVL3ImagePreprocessor

warnings.filterwarnings("ignore")

INTERNVL3_MODEL_PATH = "/efs/shared/PTM/InternVL3-8B"


class DocumentAwareInternVL3HybridProcessor:
    """Hybrid processor: InternVL3 model + Llama's proven processing pipeline."""

    def __init__(
        self,
        field_list: list[str],
        model_path: str = None,
        device: str = "cuda",
        debug: bool = False,
        batch_size: int | None = None,
        skip_model_loading: bool = False,
        pre_loaded_model=None,
        pre_loaded_tokenizer=None,
        prompt_config: dict | None = None,
        max_tiles: int = None,
        field_definitions: dict[str, list[str]] | None = None,
    ):
        """
        Initialize hybrid processor with InternVL3 model and Llama processing logic.

        Args:
            field_list (list[str]): Fields to extract
            model_path (str): Path to InternVL3 model
            device (str): Device to run on
            debug (bool): Enable debug output
            batch_size (int): Batch size (auto-detected if None)
            skip_model_loading (bool): Skip model loading (use pre-loaded)
            pre_loaded_model: Pre-loaded model instance
            pre_loaded_tokenizer: Pre-loaded tokenizer instance
            prompt_config (Dict): Configuration for prompts (single source of truth)
            max_tiles (int): Max image tiles for preprocessing (REQUIRED - set in notebook CONFIG)
            field_definitions: Pre-loaded field definitions dict. If None, loads from YAML.
        """
        self.field_list = field_list
        self.field_count = len(field_list)
        self.model_path = model_path or INTERNVL3_MODEL_PATH
        self.device = device
        self.debug = debug
        self.prompt_config = prompt_config
        if not self.prompt_config:
            raise ValueError(
                "prompt_config is required — must contain "
                "'detection_file', 'detection_key', 'extraction_files'"
            )
        missing = {"detection_file", "detection_key", "extraction_files"} - set(
            self.prompt_config
        )
        if missing:
            raise ValueError(f"prompt_config missing required keys: {missing}")
        self.max_tiles = max_tiles  # REQUIRED: Notebook-configurable tile count

        # Read fallback_type from detection YAML settings (avoids hardcoding "INVOICE")
        self._fallback_type = "INVOICE"  # safe default if YAML read fails
        try:
            from pathlib import Path

            import yaml

            detection_path = Path(self.prompt_config["detection_file"])
            if detection_path.exists():
                with detection_path.open() as f:
                    det_cfg = yaml.safe_load(f)
                self._fallback_type = det_cfg.get("settings", {}).get(
                    "fallback_type", "INVOICE"
                )
        except Exception:
            pass

        # Image preprocessing pipeline (extracted for testability and reuse)
        self.image_preprocessor = InternVL3ImagePreprocessor(
            max_tiles=max_tiles, debug=debug
        )

        # Initialize components (InternVL3 specific)
        self.model = pre_loaded_model
        self.tokenizer = pre_loaded_tokenizer
        self.generation_config = None

        # Fix pad_token_id if tokenizer is pre-loaded to suppress warnings
        if self.tokenizer is not None and self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Suppress generation warnings on model's own generation_config
        if self.model is not None and self.tokenizer is not None:
            if hasattr(self.model, "generation_config"):
                self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
                # Clear temperature/top_p — irrelevant with do_sample=False and
                # causes "not valid and may be ignored" warnings
                self.model.generation_config.temperature = None
                self.model.generation_config.top_p = None
            elif hasattr(self.model.config, "pad_token_id"):
                self.model.config.pad_token_id = self.tokenizer.eos_token_id

        # Detect model variant (2B vs 8B) for tile optimization
        self.is_8b_model = "8B" in self.model_path

        # Initialize extraction cleaner for value normalization (🧹 CLEANER CALLED output)
        self.cleaner = ExtractionCleaner(debug=debug)

        # Document-specific field lists - loaded from config/field_definitions.yaml
        # SINGLE SOURCE OF TRUTH - no hardcoding here
        self.document_field_lists = (
            field_definitions or load_document_field_definitions()
        )

        if self.debug:
            print(
                f"🎯 InternVL3 Hybrid processor initialized for {self.field_count} fields: {field_list[0]} → {field_list[-1]}"
            )

        # Configure CUDA memory allocation
        configure_cuda_memory_allocation()

        # Configure batch processing
        self._configure_batch_processing(batch_size)

        # Configure generation parameters for dynamic field count
        self._configure_generation()

        # Load model and tokenizer (unless skipping for reuse or already provided)
        if not skip_model_loading and (self.model is None or self.tokenizer is None):
            self._load_model()
        elif self.model is not None and self.tokenizer is not None:
            if self.debug:
                print("✅ Using pre-loaded InternVL3 model and tokenizer")
                print(f"🔧 Device: {self.model.device}")
                print(
                    f"💾 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
                )
            # Apply GPU optimizations to pre-loaded model
            model_dtype = next(self.model.parameters()).dtype
            optimize_model_for_gpu(self.model, dtype=model_dtype)

    def _configure_batch_processing(self, batch_size: int | None):
        """Configure batch processing parameters."""
        if batch_size is not None:
            self.batch_size = max(1, batch_size)
            if self.debug:
                print(f"🎯 Using manual batch size: {self.batch_size}")
        else:
            # Auto-detect batch size based on available memory
            available_memory = get_available_gpu_memory(self.device)
            self.batch_size = get_auto_batch_size("internvl3", available_memory)
            if self.debug:
                print(
                    f"🤖 Auto-detected batch size: {self.batch_size} (GPU Memory: {available_memory:.1f}GB)"
                )

    def _configure_generation(self):
        """Configure generation parameters for InternVL3."""
        # InternVL3 generation config - chat() method only accepts max_new_tokens and do_sample
        # NOTE: temperature and top_p are NOT valid for InternVL3 chat() and cause warnings
        self.generation_config = {
            "max_new_tokens": get_max_new_tokens(field_count=self.field_count),
            "do_sample": False,  # Greedy decoding for consistent extraction
            "use_cache": True,
            "pad_token_id": self.tokenizer.eos_token_id if self.tokenizer else None,
        }

        if self.debug:
            performance_gain = max(
                0, (29 - self.field_count) / 29 * 100
            )  # vs old invoice max
            print(
                f"🎯 DOCUMENT AWARE REDUCTION: {self.field_count} fields (~{performance_gain:.0f}% fewer than original 29)"
            )
            print(
                f"🎯 Generation config: max_new_tokens={self.generation_config['max_new_tokens']}, "
                f"do_sample={self.generation_config['do_sample']} (greedy decoding)"
            )

    def _load_model(self):
        """Load InternVL3 model and tokenizer with optimal configuration."""
        if self.debug:
            print(f"🔄 Loading InternVL3 model from: {self.model_path}")

        try:
            # Load InternVL3 model with V100 optimizations - REVERTED TO WORKING CONFIG
            self.model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",  # REVERTED: Custom device mapping broke the model
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                # use_flash_attn=True,  # Disabled: Poor V100 compatibility
            ).eval()

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=False,  # More reliable for structured tasks
            )

            # Fix pad_token_id to suppress warnings during generation
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            if self.debug:
                print("✅ InternVL3 model loaded successfully")
                print(f"🔧 Device: {self.model.device}")
                print(
                    f"💾 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
                )

            # Apply GPU optimizations
            model_dtype = next(self.model.parameters()).dtype
            optimize_model_for_gpu(self.model, dtype=model_dtype)

        except Exception as e:
            if self.debug:
                print(f"❌ Error loading InternVL3 model: {e}")
            raise

    def get_extraction_prompt(self, document_type: str = None) -> str:
        """
        Get extraction prompt for document type - uses Llama prompts for consistency.

        Args:
            document_type: Document type ('invoice', 'receipt', 'bank_statement')
                          If None, uses 'universal' prompt

        Returns:
            Complete prompt string loaded from YAML
        """
        if document_type is None:
            document_type = "universal"

        if self.debug:
            print(f"📝 Loading {document_type} prompt for InternVL3 Hybrid")

        try:
            return load_internvl3_prompt(document_type)
        except Exception as e:
            if self.debug:
                print(
                    f"⚠️ Failed to load {document_type} prompt, falling back to universal"
                )
            return load_internvl3_prompt("universal")

    def get_supported_document_types(self) -> list[str]:
        """Get list of supported document types."""
        return SimplePromptLoader.get_available_prompts("internvl3_prompts.yaml")

    def detect_and_classify_document(
        self, image_path: str, verbose: bool = False
    ) -> dict:
        """
        Detect document type using InternVL3 model and document detection prompt.
        Compatible with BatchDocumentProcessor workflow.

        Args:
            image_path: Path to the image to classify
            verbose: Whether to show detailed output

        Returns:
            Dict with classification info including document_type
        """
        try:
            from pathlib import Path

            import yaml

            # Use prompt_config if provided (single source of truth), otherwise fallback to YAML
            if self.prompt_config:
                detection_path = Path(self.prompt_config["detection_file"])
                detection_key = self.prompt_config["detection_key"]
                if verbose:
                    print(
                        f"🔧 CONFIG DEBUG - Using prompt_config: detection_key='{detection_key}'"
                    )
            else:
                # Fallback to hardcoded YAML (legacy behavior)
                detection_path = Path("prompts/document_type_detection.yaml")
                detection_key = "detection"
                if verbose:
                    print(
                        f"🔧 CONFIG DEBUG - Using fallback: detection_key='{detection_key}'"
                    )

            with detection_path.open("r") as f:
                detection_config = yaml.safe_load(f)

            detection_prompt = detection_config["prompts"][detection_key]["prompt"]
            max_tokens = detection_config.get("settings", {}).get("max_new_tokens", 50)

            if verbose:
                # Use direct stdout to bypass Rich console completely for detection
                import sys

                sys.stdout.write(
                    f"🔍 Using InternVL3 document detection prompt: {detection_key}\n"
                )
                sys.stdout.write(f"📝 Prompt: {detection_prompt[:100]}...\n")
                sys.stdout.flush()

            # Load and preprocess image using InternVL3 pipeline
            pixel_values = self.image_preprocessor.load_image(image_path, self.model)

            # Ensure on correct device (backup check)
            model_device = InternVL3ImagePreprocessor.get_model_device(self.model)
            if pixel_values.device != model_device:
                pixel_values = pixel_values.to(model_device)
                if verbose:
                    print(f"🔧 BACKUP_DEVICE_FIX: Moved tensor to {model_device}")

            # Generate response using InternVL3 model with detection-specific limits
            response = self._resilient_generate(
                pixel_values=pixel_values,
                question=detection_prompt,
                max_new_tokens=max_tokens,
                do_sample=False,
                is_detection=True,  # Enable strict token limits for detection
            )

            # Free detection tensors before extraction allocates its own
            del pixel_values
            torch.cuda.empty_cache()

            if verbose:
                # Use direct stdout to bypass Rich console completely for detection
                import sys

                sys.stdout.write(f"🤖 Model response: {response}\n")
                sys.stdout.flush()

            # Parse document type from response
            document_type = self._parse_document_type_response(
                response, detection_config
            )

            if verbose:
                # Use direct stdout to bypass Rich console completely for detection
                import sys

                sys.stdout.write(f"✅ Detected document type: {document_type}\n")
                sys.stdout.flush()

            return {
                "document_type": document_type,
                "confidence": 1.0,  # InternVL3 doesn't provide confidence scores
                "raw_response": response,
                "prompt_used": detection_key,
            }

        except Exception as e:
            # ALWAYS show detection errors - critical for debugging
            import sys

            sys.stdout.write(f"❌ DETECTION ERROR: {e}\n")
            sys.stdout.flush()
            if self.debug:
                import traceback

                sys.stdout.write("❌ DETECTION ERROR TRACEBACK:\n")
                sys.stdout.flush()
                traceback.print_exc()

            # Fallback to simple heuristic (type from detection YAML settings)
            return {
                "document_type": self._fallback_type,
                "confidence": 0.1,
                "raw_response": "",
                "prompt_used": "fallback_heuristic",
                "error": str(e),
            }

    def _classify_bank_structure(self, image_path: str, verbose: bool = False) -> str:
        """Classify bank statement structure using vision model.

        Returns:
            Structure-specific prompt key (e.g. 'bank_statement_flat')
        """
        try:
            from common.vision_bank_statement_classifier import (
                classify_bank_statement_structure_vision,
            )
        except ImportError:
            import sys
            from pathlib import Path

            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from common.vision_bank_statement_classifier import (
                classify_bank_statement_structure_vision,
            )

        if verbose:
            print("🔍 Running vision-based structure classification for bank statement")

        structure_type = classify_bank_statement_structure_vision(
            image_path,
            model=self,
            processor=None,
            verbose=verbose,
        )

        if verbose:
            print(f"🏗️ Bank statement structure: {structure_type}")
            print(f"📝 Using prompt key: bank_statement_{structure_type}")

        return f"bank_statement_{structure_type}"

    def process_document_aware(
        self, image_path: str, classification_info: dict, verbose: bool = False
    ) -> dict:
        """
        Process document using document-specific extraction based on detected type.
        Compatible with BatchDocumentProcessor workflow.

        Args:
            image_path: Path to the image to process
            classification_info: Result from detect_and_classify_document()
            verbose: Whether to show detailed output

        Returns:
            Dict with extraction results in BatchDocumentProcessor format
        """
        try:
            document_type = classification_info["document_type"].lower()

            if verbose:
                print(f"📊 Processing {document_type.upper()} document with InternVL3")

            # For bank statements, use vision-based structure classification
            if document_type == "bank_statement":
                try:
                    document_type = self._classify_bank_structure(image_path, verbose)
                except Exception as e:
                    if verbose:
                        print(f"⚠️ Vision classification failed: {e}")
                        print("📝 Falling back to bank_statement_flat prompt")
                    document_type = "bank_statement_flat"

            # Get document-specific prompt using prompt_config (single source of truth)
            # Strip structure suffixes to get base document type
            doc_type_upper = strip_structure_suffixes(document_type).upper()
            extraction_files = self.prompt_config["extraction_files"]
            if doc_type_upper not in extraction_files:
                raise ValueError(
                    f"No extraction file configured for '{doc_type_upper}'. "
                    f"Available: {list(extraction_files.keys())}. "
                    f"Add it to prompt_config['extraction_files']."
                )
            extraction_file = extraction_files[doc_type_upper]

            # Get the prompt key from config (or derive from document type if not specified)
            extraction_keys = self.prompt_config.get("extraction_keys", {})

            if doc_type_upper in extraction_keys:
                # Use explicitly configured key
                extraction_key = extraction_keys[doc_type_upper]
            else:
                # Derive key from document type (already includes structure suffix if present)
                extraction_key = document_type

            # For bank statements ONLY: if key doesn't include structure suffix, append it
            # This allows config to override by specifying full key like "bank_statement_flat"
            if (
                document_type.startswith("bank_statement")
                and doc_type_upper == "BANK_STATEMENT"
            ):
                if (
                    "_flat" not in extraction_key
                    and "_date_grouped" not in extraction_key
                ):
                    # Only append if document_type has a structure suffix
                    if "_flat" in document_type:
                        extraction_key = f"{extraction_key}_flat"
                    elif "_date_grouped" in document_type:
                        extraction_key = f"{extraction_key}_date_grouped"

            from pathlib import Path

            from common.simple_prompt_loader import SimplePromptLoader

            loader = SimplePromptLoader()
            extraction_prompt = loader.load_prompt(
                Path(extraction_file).name, extraction_key
            )

            if verbose:
                print(
                    f"📝 Using {document_type} prompt (prompt_config): {len(extraction_prompt)} characters"
                )

            # Get document-specific field list from cached config (single source of truth)
            doc_type_fields = dict(self.document_field_lists)
            # Add structure-specific bank statement aliases (same fields as bank_statement)
            if "bank_statement" in doc_type_fields:
                doc_type_fields["bank_statement_flat"] = doc_type_fields[
                    "bank_statement"
                ]
                doc_type_fields["bank_statement_date_grouped"] = doc_type_fields[
                    "bank_statement"
                ]

            # Resolve document-specific field list (no mutation of self.field_list)
            doc_field_list = doc_type_fields.get(
                document_type, doc_type_fields["invoice"]
            )

            # Calculate document-specific max tokens
            from common.model_config import get_max_new_tokens

            # Use base document type for max tokens calculation
            base_doc_type = strip_structure_suffixes(document_type)
            doc_specific_tokens = get_max_new_tokens(
                field_count=len(doc_field_list), document_type=base_doc_type
            )

            # Process with document-specific settings
            # CRITICAL: Pass extraction_prompt to avoid duplicate detection
            result = self.process_single_image(
                image_path,
                custom_prompt=extraction_prompt,
                custom_max_tokens=doc_specific_tokens,
                field_list=doc_field_list,
            )

            if verbose:
                extracted_data = result.get("extracted_data", {})
                found_fields = sum(
                    1 for v in extracted_data.values() if v != "NOT_FOUND"
                )
                print(f"✅ Extracted {found_fields}/{len(extracted_data)} fields")

            return result

        except Exception as e:
            if verbose:
                print(f"❌ Error in document-aware processing: {e}")

            # Fallback to universal processing
            return self.process_single_image(image_path)

    def _parse_document_type_response(
        self, response: str, detection_config: dict
    ) -> str:
        """
        Parse document type from model response using type mappings.

        Args:
            response: Raw model response
            detection_config: Detection configuration with type mappings

        Returns:
            Normalized document type
        """
        response_lower = response.lower().strip()

        if self.debug:
            # Use direct stdout to bypass Rich console completely for detection debug
            import sys

            sys.stdout.write(f"🔍 PARSING DEBUG - Raw response: '{response}'\n")
            sys.stdout.write(
                f"🔍 PARSING DEBUG - Cleaned response: '{response_lower}'\n"
            )
            sys.stdout.flush()

        # Get type mappings from config
        type_mappings = detection_config.get("type_mappings", {})

        # Direct mapping check
        for variant, canonical in type_mappings.items():
            if variant.lower() in response_lower:
                if self.debug:
                    import sys

                    sys.stdout.write(
                        f"✅ PARSING DEBUG - Found mapping: '{variant}' -> '{canonical}'\n"
                    )
                    sys.stdout.flush()
                return canonical

        # Fallback keyword detection (YAML-driven from fallback_keywords section)
        fallback_keywords = detection_config.get("fallback_keywords", {})
        for canonical_type, keywords in fallback_keywords.items():
            if any(kw in response_lower for kw in keywords):
                if self.debug:
                    import sys

                    sys.stdout.write(
                        f"✅ PARSING DEBUG - Keyword match: {canonical_type}\n"
                    )
                    sys.stdout.flush()
                return canonical_type

        # Final fallback
        fallback = detection_config.get("settings", {}).get("fallback_type", "INVOICE")
        if self.debug:
            import sys

            sys.stdout.write(
                f"❌ PARSING DEBUG - No matches found, using fallback: '{fallback}'\n"
            )
            sys.stdout.flush()
        return fallback

    def load_document_image(self, image_path: str) -> Image.Image:
        """Load document image with error handling."""
        try:
            return Image.open(image_path)
        except Exception as e:
            if self.debug:
                print(f"❌ Error loading image {image_path}: {e}")
            raise

    def _resilient_generate(self, pixel_values, question, **generation_kwargs):
        """Resilient generation with OOM fallback using InternVL3 chat method.

        IMPORTANT — OOM cleanup architecture:
        All ``torch.cuda.empty_cache()`` calls MUST happen **outside** ``except``
        blocks.  While inside an ``except`` handler, Python's traceback object
        holds references to every frame in the failed call-stack — which includes
        the model's intermediate activation tensors.  ``empty_cache()`` cannot
        reclaim memory that is still referenced, so calling it inside ``except``
        is a no-op.  Exiting the ``except`` block releases the traceback and
        allows the tensors to be freed.
        """

        # Build clean generation parameters
        max_tokens = generation_kwargs.get(
            "max_new_tokens", self.generation_config["max_new_tokens"]
        )
        is_detection = generation_kwargs.get("is_detection", False)

        if is_detection:
            max_tokens = min(max_tokens, 100)

        clean_generation_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": generation_kwargs.get(
                "do_sample", self.generation_config["do_sample"]
            ),
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        minimal_config = {
            "max_new_tokens": 50,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        # Retry loop — attempt 0: normal, attempt 1: after cleanup, attempt 2: minimal tokens
        response = None
        for attempt in range(3):
            gen_config = minimal_config if attempt == 2 else clean_generation_kwargs
            oom = False

            try:
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    question,
                    generation_config=gen_config,
                    history=None,
                    return_history=False,
                )
                break  # Success — exit retry loop

            except torch.cuda.OutOfMemoryError:
                oom = True  # Flag to handle OUTSIDE except block

            except Exception as e:
                if attempt == 2:
                    # Last resort on minimal generation — return fallback
                    response = (
                        "invoice" if "document" in question.lower() else "NOT_FOUND"
                    )
                    break

                # Non-OOM error on attempts 0-1 — report and return empty
                if self.debug:
                    import sys

                    sys.stdout.write(
                        f"❌ CRITICAL: Generation failed: {type(e).__name__}: {e}\n"
                    )
                    sys.stdout.flush()
                return ""

            # ── OUTSIDE except block ──
            # The traceback from the failed forward pass is now released,
            # so gc.collect() + empty_cache() can actually reclaim the
            # intermediate activation tensors.
            if oom:
                if self.debug:
                    if attempt == 0:
                        print("⚠️ CUDA OOM during generation, attempting recovery...")
                    else:
                        print("❌ Still OOM after cleanup, using minimal generation")
                gc.collect()
                torch.cuda.empty_cache()
                if attempt == 0:
                    emergency_cleanup(verbose=False)

        # Total failure — all 3 attempts exhausted without break
        if response is None:
            response = "invoice" if "document" in question.lower() else "NOT_FOUND"

        # Post-process response (common to all paths)
        if self._detect_recursion_pattern(response):
            if self.debug:
                import sys

                sys.stdout.write(
                    f"⚠️ RECURSION DETECTED: Truncating response at {len(response)} chars\n"
                )
                sys.stdout.flush()
            response = response[:200]
        elif len(response) > 2000:
            response = response[:2000]

        return response

    def _detect_recursion_pattern(self, response: str) -> bool:
        """Detect infinite recursion patterns in model responses."""
        if not response or len(response) < 50:
            return False

        # Check for repeated text patterns that indicate recursion
        response_lines = response.split("\n")

        # Look for lines that are repeated many times
        line_counts = {}
        for line in response_lines:
            line = line.strip()
            if len(line) > 10:  # Only count substantial lines
                line_counts[line] = line_counts.get(line, 0) + 1

        # If any line appears more than 5 times, it's likely recursion
        max_repetitions = max(line_counts.values()) if line_counts else 0
        if max_repetitions > 5:
            return True

        # Check for the specific "Answer with one of:" repetition pattern
        if (
            "Answer with one of:" in response
            and response.count("Answer with one of:") > 3
        ):
            return True

        # Check for repeated prompt instructions
        if "INVOICE" in response and response.count("INVOICE") > 10:
            return True

        return False

    def process_single_image(
        self,
        image_path: str,
        custom_prompt: str | None = None,
        custom_max_tokens: int | None = None,
        field_list: list[str] | None = None,
    ) -> dict:
        """Process single image with document-aware extraction using InternVL3.

        Args:
            image_path: Path to the image to process.
            custom_prompt: Pre-built extraction prompt (skips detection).
            custom_max_tokens: Override max_new_tokens for generation.
            field_list: Document-specific fields to extract. Falls back to
                self.field_list when None (standalone / no-detection path).
        """

        try:
            from pathlib import Path

            start_time = time.time()

            # Resolve field list: explicit param > self.field_list default
            document_fields = field_list if field_list is not None else self.field_list

            # Memory cleanup
            emergency_cleanup(verbose=False)

            # Use custom prompt if provided, otherwise generate from schema
            if custom_prompt:
                prompt = custom_prompt
                document_type = "CUSTOM"  # Indicate custom prompt usage
            else:
                # Get document-aware prompt using PROPER model-based detection
                detection_result = self.detect_and_classify_document(
                    image_path, verbose=self.debug
                )
                detected_type = detection_result.get(
                    "document_type", self._fallback_type
                )

                # Map uppercase detection result to lowercase prompt key
                # Derived from prompt_config extraction_files (YAML-first)
                fallback_type = detected_type.lower()
                supported_types = {
                    k: k.lower() for k in self.prompt_config["extraction_files"]
                }
                document_type = supported_types.get(detected_type, fallback_type)

                # For bank statements, use vision-based structure classification
                if detected_type == "BANK_STATEMENT":
                    try:
                        document_type = self._classify_bank_structure(
                            image_path, self.debug
                        )
                    except Exception as e:
                        if self.debug:
                            print(f"⚠️ Vision classification failed: {e}")
                            print("📝 Falling back to bank_statement_flat prompt")
                        document_type = "bank_statement_flat"

                if self.debug:
                    print(f"📋 DOCUMENT DETECTION RESULT: {detection_result}")
                    print(
                        f"🎯 DETECTED DOCUMENT TYPE: '{detected_type}' → MAPPED TO: '{document_type}'"
                    )
                    print(f"📝 LOADING EXTRACTION PROMPT FOR: '{document_type}'")

                prompt = self.get_extraction_prompt(document_type=document_type)

                # Get document-specific field list for accurate processing
                # Normalize document_type by stripping structure suffix for field list lookup
                base_doc_type = strip_structure_suffixes(document_type)
                document_fields = self.document_field_lists.get(
                    base_doc_type, document_fields
                )

                if self.debug:
                    print(
                        f"📋 DOCUMENT-SPECIFIC FIELDS: {len(document_fields)} fields for '{document_type}'"
                    )
                    print(
                        f"   Fields: {document_fields[:5]}{'...' if len(document_fields) > 5 else ''}"
                    )

            if self.debug:
                # Use direct stdout to bypass Rich console completely
                import sys

                if custom_prompt:
                    sys.stdout.write("📝 Using custom YAML prompt\n")
                    sys.stdout.write(f"🔍 CUSTOM YAML PROMPT ({len(prompt)} chars):\n")
                else:
                    sys.stdout.write(
                        f"📝 Generated prompt for {self.field_count} fields\n"
                    )
                    sys.stdout.write(
                        f"   Fields: {self.field_list[:3]}{'...' if len(self.field_list) > 3 else ''}\n"
                    )
                    sys.stdout.write(
                        f"🔍 DOCUMENT-AWARE PROMPT ({len(prompt)} chars):\n"
                    )
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.write(prompt + "\n")
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.flush()

            # Load and preprocess image using InternVL3 pipeline
            pixel_values = self.image_preprocessor.load_image(image_path, self.model)

            # Note: load_image() already handles device placement and dtype conversion
            # based on the actual model requirements, so no additional conversion needed here

            # Prepare question for InternVL3
            question = f"<image>\n{prompt}"

            # Use custom max_tokens if provided (for YAML prompts)
            generation_config = self.generation_config.copy()
            if custom_max_tokens:
                generation_config["max_new_tokens"] = custom_max_tokens

            if self.debug:
                print(f"🖼️  Input tensor shape: {pixel_values.shape}")
                print(
                    f"💭 Generating with max_new_tokens={generation_config['max_new_tokens']}"
                )

            # Generate response using InternVL3 chat method for extraction
            generation_config["is_detection"] = (
                False  # Enable full token limits for extraction
            )
            response = self._resilient_generate(
                pixel_values, question, **generation_config
            )

            if self.debug:
                # Use direct stdout to bypass Rich console completely
                import sys

                sys.stdout.write(f"📄 RAW MODEL RESPONSE ({len(response)} chars):\n")
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.write(response + "\n")
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.flush()

            # Parse response using hybrid parser (handles both JSON and plain text)
            from common.extraction_parser import hybrid_parse_response

            extracted_data = hybrid_parse_response(
                response, expected_fields=document_fields
            )

            # Apply ExtractionCleaner for value normalization (🧹 CLEANER CALLED output)
            cleaned_data = {}
            for field in document_fields:
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
                for field in document_fields:
                    value = extracted_data.get(field, "NOT_FOUND")
                    status = "✅" if value != "NOT_FOUND" else "❌"
                    sys.stdout.write(f'  {status} {field}: "{value}"\n')
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.flush()

                found_fields = [
                    k for k, v in extracted_data.items() if v != "NOT_FOUND"
                ]
                print(f"✅ Extracted {len(found_fields)}/{len(document_fields)} fields")
                if found_fields:
                    print(
                        f"   Found: {found_fields[:3]}{'...' if len(found_fields) > 3 else ''}"
                    )

            # Calculate metrics
            extracted_fields_count = len(
                [k for k in extracted_data.keys() if k in document_fields]
            )
            document_field_count = len(document_fields)
            response_completeness = (
                extracted_fields_count / document_field_count
                if document_field_count > 0
                else 0
            )
            content_coverage = (
                extracted_fields_count / document_field_count
                if document_field_count > 0
                else 0
            )

            # Cleanup with V100 optimizations
            del pixel_values
            emergency_cleanup(verbose=self.debug)

            return {
                "image_name": Path(image_path).name,
                "extracted_data": extracted_data,
                "raw_response": response,
                "processing_time": time.time() - start_time,
                "response_completeness": response_completeness,
                "content_coverage": content_coverage,
                "extracted_fields_count": extracted_fields_count,
                "field_count": document_field_count,
                "document_type": document_type,
            }

        except Exception as e:
            if self.debug:
                print(f"❌ Error processing {image_path}: {e}")
                import traceback

                traceback.print_exc()

            # Return error result with dynamic fields
            # document_fields is set before the try body, so always available
            error_fields = field_list if field_list is not None else self.field_list
            return {
                "image_name": Path(image_path).name,
                "extracted_data": {field: "NOT_FOUND" for field in error_fields},
                "raw_response": f"Error: {str(e)}",
                "processing_time": 0,
                "response_completeness": 0,
                "content_coverage": 0,
                "extracted_fields_count": 0,
                "field_count": len(error_fields),
            }

    # ========================================================================
    # BATCH INFERENCE METHODS - Uses model.batch_chat() for GPU utilization
    # ========================================================================

    def _load_detection_config(self):
        """Load detection prompt and config from YAML (cached after first call).

        Returns:
            Tuple of (detection_prompt, max_tokens, detection_config)
        """
        from pathlib import Path

        import yaml

        if self.prompt_config:
            detection_path = Path(self.prompt_config["detection_file"])
            detection_key = self.prompt_config["detection_key"]
        else:
            detection_path = Path("prompts/document_type_detection.yaml")
            detection_key = "detection"

        with detection_path.open("r") as f:
            detection_config = yaml.safe_load(f)

        detection_prompt = detection_config["prompts"][detection_key]["prompt"]
        max_tokens = detection_config.get("settings", {}).get("max_new_tokens", 50)

        return detection_prompt, max_tokens, detection_config

    def batch_detect_documents(
        self, image_paths: list[str], verbose: bool = False
    ) -> list[dict]:
        """Detect document types for a batch of images using model.batch_chat().

        Concatenates pixel_values from all images and calls batch_chat() once
        with the same detection prompt for all images.

        Args:
            image_paths: List of image file paths to classify
            verbose: Whether to show detailed output

        Returns:
            List of classification_info dicts (same format as detect_and_classify_document)
        """
        if not image_paths:
            return []

        detection_prompt, max_tokens, detection_config = self._load_detection_config()

        if verbose:
            import sys

            sys.stdout.write(
                f"🔍 Batch detecting {len(image_paths)} images with batch_chat()\n"
            )
            sys.stdout.flush()

        # Load and concatenate pixel values for all images
        all_pixel_values = []
        num_patches_list = []
        for image_path in image_paths:
            pv = self.image_preprocessor.load_image(image_path, self.model)
            all_pixel_values.append(pv)
            num_patches_list.append(pv.size(0))

        pixel_values = torch.cat(all_pixel_values, dim=0)

        # Same detection prompt for all images
        questions = [detection_prompt] * len(image_paths)

        generation_config = {
            "max_new_tokens": max_tokens,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        # Call batch_chat with OOM fallback.
        # IMPORTANT: cleanup MUST happen outside the except block — see
        # _resilient_generate docstring for why.
        oom = False
        try:
            responses = self.model.batch_chat(
                self.tokenizer,
                pixel_values,
                questions,
                generation_config=generation_config,
                num_patches_list=num_patches_list,
            )
        except torch.cuda.OutOfMemoryError:
            oom = True

        if oom:
            # Outside except — traceback released, tensors can be freed
            del pixel_values, all_pixel_values, num_patches_list
            gc.collect()
            torch.cuda.empty_cache()
            if len(image_paths) <= 1:
                return [
                    self.detect_and_classify_document(image_paths[0], verbose=verbose)
                ]
            mid = len(image_paths) // 2
            if verbose:
                import sys

                sys.stdout.write(
                    f"⚠️ OOM in batch detection — splitting {len(image_paths)} -> {mid} + {len(image_paths) - mid}\n"
                )
                sys.stdout.flush()
            r1 = self.batch_detect_documents(image_paths[:mid], verbose=verbose)
            r2 = self.batch_detect_documents(image_paths[mid:], verbose=verbose)
            return r1 + r2

        # Parse each response into classification_info
        results = []
        for i, response in enumerate(responses):
            # Truncate if recursion detected
            if self._detect_recursion_pattern(response):
                response = response[:200]

            document_type = self._parse_document_type_response(
                response, detection_config
            )

            if verbose:
                import sys
                from pathlib import Path as _Path

                sys.stdout.write(
                    f"  [{i + 1}/{len(image_paths)}] {_Path(image_paths[i]).name}: {document_type}\n"
                )
                sys.stdout.flush()

            results.append(
                {
                    "document_type": document_type,
                    "confidence": 1.0,
                    "raw_response": response,
                    "prompt_used": "batch_detection",
                }
            )

        # Cleanup
        del pixel_values, all_pixel_values
        emergency_cleanup(verbose=False)

        return results

    def batch_extract_documents(
        self,
        image_paths: list[str],
        classification_infos: list[dict],
        verbose: bool = False,
    ) -> list[dict]:
        """Extract fields from a batch of images using model.batch_chat().

        Supports different extraction prompts per image (batch_chat handles
        different questions natively). NOT used for bank statements (which
        require multi-turn via adapter).

        Args:
            image_paths: List of image file paths
            classification_infos: List of classification dicts from batch_detect_documents
            verbose: Whether to show detailed output

        Returns:
            List of extraction result dicts (same format as process_document_aware)
        """
        from pathlib import Path

        from common.extraction_parser import hybrid_parse_response
        from common.model_config import get_max_new_tokens

        if not image_paths:
            return []

        if verbose:
            import sys

            sys.stdout.write(
                f"📊 Batch extracting {len(image_paths)} images with batch_chat()\n"
            )
            sys.stdout.flush()

        # Load pixel values and build per-image prompts
        all_pixel_values = []
        num_patches_list = []
        questions = []
        field_lists_per_image = []
        max_tokens_needed = 0

        for image_path, classification_info in zip(
            image_paths, classification_infos, strict=False
        ):
            # Load image
            pv = self.image_preprocessor.load_image(image_path, self.model)
            all_pixel_values.append(pv)
            num_patches_list.append(pv.size(0))

            # Build per-image extraction prompt (same logic as process_document_aware)
            document_type = classification_info["document_type"].lower()

            # Get document-specific field list
            doc_type_fields = dict(self.document_field_lists)
            if "bank_statement" in doc_type_fields:
                doc_type_fields["bank_statement_flat"] = doc_type_fields[
                    "bank_statement"
                ]
                doc_type_fields["bank_statement_date_grouped"] = doc_type_fields[
                    "bank_statement"
                ]

            doc_field_list = doc_type_fields.get(
                document_type, doc_type_fields.get("invoice", self.field_list)
            )
            field_lists_per_image.append(doc_field_list)

            # Build extraction prompt
            extraction_prompt = self._get_batch_extraction_prompt(document_type)
            questions.append(extraction_prompt)

            # Track max tokens needed
            base_doc_type = strip_structure_suffixes(document_type)
            tokens = get_max_new_tokens(
                field_count=len(doc_field_list), document_type=base_doc_type
            )
            max_tokens_needed = max(max_tokens_needed, tokens)

        pixel_values = torch.cat(all_pixel_values, dim=0)

        generation_config = {
            "max_new_tokens": max_tokens_needed,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        # Call batch_chat with OOM fallback.
        # IMPORTANT: cleanup MUST happen outside the except block — see
        # _resilient_generate docstring for why.
        oom = False
        try:
            responses = self.model.batch_chat(
                self.tokenizer,
                pixel_values,
                questions,
                generation_config=generation_config,
                num_patches_list=num_patches_list,
            )
        except torch.cuda.OutOfMemoryError:
            oom = True

        if oom:
            # Outside except — traceback released, tensors can be freed
            del pixel_values, all_pixel_values, num_patches_list
            gc.collect()
            torch.cuda.empty_cache()
            if len(image_paths) <= 1:
                return [
                    self.process_document_aware(
                        image_paths[0], classification_infos[0], verbose=verbose
                    )
                ]
            mid = len(image_paths) // 2
            if verbose:
                import sys

                sys.stdout.write(
                    f"⚠️ OOM in batch extraction — splitting {len(image_paths)} -> {mid} + {len(image_paths) - mid}\n"
                )
                sys.stdout.flush()
            r1 = self.batch_extract_documents(
                image_paths[:mid], classification_infos[:mid], verbose=verbose
            )
            r2 = self.batch_extract_documents(
                image_paths[mid:], classification_infos[mid:], verbose=verbose
            )
            return r1 + r2

        # Parse responses and clean extracted data
        results = []
        for i, response in enumerate(responses):
            image_path = image_paths[i]
            doc_field_list = field_lists_per_image[i]
            document_type = classification_infos[i]["document_type"]

            # Truncate if recursion detected
            if self._detect_recursion_pattern(response):
                response = response[:2000]

            # Parse response
            extracted_data = hybrid_parse_response(
                response, expected_fields=doc_field_list
            )

            # Apply ExtractionCleaner
            cleaned_data = {}
            for field_name in doc_field_list:
                raw_value = extracted_data.get(field_name, "NOT_FOUND")
                if raw_value != "NOT_FOUND":
                    cleaned_data[field_name] = self.cleaner.clean_field_value(
                        field_name, raw_value
                    )
                else:
                    cleaned_data[field_name] = "NOT_FOUND"

            extracted_fields_count = sum(
                1 for v in cleaned_data.values() if v != "NOT_FOUND"
            )
            document_field_count = len(doc_field_list)

            if verbose:
                import sys

                sys.stdout.write(
                    f"  [{i + 1}/{len(image_paths)}] {Path(image_path).name}: "
                    f"{extracted_fields_count}/{document_field_count} fields\n"
                )
                sys.stdout.flush()

            results.append(
                {
                    "image_name": Path(image_path).name,
                    "extracted_data": cleaned_data,
                    "raw_response": response,
                    "processing_time": 0,  # Set by caller
                    "response_completeness": extracted_fields_count
                    / document_field_count
                    if document_field_count
                    else 0,
                    "content_coverage": extracted_fields_count / document_field_count
                    if document_field_count
                    else 0,
                    "extracted_fields_count": extracted_fields_count,
                    "field_count": document_field_count,
                    "document_type": document_type,
                }
            )

        # Cleanup
        del pixel_values, all_pixel_values
        emergency_cleanup(verbose=False)

        return results

    def _get_batch_extraction_prompt(self, document_type: str) -> str:
        """Get extraction prompt for batch inference.

        Uses prompt_config (single source of truth).
        """
        doc_type_upper = strip_structure_suffixes(document_type).upper()
        extraction_files = self.prompt_config["extraction_files"]
        if doc_type_upper not in extraction_files:
            raise ValueError(
                f"No extraction file configured for '{doc_type_upper}'. "
                f"Available: {list(extraction_files.keys())}. "
                f"Add it to prompt_config['extraction_files']."
            )
        extraction_file = extraction_files[doc_type_upper]
        extraction_keys = self.prompt_config.get("extraction_keys", {})
        extraction_key = extraction_keys.get(doc_type_upper, document_type)

        from pathlib import Path

        from common.simple_prompt_loader import SimplePromptLoader

        loader = SimplePromptLoader()
        return loader.load_prompt(Path(extraction_file).name, extraction_key)

    def get_model_info(self) -> dict:
        """Get information about the loaded model for debugging."""
        if self.model is None:
            return {"status": "not_loaded", "model_path": self.model_path}

        return {
            "status": "loaded",
            "model_path": self.model_path,
            "device": str(self.model.device),
            "is_8b_model": self.is_8b_model,
            "field_count": self.field_count,
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
        }
