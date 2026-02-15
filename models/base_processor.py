"""Base class for document extraction processors.

Provides shared logic for detection, classification, prompt resolution,
and extraction orchestration.  Subclasses implement:

    generate(image, prompt, max_tokens) -> str   # model-specific inference
    _calculate_max_tokens(field_count, doc_type) -> int

The consumer-facing Protocol (models/protocol.py) is unchanged.
BaseDocumentProcessor is purely an *implementation detail* shared by
the InternVL3 and Llama concrete processors.
"""

import abc
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from common.batch_processor import load_document_field_definitions
from common.extraction_cleaner import ExtractionCleaner
from common.model_config import get_auto_batch_size
from common.pipeline_config import strip_structure_suffixes
from common.simple_prompt_loader import SimplePromptLoader


class BaseDocumentProcessor(abc.ABC):
    """Shared implementation for vision-language document processors."""

    # -- abstract interface ----------------------------------------------------

    @abc.abstractmethod
    def generate(self, image: Image.Image, prompt: str, max_tokens: int = 1024) -> str:
        """Run model inference on *image* with *prompt*.

        This is the single method where every model truly differs.

        Args:
            image: PIL Image to process.
            prompt: Text prompt for the model.
            max_tokens: Maximum tokens to generate.

        Returns:
            Raw model response string.
        """

    @abc.abstractmethod
    def _calculate_max_tokens(self, field_count: int, document_type: str) -> int:
        """Calculate max_new_tokens for generation.

        Args:
            field_count: Number of fields to extract.
            document_type: Base document type (e.g. 'invoice').

        Returns:
            Token budget for generation.
        """

    # -- shared initialisation helpers -----------------------------------------

    def _init_shared(
        self,
        *,
        field_list: list[str],
        prompt_config: dict[str, Any],
        field_definitions: dict[str, list[str]] | None = None,
        debug: bool = False,
        device: str = "cuda",
        batch_size: int | None = None,
        model_type_key: str = "internvl3",
    ) -> None:
        """Common initialisation used by both processors.

        Must be called from the subclass ``__init__`` *after* setting
        ``self.model``, ``self.tokenizer``, etc.
        """
        self.field_list = field_list
        self.field_count = len(field_list)
        self.debug = debug
        self.device = device
        self.prompt_config = prompt_config

        # Validate prompt_config
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

        # Read fallback_type from detection YAML settings (YAML is source of truth)
        detection_path = Path(self.prompt_config["detection_file"])
        try:
            with detection_path.open() as f:
                det_cfg = yaml.safe_load(f)
            self._fallback_type = det_cfg.get("settings", {}).get("fallback_type")
            if self._fallback_type is None:
                raise ValueError(
                    f"Detection YAML {detection_path} missing settings.fallback_type"
                )
        except (OSError, yaml.YAMLError, KeyError, ValueError) as e:
            raise ValueError(
                f"Cannot read fallback_type from {detection_path}: {e}\n"
                f"Detection YAML must contain settings.fallback_type"
            ) from e

        # Resolve prompt YAML filename from extraction_files (single source of truth)
        extraction_files = self.prompt_config["extraction_files"]
        if not extraction_files:
            raise ValueError(
                "prompt_config['extraction_files'] is empty — "
                "must map document types to extraction YAML paths"
            )
        self._prompt_yaml = Path(next(iter(extraction_files.values()))).name

        # Extraction cleaner for value normalisation
        self.cleaner = ExtractionCleaner(debug=debug)

        # Document-specific field lists (single source of truth)
        self.document_field_lists: dict[str, list[str]] = (
            field_definitions
            if field_definitions is not None
            else load_document_field_definitions()
        )

        # Batch processing
        self._configure_batch_processing(model_type_key, batch_size)

    # -- shared methods --------------------------------------------------------

    def _configure_batch_processing(
        self, model_type_key: str, batch_size: int | None
    ) -> None:
        """Configure batch processing parameters."""
        if batch_size is not None:
            self.batch_size = max(1, batch_size)
            if self.debug:
                print(f"🎯 Using manual batch size: {self.batch_size}")
        else:
            from common.gpu_optimization import get_available_gpu_memory

            available_memory = get_available_gpu_memory(self.device)
            self.batch_size = get_auto_batch_size(model_type_key, available_memory)
            if self.debug:
                print(
                    f"🤖 Auto-detected batch size: {self.batch_size} "
                    f"(GPU Memory: {available_memory:.1f}GB)"
                )

    def load_document_image(self, image_path: str) -> Image.Image:
        """Load document image with error handling."""
        try:
            return Image.open(image_path)
        except Exception as e:
            if self.debug:
                print(f"❌ Error loading image {image_path}: {e}")
            raise

    def get_extraction_prompt(self, document_type: str | None = None) -> str:
        """Get extraction prompt for *document_type* via SimplePromptLoader.

        The prompt YAML filename is derived from prompt_config['extraction_files'].
        """
        if document_type is None:
            document_type = "universal"

        if self.debug:
            print(f"📝 Loading {document_type} prompt")

        loader = SimplePromptLoader()
        try:
            return loader.load_prompt(self._prompt_yaml, document_type)
        except Exception:
            if self.debug:
                print(
                    f"⚠️ Failed to load {document_type} prompt, falling back to universal"
                )
            return loader.load_prompt(self._prompt_yaml, "universal")

    def get_supported_document_types(self) -> list[str]:
        """Get list of supported document types from prompt YAML."""
        return SimplePromptLoader.get_available_prompts(self._prompt_yaml)

    def _parse_document_type_response(
        self, response: str, detection_config: dict
    ) -> str:
        """Parse document type from model response using YAML-driven type mappings.

        Uses type_mappings and fallback_keywords from the detection YAML config.
        """
        response_lower = response.lower().strip()

        if self.debug:
            sys.stdout.write(f"🔍 PARSING DEBUG - Raw response: '{response}'\n")
            sys.stdout.write(
                f"🔍 PARSING DEBUG - Cleaned response: '{response_lower}'\n"
            )
            sys.stdout.flush()

        # Direct mapping check (YAML type_mappings)
        type_mappings = detection_config.get("type_mappings", {})
        for variant, canonical in type_mappings.items():
            if variant.lower() in response_lower:
                if self.debug:
                    sys.stdout.write(
                        f"✅ PARSING DEBUG - Found mapping: '{variant}' -> '{canonical}'\n"
                    )
                    sys.stdout.flush()
                return canonical

        # Fallback keyword detection (YAML fallback_keywords)
        fallback_keywords = detection_config.get("fallback_keywords", {})
        for canonical_type, keywords in fallback_keywords.items():
            if any(kw in response_lower for kw in keywords):
                if self.debug:
                    sys.stdout.write(
                        f"✅ PARSING DEBUG - Keyword match: {canonical_type}\n"
                    )
                    sys.stdout.flush()
                return canonical_type

        # Final fallback — use YAML-loaded value from _init_shared
        fallback = detection_config.get("settings", {}).get(
            "fallback_type", self._fallback_type
        )
        if self.debug:
            sys.stdout.write(
                f"❌ PARSING DEBUG - No matches found, using fallback: '{fallback}'\n"
            )
            sys.stdout.flush()
        return fallback

    def detect_and_classify_document(
        self, image_path: str, verbose: bool = False
    ) -> dict:
        """Detect document type by running the detection prompt through generate().

        Compatible with BatchDocumentProcessor workflow.
        """
        try:
            # Load detection config
            if self.prompt_config:
                detection_path = Path(self.prompt_config["detection_file"])
                detection_key = self.prompt_config["detection_key"]
            else:
                detection_path = Path("prompts/document_type_detection.yaml")
                detection_key = "detection"

            if verbose:
                sys.stdout.write(f"🔧 CONFIG DEBUG - detection_key='{detection_key}'\n")
                sys.stdout.flush()

            with detection_path.open("r") as f:
                detection_config = yaml.safe_load(f)

            detection_prompt = detection_config["prompts"][detection_key]["prompt"]
            max_tokens = detection_config.get("settings", {}).get("max_new_tokens", 50)

            if verbose:
                sys.stdout.write(
                    f"🔍 Using document detection prompt: {detection_key}\n"
                )
                sys.stdout.write(f"📝 Prompt: {detection_prompt[:100]}...\n")
                sys.stdout.flush()

            # Load image and call model-specific generate()
            image = self.load_document_image(image_path)
            response = self.generate(image, detection_prompt, max_tokens)

            if verbose:
                sys.stdout.write(f"🤖 Model response: {response}\n")
                sys.stdout.flush()

            # Parse document type from response
            document_type = self._parse_document_type_response(
                response, detection_config
            )

            if verbose:
                sys.stdout.write(f"✅ Detected document type: {document_type}\n")
                sys.stdout.flush()

            return {
                "document_type": document_type,
                "confidence": 1.0,
                "raw_response": response,
                "prompt_used": detection_key,
            }

        except Exception as e:
            sys.stdout.write(f"❌ DETECTION ERROR: {e}\n")
            sys.stdout.flush()
            if self.debug:
                sys.stdout.write("❌ DETECTION ERROR TRACEBACK:\n")
                sys.stdout.flush()
                traceback.print_exc()

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
        """Process document using document-specific extraction based on detected type.

        Compatible with BatchDocumentProcessor workflow.
        """
        try:
            document_type = classification_info["document_type"].lower()

            if verbose:
                print(f"📊 Processing {document_type.upper()} document")

            # For bank statements, use vision-based structure classification
            if document_type == "bank_statement":
                try:
                    document_type = self._classify_bank_structure(image_path, verbose)
                except Exception as e:
                    if verbose:
                        print(f"⚠️ Vision classification failed: {e}")
                        print("📝 Falling back to bank_statement_flat prompt")
                    document_type = "bank_statement_flat"

            # Get document-specific prompt using prompt_config
            doc_type_upper = strip_structure_suffixes(document_type).upper()
            extraction_files = self.prompt_config["extraction_files"]
            if doc_type_upper not in extraction_files:
                raise ValueError(
                    f"No extraction file configured for '{doc_type_upper}'. "
                    f"Available: {list(extraction_files.keys())}. "
                    f"Add it to prompt_config['extraction_files']."
                )
            extraction_file = extraction_files[doc_type_upper]

            # Derive extraction key
            extraction_keys = self.prompt_config.get("extraction_keys", {})
            if doc_type_upper in extraction_keys:
                extraction_key = extraction_keys[doc_type_upper]
            else:
                extraction_key = document_type

            # For bank statements: append structure suffix if missing
            if (
                document_type.startswith("bank_statement")
                and doc_type_upper == "BANK_STATEMENT"
            ):
                if (
                    "_flat" not in extraction_key
                    and "_date_grouped" not in extraction_key
                ):
                    if "_flat" in document_type:
                        extraction_key = f"{extraction_key}_flat"
                    elif "_date_grouped" in document_type:
                        extraction_key = f"{extraction_key}_date_grouped"

            loader = SimplePromptLoader()
            extraction_prompt = loader.load_prompt(
                Path(extraction_file).name, extraction_key
            )

            if verbose:
                print(
                    f"📝 Using {document_type} prompt: {len(extraction_prompt)} characters"
                )

            # Resolve document-specific field list
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

            # Calculate document-specific max tokens (delegated to subclass)
            base_doc_type = strip_structure_suffixes(document_type)
            doc_specific_tokens = self._calculate_max_tokens(
                len(doc_field_list), base_doc_type
            )

            # Process with document-specific settings
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
            return self.process_single_image(image_path)
