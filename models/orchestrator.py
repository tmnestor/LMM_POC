"""DocumentOrchestrator — unified document extraction pipeline.

Replaces BaseDocumentProcessor + SimpleDocumentProcessor with a single
class that uses composition (has-a ModelBackend) instead of inheritance.

Satisfies the DocumentProcessor protocol from models/protocol.py.
"""

from __future__ import annotations

import gc
import sys
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from PIL import Image

from common.extraction_cleaner import ExtractionCleaner
from common.extraction_parser import hybrid_parse_response
from common.field_schema import get_field_schema
from common.pipeline_config import strip_structure_suffixes
from common.prompt_catalog import PromptCatalog
from models.backend import BatchInference, GenerationParams, ModelBackend

if TYPE_CHECKING:
    from common.app_config import AppConfig


class DocumentOrchestrator:
    """Unified document extraction orchestrator.

    Owns all shared logic: detection, classification, prompt resolution,
    extraction, parsing, cleaning, OOM recovery, and batch routing.

    The backend (ModelBackend) provides only raw generate() / generate_batch().

    Attributes:
        model: Underlying model object (delegates to backend).
        tokenizer: Tokenizer / processor (delegates to backend).
        batch_size: Current batch size for inference.
    """

    def __init__(
        self,
        backend: ModelBackend,
        *,
        field_list: list[str],
        prompt_config: dict[str, Any],
        field_definitions: dict[str, list[str]] | None = None,
        debug: bool = False,
        device: str = "cuda",
        batch_size: int | None = None,
        model_type_key: str = "internvl3",
        app_config: AppConfig,
        has_oom_recovery: bool = True,
    ) -> None:
        self._backend = backend
        self.debug = debug
        self.device = device
        self._model_type_key = model_type_key
        self._has_oom_recovery = has_oom_recovery

        # Validate app_config
        self.app_config: AppConfig = app_config

        # Field configuration
        self.field_list = field_list
        self.field_count = len(field_list)

        # Validate prompt_config
        if not prompt_config:
            raise ValueError(
                "prompt_config is required -- must contain "
                "'detection_file', 'detection_key', 'extraction_files'"
            )
        self.prompt_config: dict[str, Any] = prompt_config
        missing = {"detection_file", "detection_key", "extraction_files"} - set(
            self.prompt_config
        )
        if missing:
            raise ValueError(f"prompt_config missing required keys: {missing}")

        # Read fallback_type from detection YAML settings
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

        # Resolve prompt YAML filename from extraction_files
        extraction_files = self.prompt_config["extraction_files"]
        if not extraction_files:
            raise ValueError(
                "prompt_config['extraction_files'] is empty -- "
                "must map document types to extraction YAML paths"
            )
        self._prompt_yaml = Path(next(iter(extraction_files.values()))).name

        # Extraction cleaner for value normalisation
        self.cleaner = ExtractionCleaner(debug=debug)

        # Document-specific field lists
        self.document_field_lists: dict[str, list[str]] = (
            field_definitions
            if field_definitions is not None
            else get_field_schema().get_all_doc_type_fields()
        )

        # Batch processing
        self._configure_batch_processing(model_type_key, batch_size)

        # Generation config
        self._configure_generation()

        if self.debug:
            print(
                f"DocumentOrchestrator initialized: "
                f"{self.field_count} fields, batch_size={self.batch_size}, "
                f"model_type={model_type_key}"
            )

    # -- Protocol-required attributes ------------------------------------------

    @property
    def model(self) -> Any:
        """Underlying model object (for DocumentProcessor protocol)."""
        return self._backend.model

    @property
    def tokenizer(self) -> Any:
        """Tokenizer / processor (for DocumentProcessor protocol)."""
        return self._backend.processor

    @property
    def supports_batch(self) -> bool:
        """Whether the backend supports batched inference."""
        return isinstance(self._backend, BatchInference)

    # -- Batch processing config -----------------------------------------------

    def _configure_batch_processing(
        self, model_type_key: str, batch_size: int | None
    ) -> None:
        """Configure batch processing parameters."""
        if batch_size is not None:
            self.batch_size = max(1, batch_size)
            if self.debug:
                print(f"Using manual batch size: {self.batch_size}")
        else:
            from common.gpu_memory import get_available_memory

            available_memory = get_available_memory(self.device)
            self.batch_size = self.app_config.get_auto_batch_size(
                model_type_key, available_memory
            )
            if self.debug:
                print(
                    f"Auto-detected batch size: {self.batch_size} "
                    f"(GPU Memory: {available_memory:.1f}GB)"
                )

    # -- Generation config -----------------------------------------------------

    def _configure_generation(self) -> None:
        """Load generation hyper-parameters from AppConfig."""
        self.gen_config: dict[str, Any] = self.app_config.get_generation_config(
            self._model_type_key
        )

        self.fallback_max_tokens = max(
            int(self.gen_config.get("max_new_tokens_base", 512)),
            self.field_count * int(self.gen_config.get("max_new_tokens_per_field", 64)),
        )

        if self.debug:
            print(
                f"Generation config: max_new_tokens={self.fallback_max_tokens}, "
                f"do_sample={self.gen_config.get('do_sample', False)}"
            )

    def _calculate_max_tokens(self, field_count: int, document_type: str) -> int:
        """Calculate token budget based on field count and document type."""
        base = int(self.gen_config.get("max_new_tokens_base", 512))
        per_field = int(self.gen_config.get("max_new_tokens_per_field", 64))
        tokens = base + (field_count * per_field)

        if document_type == "bank_statement":
            tokens = max(tokens, 1500)
        return tokens

    # -- Image loading ---------------------------------------------------------

    def load_document_image(self, image_path: str) -> Image.Image:
        """Load document image with error handling."""
        try:
            return Image.open(image_path)
        except Exception as e:
            if self.debug:
                print(f"Error loading image {image_path}: {e}")
            raise

    # -- Prompt loading --------------------------------------------------------

    def get_extraction_prompt(self, document_type: str | None = None) -> str:
        """Get extraction prompt for *document_type* via PromptCatalog."""
        if document_type is None:
            document_type = "universal"

        if self.debug:
            print(f"Loading {document_type} prompt")

        catalog = PromptCatalog()
        try:
            return catalog.get_prompt(self._model_type_key, document_type)
        except KeyError:
            if self.debug:
                print(
                    f"Failed to load {document_type} prompt, falling back to universal"
                )
            return catalog.get_prompt(self._model_type_key, "universal")

    def get_supported_document_types(self) -> list[str]:
        """Get list of supported document types from prompt YAML."""
        return PromptCatalog().list_keys(self._model_type_key)

    # -- Document type parsing -------------------------------------------------

    def _parse_document_type_response(
        self, response: str, detection_config: dict
    ) -> str:
        """Parse document type from model response using YAML-driven type mappings."""
        response_lower = response.lower().strip()

        if self.debug:
            sys.stdout.write(f"PARSING DEBUG - Raw response: '{response}'\n")
            sys.stdout.write(f"PARSING DEBUG - Cleaned response: '{response_lower}'\n")
            sys.stdout.flush()

        # Direct mapping check (YAML type_mappings)
        type_mappings = detection_config.get("type_mappings", {})
        for variant, canonical in type_mappings.items():
            if variant.lower() in response_lower:
                if self.debug:
                    sys.stdout.write(
                        f"PARSING DEBUG - Found mapping: '{variant}' -> '{canonical}'\n"
                    )
                    sys.stdout.flush()
                return canonical

        # Fallback keyword detection
        fallback_keywords = detection_config.get("fallback_keywords", {})
        for canonical_type, keywords in fallback_keywords.items():
            if any(kw in response_lower for kw in keywords):
                if self.debug:
                    sys.stdout.write(
                        f"PARSING DEBUG - Keyword match: {canonical_type}\n"
                    )
                    sys.stdout.flush()
                return canonical_type

        # Final fallback
        fallback = detection_config.get("settings", {}).get(
            "fallback_type", self._fallback_type
        )
        if self.debug:
            sys.stdout.write(
                f"PARSING DEBUG - No matches found, using fallback: '{fallback}'\n"
            )
            sys.stdout.flush()
        return fallback

    # -- Bank structure classification -----------------------------------------

    def _classify_bank_structure(self, image_path: str, verbose: bool = False) -> str:
        """Classify bank statement structure using vision model.

        Returns:
            Structure-specific prompt key (e.g. 'bank_statement_flat')
        """
        from common.vision_bank_statement_classifier import (
            classify_bank_statement_structure_vision,
        )

        if verbose:
            print("Running vision-based structure classification for bank statement")

        structure_type = classify_bank_statement_structure_vision(
            image_path,
            model=self,
            processor=None,
            verbose=verbose,
        )

        if verbose:
            print(f"Bank statement structure: {structure_type}")
            print(f"Using prompt key: bank_statement_{structure_type}")

        return f"bank_statement_{structure_type}"

    # -- Core generate (delegates to backend with OOM recovery) ----------------

    def generate(self, image: Image.Image, prompt: str, max_tokens: int = 1024) -> str:
        """Run model inference, with optional OOM recovery.

        This is the Protocol-required generate() method.
        """
        params = GenerationParams(max_tokens=max_tokens)
        if self._has_oom_recovery:
            return self._resilient_generate(image, prompt, params)
        return self._backend.generate(image, prompt, params)

    def _resilient_generate(
        self, image: Image.Image, prompt: str, params: GenerationParams
    ) -> str:
        """Generate with OOM recovery (halve tokens and retry).

        Cleanup happens OUTSIDE the except block -- see MEMORY.md for why.
        """
        import torch

        oom = False
        try:
            return self._backend.generate(image, prompt, params)
        except torch.cuda.OutOfMemoryError:
            oom = True

        assert oom  # noqa: S101 -- always True; satisfies mypy reachability
        gc.collect()
        torch.cuda.empty_cache()
        if self.debug:
            print(
                f"OOM at {params.max_tokens} tokens, "
                f"retrying at {params.max_tokens // 2}"
            )
        retry_params = GenerationParams(
            max_tokens=params.max_tokens // 2,
            do_sample=params.do_sample,
            temperature=params.temperature,
            top_p=params.top_p,
        )
        return self._backend.generate(image, prompt, retry_params)

    # -- Detection pipeline ----------------------------------------------------

    def detect_and_classify_document(
        self, image_path: str, verbose: bool = False
    ) -> dict:
        """Detect document type by running the detection prompt through generate()."""
        try:
            detection_path = Path(self.prompt_config["detection_file"])
            detection_key = self.prompt_config["detection_key"]

            if verbose:
                sys.stdout.write(f"CONFIG DEBUG - detection_key='{detection_key}'\n")
                sys.stdout.flush()

            with detection_path.open("r") as f:
                detection_config = yaml.safe_load(f)

            detection_prompt = detection_config["prompts"][detection_key]["prompt"]
            max_tokens = detection_config.get("settings", {}).get("max_new_tokens", 50)

            if verbose:
                sys.stdout.write(f"Using document detection prompt: {detection_key}\n")
                sys.stdout.write(f"Prompt: {detection_prompt[:100]}...\n")
                sys.stdout.flush()

            image = self.load_document_image(image_path)
            response = self.generate(image, detection_prompt, max_tokens)

            if verbose:
                sys.stdout.write(f"Model response: {response}\n")
                sys.stdout.flush()

            document_type = self._parse_document_type_response(
                response, detection_config
            )

            if verbose:
                sys.stdout.write(f"Detected document type: {document_type}\n")
                sys.stdout.flush()

            return {
                "document_type": document_type,
                "confidence": 1.0,
                "raw_response": response,
                "prompt_used": detection_key,
            }

        except Exception as e:
            sys.stdout.write(f"DETECTION ERROR: {e}\n")
            sys.stdout.flush()
            if self.debug:
                sys.stdout.write("DETECTION ERROR TRACEBACK:\n")
                sys.stdout.flush()
                traceback.print_exc()

            return {
                "document_type": self._fallback_type,
                "confidence": 0.1,
                "raw_response": "",
                "prompt_used": "fallback_heuristic",
                "error": str(e),
            }

    # -- Extraction pipeline ---------------------------------------------------

    def process_document_aware(
        self, image_path: str, classification_info: dict, verbose: bool = False
    ) -> dict:
        """Process document using document-specific extraction."""
        try:
            document_type = classification_info["document_type"].lower()

            if verbose:
                print(f"Processing {document_type.upper()} document")

            # For bank statements, use vision-based structure classification
            if document_type == "bank_statement":
                try:
                    document_type = self._classify_bank_structure(image_path, verbose)
                except Exception as e:
                    if verbose:
                        print(f"Vision classification failed: {e}")
                        print("Falling back to bank_statement_flat prompt")
                    document_type = "bank_statement_flat"

            # Resolve extraction prompt
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

            catalog = PromptCatalog()
            extraction_prompt = catalog.get_prompt(self._model_type_key, extraction_key)

            if verbose:
                print(
                    f"Using {document_type} prompt: {len(extraction_prompt)} characters"
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

            # Calculate document-specific max tokens
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
                print(f"Extracted {found_fields}/{len(extracted_data)} fields")

            return result

        except Exception as e:
            if verbose:
                print(f"Error in document-aware processing: {e}")
            return self.process_single_image(image_path)

    # -- Single image processing -----------------------------------------------

    def process_single_image(
        self,
        image_path: str,
        custom_prompt: str | None = None,
        custom_max_tokens: int | None = None,
        field_list: list[str] | None = None,
    ) -> dict[str, Any]:
        """Process one document image end-to-end."""
        active_fields = field_list or self.field_list
        active_count = len(active_fields)
        start_time = time.time()
        image_name = Path(image_path).name

        try:
            if self._has_oom_recovery:
                from common.gpu_memory import release_memory

                release_memory(threshold_gb=1.0)

            image = self.load_document_image(image_path)

            prompt = custom_prompt or self.get_extraction_prompt()
            max_tokens = custom_max_tokens or self._calculate_max_tokens(
                active_count, "universal"
            )

            if self.debug:
                sys.stdout.write(f"Processing {image_name} ({active_count} fields)\n")
                sys.stdout.write(
                    f"Prompt: {len(prompt)} chars, max_tokens: {max_tokens}\n"
                )
                sys.stdout.flush()

            raw_response = self.generate(image, prompt, max_tokens)

            processing_time = time.time() - start_time

            if self.debug:
                sys.stdout.write(f"Response ({len(raw_response)} chars):\n")
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.write(raw_response + "\n")
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.flush()

            extracted_data = hybrid_parse_response(
                raw_response, expected_fields=active_fields
            )

            for field_name, value in extracted_data.items():
                extracted_data[field_name] = self.cleaner.clean_field_value(
                    field_name, value
                )

            found = sum(1 for v in extracted_data.values() if v != "NOT_FOUND")

            if self.debug:
                print(f"Extracted {found}/{active_count} fields")

            del image
            if self._has_oom_recovery:
                import torch

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

    # -- Batch methods (used by DocumentPipeline when supports_batch is True) --

    def detect_batch(self, image_paths: list[str], verbose: bool = False) -> list[dict]:
        """Batch document detection via backend.generate_batch().

        Called by DocumentPipeline only when supports_batch is True.
        No isinstance check here -- the pipeline handles routing.
        """
        if not image_paths:
            return []

        # Load detection config
        detection_path = Path(self.prompt_config["detection_file"])
        detection_key = self.prompt_config["detection_key"]

        with detection_path.open("r") as f:
            detection_config = yaml.safe_load(f)

        detection_prompt = detection_config["prompts"][detection_key]["prompt"]
        max_tokens = detection_config.get("settings", {}).get("max_new_tokens", 50)

        if verbose:
            sys.stdout.write(f"Batch detecting {len(image_paths)} images\n")
            sys.stdout.flush()

        images = [self.load_document_image(p) for p in image_paths]
        prompts = [detection_prompt] * len(image_paths)
        params = GenerationParams(max_tokens=max_tokens)

        # Safe: pipeline only calls this when supports_batch is True
        backend = self._backend
        assert isinstance(backend, BatchInference)  # noqa: S101
        responses = backend.generate_batch(images, prompts, params)

        results = []
        for i, response in enumerate(responses):
            document_type = self._parse_document_type_response(
                response, detection_config
            )
            if verbose:
                sys.stdout.write(
                    f"  [{i + 1}/{len(image_paths)}] "
                    f"{Path(image_paths[i]).name}: {document_type}\n"
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

        return results

    def extract_batch(
        self,
        image_paths: list[str],
        classification_infos: list[dict],
        verbose: bool = False,
    ) -> list[dict]:
        """Batch document extraction via backend.generate_batch().

        Called by DocumentPipeline only when supports_batch is True.
        No isinstance check here -- the pipeline handles routing.
        """
        if not image_paths:
            return []

        # Build per-image prompts and field lists
        images = []
        prompts = []
        field_lists_per_image = []
        max_tokens_needed = 0

        for image_path, classification_info in zip(
            image_paths, classification_infos, strict=False
        ):
            images.append(self.load_document_image(image_path))
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

            # Resolve extraction prompt
            extraction_prompt = self._resolve_extraction_prompt(document_type)
            prompts.append(extraction_prompt)

            # Track max tokens needed
            base_doc_type = strip_structure_suffixes(document_type)
            tokens = self._calculate_max_tokens(len(doc_field_list), base_doc_type)
            max_tokens_needed = max(max_tokens_needed, tokens)

        params = GenerationParams(max_tokens=max_tokens_needed)

        # Safe: pipeline only calls this when supports_batch is True
        backend = self._backend
        assert isinstance(backend, BatchInference)  # noqa: S101
        responses = backend.generate_batch(images, prompts, params)

        # Parse responses and clean extracted data
        results = []
        for i, response in enumerate(responses):
            doc_field_list = field_lists_per_image[i]
            document_type = classification_infos[i]["document_type"]

            extracted_data = hybrid_parse_response(
                response, expected_fields=doc_field_list
            )

            for field_name in doc_field_list:
                raw_value = extracted_data.get(field_name, "NOT_FOUND")
                if raw_value != "NOT_FOUND":
                    extracted_data[field_name] = self.cleaner.clean_field_value(
                        field_name, raw_value
                    )
                else:
                    extracted_data[field_name] = "NOT_FOUND"

            extracted_fields_count = sum(
                1 for v in extracted_data.values() if v != "NOT_FOUND"
            )
            document_field_count = len(doc_field_list)

            if verbose:
                sys.stdout.write(
                    f"  [{i + 1}/{len(image_paths)}] "
                    f"{Path(image_paths[i]).name}: "
                    f"{extracted_fields_count}/{document_field_count} fields\n"
                )
                sys.stdout.flush()

            results.append(
                {
                    "image_name": Path(image_paths[i]).name,
                    "extracted_data": extracted_data,
                    "raw_response": response,
                    "processing_time": 0,
                    "response_completeness": (
                        extracted_fields_count / document_field_count
                        if document_field_count
                        else 0
                    ),
                    "content_coverage": (
                        extracted_fields_count / document_field_count
                        if document_field_count
                        else 0
                    ),
                    "extracted_fields_count": extracted_fields_count,
                    "field_count": document_field_count,
                    "document_type": document_type,
                }
            )

        return results

    def _resolve_extraction_prompt(self, document_type: str) -> str:
        """Resolve extraction prompt for a document type."""
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

        catalog = PromptCatalog()
        return catalog.get_prompt(self._model_type_key, extraction_key)

    # -- Model info ------------------------------------------------------------

    def get_model_info(self) -> dict:
        """Return model metadata for reporting."""
        return {
            "model_type": self._model_type_key,
            "model_path": getattr(self._backend, "model_path", "unknown"),
            "batch_size": self.batch_size,
        }
