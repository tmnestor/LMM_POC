"""Intermediate base class for standard HuggingFace VLM processors.

Captures the ~140 LOC of boilerplate that was copy-pasted across
Llama4, Qwen3-VL, Qwen3.5, Nemotron, and vLLM processors.

Subclasses MUST override:
    model_type_key  -- class attribute, e.g. "gemma4"
    _load_model()   -- load self.model and self.processor from disk
    generate()      -- model-specific inference

Subclasses MAY override:
    has_oom_recovery  -- False to skip OOM retry (vLLM)
    tokenizer_attr    -- attribute name on self.processor for the tokenizer

InternVL3 and Llama (3.2) remain direct BaseDocumentProcessor subclasses
because their inference APIs are fundamentally different.
"""

from __future__ import annotations

import abc
import gc
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, override

if TYPE_CHECKING:
    from PIL import Image

    from common.app_config import AppConfig

import torch

from common.extraction_parser import parse_extraction_response
from common.model_config import get_generation_config
from models.base_processor import BaseDocumentProcessor


class SimpleDocumentProcessor(BaseDocumentProcessor):
    """Concrete base for standard HuggingFace VLM processors.

    Provides shared __init__, tokenizer property, _configure_generation,
    _calculate_max_tokens, process_single_image, _resilient_generate,
    and get_model_info.
    """

    # -- MUST override in subclass --
    model_type_key: str  # No default -- forces subclass to declare

    # -- MAY override in subclass --
    has_oom_recovery: bool = True
    tokenizer_attr: str = "tokenizer"

    # -- Model loading (MUST override) --

    @abc.abstractmethod
    def _load_model(self) -> None:
        """Load self.model and self.processor from disk."""

    # -- Shared __init__ ---------------------------------------------------

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
        model_type_key: str | None = None,
        app_config: AppConfig | None = None,
    ):
        self.model_path = model_path
        self.model = pre_loaded_model
        self.processor = pre_loaded_processor

        # Allow runtime override (vLLM serves multiple architectures)
        if model_type_key is not None:
            self._runtime_model_type_key = model_type_key

        if self.has_oom_recovery:
            from common.gpu_optimization import configure_cuda_memory_allocation

            configure_cuda_memory_allocation()

        if self.model is None:
            self._load_model()

        self._init_shared(
            field_list=field_list,
            prompt_config=prompt_config,
            field_definitions=field_definitions,
            debug=debug,
            device=device,
            batch_size=batch_size,
            model_type_key=self._effective_model_type_key,
            app_config=app_config,
        )

        self._configure_generation()

        if self.debug:
            print(
                f"{self.__class__.__name__} initialized for {self.field_count} fields"
            )

    @property
    def _effective_model_type_key(self) -> str:
        """Return runtime override if set, else class-level model_type_key."""
        return getattr(self, "_runtime_model_type_key", self.model_type_key)

    # -- Protocol compatibility --------------------------------------------

    @property
    def tokenizer(self):
        """Return tokenizer for Protocol / UnifiedBankExtractor compatibility."""
        if self.processor is not None:
            return getattr(self.processor, self.tokenizer_attr, None)
        return None

    # -- Generation config -------------------------------------------------

    def _configure_generation(self) -> None:
        """Load generation hyper-parameters from model_config registry."""
        if self.app_config is not None:
            self.gen_config: dict[str, Any] = self.app_config.get_generation_config(
                self._effective_model_type_key
            )
        else:
            self.gen_config = get_generation_config(self._effective_model_type_key)

        self.fallback_max_tokens = max(
            int(self.gen_config.get("max_new_tokens_base", 512)),
            self.field_count * int(self.gen_config.get("max_new_tokens_per_field", 64)),
        )

        if self.debug:
            print(
                f"Generation config: max_new_tokens={self.fallback_max_tokens}, "
                f"do_sample={self.gen_config.get('do_sample', False)}"
            )

    # -- Abstract method implementations -----------------------------------

    @override
    def _calculate_max_tokens(self, field_count: int, document_type: str) -> int:
        """Calculate token budget based on field count and document type."""
        base = int(self.gen_config.get("max_new_tokens_base", 512))
        per_field = int(self.gen_config.get("max_new_tokens_per_field", 64))
        tokens = base + (field_count * per_field)

        if document_type == "bank_statement":
            tokens = max(tokens, 1500)
        return tokens

    @override
    def process_single_image(
        self,
        image_path: str,
        custom_prompt: str | None = None,
        custom_max_tokens: int | None = None,
        field_list: list[str] | None = None,
    ) -> dict[str, Any]:
        """Process one document image end-to-end.

        Called by the inherited process_document_aware() method.
        """
        active_fields = field_list or self.field_list
        active_count = len(active_fields)
        start_time = time.time()
        image_name = Path(image_path).name

        try:
            if self.has_oom_recovery:
                from common.gpu_optimization import handle_memory_fragmentation

                handle_memory_fragmentation(threshold_gb=1.0, aggressive=True)

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

            if self.has_oom_recovery:
                raw_response = self._resilient_generate(image, prompt, max_tokens)
            else:
                raw_response = self.generate(image, prompt, max_tokens)

            processing_time = time.time() - start_time

            if self.debug:
                sys.stdout.write(f"Response ({len(raw_response)} chars):\n")
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.write(raw_response + "\n")
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.flush()

            extracted_data = parse_extraction_response(
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
            if self.has_oom_recovery and torch.cuda.is_available():
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

    # -- OOM recovery ------------------------------------------------------

    def _resilient_generate(
        self, image: Image.Image, prompt: str, max_tokens: int
    ) -> str:
        """Generate with OOM recovery (halve tokens and retry).

        Cleanup happens OUTSIDE the except block -- see MEMORY.md for why.
        """
        oom = False
        try:
            return self.generate(image, prompt, max_tokens)
        except torch.cuda.OutOfMemoryError:
            oom = True

        assert oom  # noqa: S101 -- always True; satisfies mypy reachability
        gc.collect()
        torch.cuda.empty_cache()
        if self.debug:
            print(f"OOM at {max_tokens} tokens, retrying at {max_tokens // 2}")
        return self.generate(image, prompt, max_tokens // 2)

    # -- Model info --------------------------------------------------------

    def get_model_info(self) -> dict:
        """Return model metadata for reporting."""
        return {
            "model_type": self._effective_model_type_key,
            "model_path": self.model_path,
            "batch_size": self.batch_size,
        }
