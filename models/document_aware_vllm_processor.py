"""vLLM-backed document extraction processor.

Inherits shared detection, classification, prompt resolution, and extraction
orchestration from BaseDocumentProcessor.  Only model-specific inference
(generate, token calculation, single-image processing) is implemented here.

Uses vLLM offline LLM engine with tensor parallelism for multi-GPU inference.
The vLLM engine handles memory management (PagedAttention), batching, and
tokenization internally — no manual torch.cuda.empty_cache() needed.
"""

import time
from pathlib import Path
from typing import Any, override

from PIL import Image

from common.extraction_parser import parse_extraction_response
from models.base_processor import BaseDocumentProcessor


class DocumentAwareVllmProcessor(BaseDocumentProcessor):
    """Document extraction processor backed by vLLM offline engine.

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
        model_type_key: str = "llama4scout",
    ):
        self.model_path = model_path
        # pre_loaded_model is a vllm.LLM engine
        self.llm_engine = pre_loaded_model
        self.model = pre_loaded_model  # base class compat
        self.processor = pre_loaded_processor  # None for vLLM

        if self.llm_engine is None:
            msg = (
                "DocumentAwareVllmProcessor requires a pre-loaded vLLM engine. "
                "Pass it as pre_loaded_model from the registry loader."
            )
            raise ValueError(msg)

        self._model_type_key = model_type_key

        # Shared init: validates config, loads field defs, sets batch size
        self._init_shared(
            field_list=field_list,
            prompt_config=prompt_config,
            field_definitions=field_definitions,
            debug=debug,
            device=device,
            batch_size=batch_size,
            model_type_key=model_type_key,
        )

        self._configure_generation()

        if self.debug:
            print(f"vLLM processor initialized for {self.field_count} fields")

    # -- Protocol compatibility ------------------------------------------------

    @property
    def tokenizer(self):
        """Return tokenizer for Protocol / BankStatementAdapter compatibility."""
        return getattr(self.llm_engine, "tokenizer", None)

    # -- Generation config -----------------------------------------------------

    def _configure_generation(self) -> None:
        """Load generation hyper-parameters from model_config."""
        from common.model_config import (
            INTERNVL3_GENERATION_CONFIG,
            LLAMA4SCOUT_GENERATION_CONFIG,
            QWEN3VL_GENERATION_CONFIG,
        )

        if self._model_type_key == "internvl3":
            self.gen_config = dict(INTERNVL3_GENERATION_CONFIG)
        elif self._model_type_key.startswith(("qwen3vl", "qwen35")):
            self.gen_config = dict(QWEN3VL_GENERATION_CONFIG)
        else:
            self.gen_config = dict(LLAMA4SCOUT_GENERATION_CONFIG)

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
        """Run inference via vLLM engine.

        Uses vLLM's chat API with OpenAI-compatible message format.
        The engine handles chat template application, image tokenization,
        and tensor-parallel inference internally.
        """
        import base64
        import io

        from vllm import SamplingParams

        # vLLM chat API requires a string URL, not a PIL Image object.
        # Convert to base64 data URI.
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        sampling = SamplingParams(
            max_tokens=max_tokens,
            temperature=0,
        )

        # Qwen3.5 enables thinking mode by default — disable it to avoid
        # <think>...</think> blocks in extraction output.
        chat_kwargs: dict[str, Any] = {}
        if self._model_type_key.startswith("qwen35"):
            chat_kwargs["chat_template_kwargs"] = {"enable_thinking": False}

        outputs = self.llm_engine.chat(
            messages=messages,
            sampling_params=sampling,
            use_tqdm=False,
            **chat_kwargs,
        )

        return outputs[0].outputs[0].text.strip()

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
        vLLM manages GPU memory internally — no manual cleanup needed.
        """
        active_fields = field_list or self.field_list
        active_count = len(active_fields)
        start_time = time.time()
        image_name = Path(image_path).name

        try:
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

            raw_response = self.generate(image, prompt, max_tokens)
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

            del image

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

    def get_model_info(self) -> dict:
        """Return model metadata for reporting."""
        return {
            "model_type": "llama4scout-w4a16-vllm",
            "model_path": self.model_path,
            "batch_size": self.batch_size,
        }
