"""vLLM-backed document extraction processor.

Inherits shared orchestration from SimpleDocumentProcessor.
Only model-specific inference is implemented here.

Uses vLLM offline LLM engine with tensor parallelism for multi-GPU inference.
The vLLM engine handles memory management (PagedAttention), batching, and
tokenization internally -- no manual torch.cuda.empty_cache() needed.
"""

from typing import Any, override

from PIL import Image

from models.simple_processor import SimpleDocumentProcessor


class DocumentAwareVllmProcessor(SimpleDocumentProcessor):
    """Document extraction processor backed by vLLM offline engine."""

    model_type_key = "llama4scout"  # Default, overridden at construction
    has_oom_recovery = False

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
        super().__init__(
            field_list=field_list,
            model_path=model_path,
            device=device,
            debug=debug,
            batch_size=batch_size,
            pre_loaded_model=pre_loaded_model,
            pre_loaded_processor=pre_loaded_processor,
            prompt_config=prompt_config,
            field_definitions=field_definitions,
            model_type_key=model_type_key,
        )
        self.llm_engine = self.model

        if self.llm_engine is None:
            msg = (
                "DocumentAwareVllmProcessor requires a pre-loaded vLLM engine. "
                "Pass it as pre_loaded_model from the registry loader."
            )
            raise ValueError(msg)

    @property
    def tokenizer(self):
        """Return tokenizer from the vLLM engine."""
        return getattr(self.llm_engine, "tokenizer", None)

    @override
    def _load_model(self) -> None:
        """vLLM processor requires a pre-loaded engine -- never called."""
        msg = (
            "DocumentAwareVllmProcessor requires a pre-loaded vLLM engine. "
            "Pass it as pre_loaded_model from the registry loader."
        )
        raise ValueError(msg)

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

        chat_kwargs: dict[str, Any] = {}
        if self._effective_model_type_key.startswith(("qwen35", "gemma4")):
            chat_kwargs["chat_template_kwargs"] = {"enable_thinking": False}

        outputs = self.llm_engine.chat(
            messages=messages,
            sampling_params=sampling,
            use_tqdm=False,
            **chat_kwargs,
        )

        return outputs[0].outputs[0].text.strip()
