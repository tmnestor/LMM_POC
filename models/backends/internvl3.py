"""InternVL3 model backend.

Extracted from document_aware_internvl3_processor.py.
Uses the InternVL3 model.chat() / model.batch_chat() API.
"""

import gc
from typing import Any

import torch
from PIL import Image

from models.backend import BatchInference, GenerationParams, ModelBackend
from models.internvl3_image_preprocessor import InternVL3ImagePreprocessor


class InternVL3Backend:
    """Backend for InternVL3 models using .chat() / .batch_chat() API.

    Implements both ModelBackend and BatchInference protocols.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        max_tiles: int = 12,
        min_tiles: int | None = None,
        debug: bool = False,
    ) -> None:
        self.model = model
        self.processor = tokenizer  # InternVL3 uses AutoTokenizer
        self._debug = debug

        self.image_preprocessor = InternVL3ImagePreprocessor(
            max_tiles=max_tiles, debug=debug, min_tiles=min_tiles
        )

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        params: GenerationParams,
    ) -> str:
        """Run InternVL3 inference via model.chat()."""
        pixel_values = self.image_preprocessor.load_image_from_pil(image, self.model)
        model_device = InternVL3ImagePreprocessor.get_model_device(self.model)
        if pixel_values.device != model_device:
            pixel_values = pixel_values.to(model_device)

        gen_config = {
            "max_new_tokens": params.max_tokens,
            "do_sample": params.do_sample,
            "pad_token_id": self.processor.eos_token_id,
        }

        response = self.model.chat(
            self.processor,
            pixel_values,
            prompt,
            generation_config=gen_config,
            history=None,
            return_history=False,
        )

        del pixel_values
        torch.cuda.empty_cache()
        return response

    def generate_batch(
        self,
        images: list[Image.Image],
        prompts: list[str],
        params: GenerationParams,
    ) -> list[str]:
        """Run batched InternVL3 inference via model.batch_chat().

        Concatenates pixel_values and passes num_patches_list.
        Falls back to sequential on OOM (halving batch).
        """
        if not images:
            return []

        all_pixel_values = []
        num_patches_list = []
        for image in images:
            pv = self.image_preprocessor.load_image_from_pil(image, self.model)
            all_pixel_values.append(pv)
            num_patches_list.append(pv.size(0))

        pixel_values = torch.cat(all_pixel_values, dim=0)

        gen_config = {
            "max_new_tokens": params.max_tokens,
            "do_sample": params.do_sample,
            "pad_token_id": self.processor.eos_token_id,
        }

        # OOM fallback with cleanup outside except block
        oom = False
        try:
            responses = self.model.batch_chat(
                self.processor,
                pixel_values,
                prompts,
                generation_config=gen_config,
                num_patches_list=num_patches_list,
            )
        except torch.cuda.OutOfMemoryError:
            oom = True

        if oom:
            del pixel_values, all_pixel_values, num_patches_list
            gc.collect()
            torch.cuda.empty_cache()

            if len(images) <= 1:
                return [self.generate(images[0], prompts[0], params)]

            mid = len(images) // 2
            r1 = self.generate_batch(images[:mid], prompts[:mid], params)
            r2 = self.generate_batch(images[mid:], prompts[mid:], params)
            return r1 + r2

        del pixel_values, all_pixel_values
        torch.cuda.empty_cache()
        return responses


# Verify protocol compliance at import time
assert isinstance(InternVL3Backend, type)  # noqa: S101
_dummy_check: type[ModelBackend] = InternVL3Backend  # type: ignore[assignment]
_dummy_batch: type[BatchInference] = InternVL3Backend  # type: ignore[assignment]
