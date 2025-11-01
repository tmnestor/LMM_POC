"""Llama Vision Multi-Turn Chat Utilities.

This module provides utilities for multi-turn conversational interactions with
Llama vision models, optimized for deterministic OCR extraction tasks.
"""

from typing import Any

from IPython.display import display
from tqdm.notebook import tqdm
from transformers import AutoProcessor, MllamaForConditionalGeneration
from transformers.image_utils import load_image


def chat_with_mllm(
    model: MllamaForConditionalGeneration,
    processor: AutoProcessor,
    prompt: str,
    images_path: list[str] | str | None = None,
    do_sample: bool = False,
    temperature: float = 0.0,
    repetition_penalty: float = 1.0,
    show_image: bool = False,
    max_new_tokens: int = 2000,
    messages: list[dict[str, Any]] | None = None,
    images: list[Any] | None = None,
) -> tuple[str, list[dict[str, Any]], list[Any]]:
    """Chat with Llama vision model in multi-turn conversation mode.

    Optimized for deterministic OCR extraction with proper generation parameters.

    Args:
        model: Loaded Llama vision model
        processor: AutoProcessor for the model
        prompt: User's text prompt
        images_path: Path(s) to image files (string or list). Defaults to None.
        do_sample: Enable sampling (if True, uses temperature). Default False for
            deterministic output.
        temperature: Sampling temperature (default 0.0 for deterministic OCR).
        repetition_penalty: Penalty for token repetition (default 1.0 = no penalty,
            important for OCR to preserve repeated data like transaction patterns).
        show_image: Display image in notebook (default False).
        max_new_tokens: Maximum tokens to generate (default 2000).
        messages: Conversation history (None or empty list for new conversation).
        images: Loaded image objects (None or empty list to load from paths).

    Returns:
        tuple: (generated_text, updated_messages, images)

    Note:
        For deterministic OCR extraction, use default settings:
        - do_sample=False (greedy decoding)
        - temperature=0.0 (not used in greedy mode, but set for clarity)
        - repetition_penalty=1.0 (preserve repeated transaction patterns)

    Example:
        >>> from common.llama_model_loader_robust import load_llama_model_robust
        >>> from common.llama_multiturn_chat import chat_with_mllm
        >>>
        >>> # Load model
        >>> model, processor = load_llama_model_robust(model_path="path/to/model")
        >>>
        >>> # Start new conversation
        >>> messages = []
        >>> images = []
        >>>
        >>> response, messages, images = chat_with_mllm(
        ...     model, processor,
        ...     prompt="Extract the transaction table from this bank statement",
        ...     images_path=["path/to/image.png"],
        ...     messages=messages,
        ...     images=images
        ... )
        >>>
        >>> # Continue conversation
        >>> response2, messages, images = chat_with_mllm(
        ...     model, processor,
        ...     prompt="Now extract only the debit amounts",
        ...     messages=messages,
        ...     images=images
        ... )
    """
    # Handle default mutable arguments
    if images_path is None:
        images_path = []
    if messages is None:
        messages = []
    if images is None:
        images = []

    # Ensure images_path is a list
    if not isinstance(images_path, list):
        images_path = [images_path]

    # Load images if needed
    if len(images) == 0 and len(images_path) > 0:
        for image_path in tqdm(images_path, desc="Loading images"):
            image = load_image(image_path)
            images.append(image)
            if show_image:
                display(image)

    # Build conversation messages
    if len(messages) == 0:
        # Starting a new conversation about an image
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
    else:
        # Continuing conversation on the image
        messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})

    # Process input data
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=images, text=text, return_tensors="pt").to(model.device)

    # Configure generation arguments for deterministic OCR
    generation_args = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty,
    }

    if do_sample:
        # Sampling mode: use temperature
        generation_args["temperature"] = temperature
    else:
        # Greedy decoding mode (deterministic): disable sampling parameters
        generation_args["temperature"] = None
        generation_args["top_p"] = None
        generation_args["top_k"] = None

    # Generate response
    generate_ids = model.generate(**inputs, **generation_args)

    # Trim input tokens from output
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1] : -1]
    generated_texts = processor.decode(generate_ids[0], clean_up_tokenization_spaces=False)

    # Append the model's response to the conversation history
    messages.append(
        {"role": "assistant", "content": [{"type": "text", "text": generated_texts}]}
    )

    return generated_texts, messages, images
