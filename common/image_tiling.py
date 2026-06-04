"""Pure-PIL InternVL dynamic tiling — the single source for tile cropping.

The official InternVL ``dynamic_preprocess`` algorithm, with no torch / torchvision
dependency so it can be used in the vLLM pre-tiling path (where images are cropped
ourselves and handed to vLLM as separate sub-images) as well as the HF preprocessor.

Both ``models/internvl3_image_preprocessor.py`` (HF path) and
``models/backends/vllm_backend.py`` (vLLM pre-tiling path) delegate here, so the two
paths tile identically — see ``plans/2026-06-04-adaptive-pre-tiling.md``.
"""

from PIL import Image


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    """Standard InternVL3 find_closest_aspect_ratio from official documentation."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    *,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> list[Image.Image]:
    """Crop *image* into up to *max_num* ``image_size``-square tiles.

    Returns the tile crops in row-major order; when *use_thumbnail* is set and more
    than one tile was produced, a single down-scaled thumbnail of the whole image is
    appended last (the native InternVL ordering: detail tiles first, global view last).
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    ratio_set = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    target_ratios = sorted(ratio_set, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []

    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images
