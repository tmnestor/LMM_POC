"""InternVL3 Image Preprocessor.

Handles image loading, dynamic tiling, transforms, and tensor preparation
for InternVL3 vision-language models.

Separates pure image computation from model loading and generation logic,
enabling independent testing and reuse (e.g. by UnifiedBankExtractor).
"""

from dataclasses import dataclass

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter, ImageStat

# ImageNet normalization constants (for vision transformers)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Default image size for processing
DEFAULT_IMAGE_SIZE = 448

# Quality assessment reference values
_SHARPNESS_REF = 500.0
_CONTRAST_REF = 80.0
_SHARPNESS_WEIGHT = 0.7
_CONTRAST_WEIGHT = 0.3

# 3x3 Laplacian kernel for sharpness estimation
_LAPLACIAN_KERNEL = ImageFilter.Kernel(
    size=(3, 3),
    kernel=[0, 1, 0, 1, -4, 1, 0, 1, 0],
    scale=1,
    offset=128,
)


@dataclass
class ImageQualityMetrics:
    """Per-image quality assessment results."""

    sharpness_score: float
    contrast_score: float
    composite_score: float
    recommended_tiles: int
    width: int
    height: int


class InternVL3ImagePreprocessor:
    """Image preprocessing pipeline for InternVL3 models.

    Handles dynamic tiling, ImageNet normalization, and tensor
    preparation with correct dtype/device for model inference.
    """

    def __init__(
        self,
        max_tiles: int,
        debug: bool = False,
        min_tiles: int | None = None,
    ):
        self.max_tiles = max_tiles
        self.min_tiles = min_tiles
        self.adaptive_enabled = min_tiles is not None and min_tiles < max_tiles
        self.debug = debug

    def assess_image_quality(self, image: Image.Image) -> ImageQualityMetrics:
        """Assess image quality using sharpness and contrast metrics.

        Args:
            image: PIL Image to assess.

        Returns:
            ImageQualityMetrics with per-metric scores and recommended tile count.
        """
        gray = image.convert("L")
        w, h = image.size

        # Sharpness: variance of Laplacian response
        laplacian = gray.filter(_LAPLACIAN_KERNEL)
        lap_arr = np.asarray(laplacian, dtype=np.float64) - 128.0
        sharpness_raw = float(np.var(lap_arr))
        sharpness_score = min(sharpness_raw / _SHARPNESS_REF, 1.0)

        # Contrast: std deviation of grayscale pixel values
        contrast_raw = ImageStat.Stat(gray).stddev[0]
        contrast_score = min(contrast_raw / _CONTRAST_REF, 1.0)

        composite = (
            _SHARPNESS_WEIGHT * sharpness_score + _CONTRAST_WEIGHT * contrast_score
        )

        # Low quality → more tiles; high quality → fewer tiles
        tile_range = self.max_tiles - (self.min_tiles or self.max_tiles)
        recommended = self.max_tiles - int(composite * tile_range)

        return ImageQualityMetrics(
            sharpness_score=sharpness_score,
            contrast_score=contrast_score,
            composite_score=composite,
            recommended_tiles=recommended,
            width=w,
            height=h,
        )

    def get_tile_info(self, image_file) -> str:
        """Return a one-line summary of tile count and quality for an image file.

        Lightweight — opens the image for quality assessment only, no tensor work.
        """
        from pathlib import Path

        image = Image.open(image_file).convert("RGB")
        resolved = self._resolve_tile_count(image, None)
        tiles = self.dynamic_preprocess(
            image, min_num=1, max_num=resolved, image_size=448, use_thumbnail=True
        )
        if self.adaptive_enabled:
            metrics = self.assess_image_quality(image)
            return (
                f"{Path(image_file).name}: {len(tiles)} tiles "
                f"(quality: {metrics.composite_score:.2f})"
            )
        return f"{Path(image_file).name}: {len(tiles)} tiles"

    def _resolve_tile_count(self, image: Image.Image, max_num: int | None) -> int:
        """Determine tile count for an image.

        Priority:
        1. Explicit max_num override (from caller / notebook) → use it directly.
        2. Adaptive OFF → self.max_tiles.
        3. Adaptive ON → quality-based selection in [min_tiles, max_tiles].
        """
        if max_num is not None:
            return max_num

        if not self.adaptive_enabled:
            return self.max_tiles

        metrics = self.assess_image_quality(image)
        if self.debug:
            print(
                f"  quality: sharpness={metrics.sharpness_score:.2f} "
                f"contrast={metrics.contrast_score:.2f} "
                f"composite={metrics.composite_score:.2f}"
            )
        return metrics.recommended_tiles

    @staticmethod
    def get_model_device(model) -> torch.device:
        """Get the correct device for pixel_values tensor placement.

        For multi-GPU models with device_map="auto", pixel_values must be
        placed on the device where the vision model's embedding layer resides,
        since that's where image tensors enter the model.
        """
        if hasattr(model, "vision_model") and hasattr(model.vision_model, "embeddings"):
            try:
                return next(model.vision_model.embeddings.parameters()).device
            except (StopIteration, AttributeError):
                pass

        if hasattr(model, "device"):
            return model.device

        try:
            return next(model.parameters()).device
        except (StopIteration, AttributeError):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def resolve_pixel_dtype(model, debug: bool = False) -> torch.dtype:
        """Resolve the correct dtype for pixel tensors based on model weights.

        Checks vision model embeddings first (most reliable for quantized
        models), then falls back through model.dtype, vision_model.dtype,
        parameter dtype, and finally float32 for V100 compatibility.
        """
        # Strategy 1: Vision model embedding layer (most reliable)
        if hasattr(model, "vision_model") and hasattr(model.vision_model, "embeddings"):
            try:
                dtype = next(model.vision_model.embeddings.parameters()).dtype
                if debug:
                    print(f"🔧 TENSOR_DTYPE: Using vision model dtype {dtype}")
                return dtype
            except (StopIteration, AttributeError):
                pass

        # Strategy 2: model.dtype attribute
        if hasattr(model, "dtype"):
            if debug:
                print(f"🔧 TENSOR_DTYPE: Using model.dtype {model.dtype}")
            return model.dtype

        # Strategy 3: vision_model.dtype
        if hasattr(model, "vision_model") and hasattr(model.vision_model, "dtype"):
            if debug:
                print(
                    f"🔧 TENSOR_DTYPE: Using vision_model.dtype {model.vision_model.dtype}"
                )
            return model.vision_model.dtype

        # Strategy 4: First model parameter
        try:
            dtype = next(model.parameters()).dtype
            if debug:
                print(f"🔧 TENSOR_DTYPE: Using parameter dtype {dtype}")
            return dtype
        except (StopIteration, AttributeError):
            pass

        # Last resort: float32 for V100 compatibility
        if debug:
            print("🔧 TENSOR_DTYPE: Using float32 fallback (V100 safe)")
        return torch.float32

    def build_transform(self, input_size: int = DEFAULT_IMAGE_SIZE) -> T.Compose:
        """Build InternVL3 image transformation pipeline."""
        return T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(
                    (input_size, input_size),
                    interpolation=T.InterpolationMode.BICUBIC,
                ),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def find_closest_aspect_ratio(
        self, aspect_ratio, target_ratios, width, height, image_size
    ):
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
        self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
    ):
        """Standard InternVL3 dynamic_preprocess from official documentation."""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = self.find_closest_aspect_ratio(
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
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(
        self, image_file, model, input_size: int = 448, max_num: int | None = None
    ) -> torch.Tensor:
        """Complete InternVL3 image loading and preprocessing pipeline.

        Args:
            image_file: Path to image file.
            model: The InternVL3 model (used for dtype/device resolution).
            input_size: Tile size in pixels (default 448).
            max_num: Max tiles override. Uses self.max_tiles if None.

        Returns:
            Preprocessed pixel values tensor ready for model inference.
        """
        image = Image.open(image_file).convert("RGB")

        resolved = self._resolve_tile_count(image, max_num)
        if resolved is None:
            raise ValueError(
                "max_tiles not configured!\n"
                "Add MAX_TILES to your notebook CONFIG:\n"
                "   CONFIG = {\n"
                "       ...\n"
                "       'MAX_TILES': 36,  # H200: 36, V100-8B: 14, 2B: 18\n"
                "   }\n"
                "Pass to processor initialization:\n"
                "   hybrid_processor = DocumentAwareInternVL3HybridProcessor(\n"
                "       ...\n"
                "       max_tiles=CONFIG['MAX_TILES']\n"
                "   )"
            )

        if self.debug:
            print(f"LOAD_IMAGE: max_num={resolved}, input_size={input_size}")

        images = self.dynamic_preprocess(
            image,
            min_num=1,
            max_num=resolved,
            image_size=input_size,
            use_thumbnail=True,
        )
        # Force output past Rich progress bar
        import sys
        from pathlib import Path

        if self.adaptive_enabled and max_num is None:
            metrics = self.assess_image_quality(image)
            sys.stdout.write(
                f"{Path(image_file).name}: {len(images)} tiles "
                f"(quality: {metrics.composite_score:.2f})\n"
            )
        else:
            sys.stdout.write(f"{Path(image_file).name}: {len(images)} tiles\n")
        sys.stdout.flush()

        transform = self.build_transform(input_size=input_size)
        pixel_values = torch.stack([transform(img) for img in images])

        # Convert to model's dtype
        pixel_dtype = self.resolve_pixel_dtype(model, debug=self.debug)
        pixel_values = pixel_values.to(dtype=pixel_dtype)

        # Move to model's device
        model_device = self.get_model_device(model)
        if pixel_values.device != model_device:
            pixel_values = pixel_values.to(model_device)
            if self.debug:
                print(f"🔧 DEVICE_MOVE: Moved tensor to {model_device}")
        elif self.debug:
            print(f"✅ DEVICE_OK: Tensor already on {model_device}")

        # Debug verification
        if self.debug:
            try:
                model_param = next(iter(model.parameters()))
                print(
                    f"🔍 DTYPE_CHECK: pixel_values={pixel_values.dtype}, model_param={model_param.dtype}"
                )
                print(
                    f"🔍 DEVICE_CHECK: pixel_values={pixel_values.device}, model_param={model_param.device}"
                )
                if pixel_values.dtype != model_param.dtype:
                    print(
                        f"⚠️ DTYPE_MISMATCH: pixel_values ({pixel_values.dtype}) != model ({model_param.dtype})"
                    )
            except Exception as debug_err:
                print(f"⚠️ Debug check failed: {debug_err}")

            print(
                f"📐 TENSOR_SHAPE: {pixel_values.shape} (batch_size={pixel_values.shape[0]} tiles)"
            )
            print(f"📊 TENSOR_DTYPE: {pixel_values.dtype}")
            print(f"📍 TENSOR_DEVICE: {pixel_values.device}")

        return pixel_values

    def load_image_from_pil(
        self,
        image: Image.Image,
        model,
        input_size: int = 448,
        max_num: int | None = None,
    ) -> torch.Tensor:
        """Same as load_image but accepts a PIL Image instead of a file path.

        Args:
            image: PIL Image (already loaded).
            model: The InternVL3 model (used for dtype/device resolution).
            input_size: Tile size in pixels (default 448).
            max_num: Max tiles override. Uses self.max_tiles if None.

        Returns:
            Preprocessed pixel values tensor ready for model inference.
        """
        rgb_image = image.convert("RGB")

        resolved = self._resolve_tile_count(rgb_image, max_num)
        if resolved is None:
            raise ValueError(
                "max_tiles not configured!\n"
                "Set max_tiles when constructing the processor."
            )

        images = self.dynamic_preprocess(
            rgb_image,
            min_num=1,
            max_num=resolved,
            image_size=input_size,
            use_thumbnail=True,
        )

        transform = self.build_transform(input_size=input_size)
        pixel_values = torch.stack([transform(img) for img in images])

        pixel_dtype = self.resolve_pixel_dtype(model, debug=self.debug)
        pixel_values = pixel_values.to(dtype=pixel_dtype)

        model_device = self.get_model_device(model)
        if pixel_values.device != model_device:
            pixel_values = pixel_values.to(model_device)

        return pixel_values
