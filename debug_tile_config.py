#!/usr/bin/env python3
"""
Debug InternVL3-8B Tile Configuration Issue

This script investigates the tile-to-token mapping and finds the correct max_num
parameter to avoid the broadcasting error.

Run this on the H200/V100 machine to investigate the actual model configuration.
"""

from pathlib import Path

from transformers import AutoConfig


def analyze_model_config(model_path):
    """Analyze the InternVL3-8B configuration for vision token limits."""
    print("🔍 LOADING MODEL CONFIGURATION...")

    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        print("\n📋 MODEL CONFIGURATION ANALYSIS:")
        print(f"Model class: {config.__class__.__name__}")

        # Check for vision configuration
        if hasattr(config, 'vision_config'):
            vc = config.vision_config
            print(f"Vision model: {vc.__class__.__name__}")
            print(f"Image size: {getattr(vc, 'image_size', 'Not found')}")
            print(f"Patch size: {getattr(vc, 'patch_size', 'Not found')}")
            print(f"Hidden size: {getattr(vc, 'hidden_size', 'Not found')}")

        # Check for LLM configuration
        if hasattr(config, 'llm_config'):
            llm = config.llm_config
            print(f"LLM hidden size: {getattr(llm, 'hidden_size', 'Not found')}")
            print(f"LLM vocab size: {getattr(llm, 'vocab_size', 'Not found')}")

        # Look for max_num or similar parameters
        config_dict = config.to_dict()
        relevant_keys = [k for k in config_dict.keys() if 'max' in k.lower() or 'num' in k.lower() or 'tile' in k.lower()]
        print(f"\nRelevant config keys: {relevant_keys}")

        for key in relevant_keys:
            print(f"{key}: {config_dict[key]}")

        return config

    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None

def test_tile_generation(image_path, model_path):
    """Test different max_num values to find the working configuration."""
    print(f"\n🖼️  TESTING TILE GENERATION WITH: {Path(image_path).name}")

    # Import required modules and define load_image function locally
    import torch
    import torchvision.transforms as T
    from PIL import Image
    from torchvision.transforms.functional import InterpolationMode

    # Define the image processing functions directly
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def build_transform(input_size):
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return transform

    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
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

    def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

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
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(image_file, input_size=448, max_num=12):
        image = Image.open(image_file).convert('RGB')
        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    # Test different max_num values
    test_configs = [
        (448, 1), (448, 2), (448, 3), (448, 4), (448, 5), (448, 6), (448, 7),
        (560, 1), (560, 2), (560, 3), (560, 4), (560, 5), (560, 6),
        (672, 1), (672, 2), (672, 3), (672, 4), (672, 5), (672, 6)
    ]

    print("\n📊 TILE GENERATION ANALYSIS:")
    print(f"{'Resolution':<10} {'Max Tiles':<10} {'Actual Tiles':<12} {'Total Tokens':<12} {'Status'}")
    print("-" * 70)

    for input_size, max_num in test_configs:
        try:
            pixel_values = load_image(image_path, input_size=input_size, max_num=max_num)
            actual_tiles = pixel_values.shape[0]
            total_tokens = actual_tiles * 256  # 256 tokens per tile

            # Check if this would work (target: 1792 tokens or less)
            if total_tokens <= 1792:
                status = "✅ SAFE"
            elif total_tokens <= 2048:
                status = "⚠️  RISKY"
            else:
                status = "❌ OVER"

            print(f"{input_size}px     {max_num:<10} {actual_tiles:<12} {total_tokens:<12} {status}")

        except Exception as e:
            print(f"{input_size}px     {max_num:<10} ERROR: {str(e)[:20]}")

    return test_configs

def find_model_token_limit(model_path):
    """Try to determine the actual token limit from model architecture."""
    print("\n🎯 INVESTIGATING MODEL TOKEN CAPACITY...")

    try:
        # Load just the config to inspect architecture
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        # Look for position embeddings or sequence length limits
        if hasattr(config, 'llm_config'):
            llm_config = config.llm_config
            max_pos = getattr(llm_config, 'max_position_embeddings', None)
            seq_len = getattr(llm_config, 'max_sequence_length', None)

            print(f"LLM max position embeddings: {max_pos}")
            print(f"LLM max sequence length: {seq_len}")

        # Check vision config for limits
        if hasattr(config, 'vision_config'):
            vision_config = config.vision_config
            image_size = getattr(vision_config, 'image_size', 448)
            patch_size = getattr(vision_config, 'patch_size', 14)

            # Calculate maximum possible tokens per image
            patches_per_side = image_size // patch_size
            max_tokens_per_image = patches_per_side * patches_per_side

            print(f"Vision image size: {image_size}")
            print(f"Vision patch size: {patch_size}")
            print(f"Max tokens per 448x448 image: {max_tokens_per_image}")

            # Calculate safe tile count
            if max_tokens_per_image > 0:
                safe_tiles_256_tokens = 1792 // 256  # Assuming 256 tokens per tile
                safe_tiles_patches = 1792 // max_tokens_per_image

                print(f"Safe tile count (256 tokens/tile): {safe_tiles_256_tokens}")
                print(f"Safe tile count (patches/tile): {safe_tiles_patches}")

        return config

    except Exception as e:
        print(f"❌ Error analyzing model limits: {e}")
        return None

def main():
    """Main analysis function."""
    model_path = "/home/jovyan/nfs_share/models/InternVL3-8B"
    image_path = "/home/jovyan/nfs_share/tod/LMM_POC/evaluation_data/image_008.png"

    print("🚨 INTERNVL3-8B TILE CONFIGURATION DEBUG")
    print("=" * 60)

    # 1. Analyze model configuration
    config = analyze_model_config(model_path)

    # 2. Test tile generation with different parameters
    test_tile_generation(image_path, model_path)

    # 3. Find model token limits
    find_model_token_limit(model_path)

    print("\n🎯 RECOMMENDATIONS:")
    print("1. Test max_num=7 with 448px resolution (7 × 256 = 1792 tokens)")
    print("2. Test max_num=6 as conservative fallback (6 × 256 = 1536 tokens)")
    print("3. Avoid max_num > 7 to stay under 1792 token limit")
    print("4. The 1792 appears to be a position embedding or attention limit")

if __name__ == "__main__":
    main()