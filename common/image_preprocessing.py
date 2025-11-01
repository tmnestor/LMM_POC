"""
Bank Statement Image Preprocessing Module

PIL-only preprocessing functions for Australian bank statements
to improve OCR performance with Llama Vision 3.2

Note: This module uses only PIL/Pillow (no OpenCV dependency)
"""

from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def enhance_for_llama(image_path, target_dpi=300):
    """
    Enhancement strategy for Llama Vision 3.2

    Args:
        image_path: Path to the bank statement image
        target_dpi: Target DPI for upscaling (default: 300)

    Returns:
        PIL Image: Enhanced image
    """
    img = Image.open(image_path)

    # Upscale low-resolution scans
    min_dimension = 2000  # Llama works better with higher res
    if min(img.size) < min_dimension:
        scale = min_dimension / min(img.size)
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.LANCZOS)

    # Enhance sharpness moderately
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.4)

    # Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)

    return img


def enhance_statement_quality(image_path):
    """
    Light enhancement specifically for bank statements (recommended)

    This is the recommended preprocessing approach for high-quality scans.
    Uses gentle enhancements that preserve image fidelity.

    Args:
        image_path: Path to the bank statement image

    Returns:
        PIL Image: Enhanced image
    """
    img = Image.open(image_path)

    # Check if image needs upscaling (low DPI scans)
    if img.size[0] < 1500:
        scale_factor = 1500 / img.size[0]
        new_size = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
        img = img.resize(new_size, Image.LANCZOS)

    # Moderate sharpness (too much breaks numbers)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.3)

    # Increase contrast slightly
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)

    return img


def preprocess_statement_for_llama(image_path):
    """
    Preprocessing optimized for Llama Vision 3.2's OCR (PIL-only version)

    This function applies more aggressive preprocessing including grayscale
    conversion and thresholding. May help with low-quality scans but could
    hurt performance on high-quality images.

    Args:
        image_path: Path to the bank statement image

    Returns:
        PIL Image: Preprocessed image ready for Llama Vision
    """
    img = Image.open(image_path)

    # Convert to grayscale
    gray = img.convert('L')

    # Apply smoothing filter (gentle denoising)
    smoothed = gray.filter(ImageFilter.SMOOTH)

    # Auto contrast enhancement
    contrasted = ImageOps.autocontrast(smoothed)

    # Apply sharpening to enhance text edges
    sharpened = contrasted.filter(ImageFilter.SHARPEN)

    # Convert back to RGB for model compatibility
    rgb = sharpened.convert('RGB')

    return rgb


def preprocess_bank_statement(image_path):
    """
    Optimized preprocessing for bank statements (PIL-only alternative)

    Gentler preprocessing than preprocess_statement_for_llama.
    Good balance for moderately low-quality scans.

    Args:
        image_path: Path to the bank statement image

    Returns:
        PIL Image: Preprocessed image
    """
    img = Image.open(image_path)

    # Convert to grayscale
    gray = img.convert('L')

    # Light smoothing only
    smoothed = gray.filter(ImageFilter.SMOOTH_MORE)

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(smoothed)
    contrasted = enhancer.enhance(1.5)

    # Slight sharpening
    sharpened = contrasted.filter(ImageFilter.SHARPEN)

    # Convert back to RGB for model
    rgb = sharpened.convert('RGB')

    return rgb


def binarize_statement(image_path, threshold=128):
    """
    Simple binarization for very low-quality scans

    Converts image to black and white using a threshold.
    Use with caution - may lose information in gradients.

    Args:
        image_path: Path to the bank statement image
        threshold: Threshold value (0-255, default 128)

    Returns:
        PIL Image: Binarized image
    """
    img = Image.open(image_path)

    # Convert to grayscale
    gray = img.convert('L')

    # Apply threshold
    binary = gray.point(lambda x: 255 if x > threshold else 0, mode='L')

    # Convert to RGB for model
    rgb = binary.convert('RGB')

    return rgb


def extract_regions(image_path, output_dir=None):
    """
    Split image into regions for region-based extraction (PIL-only version)

    Args:
        image_path: Path to the bank statement image
        output_dir: Optional directory to save region images (default: temp files)

    Returns:
        dict: Dictionary with region names and their file paths
    """
    img = Image.open(image_path)
    width, height = img.size

    # Define regions (adjust based on statement layout)
    regions = {
        'header': (0, 0, width, int(height * 0.15)),           # Top 15%
        'summary': (0, int(height * 0.15), width, int(height * 0.25)),  # Next 10%
        'transactions': (0, int(height * 0.25), width, height)  # Rest
    }

    # Prepare output directory
    if output_dir is None:
        output_dir = Path(image_path).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Extract and save regions
    region_paths = {}
    for region_name, bbox in regions.items():
        region_img = img.crop(bbox)
        region_path = output_dir / f"temp_{region_name}.png"
        region_img.save(region_path)
        region_paths[region_name] = str(region_path)

    return region_paths


def adaptive_enhance(image_path):
    """
    Adaptive enhancement that analyzes image and applies appropriate processing

    Checks image quality metrics and applies enhancement accordingly.

    Args:
        image_path: Path to the bank statement image

    Returns:
        PIL Image: Adaptively enhanced image
    """
    img = Image.open(image_path)

    # Check if image is already high resolution
    if min(img.size) >= 2000:
        # High quality - use gentle enhancement
        return enhance_statement_quality(image_path)
    else:
        # Lower quality - use moderate enhancement
        return enhance_for_llama(image_path)


# Recommended preprocessing function for most use cases
def preprocess_recommended(image_path):
    """
    Recommended preprocessing for bank statements

    This is a balanced approach suitable for most bank statement images.
    Alias for enhance_statement_quality.

    Args:
        image_path: Path to the bank statement image

    Returns:
        PIL Image: Enhanced image
    """
    return enhance_statement_quality(image_path)
