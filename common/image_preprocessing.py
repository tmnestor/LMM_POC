"""
Bank Statement Image Preprocessing Module

Optimized preprocessing functions for Australian bank statements
to improve OCR performance with Llama Vision 3.2
"""

import cv2
from PIL import Image, ImageEnhance


def preprocess_statement_for_llama(image_path):
    """
    Preprocessing optimized for Llama Vision 3.2's OCR
    
    Args:
        image_path: Path to the bank statement image
        
    Returns:
        PIL Image: Preprocessed image ready for Llama Vision
    """
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Light denoising (Llama handles noise reasonably well)
    denoised = cv2.fastNlMeansDenoising(gray, h=8)
    
    # Adaptive binarization - works best for tables
    binary = cv2.adaptiveThreshold(
        denoised, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 
        15, 3
    )
    
    # Remove table lines that can confuse OCR
    # Horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    detect_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # Vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    detect_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    # Remove lines
    binary_cleaned = cv2.subtract(binary, detect_horizontal)
    binary_cleaned = cv2.subtract(binary_cleaned, detect_vertical)
    
    # Convert to RGB for Llama
    rgb = cv2.cvtColor(binary_cleaned, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(rgb)


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


def preprocess_bank_statement(image_path):
    """
    Optimized preprocessing for bank statements (alternative approach)
    
    Args:
        image_path: Path to the bank statement image
        
    Returns:
        PIL Image: Preprocessed image
    """
    img = cv2.imread(image_path)
    
    # Bank statements are typically high-quality scans, so be gentle
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Slight denoising only (statements usually clean)
    denoised = cv2.fastNlMeansDenoising(gray, h=7)
    
    # Adaptive thresholding works better for tables with grid lines
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Remove horizontal/vertical lines (can interfere with text)
    # This helps with table borders
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    
    detect_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    detect_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    # Remove lines from binary image
    binary = cv2.bitwise_and(binary, cv2.bitwise_not(detect_horizontal))
    binary = cv2.bitwise_and(binary, cv2.bitwise_not(detect_vertical))
    
    # Convert back to RGB for model
    rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(rgb)


def enhance_statement_quality(image_path):
    """
    Enhancement specifically for bank statements
    
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


def extract_regions(image_path):
    """
    Split image into regions for region-based extraction
    
    Args:
        image_path: Path to the bank statement image
        
    Returns:
        dict: Dictionary with region names and their image arrays
    """
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    # Define regions (adjust based on statement layout)
    regions = {
        'header': img[0:int(height*0.15), :],           # Top 15%
        'summary': img[int(height*0.15):int(height*0.25), :],  # Next 10%
        'transactions': img[int(height*0.25):, :]       # Rest
    }
    
    # Save regions to temporary files
    region_paths = {}
    for region_name, region_img in regions.items():
        region_path = f"temp_{region_name}.png"
        cv2.imwrite(region_path, region_img)
        region_paths[region_name] = region_path
    
    return region_paths
