"""
InternVL3-specific processor for vision model evaluation.

This module contains all InternVL3-specific code including model loading,
image preprocessing, and batch processing logic.
"""

import time
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from common.config import (
    DEFAULT_IMAGE_SIZE,
    EXTRACTION_FIELDS,
    FIELD_COUNT,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INTERNVL3_MODEL_PATH,
)
from common.evaluation_utils import parse_extraction_response


class InternVL3Processor:
    """Processor for InternVL3 vision-language model."""
    
    def __init__(self, model_path=None, device='cuda'):
        """
        Initialize InternVL3 processor with model and tokenizer.
        
        Args:
            model_path (str): Path to model weights (uses default if None)
            device (str): Device to run model on
        """
        self.model_path = model_path or INTERNVL3_MODEL_PATH
        self.device = device
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        
        # Initialize model and tokenizer
        self._load_model()
        
        # Setup generation config
        self.generation_config = dict(
            max_new_tokens=max(1000, FIELD_COUNT * 50),  # Scale tokens with field count
            do_sample=False,  # Deterministic for consistent field extraction
            pad_token_id=self.tokenizer.eos_token_id,  # Prevent pad_token_id warnings
        )
    
    def _load_model(self):
        """Load InternVL3 model and tokenizer with compatibility settings."""
        print(f"🔧 Loading InternVL3-2B model from: {self.model_path}")
        
        try:
            # Load model with compatibility settings
            self.model = (
                AutoModel.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,  # Official recommendation
                    low_cpu_mem_usage=True,
                    use_flash_attn=False,  # Disabled for compatibility
                    trust_remote_code=True,
                )
                .eval()
                .to(self.device)
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=False,  # More reliable for structured tasks
            )
            
            print("✅ InternVL3 model and tokenizer loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading InternVL3 model: {e}")
            raise
    
    def get_extraction_prompt(self):
        """Get the extraction prompt for InternVL3."""
        prompt = f"""Extract data from this business document. 
Output ALL fields below with their exact keys. 
Use "N/A" if field is not visible or not present.

OUTPUT FORMAT ({FIELD_COUNT} required fields):
"""
        # Add all fields with format guidance
        for field in EXTRACTION_FIELDS:
            prompt += f"{field}: [value or N/A]\n"
        
        prompt += f"""
INSTRUCTIONS:
- Keep field names EXACTLY as shown above
- Use "N/A" for any missing/unclear information
- Do not add explanations or comments
- Extract actual values from the document image
- Output exactly {FIELD_COUNT} lines, one for each field"""
        
        return prompt
    
    def build_transform(self, input_size=DEFAULT_IMAGE_SIZE):
        """
        Build InternVL3 image transformation pipeline.
        
        Args:
            input_size: Target size for image resizing
            
        Returns:
            torchvision.transforms.Compose: Transformation pipeline
        """
        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        return transform
    
    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        """Find closest aspect ratio for InternVL3 dynamic preprocessing."""
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
    
    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=DEFAULT_IMAGE_SIZE, use_thumbnail=False):
        """
        InternVL3 dynamic preprocessing algorithm.
        
        Args:
            image: PIL Image to preprocess
            min_num: Minimum number of tiles
            max_num: Maximum number of tiles
            image_size: Size of each tile
            use_thumbnail: Whether to include thumbnail
            
        Returns:
            list: List of preprocessed image tiles
        """
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        
        # Calculate target ratios
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        
        # Find closest aspect ratio
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )
        
        # Calculate target dimensions
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        
        # Resize image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        
        # Split into tiles
        for i in range(target_aspect_ratio[0]):
            for j in range(target_aspect_ratio[1]):
                box = (
                    i * image_size,
                    j * image_size,
                    (i + 1) * image_size,
                    (j + 1) * image_size,
                )
                split_img = resized_img.crop(box)
                processed_images.append(split_img)
        
        # Add thumbnail if requested
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        
        return processed_images
    
    def load_image(self, image_file, input_size=DEFAULT_IMAGE_SIZE, max_num=12):
        """
        Complete InternVL3 image loading and preprocessing pipeline.
        
        Args:
            image_file: Path to image file or PIL Image
            input_size: Size for each tile
            max_num: Maximum number of tiles
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Load image if path provided
        if isinstance(image_file, str):
            image = Image.open(image_file).convert("RGB")
        else:
            image = image_file.convert("RGB")
        
        # Apply dynamic preprocessing
        images = self.dynamic_preprocess(image, image_size=input_size, max_num=max_num)
        
        # Apply transforms
        transform = self.build_transform(input_size=input_size)
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)
        
        return pixel_values
    
    def process_single_image(self, image_path):
        """
        Process a single image through InternVL3 extraction pipeline.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            dict: Extraction results with metadata
        """
        try:
            start_time = time.time()
            
            # Load and preprocess image
            pixel_values = self.load_image(image_path)
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            
            # Prepare conversation
            question = f"<image>\n{self.get_extraction_prompt()}"
            
            # Generate response
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                self.generation_config,
                history=None,
                return_history=False,
            )
            
            processing_time = time.time() - start_time
            
            # Parse response
            extracted_data = parse_extraction_response(response)
            
            # Calculate metrics
            extracted_fields_count = sum(1 for v in extracted_data.values() if v != "N/A")
            response_completeness = len([k for k in extracted_data.keys() if k in EXTRACTION_FIELDS]) / len(EXTRACTION_FIELDS)
            content_coverage = extracted_fields_count / len(EXTRACTION_FIELDS)
            
            return {
                'image_name': Path(image_path).name,
                'extracted_data': extracted_data,
                'raw_response': response,
                'processing_time': processing_time,
                'response_completeness': response_completeness,
                'content_coverage': content_coverage,
                'extracted_fields_count': extracted_fields_count,
                'raw_response_length': len(response)
            }
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            return {
                'image_name': Path(image_path).name,
                'extracted_data': {field: "N/A" for field in EXTRACTION_FIELDS},
                'raw_response': f"Error: {str(e)}",
                'processing_time': 0,
                'response_completeness': 0,
                'content_coverage': 0,
                'extracted_fields_count': 0,
                'raw_response_length': 0
            }
    
    def process_image_batch(self, image_files, progress_callback=None):
        """
        Process batch of images through InternVL3 extraction pipeline.
        
        Args:
            image_files (list): List of image file paths
            progress_callback (callable): Optional callback for progress updates
            
        Returns:
            tuple: (results, statistics) - Extraction results and batch statistics
        """
        results = []
        total_processing_time = 0
        successful_extractions = 0
        
        print(f"\n🚀 Processing {len(image_files)} images with InternVL3...")
        
        for idx, image_path in enumerate(image_files, 1):
            # Progress update
            if progress_callback:
                progress_callback(idx, len(image_files), image_path)
            else:
                print(f"\n[{idx}/{len(image_files)}] Processing: {Path(image_path).name}")
            
            # Process image
            result = self.process_single_image(image_path)
            results.append(result)
            
            # Update statistics
            total_processing_time += result['processing_time']
            if result['response_completeness'] > 0:
                successful_extractions += 1
            
            # Show extraction status
            print(f"   ⏱️ Processing time: {result['processing_time']:.2f}s")
            print(f"   📊 Fields extracted: {result['extracted_fields_count']}/{FIELD_COUNT}")
            print(f"   ✅ Response completeness: {result['response_completeness']:.1%}")
        
        # Calculate batch statistics
        batch_statistics = {
            'total_images': len(image_files),
            'successful_extractions': successful_extractions,
            'total_processing_time': total_processing_time,
            'average_processing_time': total_processing_time / len(image_files) if image_files else 0,
            'success_rate': successful_extractions / len(image_files) if image_files else 0
        }
        
        print("\n📊 Batch Processing Complete:")
        print(f"   Total images: {batch_statistics['total_images']}")
        print(f"   Successful extractions: {batch_statistics['successful_extractions']}")
        print(f"   Success rate: {batch_statistics['success_rate']:.1%}")
        print(f"   Average processing time: {batch_statistics['average_processing_time']:.2f}s")
        
        return results, batch_statistics