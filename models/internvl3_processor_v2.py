#!/usr/bin/env python3
"""
Document-Type-Aware InternVL3 Processor - Phase 3 Implementation

Enhanced InternVL3 processor with automatic document type detection and 
schema-driven field extraction for improved performance and accuracy.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from common.config import (
    CLEAR_GPU_CACHE_AFTER_BATCH,
)
from common.document_processor_base import DocumentAwareProcessor
from common.extraction_parser import parse_extraction_response
from common.gpu_optimization import (
    clear_model_caches,
    handle_memory_fragmentation,
)

# Import from original processor and document-aware components
from models.internvl3_processor import InternVL3Processor


class DocumentAwareInternVL3Processor(InternVL3Processor, DocumentAwareProcessor):
    """
    Document-type-aware InternVL3 processor combining automatic document detection
    with targeted field extraction for improved efficiency and accuracy.
    """
    
    def __init__(
        self,
        model_path=None,
        device="cuda",
        batch_size=None,
        extraction_mode=None,
        debug=False,
        grouping_strategy="detailed_grouped",
        enable_document_awareness=True
    ):
        """
        Initialize document-aware InternVL3 processor.
        
        Args:
            model_path (str): Path to model weights (uses default if None)
            device (str): Device to run model on
            batch_size (int): Batch size for processing (auto-detected if None)
            extraction_mode (str): Extraction mode ('single_pass', 'grouped', 'adaptive')
            debug (bool): Enable debug logging for extraction
            grouping_strategy (str): Grouping strategy ('8_groups' or '6_groups')
            enable_document_awareness (bool): Enable document-type-specific processing
        """
        # Initialize parent classes
        InternVL3Processor.__init__(
            self, model_path, device, batch_size, extraction_mode, debug, grouping_strategy
        )
        DocumentAwareProcessor.__init__(self)
        
        # Configure document awareness
        self.enable_document_awareness = enable_document_awareness
        
        if self.enable_document_awareness:
            print("🚀 Initializing document-type-aware InternVL3 processor...")
            
            # Initialize document awareness after model loading
            self.initialize_document_awareness()
            
            # Set up document detector with this processor instance
            self.set_document_detector(self)
            
            print("✅ Document-aware InternVL3 processor ready")
        else:
            print("📋 Document awareness disabled - using unified extraction")
    
    def process_single_image(self, image_path: str, **kwargs) -> Dict:
        """
        Enhanced single image processing with document-type awareness.
        
        Args:
            image_path (str): Path to image file
            **kwargs: Additional processing parameters
            
        Returns:
            dict: Enhanced extraction results with document-aware metadata
        """
        overall_start_time = time.time()
        
        try:
            # Step 1: Document-aware processing (if enabled)
            if self.enable_document_awareness:
                doc_aware_result = self.process_single_image_with_document_awareness(
                    image_path, model_type="internvl3", **kwargs
                )
                
                # Use document-specific prompt if available
                if "targeted_prompt" in doc_aware_result and doc_aware_result["targeted_prompt"]:
                    custom_prompt = doc_aware_result["targeted_prompt"]
                    print(f"🎯 Using targeted prompt for {doc_aware_result['document_type']} ({doc_aware_result['field_count']} fields)")
                else:
                    custom_prompt = None
                    print("📋 Using standard unified prompt")
            else:
                doc_aware_result = {}
                custom_prompt = None
            
            # Step 2: Process with original InternVL3 logic (with optional custom prompt)
            extraction_result = self._process_with_internvl3_model(image_path, custom_prompt)
            
            # Step 3: Combine results
            final_result = {
                **extraction_result,
                "document_awareness": doc_aware_result,
                "total_processing_time": time.time() - overall_start_time,
                "enhanced_processor": True
            }
            
            # Add efficiency metrics if document-aware
            if self.enable_document_awareness and doc_aware_result:
                doc_type = doc_aware_result.get("document_type", "unknown")
                field_count = doc_aware_result.get("field_count", 25)
                
                final_result.update({
                    "detected_document_type": doc_type,
                    "fields_extracted": field_count,
                    "field_reduction": 25 - field_count if field_count < 25 else 0,
                    "efficiency_gain": f"{((25 - field_count) / 25) * 100:.1f}%" if field_count < 25 else "0%"
                })
            
            return final_result
            
        except Exception as e:
            print(f"❌ Enhanced processing failed: {e}")
            # Fallback to original processing
            print("🔄 Falling back to standard InternVL3 processing...")
            return InternVL3Processor.process_single_image(self, image_path)
    
    def _process_with_internvl3_model(self, image_path: str, custom_prompt: Optional[str] = None) -> Dict:
        """
        Process image with InternVL3 model using custom or standard prompt.
        
        Args:
            image_path (str): Path to image file
            custom_prompt (str, optional): Custom prompt to use instead of default
            
        Returns:
            dict: InternVL3 processing results
        """
        try:
            start_time = time.time()
            
            # Memory optimization - V100 optimized
            handle_memory_fragmentation(threshold_gb=0.2, aggressive=True)
            
            # Additional V100 safety check
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                if reserved > 20.0:  # Alert if approaching V100 limits
                    print(f"🚨 HIGH MEMORY WARNING: {reserved:.1f}GB reserved - potential V100 crash risk")
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            
            # Load and preprocess image
            image = self.load_image(image_path)
            
            # Use custom prompt if provided, otherwise use standard
            if custom_prompt:
                prompt_text = custom_prompt
                print(f"🎯 Using custom prompt ({len(custom_prompt)} chars)")
            else:
                prompt_text = self.get_extraction_prompt()
                print(f"📋 Using standard prompt ({len(prompt_text)} chars)")
            
            # Generate response
            response = self._generate_response(image, prompt_text)
            
            processing_time = time.time() - start_time
            
            # Parse extraction response
            parsed_data = parse_extraction_response(response)
            
            # Clean up
            if CLEAR_GPU_CACHE_AFTER_BATCH:
                clear_model_caches(self.model)
            
            return {
                "image_path": image_path,
                "processing_time": processing_time,
                "raw_response": response,
                "parsed_data": parsed_data,
                "extraction_mode": self.extraction_mode,
                "prompt_used": "custom" if custom_prompt else "standard",
                "prompt_length": len(prompt_text),
                "model_size": "8B" if self.is_8b_model else "2B"
            }
            
        except Exception as e:
            print(f"❌ InternVL3 model processing failed: {e}")
            raise
    
    def _generate_response(self, image: torch.Tensor, prompt: str) -> str:
        """Generate response using InternVL3 model."""
        try:
            # Prepare inputs - InternVL3 load_image already returns properly batched tensor
            pixel_values = image.to(torch.bfloat16).to(self.device)
            
            # InternVL3 expects prompts with <image> token
            formatted_prompt = f"<image>\n{prompt}"
            
            # Method selection based on model capabilities
            if hasattr(self.model, 'chat') and callable(self.model.chat):
                # Use chat method (preferred for most InternVL3 models)
                response, _ = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    formatted_prompt,
                    generation_config=self.generation_config,
                    history=None,
                    return_history=True
                )
            elif hasattr(self.model, 'generate') and callable(self.model.generate):
                # Fallback to generate method
                with torch.no_grad():
                    response = self.model.generate(
                        pixel_values=pixel_values,
                        input_ids=self.tokenizer.encode(prompt, return_tensors='pt').to(self.device),
                        **self.generation_config.__dict__
                    )
                    response = self.tokenizer.decode(response[0], skip_special_tokens=True)
            else:
                raise AttributeError("Model does not have 'chat' or 'generate' method")
            
            return response
            
        except Exception as e:
            print(f"❌ Response generation failed: {e}")
            raise
    
    def _extract_with_custom_prompt(self, image_path: str, prompt: str, **generation_kwargs) -> str:
        """
        Extract using custom prompt - used by document detector.
        
        Args:
            image_path (str): Path to image file
            prompt (str): Custom extraction prompt
            **generation_kwargs: Generation parameters
            
        Returns:
            str: Raw model response
        """
        try:
            # Load and preprocess image
            image = self.load_image(image_path)
            pixel_values = image.to(torch.bfloat16).to(self.device)
            
            # Create temporary generation config with provided kwargs
            temp_generation_config = self.generation_config
            if generation_kwargs:
                # Update generation config with custom parameters
                for key, value in generation_kwargs.items():
                    if hasattr(temp_generation_config, key):
                        setattr(temp_generation_config, key, value)
            
            # InternVL3 expects prompts with <image> token
            formatted_prompt = f"<image>\n{prompt}"
            
            # Generate response using appropriate method
            if hasattr(self.model, 'chat') and callable(self.model.chat):
                response, _ = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    formatted_prompt,
                    generation_config=temp_generation_config,
                    history=None,
                    return_history=True
                )
            else:
                # Fallback method
                with torch.no_grad():
                    output = self.model.generate(
                        pixel_values=pixel_values,
                        input_ids=self.tokenizer.encode(prompt, return_tensors='pt').to(self.device),
                        max_new_tokens=generation_kwargs.get("max_new_tokens", 150),
                        temperature=generation_kwargs.get("temperature", 0.0),
                        do_sample=generation_kwargs.get("do_sample", False)
                    )
                    response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            print(f"❌ Custom prompt extraction failed: {e}")
            raise
    
    def process_image_batch(self, image_files: List[str], **kwargs) -> Tuple[List[Dict], Dict]:
        """
        Enhanced batch processing with document-type awareness.
        
        Args:
            image_files (List[str]): List of image file paths
            **kwargs: Additional processing parameters
            
        Returns:
            Tuple[List[Dict], Dict]: (results, statistics)
        """
        print(f"🚀 Processing batch of {len(image_files)} images with document awareness...")
        
        results = []
        batch_start_time = time.time()
        
        # Track document-aware statistics
        doc_type_counts = {}
        total_field_reduction = 0
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n📄 Processing {i}/{len(image_files)}: {Path(image_path).name}")
            
            try:
                # Process single image with enhancements
                result = self.process_single_image(image_path, **kwargs)
                results.append(result)
                
                # Track document-aware statistics
                if self.enable_document_awareness and "document_awareness" in result:
                    doc_type = result.get("detected_document_type", "unknown")
                    doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1
                    
                    field_reduction = result.get("field_reduction", 0)
                    total_field_reduction += field_reduction
                
            except Exception as e:
                print(f"❌ Failed to process {Path(image_path).name}: {e}")
                results.append({
                    "image_path": image_path,
                    "error": str(e),
                    "processing_time": 0
                })
        
        batch_processing_time = time.time() - batch_start_time
        
        # Calculate enhanced statistics
        successful_results = [r for r in results if "error" not in r]
        avg_processing_time = sum(r["processing_time"] for r in successful_results) / len(successful_results) if successful_results else 0
        
        stats = {
            "total_images": len(image_files),
            "successful": len(successful_results),
            "failed": len(image_files) - len(successful_results),
            "batch_processing_time": batch_processing_time,
            "avg_processing_time": avg_processing_time,
            "document_awareness_enabled": self.enable_document_awareness,
            "model_size": "8B" if self.is_8b_model else "2B"
        }
        
        # Add document-aware statistics
        if self.enable_document_awareness and doc_type_counts:
            avg_field_reduction = total_field_reduction / len(successful_results) if successful_results else 0
            
            stats.update({
                "document_type_distribution": doc_type_counts,
                "avg_field_reduction": avg_field_reduction,
                "efficiency_gain_estimate": f"{(avg_field_reduction / 25) * 100:.1f}%",
                "performance_improvement_estimate": f"{(avg_field_reduction / 25) * 40:.1f}%"  # InternVL3 benefits more from field reduction
            })
        
        return results, stats
    
    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report including document-aware metrics."""
        base_report = DocumentAwareProcessor.get_performance_report(self)
        
        # Add InternVL3-specific metrics
        internvl3_metrics = {
            "model_type": "InternVL3",
            "model_size": "8B" if self.is_8b_model else "2B",
            "extraction_mode": self.extraction_mode,
            "device": str(self.device),
            "batch_size": self.batch_size,
            "document_awareness_enabled": self.enable_document_awareness
        }
        
        return {
            **base_report,
            "internvl3_specific_metrics": internvl3_metrics
        }


def main():
    """Test document-aware InternVL3 processor."""
    print("🚀 Document-Aware InternVL3 Processor - Phase 3 Testing")
    
    try:
        # Initialize processor with document awareness
        processor = DocumentAwareInternVL3Processor(
            debug=True,
            enable_document_awareness=True
        )
        
        print("✅ Processor initialized successfully")
        print(f"📊 Document awareness enabled: {processor.enable_document_awareness}")
        print(f"📊 Model size: {'8B' if processor.is_8b_model else '2B'}")
        
        # Test single image processing
        test_image = "evaluation_data/synthetic_invoice_001.png"
        if Path(test_image).exists():
            print(f"\n📄 Testing with: {Path(test_image).name}")
            
            result = processor.process_single_image(test_image)
            
            print("✅ Processing completed")
            print(f"   Document type: {result.get('detected_document_type', 'unknown')}")
            print(f"   Fields extracted: {result.get('fields_extracted', 'unknown')}")
            print(f"   Processing time: {result.get('total_processing_time', 0):.2f}s")
        else:
            print(f"⚠️ Test image not found: {test_image}")
        
        print("\n✅ Document-aware InternVL3 processor testing completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()