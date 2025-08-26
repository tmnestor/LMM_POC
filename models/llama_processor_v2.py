#!/usr/bin/env python3
"""
Document-Type-Aware Llama Processor - Phase 3 Implementation

Enhanced Llama processor with automatic document type detection and 
schema-driven field extraction for improved performance and accuracy.
"""

import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from common.config import (
    CLEAR_GPU_CACHE_AFTER_BATCH,
)
from common.document_processor_base import DocumentAwareProcessor
from common.extraction_parser import parse_extraction_response
from common.gpu_optimization import (
    comprehensive_memory_cleanup,
    handle_memory_fragmentation,
)

# Import from original processor and document-aware components
from models.llama_processor import LlamaProcessor

warnings.filterwarnings("ignore")


class DocumentAwareLlamaProcessor(LlamaProcessor, DocumentAwareProcessor):
    """
    Document-type-aware Llama processor combining automatic document detection
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
        Initialize document-aware Llama processor.
        
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
        LlamaProcessor.__init__(
            self, model_path, device, batch_size, extraction_mode, debug, grouping_strategy
        )
        DocumentAwareProcessor.__init__(self)
        
        # Configure document awareness
        self.enable_document_awareness = enable_document_awareness
        
        if self.enable_document_awareness:
            print("🚀 Initializing document-type-aware Llama processor...")
            
            # Initialize document awareness after model loading
            self.initialize_document_awareness()
            
            # Set up document detector with this processor instance
            self.set_document_detector(self)
            
            print("✅ Document-aware Llama processor ready")
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
                    image_path, model_type="llama", **kwargs
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
            
            # Step 2: Process with original Llama logic (with optional custom prompt)
            extraction_result = self._process_with_llama_model(image_path, custom_prompt)
            
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
            print("🔄 Falling back to standard Llama processing...")
            return LlamaProcessor.process_single_image(self, image_path)
    
    def _process_with_llama_model(self, image_path: str, custom_prompt: Optional[str] = None) -> Dict:
        """
        Process image with Llama model using custom or standard prompt.
        
        Args:
            image_path (str): Path to image file
            custom_prompt (str, optional): Custom prompt to use instead of default
            
        Returns:
            dict: Llama processing results
        """
        try:
            start_time = time.time()
            
            # Memory optimization
            handle_memory_fragmentation(threshold_gb=1.0, aggressive=True)
            
            # Load image
            image = self.load_document_image(image_path)
            
            # Use custom prompt if provided, otherwise use standard
            if custom_prompt:
                prompt_text = custom_prompt
                print(f"🎯 Using custom prompt ({len(custom_prompt)} chars)")
            else:
                prompt_text = self.get_extraction_prompt()
                print(f"📋 Using standard prompt ({len(prompt_text)} chars)")
            
            # Create multimodal conversation
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            
            # Apply chat template
            input_text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(image, input_text, return_tensors="pt").to(
                self.model.device
            )
            
            # Generation configuration
            generation_kwargs = {
                "max_new_tokens": self.generation_config["max_new_tokens"],
                "temperature": self.generation_config["temperature"],
                "do_sample": self.generation_config["do_sample"],
                "top_p": self.generation_config["top_p"],
                "use_cache": self.generation_config["use_cache"],
                "pad_token_id": self.processor.tokenizer.eos_token_id,
            }
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(**inputs, **generation_kwargs)
            
            # Decode response
            prompt_len = inputs['input_ids'].shape[1]
            generated_tokens = output[0][prompt_len:]
            response = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            processing_time = time.time() - start_time
            
            # Parse extraction response
            parsed_data = parse_extraction_response(response)
            
            # Clean up
            if CLEAR_GPU_CACHE_AFTER_BATCH:
                comprehensive_memory_cleanup()
            
            return {
                "image_path": image_path,
                "processing_time": processing_time,
                "raw_response": response,
                "parsed_data": parsed_data,
                "extraction_mode": self.extraction_mode,
                "prompt_used": "custom" if custom_prompt else "standard",
                "prompt_length": len(prompt_text)
            }
            
        except Exception as e:
            print(f"❌ Llama model processing failed: {e}")
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
            # Load image
            image = self.load_document_image(image_path)
            
            # Create multimodal conversation
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            # Apply chat template
            input_text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(image, input_text, return_tensors="pt").to(
                self.model.device
            )
            
            # Merge generation config with provided kwargs
            final_generation_kwargs = {
                "max_new_tokens": generation_kwargs.get("max_new_tokens", 150),
                "temperature": generation_kwargs.get("temperature", 0.0),
                "do_sample": generation_kwargs.get("do_sample", False),
                "pad_token_id": self.processor.tokenizer.eos_token_id,
            }
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(**inputs, **final_generation_kwargs)
            
            # Decode response
            prompt_len = inputs['input_ids'].shape[1]
            generated_tokens = output[0][prompt_len:]
            response = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
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
            "document_awareness_enabled": self.enable_document_awareness
        }
        
        # Add document-aware statistics
        if self.enable_document_awareness and doc_type_counts:
            avg_field_reduction = total_field_reduction / len(successful_results) if successful_results else 0
            
            stats.update({
                "document_type_distribution": doc_type_counts,
                "avg_field_reduction": avg_field_reduction,
                "efficiency_gain_estimate": f"{(avg_field_reduction / 25) * 100:.1f}%",
                "performance_improvement_estimate": f"{(avg_field_reduction / 25) * 30:.1f}%"  # Rough estimate
            })
        
        return results, stats
    
    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report including document-aware metrics."""
        base_report = DocumentAwareProcessor.get_performance_report(self)
        
        # Add Llama-specific metrics
        llama_metrics = {
            "model_type": "Llama-3.2-11B-Vision",
            "extraction_mode": self.extraction_mode,
            "device": str(self.device),
            "batch_size": self.batch_size,
            "document_awareness_enabled": self.enable_document_awareness
        }
        
        return {
            **base_report,
            "llama_specific_metrics": llama_metrics
        }


def main():
    """Test document-aware Llama processor."""
    print("🚀 Document-Aware Llama Processor - Phase 3 Testing")
    
    try:
        # Initialize processor with document awareness
        processor = DocumentAwareLlamaProcessor(
            debug=True,
            enable_document_awareness=True
        )
        
        print("✅ Processor initialized successfully")
        print(f"📊 Document awareness enabled: {processor.enable_document_awareness}")
        
        # Test single image processing
        test_image = "evaluation_data/synthetic_invoice_001.png"
        if Path(test_image).exists():
            print(f"\n📄 Testing with: {Path(test_image).name}")
            
            result = processor.process_single_image(test_image)
            
            print("✅ Processing completed")
            print(f"   Document type: {result.get('detected_document_type', 'unknown')}")
            print(f"   Fields extracted: {result.get('fields_extracted', 'unknown')}")
            print(f"   Efficiency gain: {result.get('efficiency_gain', '0%')}")
            print(f"   Processing time: {result.get('total_processing_time', 0):.2f}s")
        else:
            print(f"⚠️ Test image not found: {test_image}")
        
        print("\n✅ Document-aware Llama processor testing completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()