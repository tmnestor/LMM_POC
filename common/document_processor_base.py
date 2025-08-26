#!/usr/bin/env python3
"""
Document-Type-Aware Processor Base - Phase 3 Implementation

Base class providing document-type-specific extraction capabilities.
Integrates document detection with schema-driven field extraction.
"""

import time
from typing import Any, Dict, Optional

from common.document_schema_loader import DocumentTypeFieldSchema
from common.document_type_detector import DocumentTypeDetector


class DocumentAwareProcessor:
    """
    Base class for document-type-aware processing.
    
    Provides automatic document type detection, schema selection,
    and targeted field extraction for improved performance and accuracy.
    """
    
    def __init__(self):
        """Initialize document-aware processor components."""
        self.document_detector: Optional[DocumentTypeDetector] = None
        self.schema_loader: Optional[DocumentTypeFieldSchema] = None
        
        # Performance tracking
        self.extraction_stats = {
            "total_documents": 0,
            "by_document_type": {},
            "processing_times": [],
            "field_reductions": []
        }
        
        # Configuration
        self.config = {
            "enable_document_detection": True,
            "enable_schema_optimization": True,
            "fallback_to_unified": True,
            "confidence_threshold": 0.85
        }
    
    def initialize_document_awareness(self):
        """Initialize document detection and schema systems."""
        print("🚀 Initializing document-type-aware processing...")
        
        try:
            # Initialize schema loader
            self.schema_loader = DocumentTypeFieldSchema()
            print("✅ Document-specific schema system loaded")
            
            # Initialize document detector (will be set by child class)
            # Child classes need to call set_document_detector() after model initialization
            
            print("✅ Document-aware processing initialized")
            
        except Exception as e:
            print(f"⚠️ Document-aware initialization failed: {e}")
            print("💡 Falling back to unified schema processing")
            self.config["enable_document_detection"] = False
            self.config["enable_schema_optimization"] = False
    
    def set_document_detector(self, processor_instance):
        """Set up document detector with the processor instance."""
        if not self.schema_loader:
            print("⚠️ Schema loader not available - document detection disabled")
            return
        
        try:
            self.document_detector = DocumentTypeDetector(processor_instance)
            self.schema_loader.set_document_detector(self.document_detector)
            print("✅ Document detector configured for automatic type detection")
            
        except Exception as e:
            print(f"⚠️ Document detector setup failed: {e}")
            print("💡 Continuing with unified schema processing")
    
    def detect_and_get_schema(self, image_path: str) -> Dict[str, Any]:
        """
        Detect document type and return appropriate schema.
        
        Args:
            image_path: Path to document image
            
        Returns:
            Schema configuration for the document type
        """
        if not self.config["enable_document_detection"] or not self.schema_loader:
            return self._get_unified_schema()
        
        try:
            # Get document-specific schema (includes auto-detection)
            schema = self.schema_loader.get_schema_for_image(image_path)
            
            # Track statistics
            doc_type = schema.get("document_type", "unknown")
            if doc_type not in self.extraction_stats["by_document_type"]:
                self.extraction_stats["by_document_type"][doc_type] = 0
            self.extraction_stats["by_document_type"][doc_type] += 1
            
            # Calculate field reduction
            if doc_type != "unified_fallback":
                field_reduction = 25 - schema["total_fields"]
                self.extraction_stats["field_reductions"].append(field_reduction)
            
            return schema
            
        except Exception as e:
            print(f"⚠️ Document-specific schema failed: {e}")
            return self._get_unified_schema()
    
    def _get_unified_schema(self) -> Dict[str, Any]:
        """Get unified schema as fallback."""
        try:
            if self.schema_loader:
                return self.schema_loader.get_document_schema("unknown")
            else:
                # Basic fallback schema
                return {
                    "document_type": "unified_fallback",
                    "total_fields": 25,
                    "extraction_mode": "unified",
                    "fields": [],  # Will be handled by child class
                    "optimization_level": "standard"
                }
        except Exception:
            # Ultimate fallback
            return {
                "document_type": "unified_fallback",
                "total_fields": 25,
                "extraction_mode": "unified",
                "fields": [],
                "optimization_level": "standard"
            }
    
    def generate_document_specific_prompt(self, schema: Dict[str, Any], model_type: str = "llama") -> str:
        """
        Generate targeted prompt based on document type and schema.
        
        Args:
            schema: Document schema configuration
            model_type: Model type (llama/internvl3) for prompt optimization
            
        Returns:
            Optimized prompt for the specific document type and model
        """
        doc_type = schema.get("document_type", "unknown")
        fields = schema.get("fields", [])
        field_count = len(fields)
        
        # Get model-specific templates if available
        model_templates = self.schema_loader.v2_schema.get("model_prompt_templates", {}) if self.schema_loader else {}
        type_templates = model_templates.get(model_type, {}).get("document_type_specific", {})
        
        # Document-specific prompt configuration
        if doc_type in ["invoice"] and doc_type in type_templates:
            template = type_templates[doc_type]
            opening = template.get("opening", "Extract invoice information from this business document.")
            focus = template.get("focus", "Focus on invoice-specific fields.")
            emphasis = template.get("field_emphasis", "Pay attention to all relevant fields.")
            
        elif doc_type in ["bank_statement"] and doc_type in type_templates:
            template = type_templates[doc_type]
            opening = template.get("opening", "Extract bank statement information from this financial document.")
            focus = template.get("focus", "Focus on account and transaction details.")
            emphasis = template.get("field_emphasis", "Pay attention to balances and account information.")
            
        elif doc_type in ["receipt"] and doc_type in type_templates:
            template = type_templates[doc_type]
            opening = template.get("opening", "Extract receipt information from this purchase document.")
            focus = template.get("focus", "Focus on transaction and payment details.")
            emphasis = template.get("field_emphasis", "Pay attention to items and payment information.")
            
        else:
            # Fallback to unified approach
            opening = "Extract key-value data from this business document image."
            focus = "Focus on all relevant business document fields."
            emphasis = "Extract all available information accurately."
        
        # Generate field list
        field_list = []
        for field in fields:
            if isinstance(field, dict):
                field_name = field.get("name", "")
                field_instruction = field.get("instruction", "")
                if field_name and field_instruction:
                    field_list.append(f"{field_name}: {field_instruction}")
        
        # Construct optimized prompt
        if model_type.lower() == "internvl3":
            # Concise prompt for InternVL3
            prompt = f"""{opening}

Extract these {field_count} fields:

{chr(10).join(field_list)}

Output format: KEY: value
Use NOT_FOUND if field not visible."""
        
        else:
            # Detailed prompt for Llama
            prompt = f"""{opening}

{focus} {emphasis}

Extract exactly {field_count} fields from this document:

{chr(10).join(field_list)}

CRITICAL INSTRUCTIONS:
- Output ONLY the structured data below
- Use exactly: KEY: value (colon and space)
- Include ALL {field_count} fields even if value is NOT_FOUND
- Do NOT add explanations or conversation text
- Do NOT calculate missing values - extract only what is visible

Begin extraction:"""
        
        return prompt
    
    def process_single_image_with_document_awareness(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """
        Process single image with document-type-aware extraction.
        
        This method should be called by child classes after their model-specific processing.
        
        Args:
            image_path: Path to document image
            **kwargs: Additional processing parameters
            
        Returns:
            Processing results with document-aware metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Get document-specific schema
            schema = self.detect_and_get_schema(image_path)
            doc_type = schema.get("document_type", "unknown")
            
            print(f"📋 Processing as: {doc_type} ({schema['total_fields']} fields)")
            
            # Step 2: Generate targeted prompt
            model_type = kwargs.get("model_type", "llama")
            targeted_prompt = self.generate_document_specific_prompt(schema, model_type)
            
            # Step 3: Process with child class implementation
            # (This will be called by child classes with their specific processing logic)
            
            # Step 4: Track performance
            processing_time = time.time() - start_time
            self.extraction_stats["total_documents"] += 1
            self.extraction_stats["processing_times"].append(processing_time)
            
            return {
                "document_type": doc_type,
                "schema_used": schema,
                "targeted_prompt": targeted_prompt,
                "field_count": schema["total_fields"],
                "extraction_mode": schema.get("extraction_mode", "unknown"),
                "processing_time": processing_time,
                "optimization_level": schema.get("optimization_level", "standard")
            }
            
        except Exception as e:
            print(f"❌ Document-aware processing failed: {e}")
            # Return fallback metadata
            return {
                "document_type": "processing_error",
                "schema_used": {"total_fields": 0, "extraction_mode": "error"},
                "targeted_prompt": "",
                "field_count": 0,
                "extraction_mode": "error",
                "processing_time": time.time() - start_time,
                "error": str(e)
            }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report for document-aware processing."""
        stats = self.extraction_stats
        
        if not stats["total_documents"]:
            return {"message": "No documents processed yet"}
        
        # Calculate averages
        avg_processing_time = sum(stats["processing_times"]) / len(stats["processing_times"])
        avg_field_reduction = sum(stats["field_reductions"]) / len(stats["field_reductions"]) if stats["field_reductions"] else 0
        
        # Document type distribution
        type_distribution = {}
        for doc_type, count in stats["by_document_type"].items():
            percentage = (count / stats["total_documents"]) * 100
            type_distribution[doc_type] = {
                "count": count,
                "percentage": percentage
            }
        
        return {
            "total_documents": stats["total_documents"],
            "document_type_distribution": type_distribution,
            "performance_metrics": {
                "avg_processing_time": avg_processing_time,
                "avg_field_reduction": avg_field_reduction,
                "efficiency_gain_estimate": f"{avg_field_reduction / 25 * 100:.1f}%" if avg_field_reduction > 0 else "0%"
            },
            "document_aware_enabled": self.config["enable_document_detection"],
            "schema_optimization_enabled": self.config["enable_schema_optimization"]
        }
    
    def reset_performance_stats(self):
        """Reset performance tracking statistics."""
        self.extraction_stats = {
            "total_documents": 0,
            "by_document_type": {},
            "processing_times": [],
            "field_reductions": []
        }
        print("📊 Performance statistics reset")


def main():
    """Test document-aware processor base functionality."""
    print("🚀 Document-Aware Processor Base - Phase 3 Testing")
    
    try:
        # Initialize base processor
        processor = DocumentAwareProcessor()
        processor.initialize_document_awareness()
        
        # Test schema loading
        if processor.schema_loader:
            supported_types = processor.schema_loader.get_supported_document_types()
            print(f"✅ Supported document types: {supported_types}")
            
            # Test schema retrieval for each type
            for doc_type in supported_types:
                schema = processor.schema_loader.get_document_schema(doc_type)
                print(f"✅ {doc_type}: {schema['total_fields']} fields")
        
        print("\n✅ Base processor testing completed successfully!")
        print("\n💡 This base class will be extended by LlamaProcessor and InternVL3Processor")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()