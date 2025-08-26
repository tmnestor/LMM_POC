#!/usr/bin/env python3
"""
Document-Type-Specific Schema Loader - Phase 2 Implementation

Extended schema loader supporting document-type-specific field extraction.
Maintains backward compatibility with unified schema while enabling targeted extraction.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from common.document_type_detector import DocumentTypeDetector
from common.schema_loader import FieldSchema


class DocumentTypeFieldSchema(FieldSchema):
    """
    Extended schema loader supporting document-type-specific extraction.
    
    Automatically detects document type and selects appropriate field schema,
    reducing extraction overhead and improving accuracy through targeted prompts.
    """
    
    def __init__(self, schema_file: str = "field_schema_v2.yaml", fallback_file: str = "field_schema.yaml"):
        """
        Initialize document-type-specific schema loader.
        
        Args:
            schema_file: Path to v2 schema file with document-specific definitions
            fallback_file: Path to v1 unified schema for backward compatibility
        """
        self.v2_schema_file = schema_file
        self.v1_fallback_file = fallback_file
        
        # Load v2 schema
        self.v2_schema = self._load_v2_schema()
        
        # Initialize v1 schema for fallback
        try:
            super().__init__(fallback_file)
            self.v1_available = True
        except Exception as e:
            print(f"⚠️ V1 schema not available: {e}")
            self.v1_available = False
        
        # Cache for performance
        self._document_schemas_cache = {}
        self._compiled_schemas_cache = {}
        
        # Document type detector (will be set by processor)
        self.document_detector: Optional[DocumentTypeDetector] = None
        
        print("✅ Document-type schema loader initialized")
        print(f"   V2 Schema: {self.v2_schema_file}")
        print(f"   V1 Fallback: {'Available' if self.v1_available else 'Not available'}")
        print(f"   Extraction mode: {self.v2_schema.get('extraction_mode', 'unknown')}")
    
    def _load_v2_schema(self) -> dict:
        """Load v2 schema with comprehensive error handling."""
        try:
            schema_path = Path(__file__).parent / self.v2_schema_file
            
            if not schema_path.exists():
                raise FileNotFoundError(
                    f"❌ FATAL: V2 schema file not found: {schema_path}\n"
                    f"💡 Cannot proceed without document-specific schema\n"
                    f"💡 Ensure {self.v2_schema_file} exists in common/ directory"
                )
            
            with schema_path.open("r", encoding="utf-8") as f:
                schema = yaml.safe_load(f)
            
            if not isinstance(schema, dict):
                raise ValueError(
                    f"❌ FATAL: Invalid v2 schema structure in {self.v2_schema_file}\n"
                    f"💡 Expected dictionary at root level"
                )
            
            # Validate required sections
            required_sections = ["common_fields", "document_schemas", "groups"]
            missing_sections = [s for s in required_sections if s not in schema]
            
            if missing_sections:
                raise ValueError(
                    f"❌ FATAL: Missing required sections in v2 schema: {missing_sections}\n"
                    f"💡 Required sections: {required_sections}"
                )
            
            return schema
            
        except yaml.YAMLError as e:
            raise ValueError(
                f"❌ FATAL: Invalid YAML in {self.v2_schema_file}: {e}\n"
                f"💡 Check YAML syntax and structure"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"❌ FATAL: V2 schema loading failed: {e}\n"
                f"💡 Cannot proceed without document-specific definitions"
            ) from e
    
    def set_document_detector(self, detector: DocumentTypeDetector):
        """Set document type detector for automatic classification."""
        self.document_detector = detector
        print("✅ Document detector configured for schema routing")
    
    def detect_document_type(self, image_path: str) -> str:
        """
        Detect document type for schema selection.
        
        Args:
            image_path: Path to document image
            
        Returns:
            Document type string (invoice, bank_statement, receipt, or unknown)
        """
        if not self.document_detector:
            print("⚠️ No document detector configured - using unified fallback")
            return "unknown"
        
        try:
            result = self.document_detector.detect_document_type(image_path)
            doc_type = result['type']
            confidence = result['confidence']
            
            print(f"📋 Document classified as: {doc_type} (confidence: {confidence:.2f})")
            
            # Use classification if confidence is high enough
            if confidence >= self.document_detector.confidence_threshold:
                return doc_type
            else:
                print(f"⚠️ Low confidence ({confidence:.2f}) - falling back to unified")
                return "unknown"
                
        except Exception as e:
            print(f"❌ Document detection failed: {e}")
            return "unknown"
    
    def get_document_schema(self, document_type: str) -> Dict[str, Any]:
        """
        Get complete field schema for specified document type.
        
        Args:
            document_type: Document type (invoice, bank_statement, receipt)
            
        Returns:
            Complete schema dictionary with all fields and metadata
        """
        # Use cache if available
        if document_type in self._compiled_schemas_cache:
            return self._compiled_schemas_cache[document_type]
        
        # Handle unknown/unsupported types
        if document_type == "unknown" or document_type not in self.v2_schema["document_schemas"]:
            print(f"📋 Using unified fallback for document type: {document_type}")
            return self._get_unified_fallback_schema()
        
        # Build document-specific schema
        doc_schema_def = self.v2_schema["document_schemas"][document_type]
        
        # Start with common fields if inherited
        complete_fields = []
        if doc_schema_def.get("inherits_common", False):
            complete_fields.extend(self.v2_schema["common_fields"])
        
        # Add document-specific fields
        if "specific_fields" in doc_schema_def:
            complete_fields.extend(doc_schema_def["specific_fields"])
        
        # Build complete schema
        complete_schema = {
            "fields": complete_fields,
            "total_fields": len(complete_fields),
            "document_type": document_type,
            "metadata": doc_schema_def.get("metadata", {}),
            "excluded_fields": doc_schema_def.get("excluded_fields", []),
            "validation_rules": doc_schema_def.get("validation_rules", []),
            "groups": self.v2_schema["groups"],
            "extraction_mode": "document_type_specific"
        }
        
        # Cache result
        self._compiled_schemas_cache[document_type] = complete_schema
        
        print(f"✅ Schema compiled for {document_type}: {len(complete_fields)} fields")
        print(f"   Efficiency gain: {doc_schema_def.get('metadata', {}).get('efficiency_gain', 'Unknown')}")
        
        return complete_schema
    
    def _get_unified_fallback_schema(self) -> Dict[str, Any]:
        """Get unified schema for fallback when document type is unknown."""
        if not self.v1_available:
            raise RuntimeError(
                f"❌ FATAL: Unified fallback schema not available\n"
                f"💡 Cannot process unknown document types without v1 schema\n"
                f"💡 Ensure {self.v1_fallback_file} exists for fallback support"
            )
        
        # Use parent class methods for v1 schema
        return {
            "fields": self.field_names,  # From parent FieldSchema class
            "total_fields": self.total_fields,
            "document_type": "unified_fallback",
            "metadata": {"description": "Unified 25-field schema for unknown document types"},
            "excluded_fields": [],
            "validation_rules": self.schema.get("interdependency_rules", []),
            "groups": self.schema["groups"],
            "extraction_mode": "unified"
        }
    
    def get_schema_for_image(self, image_path: str) -> Dict[str, Any]:
        """
        One-step method: detect document type and return appropriate schema.
        
        Args:
            image_path: Path to document image
            
        Returns:
            Complete schema for the detected document type
        """
        document_type = self.detect_document_type(image_path)
        return self.get_document_schema(document_type)
    
    def get_field_names_for_type(self, document_type: str) -> List[str]:
        """Get list of field names for specific document type."""
        schema = self.get_document_schema(document_type)
        return [field["name"] for field in schema["fields"]]
    
    def get_field_count_for_type(self, document_type: str) -> int:
        """Get field count for specific document type."""
        return len(self.get_field_names_for_type(document_type))
    
    def compare_schemas(self, document_type: str) -> Dict[str, Any]:
        """
        Compare document-specific schema with unified schema.
        
        Args:
            document_type: Document type to compare
            
        Returns:
            Comparison metrics and field differences
        """
        specific_schema = self.get_document_schema(document_type)
        unified_schema = self._get_unified_fallback_schema() if self.v1_available else None
        
        if not unified_schema:
            return {"error": "Unified schema not available for comparison"}
        
        specific_fields = set(f["name"] for f in specific_schema["fields"])
        unified_fields = set(f["name"] for f in unified_schema["fields"])
        
        return {
            "document_type": document_type,
            "specific_field_count": len(specific_fields),
            "unified_field_count": len(unified_fields),
            "field_reduction": len(unified_fields) - len(specific_fields),
            "field_reduction_percentage": ((len(unified_fields) - len(specific_fields)) / len(unified_fields)) * 100,
            "excluded_fields": list(unified_fields - specific_fields),
            "efficiency_gain": f"{((len(unified_fields) - len(specific_fields)) / len(unified_fields)) * 100:.0f}% fewer fields"
        }
    
    def get_supported_document_types(self) -> List[str]:
        """Get list of supported document types."""
        return list(self.v2_schema["document_schemas"].keys())
    
    def validate_document_type_schema(self, document_type: str) -> Dict[str, Any]:
        """
        Validate schema for a specific document type.
        
        Args:
            document_type: Document type to validate
            
        Returns:
            Validation results
        """
        if document_type not in self.v2_schema["document_schemas"]:
            return {
                "valid": False,
                "error": f"Document type '{document_type}' not found in schema"
            }
        
        try:
            schema = self.get_document_schema(document_type)
            
            # Basic validation
            field_names = [f["name"] for f in schema["fields"]]
            duplicate_fields = [name for name in field_names if field_names.count(name) > 1]
            
            # Check required properties
            validation_results = {
                "valid": True,
                "document_type": document_type,
                "field_count": len(schema["fields"]),
                "duplicate_fields": duplicate_fields,
                "has_critical_fields": any(f.get("group") == "critical" for f in schema["fields"]),
                "validation_rules_count": len(schema["validation_rules"])
            }
            
            if duplicate_fields:
                validation_results["valid"] = False
                validation_results["error"] = f"Duplicate fields found: {duplicate_fields}"
            
            return validation_results
            
        except Exception as e:
            return {
                "valid": False,
                "document_type": document_type,
                "error": f"Schema validation failed: {str(e)}"
            }
    
    def generate_schema_report(self) -> str:
        """Generate comprehensive report of all document schemas."""
        report = f"""
📊 DOCUMENT-TYPE-SPECIFIC SCHEMA REPORT
{'='*60}

📋 OVERVIEW:
   Schema Version: {self.v2_schema.get('schema_version', 'Unknown')}
   Extraction Mode: {self.v2_schema.get('extraction_mode', 'Unknown')}
   Supported Document Types: {len(self.get_supported_document_types())}
   V1 Fallback Available: {'Yes' if self.v1_available else 'No'}

📈 DOCUMENT TYPE SCHEMAS:
"""
        
        for doc_type in self.get_supported_document_types():
            comparison = self.compare_schemas(doc_type)
            validation = self.validate_document_type_schema(doc_type)
            
            status = "✅" if validation["valid"] else "❌"
            
            report += f"""
   {status} {doc_type.upper()}:
      Fields: {comparison['specific_field_count']} (vs {comparison['unified_field_count']} unified)
      Efficiency: {comparison['efficiency_gain']}
      Excluded: {len(comparison['excluded_fields'])} fields
      Valid: {validation['valid']}
"""
        
        # Common fields info
        common_field_count = len(self.v2_schema["common_fields"])
        report += f"""
🔄 COMMON FIELDS:
   Shared across document types: {common_field_count} fields
   
🎯 PERFORMANCE TARGETS:
   Field Reduction: 28-52% per document type
   Processing Speed: 30-50% faster
   Accuracy Target: 90%+ per document type
"""
        
        return report
    
    def get_extraction_strategy(self, document_type: str) -> Dict[str, Any]:
        """
        Get extraction strategy configuration for document type.
        
        Args:
            document_type: Document type
            
        Returns:
            Strategy configuration for processors
        """
        schema = self.get_document_schema(document_type)
        
        # Get model-specific templates if available
        model_templates = self.v2_schema.get("model_prompt_templates", {})
        
        return {
            "document_type": document_type,
            "extraction_mode": schema["extraction_mode"],
            "field_count": schema["total_fields"],
            "validation_rules": schema["validation_rules"],
            "model_templates": model_templates,
            "optimization_level": "aggressive" if schema["total_fields"] < 20 else "standard"
        }


def main():
    """Test document-type-specific schema loading."""
    print("🚀 Document-Type-Specific Schema Loader - Phase 2 Testing")
    
    try:
        # Initialize loader
        schema_loader = DocumentTypeFieldSchema()
        
        # Test each document type
        for doc_type in schema_loader.get_supported_document_types():
            print(f"\n📋 Testing {doc_type.upper()} schema:")
            
            # Get schema
            schema = schema_loader.get_document_schema(doc_type)
            print(f"   Fields: {schema['total_fields']}")
            
            # Validate
            validation = schema_loader.validate_document_type_schema(doc_type)
            print(f"   Valid: {validation['valid']}")
            
            # Compare with unified
            comparison = schema_loader.compare_schemas(doc_type)
            print(f"   Efficiency: {comparison['efficiency_gain']}")
        
        # Generate full report
        print("\n" + schema_loader.generate_schema_report())
        
        print("\n✅ Schema loader testing completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()