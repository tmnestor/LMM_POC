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


class DocumentTypeFieldSchema:
    """
    Clean document-aware schema loader - no legacy compatibility.
    
    Automatically detects document type and selects appropriate field schema,
    optimized purely for document-aware extraction with maximum efficiency.
    """
    
    def __init__(self, schema_file: str = "field_schema_v4.yaml", fallback_file: Optional[str] = None):
        """
        Initialize document-aware schema loader - clean, no legacy compatibility.
        
        Args:
            schema_file: Path to v3+ schema file with document-specific definitions
            fallback_file: Ignored - no fallback support for clean architecture
        """
        if fallback_file is not None:
            print("⚠️  Fallback file ignored - document-aware system uses clean architecture")
            
        self.schema_file = schema_file
        
        # Load document-aware schema
        self.schema = self._load_document_schema()
        
        # Cache for performance
        self._document_schemas_cache = {}
        self._compiled_schemas_cache = {}
        
        # Document type detector (will be set by processor)
        self.document_detector: Optional[DocumentTypeDetector] = None
        
        print("✅ Document-aware schema loader initialized (clean architecture)")
        print(f"   Schema: {self.schema_file}")
        print(f"   Extraction mode: {self.schema.get('extraction_mode', 'document_aware')}")
        print(f"   Supported document types: {len(self.schema.get('document_schemas', {}))}")
    
    def _load_document_schema(self) -> dict:
        """Load document-aware schema with comprehensive error handling."""
        try:
            schema_path = Path(__file__).parent / self.schema_file
            
            if not schema_path.exists():
                raise FileNotFoundError(
                    f"❌ FATAL: Document-aware schema file not found: {schema_path}\n"
                    f"💡 Cannot proceed without document-aware schema\n"
                    f"💡 Ensure {self.schema_file} exists in common/ directory"
                )
            
            with schema_path.open("r", encoding="utf-8") as f:
                schema = yaml.safe_load(f)
            
            if not isinstance(schema, dict):
                raise ValueError(
                    f"❌ FATAL: Invalid document-aware schema structure in {self.schema_file}\n"
                    f"💡 Expected dictionary at root level"
                )
            
            # Validate required sections
            required_sections = ["common_fields", "document_schemas", "groups"]
            missing_sections = [s for s in required_sections if s not in schema]
            
            if missing_sections:
                raise ValueError(
                    f"❌ FATAL: Missing required sections in document schema: {missing_sections}\n"
                    f"💡 Required sections: {required_sections}"
                )
            
            return schema
            
        except yaml.YAMLError as e:
            raise ValueError(
                f"❌ FATAL: Invalid YAML in {self.schema_file}: {e}\n"
                f"💡 Check YAML syntax and structure"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"❌ FATAL: Document-aware schema loading failed: {e}\n"
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
            Schema type string (invoice, bank_statement, receipt, or unknown)
        """
        if not self.document_detector:
            print("⚠️ No document detector configured - using unified fallback")
            return "unknown"
        
        try:
            result = self.document_detector.detect_document_type(image_path)
            detected_type = result['type']
            confidence = result['confidence']
            
            print(f"📋 Document detected as: {detected_type} (confidence: {confidence:.2f})")
            
            # Map detected type to schema type
            schema_type = self._map_to_schema_type(detected_type)
            
            if schema_type and confidence >= self.document_detector.confidence_threshold:
                if detected_type != schema_type:
                    print(f"   → Mapping '{detected_type}' to schema type '{schema_type}'")
                return schema_type
            else:
                print("⚠️ Low confidence or unknown type - falling back to unified")
                return "unknown"
                
        except Exception as e:
            print(f"❌ Document detection failed: {e}")
            return "unknown"
    
    def _map_to_schema_type(self, detected_type: str) -> Optional[str]:
        """
        Map detected document type to canonical schema type.
        
        Args:
            detected_type: The detected document type (e.g., 'estimate', 'quote')
            
        Returns:
            Canonical schema type ('invoice', 'receipt', 'bank_statement') or None
        """
        type_mapping = {
            "invoice": "invoice",
            "tax invoice": "invoice",
            "bill": "invoice",
            "estimate": "invoice",
            "quote": "invoice",
            "quotation": "invoice",
            "proforma invoice": "invoice",
            "receipt": "receipt",
            "purchase receipt": "receipt",
            "payment receipt": "receipt",
            "sales receipt": "receipt",
            "bank statement": "bank_statement",
            "account statement": "bank_statement",
            "credit card statement": "bank_statement",
            "statement": "bank_statement"
        }
        
        # Normalize the detected type
        normalized = detected_type.lower().strip()
        return type_mapping.get(normalized)
    
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
        
        # Handle unknown/unsupported types - fail fast for clean architecture
        if document_type == "unknown" or document_type not in self.schema["document_schemas"]:
            raise RuntimeError(
                f"❌ FATAL: Unsupported document type '{document_type}'\n"
                f"💡 Supported types: {list(self.schema['document_schemas'].keys())}\n"
                f"💡 Document-aware system requires proper document classification\n"
                f"💡 No fallback support - use clean document-aware approach only"
            )
        
        # Build document-specific schema
        doc_schema_def = self.schema["document_schemas"][document_type]
        
        # Start with common fields if inherited
        complete_fields = []
        if doc_schema_def.get("inherits_common", False):
            complete_fields.extend(self.schema["common_fields"])
        
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
            "groups": self.schema["groups"],
            "extraction_mode": "document_aware"
        }
        
        # Cache result
        self._compiled_schemas_cache[document_type] = complete_schema
        
        print(f"✅ Schema compiled for {document_type}: {len(complete_fields)} fields")
        
        return complete_schema
    
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
        Get document-specific schema metrics (no unified comparison in clean architecture).
        
        Args:
            document_type: Document type to analyze
            
        Returns:
            Document schema metrics
        """
        specific_schema = self.get_document_schema(document_type)
        
        # Get field names from schema
        if isinstance(specific_schema["fields"][0], dict):
            specific_fields = set(f["name"] for f in specific_schema["fields"])
        else:
            specific_fields = set(specific_schema["fields"])
        
        return {
            "document_type": document_type,
            "field_count": len(specific_fields),
            "fields": sorted(list(specific_fields)),
            "extraction_mode": specific_schema.get("extraction_mode", "document_aware"),
            "architecture": "Clean document-aware (no legacy compatibility)"
        }
    
    def get_supported_document_types(self) -> List[str]:
        """Get list of supported document types."""
        return list(self.schema["document_schemas"].keys())
    
    def validate_document_type_schema(self, document_type: str) -> Dict[str, Any]:
        """
        Validate schema for a specific document type.
        
        Args:
            document_type: Document type to validate
            
        Returns:
            Validation results
        """
        if document_type not in self.schema["document_schemas"]:
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
   Schema Version: {self.schema.get('schema_version', 'Unknown')}
   Extraction Mode: {self.schema.get('extraction_mode', 'Unknown')}
   Supported Document Types: {len(self.get_supported_document_types())}
   Architecture: Clean document-aware (no legacy compatibility)

📈 DOCUMENT TYPE SCHEMAS:
"""
        
        for doc_type in self.get_supported_document_types():
            comparison = self.compare_schemas(doc_type)
            validation = self.validate_document_type_schema(doc_type)
            
            status = "✅" if validation["valid"] else "❌"
            
            report += f"""
   {status} {doc_type.upper()}:
      Fields: {comparison['field_count']} fields
      Architecture: {comparison['architecture']}
      Valid: {validation['valid']}
"""
        
        # Common fields info
        common_field_count = len(self.schema["common_fields"])
        report += f"""
🔄 COMMON FIELDS:
   Shared across document types: {common_field_count} fields
   
🎯 PERFORMANCE TARGETS:
   Field Reduction: 20-52% per document type
   Processing Speed: 30%+ faster  
   Accuracy Target: 90%+ per document type
   Architecture: Clean, efficient, no technical debt
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
        model_templates = self.schema.get("model_prompt_templates", {})
        
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
        
        # Generate full report
        print("\n" + schema_loader.generate_schema_report())
        
        print("\n✅ Schema loader testing completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()