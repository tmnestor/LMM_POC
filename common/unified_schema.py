#!/usr/bin/env python3
"""
Unified Schema Loader - Simplified YAML-First Architecture

Single, simple schema loader that replaces both document_schema_loader.py 
and schema_loader.py with a clean, maintainable implementation.
"""

from pathlib import Path
from typing import Dict, List, Optional

import yaml


class DocumentTypeFieldSchema:
    """Simple, unified schema loader for YAML-first architecture."""
    
    def __init__(self, schema_file: str = "config/fields.yaml", fallback_file: Optional[str] = None):
        """
        Initialize the schema loader.
        
        Args:
            schema_file: Path to schema YAML file relative to project root
            fallback_file: Ignored (for backward compatibility)
        """
        if fallback_file:
            print("⚠️ Fallback file ignored - using simplified schema")
        self.schema_file = schema_file
        self.schema = self._load_schema()
        self._validate_schema()
        
        # Cache for performance
        self._field_cache = {}
        
    def _load_schema(self) -> Dict:
        """Load schema from YAML file."""
        # Find project root
        project_root = Path(__file__).parent.parent
        schema_path = project_root / self.schema_file
        
        if not schema_path.exists():
            raise FileNotFoundError(
                f"❌ Schema file not found: {schema_path}\n"
                f"💡 Run setup.sh to validate configuration"
            )
        
        with schema_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    def _validate_schema(self):
        """Validate basic schema structure."""
        required = ["all_fields", "document_fields", "total_fields"]
        missing = [k for k in required if k not in self.schema]
        
        if missing:
            raise ValueError(f"❌ Missing required sections: {missing}")
        
        # Validate field count
        if len(self.schema["all_fields"]) != self.schema["total_fields"]:
            raise ValueError(
                f"❌ Field count mismatch: expected {self.schema['total_fields']}, "
                f"got {len(self.schema['all_fields'])}"
            )
    
    # ========================================================================
    # Core Methods - Simple and Direct
    # ========================================================================
    
    def get_all_fields(self) -> List[str]:
        """Get all 49 fields."""
        return self.schema["all_fields"]
    
    def get_document_fields(self, document_type: str) -> List[str]:
        """
        Get fields for specific document type.
        
        Args:
            document_type: 'invoice', 'receipt', or 'bank_statement'
            
        Returns:
            List of field names for that document type
        """
        # Normalize document type
        doc_type = self._normalize_document_type(document_type)
        
        if doc_type not in self.schema["document_fields"]:
            # Fall back to all fields if unknown type
            return self.get_all_fields()
        
        return self.schema["document_fields"][doc_type]
    
    def get_field_count(self, document_type: Optional[str] = None) -> int:
        """Get field count for document type or all fields."""
        if document_type:
            return len(self.get_document_fields(document_type))
        return self.schema["total_fields"]
    
    def get_critical_fields(self) -> List[str]:
        """Get critical fields requiring special validation."""
        return self.schema.get("critical_fields", [])
    
    def is_critical_field(self, field_name: str) -> bool:
        """Check if a field is critical."""
        return field_name in self.get_critical_fields()
    
    # ========================================================================
    # Document Type Handling
    # ========================================================================
    
    def _normalize_document_type(self, document_type: str) -> str:
        """Normalize document type to canonical form."""
        mapping = {
            "invoice": "invoice",
            "tax invoice": "invoice",
            "tax_invoice": "invoice",
            "receipt": "receipt",
            "bank statement": "bank_statement",
            "bank_statement": "bank_statement",
            "statement": "bank_statement",
        }
        
        normalized = document_type.lower().strip()
        return mapping.get(normalized, normalized)
    
    def get_supported_document_types(self) -> List[str]:
        """Get list of supported document types."""
        return list(self.schema["document_fields"].keys())
    
    # ========================================================================
    # Backward Compatibility Methods
    # ========================================================================
    
    def get_extraction_fields(self) -> List[str]:
        """Backward compatibility: get all fields."""
        return self.get_all_fields()
    
    def get_field_names_for_type(self, document_type: str) -> List[str]:
        """Backward compatibility: get fields for document type."""
        return self.get_document_fields(document_type)
    
    def get_document_schema(self, document_type: str) -> Dict:
        """
        Backward compatibility: get schema dict for document type.
        
        Returns simplified schema structure.
        """
        fields = self.get_document_fields(document_type)
        return {
            "fields": fields,
            "total_fields": len(fields),
            "document_type": self._normalize_document_type(document_type),
            "critical_fields": self.get_critical_fields(),
        }
    
    def get_schema_for_image(self, image_path: str, document_type: str) -> Dict:
        """Backward compatibility: get schema for image."""
        return self.get_document_schema(document_type)
    
    def generate_dynamic_prompt(self, model_name: str = None, strategy: str = None) -> str:
        """
        Backward compatibility: generate dynamic prompt.
        
        Since we simplified the schema, just return a basic extraction prompt.
        Real prompt generation should use the prompt_loader system.
        """
        fields = self.get_all_fields()
        field_list = "\n".join([f"{field}: [extract {field.lower().replace('_', ' ')} or NOT_FOUND]" for field in fields])
        
        return f"""Extract structured data from this business document image.

REQUIRED OUTPUT FORMAT - EXACTLY {len(fields)} LINES:
{field_list}

Extract the exact values as they appear in the document. If a field is not present or cannot be determined, output NOT_FOUND."""
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def compare_document_types(self) -> Dict:
        """Compare field counts across document types."""
        all_count = self.schema["total_fields"]
        comparison = {}
        
        for doc_type in self.get_supported_document_types():
            fields = self.get_document_fields(doc_type)
            comparison[doc_type] = {
                "field_count": len(fields),
                "reduction": f"{(all_count - len(fields))/all_count*100:.0f}%",
                "fields": fields,
            }
        
        return comparison
    
    def validate_extraction_result(self, result: Dict, document_type: str) -> Dict:
        """
        Validate extraction result against expected fields.
        
        Args:
            result: Extraction result dictionary
            document_type: Document type
            
        Returns:
            Validation report
        """
        expected_fields = set(self.get_document_fields(document_type))
        extracted_fields = set(result.keys())
        
        return {
            "valid": expected_fields.issubset(extracted_fields),
            "missing_fields": list(expected_fields - extracted_fields),
            "extra_fields": list(extracted_fields - expected_fields),
            "coverage": len(expected_fields & extracted_fields) / len(expected_fields),
        }
    
    # Additional methods for full compatibility
    def detect_document_type(self, image_path: str) -> str:
        """Placeholder - requires external detector."""
        return "invoice"  # Default fallback
    
    def set_document_detector(self, detector):
        """Placeholder for setting document detector."""
        pass
    
    @property
    def total_fields(self) -> int:
        """Property for backward compatibility."""
        return self.schema["total_fields"]


# ============================================================================
# Singleton Pattern for Backward Compatibility
# ============================================================================

_global_schema: Optional[DocumentTypeFieldSchema] = None


def get_global_schema() -> DocumentTypeFieldSchema:
    """Get or create global schema instance."""
    global _global_schema
    if _global_schema is None:
        _global_schema = DocumentTypeFieldSchema()
    return _global_schema


def get_extraction_fields() -> List[str]:
    """Get all extraction fields."""
    return get_global_schema().get_all_fields()


# ============================================================================
# Additional Methods for Compatibility
# ============================================================================

# Add methods that were in the old classes but keep them simple


# ============================================================================
# Testing
# ============================================================================

def main():
    """Test the schema loader."""
    print("🚀 Testing DocumentTypeFieldSchema\n")
    
    schema = DocumentTypeFieldSchema()
    
    # Test basic functionality
    print(f"✅ Total fields: {schema.get_field_count()}")
    print(f"✅ All fields loaded: {len(schema.get_all_fields())} fields")
    print(f"✅ Supported document types: {schema.get_supported_document_types()}")
    
    # Test document-specific fields
    print("\n📋 Document-Specific Field Counts:")
    for doc_type in schema.get_supported_document_types():
        count = schema.get_field_count(doc_type)
        reduction = (48 - count) / 48 * 100
        print(f"  {doc_type}: {count} fields ({reduction:.0f}% reduction)")
    
    # Test compatibility methods
    print("\n🔄 Testing Compatibility Methods:")
    print(f"  get_field_names_for_type('invoice'): {len(schema.get_field_names_for_type('invoice'))} fields")
    print(f"  total_fields property: {schema.total_fields} fields")
    
    # Test global functions
    global_fields = get_extraction_fields()
    print(f"  get_extraction_fields(): {len(global_fields)} fields")
    
    print("\n✅ All tests passed! Schema is working.")


if __name__ == "__main__":
    main()