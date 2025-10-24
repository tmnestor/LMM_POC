#!/usr/bin/env python3
"""
Simple Field Definitions Loader - Replaces complex unified schema system

Loads field definitions from the simplified config/field_definitions.yaml file.
Much simpler than the old 1000+ line unified_schema.yaml system.
"""

from pathlib import Path
from typing import Dict, List

import yaml


class SimpleFieldLoader:
    """Simple field definitions loader."""

    def __init__(self, config_file: str = "config/field_definitions.yaml"):
        """Initialize with simplified field definitions."""
        self.config_file = config_file
        self.config_path = Path(config_file)
        self._config = None

    def _load_config(self) -> Dict:
        """Load the field definitions config."""
        if self._config is None:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Field definitions not found: {self.config_path}")

            with self.config_path.open("r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f)

        return self._config

    def get_document_fields(self, document_type: str) -> List[str]:
        """Get field list for a document type."""
        config = self._load_config()
        doc_fields = config.get("document_fields", {})

        if document_type not in doc_fields:
            # Return invoice fields as default
            return doc_fields.get("invoice", {}).get("fields", [])

        return doc_fields[document_type].get("fields", [])

    def get_supported_document_types(self) -> List[str]:
        """Get list of supported document types."""
        config = self._load_config()
        return config.get("supported_document_types", ["invoice", "receipt", "bank_statement"])

    def get_field_description(self, field_name: str) -> str:
        """Get description for a field."""
        config = self._load_config()
        descriptions = config.get("field_descriptions", {})
        return descriptions.get(field_name, f"Description for {field_name}")

    def get_critical_fields(self) -> List[str]:
        """Get list of critical fields for evaluation."""
        config = self._load_config()
        evaluation = config.get("evaluation", {})
        return evaluation.get("critical_fields", ["BUSINESS_ABN", "TOTAL_AMOUNT", "GST_AMOUNT"])


# Convenience functions for backward compatibility
def get_document_field_list(document_type: str = "invoice") -> List[str]:
    """Get field list for document type."""
    loader = SimpleFieldLoader()
    return loader.get_document_fields(document_type)


def get_supported_types() -> List[str]:
    """Get supported document types."""
    loader = SimpleFieldLoader()
    return loader.get_supported_document_types()


# Testing
if __name__ == "__main__":
    print("🧪 Testing SimpleFieldLoader\n")

    loader = SimpleFieldLoader()

    # Test document types
    for doc_type in ["invoice", "receipt", "bank_statement", "universal"]:
        fields = loader.get_document_fields(doc_type)
        print(f"✅ {doc_type}: {len(fields)} fields")

    # Test other functions
    supported = loader.get_supported_document_types()
    print(f"✅ Supported types: {supported}")

    critical = loader.get_critical_fields()
    print(f"✅ Critical fields: {critical}")

    print("\n✅ SimpleFieldLoader test complete!")