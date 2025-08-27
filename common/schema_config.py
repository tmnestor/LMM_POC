#!/usr/bin/env python3
"""
Clean Document-Aware Schema Configuration

Separate module to avoid circular imports - provides schema initialization
and field discovery for the document-aware system.
"""


from .document_schema_loader import DocumentTypeFieldSchema


class DocumentAwareConfig:
    """Clean schema configuration without circular dependencies."""
    
    def __init__(self):
        # Initialize document schema
        self.schema_loader = DocumentTypeFieldSchema("field_schema_v4.yaml", fallback_file=None)
        
        # Generate unified field lists
        self._build_unified_fields()
        self._build_field_types()
        self._build_field_groups()
        
        print(f"📊 Schema config initialized: {self.field_count} fields across {len(self.supported_types)} document types")
    
    def _build_unified_fields(self):
        """Build unified field list from all document types."""
        all_field_names = set()
        
        for doc_type in self.schema_loader.get_supported_document_types():
            schema = self.schema_loader.get_document_schema(doc_type)
            for field in schema["fields"]:
                all_field_names.add(field["name"])
        
        self.extraction_fields = sorted(list(all_field_names))
        self.field_count = len(self.extraction_fields)
        self.supported_types = self.schema_loader.get_supported_document_types()
    
    def _build_field_types(self):
        """Build field type mappings."""
        field_types = {}
        
        for doc_type in self.schema_loader.get_supported_document_types():
            schema = self.schema_loader.get_document_schema(doc_type)
            for field in schema["fields"]:
                field_name = field["name"]
                field_type = field.get("type", "text")
                field_types[field_name] = field_type
        
        self.field_types = field_types
        
        # Build type-specific lists
        self.phone_fields = [name for name, ftype in field_types.items() if ftype == "phone"]
        self.list_fields = [name for name, ftype in field_types.items() if ftype == "list"]
        self.monetary_fields = [name for name, ftype in field_types.items() if ftype == "monetary"]
        self.numeric_id_fields = [name for name, ftype in field_types.items() if ftype == "numeric_id"]
        self.date_fields = [name for name, ftype in field_types.items() if ftype == "date"]
        self.text_fields = [name for name, ftype in field_types.items() if ftype == "text"]
        
        # New field types in v4
        self.boolean_fields = [name for name, ftype in field_types.items() if ftype == "boolean"]
        self.calculated_fields = [name for name, ftype in field_types.items() if ftype == "calculated"]
        self.transaction_list_fields = [name for name, ftype in field_types.items() if ftype == "transaction_list"]
        
        print(f"📋 Field types: phone={len(self.phone_fields)}, list={len(self.list_fields)}, monetary={len(self.monetary_fields)}")
        print(f"📋 New v4 types: boolean={len(self.boolean_fields)}, calculated={len(self.calculated_fields)}, transaction_list={len(self.transaction_list_fields)}")
    
    def _build_field_groups(self):
        """Build field groups for extraction strategies.""" 
        groups = self.schema_loader.schema.get("groups", {})
        field_groups = []
        
        for group_name, group_config in groups.items():
            # Find fields that belong to this group across all document types  
            group_fields = []
            for doc_type in self.schema_loader.get_supported_document_types():
                doc_schema = self.schema_loader.get_document_schema(doc_type)
                for field in doc_schema["fields"]:
                    if field.get("group") == group_name and field["name"] not in [f["field"] for f in group_fields]:
                        group_fields.append({
                            "field": field["name"], 
                            "instruction": field.get("instruction", f"[{field['name']} or NOT_FOUND]")
                        })
            
            if group_fields:  # Only add groups that have fields
                field_groups.append({
                    "group_name": group_config.get("name", group_name),
                    "fields": group_fields,
                    "max_tokens": group_config.get("max_tokens", 400),
                    "temperature": group_config.get("temperature", 0.0)
                })
        
        self.field_groups = field_groups


# Global instance - initialized once
_schema_config = None

def get_schema_config() -> DocumentAwareConfig:
    """Get global schema configuration."""
    global _schema_config
    if _schema_config is None:
        _schema_config = DocumentAwareConfig()
    return _schema_config