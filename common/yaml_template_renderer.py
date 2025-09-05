#!/usr/bin/env python3
"""
Pure YAML Template Renderer - True YAML-First Architecture
Eliminates Python dynamic overrides and enables template-based prompt generation.
"""

from pathlib import Path
from typing import Any, Dict, List

import yaml


class PureYAMLRenderer:
    """Pure YAML template renderer with zero Python dynamic overrides."""

    def __init__(self, debug: bool = False):
        """Initialize renderer with unified schema loading."""
        self.debug = debug
        self.unified_schema = self._load_unified_schema()

    def _load_unified_schema(self) -> Dict[str, Any]:
        """Load unified schema configuration - single source of truth."""
        try:
            schema_path = (
                Path(__file__).parent.parent / "config" / "unified_schema.yaml"
            )
            if not schema_path.exists():
                raise FileNotFoundError(
                    f"❌ FATAL: Unified schema not found at {schema_path}\n"
                    "💡 This is required for true YAML-first architecture.\n"
                    "💡 Run the architectural remediation to create it."
                )

            with schema_path.open("r", encoding="utf-8") as f:
                schema_data = yaml.safe_load(f)

            if self.debug:
                print(
                    f"✅ Loaded unified schema v{schema_data.get('schema_version', 'unknown')}"
                )

            return schema_data

        except Exception as e:
            raise ValueError(
                f"❌ FATAL: Could not load unified schema: {e}\n"
                "💡 Check YAML syntax and file permissions\n"
                "💡 Ensure unified schema exists and is valid"
            ) from e

    def render_prompt_for_document_type(
        self, document_type: str, field_list: List[str], model_name: str = None
    ) -> str:
        """
        Render prompt using pure YAML templates - with optional model-specific overrides.

        Args:
            document_type: Document type (invoice, receipt, bank_statement)
            field_list: List of fields to extract for this document
            model_name: Optional model name for model-specific templates (llama, internvl3)

        Returns:
            Complete prompt string generated from YAML templates only
        """
        document_types = self.unified_schema.get("document_types", {})
        
        # Check model exists
        if model_name not in document_types:
            raise ValueError(f"Model '{model_name}' not found in document_types. Available models: {list(document_types.keys())}")
        
        # Check document type exists for model
        if document_type not in document_types[model_name]:
            raise ValueError(f"Document type '{document_type}' not found for model '{model_name}'. Available types: {list(document_types[model_name].keys())}")

        doc_config = document_types[model_name][document_type]
        prompt_templates = self.unified_schema["prompt_templates"]
        all_field_definitions = self.unified_schema["field_definitions"]

        # Get explicit model templates - no defaults, no overrides
        if model_name and model_name in prompt_templates:
            templates = prompt_templates[model_name]
        else:
            raise ValueError(f"Model '{model_name}' not found in prompt_templates. Available models: {list(prompt_templates.keys())}")

        # Get model-specific field definitions
        if model_name and model_name in all_field_definitions:
            field_definitions = all_field_definitions[model_name]
        else:
            raise ValueError(f"Model '{model_name}' not found in field_definitions. Available models: {list(all_field_definitions.keys())}")

        # Prepare template variables - no dynamic generation, just data substitution
        template_vars = {
            "field_count": len(field_list),
            "last_field": field_list[-1] if field_list else "TOTAL_AMOUNT",
            "document_type": document_type,
        }

        # Build prompt using pure template substitution
        prompt_parts = []

        # Add expertise frame
        expertise_frame = templates["expertise_frame"]
        prompt_parts.append(expertise_frame)

        # Add critical instructions with template substitution
        critical_header = templates["critical_instructions_header"]
        prompt_parts.append(f"\n{critical_header}")

        for instruction in templates["critical_instructions"]:
            rendered_instruction = self._render_template_string(
                instruction, template_vars
            )
            prompt_parts.append(f"- {rendered_instruction}")

        # Add output format with template substitution
        output_format = self._render_template_string(
            templates["output_format"], template_vars
        )
        prompt_parts.append(f"\n{output_format}")

        # Add field instructions in semantic order (from unified schema)
        prompt_parts.append("")  # Empty line
        for field in field_list:
            if field in field_definitions:
                instruction = field_definitions[field]["instruction"]
                prompt_parts.append(f"{field}: {instruction}")

        # Add format rules with template substitution
        format_rules_header = templates["format_rules_header"]
        prompt_parts.append(f"\n{format_rules_header}")

        for rule in templates["format_rules"]:
            rendered_rule = self._render_template_string(rule, template_vars)
            prompt_parts.append(f"- {rendered_rule}")

        # Add stop instruction with template substitution
        stop_instruction = self._render_template_string(
            templates["stop_instruction"], template_vars
        )
        prompt_parts.append(f"\n{stop_instruction}")

        return "\n".join(prompt_parts)

    def render_universal_prompt(self, model_name: str = "llama") -> str:
        """
        Render universal extraction prompt using universal_extraction templates.
        
        This method generates single-pass extraction prompts that process all 17 fields
        in one call, eliminating the need for document type detection.
        
        Args:
            model_name: Model name for model-specific templates (llama, internvl3)
            
        Returns:
            Complete universal prompt string for single-pass extraction
        """
        universal_extraction = self.unified_schema.get("universal_extraction", {})
        semantic_field_order = self.unified_schema.get("semantic_field_order", {})
        
        # Validate model exists in universal_extraction
        if model_name not in universal_extraction:
            raise ValueError(f"Model '{model_name}' not found in universal_extraction. Available models: {list(universal_extraction.keys())}")
        
        # Get model-specific universal template
        universal_template = universal_extraction[model_name]
        
        # Get semantic field order for this model (used for consistent ordering)
        if model_name in semantic_field_order:
            field_list = semantic_field_order[model_name]
        else:
            # Fallback to a default 17-field list if semantic order not available
            field_list = [
                "DOCUMENT_TYPE", "INVOICE_DATE", "SUPPLIER_NAME", "BUSINESS_ABN", "BUSINESS_ADDRESS",
                "PAYER_NAME", "PAYER_ADDRESS", "LINE_ITEM_DESCRIPTIONS", "LINE_ITEM_QUANTITIES",
                "LINE_ITEM_PRICES", "LINE_ITEM_TOTAL_PRICES", "GST_AMOUNT", "IS_GST_INCLUDED",
                "TOTAL_AMOUNT", "STATEMENT_DATE_RANGE", "TRANSACTION_DATES", "TRANSACTION_AMOUNTS_PAID"
            ]
            
        # Prepare template variables for universal extraction
        template_vars = {
            "field_count": len(field_list),
            "last_field": field_list[-1] if field_list else "TRANSACTION_AMOUNTS_PAID",
            "model_name": model_name,
        }
        
        if self.debug:
            print(f"🌍 Rendering universal prompt for {model_name} with {len(field_list)} fields")
            print(f"   Last field: {template_vars['last_field']}")
        
        # Build universal prompt using template sections
        prompt_parts = []
        
        # Add system prompt
        if "system_prompt" in universal_template:
            system_prompt = self._render_template_string(
                universal_template["system_prompt"], template_vars
            )
            prompt_parts.append(system_prompt)
        
        # Add field instructions
        if "field_instructions" in universal_template:
            field_instructions = self._render_template_string(
                universal_template["field_instructions"], template_vars
            )
            prompt_parts.append(f"\n{field_instructions}")
            
        # Add output format if available
        if "output_format" in universal_template:
            output_format = self._render_template_string(
                universal_template["output_format"], template_vars
            )
            prompt_parts.append(f"\nEXPECTED OUTPUT FORMAT:\n{output_format}")
        
        universal_prompt = "\n".join(prompt_parts)
        
        if self.debug:
            print(f"✅ Generated universal prompt ({len(universal_prompt)} chars)")
            
        return universal_prompt

    def get_universal_field_list(self, model_name: str = "llama") -> List[str]:
        """
        Get universal field list in semantic order for the specified model.
        
        Args:
            model_name: Model name (llama, internvl3)
            
        Returns:
            List of all 17 universal fields in semantic order
        """
        semantic_field_order = self.unified_schema.get("semantic_field_order", {})
        
        if model_name in semantic_field_order:
            field_list = semantic_field_order[model_name]
        else:
            # Fallback universal field list
            field_list = [
                "DOCUMENT_TYPE", "INVOICE_DATE", "SUPPLIER_NAME", "BUSINESS_ABN", "BUSINESS_ADDRESS",
                "PAYER_NAME", "PAYER_ADDRESS", "LINE_ITEM_DESCRIPTIONS", "LINE_ITEM_QUANTITIES", 
                "LINE_ITEM_PRICES", "LINE_ITEM_TOTAL_PRICES", "GST_AMOUNT", "IS_GST_INCLUDED",
                "TOTAL_AMOUNT", "STATEMENT_DATE_RANGE", "TRANSACTION_DATES", "TRANSACTION_AMOUNTS_PAID"
            ]
            
        if self.debug:
            print(f"🌍 Universal field list for {model_name}: {len(field_list)} fields")
            
        return field_list

    def _render_template_string(self, template: str, variables: Dict[str, Any]) -> str:
        """
        Render template string with variable substitution.
        Uses simple {variable} placeholder format - no complex logic.
        """
        rendered = template
        for key, value in variables.items():
            placeholder = f"{{{key}}}"
            if placeholder in rendered:
                rendered = rendered.replace(placeholder, str(value))

        return rendered

    def get_document_field_list(self, document_type: str, model_name: str = "llama") -> List[str]:
        """
        Get field list for document type from unified schema - no hardcoding.

        Args:
            document_type: Document type (invoice, receipt, bank_statement)
            model_name: Model name for model-specific configuration (llama, internvl3)

        Returns:
            List of field names in semantic order
        """
        document_types = self.unified_schema.get("document_types", {})
        
        if model_name not in document_types:
            raise ValueError(f"Model '{model_name}' not found in document_types. Available models: {list(document_types.keys())}")
            
        if document_type not in document_types[model_name]:
            raise ValueError(f"Document type '{document_type}' not found for model '{model_name}'. Available types: {list(document_types[model_name].keys())}")

        doc_config = document_types[model_name][document_type]
        return doc_config["required_fields"]

    def validate_field_consistency(self) -> bool:
        """
        Validate unified schema consistency - fail fast if configuration is invalid.

        Returns:
            True if schema is consistent, raises ValueError if not
        """
        semantic_field_order = self.unified_schema.get("semantic_field_order", {})
        field_definitions = self.unified_schema.get("field_definitions", {})
        document_types = self.unified_schema.get("document_types", {})

        # Validate each model's consistency
        for model_name in ["llama", "internvl3"]:
            # Check semantic field order exists for model
            if model_name not in semantic_field_order:
                raise ValueError(f"❌ FATAL: Model '{model_name}' not found in semantic_field_order")
                
            # Check field definitions exist for model  
            if model_name not in field_definitions:
                raise ValueError(f"❌ FATAL: Model '{model_name}' not found in field_definitions")
                
            # Check document types exist for model
            if model_name not in document_types:
                raise ValueError(f"❌ FATAL: Model '{model_name}' not found in document_types")
            
            model_semantic_order = semantic_field_order[model_name]
            model_field_definitions = field_definitions[model_name] 
            model_document_types = document_types[model_name]

            # Check that all fields in semantic order have definitions
            missing_definitions = []
            for field in model_semantic_order:
                if field not in model_field_definitions:
                    missing_definitions.append(f"{model_name}.{field}")

            if missing_definitions:
                raise ValueError(
                    f"❌ FATAL: Fields in semantic order missing definitions: {missing_definitions}\n"
                    "💡 Add field definitions to unified_schema.yaml"
                )

            # Check that all document type required fields have definitions
            missing_doc_fields = []
            for doc_type, doc_config in model_document_types.items():
                for field in doc_config.get("required_fields", []):
                    if field not in model_field_definitions:
                        missing_doc_fields.append(f"{model_name}.{doc_type}.{field}")

            if missing_doc_fields:
                raise ValueError(
                    f"❌ FATAL: Document type fields missing definitions: {missing_doc_fields}\n"
                    "💡 Add field definitions to unified_schema.yaml"
                )

        if self.debug:
            print("✅ Unified schema consistency validation passed for all models")

        return True

    def get_supported_document_types(self, model_name: str = "llama") -> List[str]:
        """Get list of supported document types from unified schema for a specific model."""
        document_types = self.unified_schema.get("document_types", {})
        
        if model_name not in document_types:
            raise ValueError(f"Model '{model_name}' not found in document_types. Available models: {list(document_types.keys())}")
            
        return list(document_types[model_name].keys())

    def get_field_definition(self, field_name: str, model_name: str = "llama") -> Dict[str, Any]:
        """Get field definition from unified schema for a specific model."""
        field_definitions = self.unified_schema.get("field_definitions", {})
        
        if model_name not in field_definitions:
            raise ValueError(f"Model '{model_name}' not found in field_definitions. Available models: {list(field_definitions.keys())}")
            
        model_field_definitions = field_definitions[model_name]
        if field_name not in model_field_definitions:
            raise ValueError(f"Field '{field_name}' not found for model '{model_name}' in unified schema")
            
        return model_field_definitions[field_name]

    def get_document_config(self, document_type: str, model_name: str = "llama") -> Dict[str, Any]:
        """Get document type configuration from unified schema for a specific model."""
        document_types = self.unified_schema.get("document_types", {})
        
        if model_name not in document_types:
            raise ValueError(f"Model '{model_name}' not found in document_types. Available models: {list(document_types.keys())}")
            
        model_document_types = document_types[model_name]
        if document_type not in model_document_types:
            raise ValueError(f"Document type '{document_type}' not found for model '{model_name}' in unified schema")
            
        return model_document_types[document_type]
