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
        self, document_type: str, field_list: List[str]
    ) -> str:
        """
        Render prompt using pure YAML templates - no Python overrides.

        Args:
            document_type: Document type (invoice, receipt, bank_statement)
            field_list: List of fields to extract for this document

        Returns:
            Complete prompt string generated from YAML templates only
        """
        if document_type not in self.unified_schema.get("document_types", {}):
            raise ValueError(f"Unsupported document type: {document_type}")

        doc_config = self.unified_schema["document_types"][document_type]
        templates = self.unified_schema["prompt_templates"]
        field_definitions = self.unified_schema["field_definitions"]

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

    def get_document_field_list(self, document_type: str) -> List[str]:
        """
        Get field list for document type from unified schema - no hardcoding.

        Args:
            document_type: Document type (invoice, receipt, bank_statement)

        Returns:
            List of field names in semantic order
        """
        if document_type not in self.unified_schema.get("document_types", {}):
            raise ValueError(f"Unsupported document type: {document_type}")

        doc_config = self.unified_schema["document_types"][document_type]
        return doc_config["required_fields"]

    def validate_field_consistency(self) -> bool:
        """
        Validate unified schema consistency - fail fast if configuration is invalid.

        Returns:
            True if schema is consistent, raises ValueError if not
        """
        semantic_order = self.unified_schema.get("semantic_field_order", [])
        field_definitions = self.unified_schema.get("field_definitions", {})
        document_types = self.unified_schema.get("document_types", {})

        # Check that all fields in semantic order have definitions
        missing_definitions = []
        for field in semantic_order:
            if field not in field_definitions:
                missing_definitions.append(field)

        if missing_definitions:
            raise ValueError(
                f"❌ FATAL: Fields in semantic order missing definitions: {missing_definitions}\n"
                "💡 Add field definitions to unified_schema.yaml"
            )

        # Check that all document type required fields have definitions
        missing_doc_fields = []
        for doc_type, doc_config in document_types.items():
            for field in doc_config.get("required_fields", []):
                if field not in field_definitions:
                    missing_doc_fields.append(f"{doc_type}.{field}")

        if missing_doc_fields:
            raise ValueError(
                f"❌ FATAL: Document type fields missing definitions: {missing_doc_fields}\n"
                "💡 Add field definitions to unified_schema.yaml"
            )

        if self.debug:
            print("✅ Unified schema consistency validation passed")

        return True

    def get_supported_document_types(self) -> List[str]:
        """Get list of supported document types from unified schema."""
        return list(self.unified_schema.get("document_types", {}).keys())

    def get_field_definition(self, field_name: str) -> Dict[str, Any]:
        """Get field definition from unified schema."""
        field_definitions = self.unified_schema.get("field_definitions", {})
        if field_name not in field_definitions:
            raise ValueError(f"Field '{field_name}' not found in unified schema")
        return field_definitions[field_name]

    def get_document_config(self, document_type: str) -> Dict[str, Any]:
        """Get document type configuration from unified schema."""
        document_types = self.unified_schema.get("document_types", {})
        if document_type not in document_types:
            raise ValueError(
                f"Document type '{document_type}' not found in unified schema"
            )
        return document_types[document_type]
