#!/usr/bin/env python3
"""
YAML Consistency Validator - True YAML-First Architecture
Comprehensive validation of unified schema configuration consistency.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import yaml
from rich.console import Console
from rich.table import Table

console = Console()


class YAMLConsistencyValidator:
    """Comprehensive validator for unified schema consistency."""

    def __init__(self, debug: bool = False):
        """Initialize validator."""
        self.debug = debug
        self.project_root = Path(__file__).parent.parent
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_all(self) -> bool:
        """
        Run comprehensive validation of all YAML configurations.

        Returns:
            True if all validations pass, False otherwise
        """
        console.print(
            "[bold blue]🔍 YAML Consistency Validation - True YAML-First Architecture[/bold blue]"
        )
        console.print()

        validation_results = []

        # Run all validation checks
        validation_results.append(
            ("Unified Schema Structure", self._validate_unified_schema_structure())
        )
        validation_results.append(
            ("Field Definitions Consistency", self._validate_field_definitions())
        )
        validation_results.append(
            ("Document Type Consistency", self._validate_document_types())
        )
        validation_results.append(
            ("Template Placeholder Consistency", self._validate_template_placeholders())
        )
        validation_results.append(
            ("Field Order Consistency", self._validate_field_order_consistency())
        )
        validation_results.append(
            ("Legacy Configuration Detection", self._detect_legacy_configurations())
        )
        validation_results.append(
            ("Python Override Detection", self._detect_python_overrides())
        )

        # Display results table
        self._display_validation_results(validation_results)

        # Display detailed errors and warnings
        if self.errors:
            console.print("\n[bold red]❌ ERRORS:[/bold red]")
            for i, error in enumerate(self.errors, 1):
                console.print(f"  {i}. {error}")

        if self.warnings:
            console.print("\n[bold yellow]⚠️  WARNINGS:[/bold yellow]")
            for i, warning in enumerate(self.warnings, 1):
                console.print(f"  {i}. {warning}")

        # Final result
        all_passed = all(result[1] for result in validation_results) and not self.errors

        if all_passed:
            console.print(
                "\n[bold green]✅ All YAML consistency validations passed![/bold green]"
            )
            console.print(
                "[green]📋 Configuration is ready for true YAML-first architecture[/green]"
            )
        else:
            console.print(
                "\n[bold red]❌ YAML consistency validation failed![/bold red]"
            )
            console.print(
                "[red]💡 Fix errors above before proceeding with YAML-first architecture[/red]"
            )

        return all_passed

    def _validate_unified_schema_structure(self) -> bool:
        """Validate unified schema file structure."""
        try:
            unified_schema_path = self.project_root / "config" / "unified_schema.yaml"

            if not unified_schema_path.exists():
                self.errors.append(f"Unified schema not found: {unified_schema_path}")
                return False

            with unified_schema_path.open("r", encoding="utf-8") as f:
                schema = yaml.safe_load(f)

            # Check required top-level sections
            required_sections = [
                "schema_version",
                "semantic_field_order",
                "field_definitions",
                "document_types",
                "prompt_templates",
            ]

            for section in required_sections:
                if section not in schema:
                    self.errors.append(
                        f"Missing required section in unified schema: {section}"
                    )
                    return False

            # Validate schema version format
            version = schema.get("schema_version", "")
            if not version.startswith("5.0"):
                self.warnings.append(
                    f"Schema version '{version}' should be 5.0-unified for new architecture"
                )

            return True

        except yaml.YAMLError as e:
            self.errors.append(f"YAML syntax error in unified schema: {e}")
            return False
        except Exception as e:
            self.errors.append(f"Error validating unified schema structure: {e}")
            return False

    def _validate_field_definitions(self) -> bool:
        """Validate field definitions consistency for all models."""
        try:
            unified_schema = self._load_unified_schema()
            if not unified_schema:
                return False

            semantic_field_order = unified_schema.get("semantic_field_order", {})
            field_definitions = unified_schema.get("field_definitions", {})

            # Validate each model's field definitions
            for model_name in ["llama", "internvl3"]:
                # Check that model exists in all sections
                if model_name not in semantic_field_order:
                    self.errors.append(
                        f"Model '{model_name}' not found in semantic_field_order"
                    )
                    return False

                if model_name not in field_definitions:
                    self.errors.append(
                        f"Model '{model_name}' not found in field_definitions"
                    )
                    return False

                model_semantic_order = semantic_field_order[model_name]
                model_field_definitions = field_definitions[model_name]

                # Check that all fields in semantic order have definitions
                missing_definitions = []
                for field in model_semantic_order:
                    if field not in model_field_definitions:
                        missing_definitions.append(f"{model_name}.{field}")

                if missing_definitions:
                    self.errors.append(
                        f"Fields in semantic order missing definitions: {missing_definitions}"
                    )
                    return False

                # Check that all field definitions have required properties
                required_props = [
                    "order",
                    "instruction",
                    "description",
                    "applicable_documents",
                    "required",
                ]
                for field, definition in model_field_definitions.items():
                    missing_props = [
                        prop for prop in required_props if prop not in definition
                    ]
                    if missing_props:
                        self.errors.append(
                            f"Field '{model_name}.{field}' missing properties: {missing_props}"
                        )
                        return False

                # Check order numbering consistency
                expected_order = 1
                for field in model_semantic_order:
                    if field in model_field_definitions:
                        actual_order = model_field_definitions[field].get("order", 0)
                        if actual_order != expected_order:
                            self.warnings.append(
                                f"Field '{model_name}.{field}' has order {actual_order}, expected {expected_order}"
                            )
                        expected_order += 1

            return True

        except Exception as e:
            self.errors.append(f"Error validating field definitions: {e}")
            return False

    def _validate_document_types(self) -> bool:
        """Validate document type configurations for all models."""
        try:
            unified_schema = self._load_unified_schema()
            if not unified_schema:
                return False

            document_types = unified_schema.get("document_types", {})
            field_definitions = unified_schema.get("field_definitions", {})

            # Validate each model's document type configurations
            for model_name in ["llama", "internvl3"]:
                # Check that model exists in all sections
                if model_name not in document_types:
                    self.errors.append(
                        f"Model '{model_name}' not found in document_types"
                    )
                    return False

                if model_name not in field_definitions:
                    self.errors.append(
                        f"Model '{model_name}' not found in field_definitions"
                    )
                    return False

                model_document_types = document_types[model_name]
                model_field_definitions = field_definitions[model_name]

                # Check each document type configuration for this model
                for doc_type, config in model_document_types.items():
                    required_fields = config.get("required_fields", [])

                    # Check that all required fields exist in field definitions
                    missing_fields = []
                    for field in required_fields:
                        if field not in model_field_definitions:
                            missing_fields.append(f"{model_name}.{doc_type}.{field}")

                    if missing_fields:
                        self.errors.append(
                            f"Document type references undefined fields: {missing_fields}"
                        )
                        return False

                    # Check that required fields are applicable to this document type
                    invalid_fields = []
                    for field in required_fields:
                        field_def = model_field_definitions.get(field, {})
                        applicable_docs = field_def.get("applicable_documents", [])
                        if doc_type not in applicable_docs:
                            invalid_fields.append(f"{model_name}.{doc_type}.{field}")

                    if invalid_fields:
                        self.warnings.append(
                            f"Document type uses fields not marked as applicable: {invalid_fields}"
                        )

            return True

        except Exception as e:
            self.errors.append(f"Error validating document types: {e}")
            return False

    def _validate_template_placeholders(self) -> bool:
        """Validate prompt template placeholders."""
        try:
            unified_schema = self._load_unified_schema()
            if not unified_schema:
                return False

            templates = unified_schema.get("prompt_templates", {})

            # Define valid placeholders
            valid_placeholders = {"field_count", "last_field", "document_type"}

            # Check each template for valid placeholders
            for template_name, template_value in templates.items():
                if isinstance(template_value, str):
                    placeholders = self._extract_placeholders(template_value)
                    invalid_placeholders = placeholders - valid_placeholders

                    if invalid_placeholders:
                        self.errors.append(
                            f"Template '{template_name}' uses invalid placeholders: {invalid_placeholders}"
                        )
                        return False

                elif isinstance(template_value, list):
                    # Handle list templates (like critical_instructions)
                    for i, item in enumerate(template_value):
                        if isinstance(item, str):
                            placeholders = self._extract_placeholders(item)
                            invalid_placeholders = placeholders - valid_placeholders

                            if invalid_placeholders:
                                self.errors.append(
                                    f"Template '{template_name}[{i}]' uses invalid placeholders: {invalid_placeholders}"
                                )
                                return False

            return True

        except Exception as e:
            self.errors.append(f"Error validating template placeholders: {e}")
            return False

    def _validate_field_order_consistency(self) -> bool:
        """Validate field order consistency across configurations for all models."""
        try:
            unified_schema = self._load_unified_schema()
            if not unified_schema:
                return False

            semantic_field_order = unified_schema.get("semantic_field_order", {})
            document_types = unified_schema.get("document_types", {})

            # Validate each model's field order consistency
            for model_name in ["llama", "internvl3"]:
                # Check that model exists in all sections
                if model_name not in semantic_field_order:
                    self.errors.append(
                        f"Model '{model_name}' not found in semantic_field_order"
                    )
                    return False

                if model_name not in document_types:
                    self.errors.append(
                        f"Model '{model_name}' not found in document_types"
                    )
                    return False

                model_semantic_order = semantic_field_order[model_name]
                model_document_types = document_types[model_name]

                # For each document type, check that required fields follow semantic order
                for doc_type, config in model_document_types.items():
                    required_fields = config.get("required_fields", [])

                    # Extract the subset of semantic order that applies to this document
                    applicable_semantic_order = [
                        field
                        for field in model_semantic_order
                        if field in required_fields
                    ]

                    # Check if required fields match semantic order
                    if required_fields != applicable_semantic_order:
                        self.warnings.append(
                            f"Document type '{model_name}.{doc_type}' field order doesn't match semantic order.\n"
                            f"  Expected: {applicable_semantic_order}\n"
                            f"  Actual: {required_fields}"
                        )

            return True

        except Exception as e:
            self.errors.append(f"Error validating field order consistency: {e}")
            return False

    def _detect_legacy_configurations(self) -> bool:
        """Detect legacy configuration files that should be removed."""
        legacy_files = [
            "config/fields.yaml",
            "config/document_metrics.yaml",
            "config/document_type_mappings.yaml",
        ]

        found_legacy = []
        for file_path in legacy_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                found_legacy.append(file_path)

        if found_legacy:
            self.warnings.append(
                f"Legacy configuration files detected (should be removed): {found_legacy}.\n"
                "  These are replaced by unified_schema.yaml in true YAML-first architecture."
            )

        return True  # This is a warning, not an error

    def _detect_python_overrides(self) -> bool:
        """Detect Python code that might override YAML configurations."""
        models_dir = self.project_root / "models"
        if not models_dir.exists():
            return True

        python_files = list(models_dir.glob("*.py"))
        problematic_patterns = [
            r"\.replace\(",
            r"f\".*\{.*field.*\}",
            r"dynamic.*field",
            r"string.*format",
        ]

        found_overrides = []

        for py_file in python_files:
            try:
                content = py_file.read_text(encoding="utf-8")
                for pattern in problematic_patterns:
                    if pattern in content.lower() or "replace(" in content:
                        # Simple check for string replacement operations
                        if ".replace(" in content and "clean" not in py_file.name:
                            found_overrides.append(
                                f"{py_file.name}: contains .replace() operations"
                            )
                            break

            except Exception as e:
                self.warnings.append(
                    f"Could not check {py_file} for Python overrides: {e}"
                )

        if found_overrides:
            self.warnings.append(
                f"Potential Python YAML overrides detected: {found_overrides}.\n"
                "  Review these for compliance with true YAML-first architecture."
            )

        return True  # This is a warning, not an error

    def _load_unified_schema(self) -> Dict[str, Any]:
        """Load unified schema with error handling."""
        try:
            schema_path = self.project_root / "config" / "unified_schema.yaml"
            with schema_path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.errors.append(f"Could not load unified schema: {e}")
            return {}

    def _extract_placeholders(self, template_string: str) -> Set[str]:
        """Extract placeholder names from template string."""
        import re

        placeholders = set()
        # Find all {placeholder} patterns
        matches = re.findall(r"\{([^}]+)\}", template_string)
        placeholders.update(matches)
        return placeholders

    def _display_validation_results(self, results: List[Tuple[str, bool]]):
        """Display validation results in a nice table."""
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Validation Check", style="cyan")
        table.add_column("Status", justify="center")

        for check_name, passed in results:
            status = "[green]✅ PASS[/green]" if passed else "[red]❌ FAIL[/red]"
            table.add_row(check_name, status)

        console.print(table)


def main():
    """Main CLI entry point for YAML validation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate YAML consistency for true YAML-first architecture"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    validator = YAMLConsistencyValidator(debug=args.debug)
    success = validator.validate_all()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
