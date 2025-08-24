"""
Field Schema Loader - Single Source of Truth for Field Definitions

This module provides dynamic field configuration loading from field_schema.yaml,
eliminating hardcoded field names throughout the codebase and ensuring consistency.
"""

from pathlib import Path
from typing import Any, Dict, List

import yaml


class FieldSchema:
    """
    Single source of truth for all field definitions.

    Dynamically loads field configurations from field_schema.yaml and provides
    methods to generate all required configurations for the extraction pipeline.
    """

    def __init__(self, schema_file: str = "field_schema.yaml"):
        """
        Initialize the field schema loader.

        Args:
            schema_file (str): Path to schema YAML file relative to common/ directory
        """
        self.schema_file = schema_file
        self.schema = self._load_schema()
        self._validate_schema()

        # Cache computed properties for performance
        self._field_names_cache = None
        self._field_metadata_cache = {}
        self._group_configs_cache = {}

    def _load_schema(self) -> dict:
        """Load schema from YAML file with comprehensive error handling."""
        try:
            # Find schema file relative to this module
            schema_path = Path(__file__).parent / self.schema_file

            if not schema_path.exists():
                raise FileNotFoundError(
                    f"❌ FATAL: Schema file not found: {schema_path}\n"
                    f"💡 Cannot proceed without field schema\n"
                    f"💡 Ensure {self.schema_file} exists in common/ directory"
                )

            with schema_path.open("r", encoding="utf-8") as f:
                schema = yaml.safe_load(f)

            if not isinstance(schema, dict):
                raise ValueError(
                    f"❌ FATAL: Invalid schema structure in {self.schema_file}\n"
                    f"💡 Expected dictionary at root level"
                )

            return schema

        except yaml.YAMLError as e:
            raise ValueError(
                f"❌ FATAL: Invalid YAML in {self.schema_file}: {e}\n"
                f"💡 Check YAML syntax and structure"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"❌ FATAL: Schema loading failed: {e}\n"
                f"💡 Cannot proceed without field definitions"
            ) from e

    def _validate_schema(self):
        """Validate schema structure and required sections."""
        required_sections = ["fields", "groups", "grouping_strategies"]
        missing_sections = []

        for section in required_sections:
            if section not in self.schema:
                missing_sections.append(section)

        if missing_sections:
            raise ValueError(
                f"❌ FATAL: Missing required sections in schema: {missing_sections}\n"
                f"💡 Schema must contain: {required_sections}"
            )

        # Validate fields structure
        if not isinstance(self.schema["fields"], list):
            raise ValueError("❌ FATAL: 'fields' must be a list")

        if len(self.schema["fields"]) != self.schema.get("total_fields", 25):
            raise ValueError(
                f"❌ FATAL: Field count mismatch. Expected {self.schema.get('total_fields', 25)}, "
                f"got {len(self.schema['fields'])}"
            )

        # Validate each field has required properties
        required_field_props = ["name", "type", "evaluation_logic", "group"]
        for i, field in enumerate(self.schema["fields"]):
            missing_props = [prop for prop in required_field_props if prop not in field]
            if missing_props:
                raise ValueError(
                    f"❌ FATAL: Field {i} missing required properties: {missing_props}\n"
                    f"💡 Required properties: {required_field_props}"
                )

    @property
    def field_names(self) -> List[str]:
        """Get ordered list of all field names from schema."""
        if self._field_names_cache is None:
            self._field_names_cache = [field["name"] for field in self.schema["fields"]]
        return self._field_names_cache

    @property
    def total_fields(self) -> int:
        """Get total number of fields."""
        return len(self.field_names)

    def get_field_metadata(self, field_name: str) -> dict:
        """
        Get complete metadata for a field.

        Args:
            field_name (str): Name of the field

        Returns:
            dict: Field metadata including type, evaluation_logic, etc.

        Raises:
            ValueError: If field name not found
        """
        if field_name not in self._field_metadata_cache:
            for field in self.schema["fields"]:
                if field["name"] == field_name:
                    self._field_metadata_cache[field_name] = field
                    break
            else:
                raise ValueError(f"Unknown field name: {field_name}")

        return self._field_metadata_cache[field_name]

    def get_fields_by_type(self, field_type: str) -> List[str]:
        """
        Get all fields of specific type.

        Args:
            field_type (str): Type of fields (monetary, text, date, etc.)

        Returns:
            List[str]: List of field names of the specified type
        """
        return [
            field["name"]
            for field in self.schema["fields"]
            if field["type"] == field_type
        ]

    def get_fields_by_group(self, group_name: str) -> List[str]:
        """
        Get all fields assigned to a specific group.

        Args:
            group_name (str): Name of the group

        Returns:
            List[str]: List of field names in the group
        """
        return [
            field["name"]
            for field in self.schema["fields"]
            if field["group"] == group_name
        ]

    def get_all_fields(self) -> List[str]:
        """Get list of all field names (all fields are required for extraction)."""
        return self.field_names

    def get_group_config(self, group_name: str) -> dict:
        """
        Get group configuration with dynamically assigned fields.

        Args:
            group_name (str): Name of the group

        Returns:
            dict: Group configuration with fields list
        """
        if group_name not in self._group_configs_cache:
            if group_name not in self.schema["groups"]:
                raise ValueError(f"Unknown group name: {group_name}")

            # Get base group config
            group_config = self.schema["groups"][group_name].copy()

            # Add dynamically generated fields list
            group_config["fields"] = self.get_fields_by_group(group_name)

            self._group_configs_cache[group_name] = group_config

        return self._group_configs_cache[group_name]

    def get_grouping_strategy(self, strategy_name: str) -> dict:
        """
        Get complete grouping strategy configuration.

        Args:
            strategy_name (str): Name of the strategy ('detailed_grouped', 'field_grouped')

        Returns:
            dict: Strategy configuration with group configs
        """
        if strategy_name not in self.schema["grouping_strategies"]:
            raise ValueError(f"Unknown grouping strategy: {strategy_name}")

        strategy = self.schema["grouping_strategies"][strategy_name].copy()

        # Generate complete group configurations
        strategy_groups = {}

        if strategy_name == "field_grouped":
            # For cognitive strategy, generate combined groups
            strategy_groups = self._generate_cognitive_groups()
        else:
            # For other strategies, use direct group mapping
            for group_name in strategy["groups"]:
                strategy_groups[group_name] = self.get_group_config(group_name)

        strategy["group_configs"] = strategy_groups
        return strategy

    def _generate_cognitive_groups(self) -> dict:
        """Generate cognitive grouping strategy groups dynamically."""
        cognitive_groups = {}

        # Regulatory and Financial Core: critical + monetary fields
        regulatory_fields = self.get_fields_by_group(
            "critical"
        ) + self.get_fields_by_type("monetary")
        cognitive_groups["regulatory_financial"] = {
            "name": "Regulatory and Financial Core",
            "fields": regulatory_fields,
            "max_tokens": 400,
            "temperature": 0.0,
            "description": "Core business validation and primary financial amounts",
            "cognitive_focus": "Essential regulatory compliance and financial totals",
        }

        # Entity Contacts: business + payer info
        entity_fields = self.get_fields_by_group(
            "business_entity"
        ) + self.get_fields_by_group("payer_info")
        cognitive_groups["entity_contacts"] = {
            "name": "Entity Contact Information",
            "fields": entity_fields,
            "max_tokens": 600,
            "temperature": 0.0,
            "description": "All contact information for involved parties",
            "cognitive_focus": "Who is involved - all participant identification",
        }

        # Transaction Details: item details
        cognitive_groups["transaction_details"] = {
            "name": "Transaction Line Items",
            "fields": self.get_fields_by_group("item_details"),
            "max_tokens": 500,
            "temperature": 0.0,
            "description": "Detailed line item transaction data",
            "cognitive_focus": "What was bought - item-level transaction details",
        }

        # Temporal Data: dates
        cognitive_groups["temporal_data"] = {
            "name": "Temporal Information",
            "fields": self.get_fields_by_group("dates"),
            "max_tokens": 350,
            "temperature": 0.0,
            "description": "All date and time-related information",
            "cognitive_focus": "When - temporal context and deadlines",
        }

        # Banking Payment: banking
        cognitive_groups["banking_payment"] = {
            "name": "Banking and Payment Details",
            "fields": self.get_fields_by_group("banking"),
            "max_tokens": 400,
            "temperature": 0.0,
            "description": "Financial institution and payment processing information",
            "cognitive_focus": "How payment is processed - banking infrastructure",
        }

        # Document Metadata: metadata + any remaining monetary fields
        metadata_fields = self.get_fields_by_group("metadata")
        # Add balance fields that aren't in critical group
        balance_fields = [
            f
            for f in self.get_fields_by_type("monetary")
            if "BALANCE" in f and f not in self.get_fields_by_group("critical")
        ]
        cognitive_groups["document_metadata"] = {
            "name": "Document Context and Balances",
            "fields": metadata_fields + balance_fields,
            "max_tokens": 350,
            "temperature": 0.0,
            "description": "Document classification and account balance information",
            "cognitive_focus": "Document type and account state context",
        }

        return cognitive_groups

    def get_validation_rules(self, group_name: str) -> dict:
        """
        Get validation rules for a group.

        Args:
            group_name (str): Name of the group

        Returns:
            dict: Validation rules for the group
        """
        return self.schema.get("validation_rules", {}).get(group_name, {})

    def generate_field_types_mapping(self) -> Dict[str, str]:
        """Generate field name to type mapping."""
        return {field["name"]: field["type"] for field in self.schema["fields"]}

    def generate_field_descriptions_mapping(self) -> Dict[str, str]:
        """Generate field name to description mapping."""
        return {field["name"]: field["description"] for field in self.schema["fields"]}

    def generate_prompt_instructions(self) -> Dict[str, str]:
        """Generate field name to instruction mapping for prompts."""
        return {field["name"]: field["instruction"] for field in self.schema["fields"]}

    def get_semantic_field_order(self) -> List[str]:
        """
        Get field names in semantic order (as defined in schema).
        This preserves the carefully designed field ordering for optimal model performance.
        """
        return self.field_names  # Already in semantic order from YAML

    def validate_field_completeness(
        self, extracted_fields: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Validate extracted fields against schema requirements.

        Args:
            extracted_fields (dict): Extracted field data

        Returns:
            dict: Validation results with metrics
        """
        expected_fields = set(self.field_names)
        actual_fields = set(extracted_fields.keys())

        missing_fields = expected_fields - actual_fields
        extra_fields = actual_fields - expected_fields

        # Check required fields
        required_fields = set(self.get_required_fields())
        missing_required = required_fields - actual_fields

        return {
            "total_expected": len(expected_fields),
            "total_extracted": len(actual_fields),
            "missing_fields": list(missing_fields),
            "extra_fields": list(extra_fields),
            "missing_required": list(missing_required),
            "completeness_ratio": len(actual_fields & expected_fields)
            / len(expected_fields),
            "validation_passed": len(missing_required) == 0 and len(extra_fields) == 0,
        }

    def get_model_prompt_template(
        self, model_name: str, strategy: str, group_name: str = None
    ) -> dict:
        """
        Get model-specific prompt template from schema.

        Args:
            model_name (str): Model identifier ('llama' or 'internvl3')
            strategy (str): Extraction strategy ('single_pass' or 'field_grouped')
            group_name (str): Group name for field_grouped strategy (optional)

        Returns:
            dict: Prompt template configuration

        Raises:
            ValueError: If template not found or invalid parameters
        """
        if "model_prompt_templates" not in self.schema:
            raise ValueError("No model prompt templates found in schema")

        templates = self.schema["model_prompt_templates"]

        if model_name not in templates:
            available_models = list(templates.keys())
            raise ValueError(
                f"Model '{model_name}' not found in templates. "
                f"Available models: {available_models}"
            )

        model_templates = templates[model_name]

        if strategy not in model_templates:
            available_strategies = list(model_templates.keys())
            raise ValueError(
                f"Strategy '{strategy}' not found for model '{model_name}'. "
                f"Available strategies: {available_strategies}"
            )

        strategy_templates = model_templates[strategy]

        # For single_pass, return the template directly
        if strategy == "single_pass":
            return strategy_templates

        # For grouped strategies, need group_name
        elif strategy in ["field_grouped", "detailed_grouped"]:
            if group_name is None:
                raise ValueError(f"group_name is required for {strategy} strategy")

            if group_name not in strategy_templates:
                available_groups = list(strategy_templates.keys())
                raise ValueError(
                    f"Group '{group_name}' not found in {strategy} templates for '{model_name}'. "
                    f"Available groups: {available_groups}"
                )

            return strategy_templates[group_name]

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def get_available_model_templates(self) -> dict:
        """
        Get all available model prompt templates.

        Returns:
            dict: Available models and their strategies
        """
        if "model_prompt_templates" not in self.schema:
            return {}

        templates = self.schema["model_prompt_templates"]
        result = {}

        for model_name, model_templates in templates.items():
            result[model_name] = {
                "strategies": list(model_templates.keys()),
                "groups": {},
            }

            # Add group information for grouped strategies
            for grouped_strategy in ["field_grouped", "detailed_grouped"]:
                if grouped_strategy in model_templates:
                    result[model_name]["groups"][grouped_strategy] = list(
                        model_templates[grouped_strategy].keys()
                    )

        return result

    def generate_dynamic_prompt(
        self,
        model_name: str,
        strategy: str,
        group_name: str = None,
        fields: list = None,
    ) -> str:
        """
        Generate complete prompt from model-specific template and schema data.

        Args:
            model_name (str): Model identifier ('llama' or 'internvl3')
            strategy (str): Extraction strategy ('single_pass' or 'field_grouped')
            group_name (str): Group name for field_grouped strategy (optional)
            fields (list): Override field list (optional, uses group fields if not provided)

        Returns:
            str: Complete generated prompt
        """
        # Get the template
        template = self.get_model_prompt_template(model_name, strategy, group_name)

        # Get field list
        if fields is None:
            if strategy == "single_pass":
                fields = self.field_names
            elif strategy == "field_grouped" and group_name:
                # For field_grouped, try cognitive groups first, then base groups
                try:
                    cognitive_strategy = self.get_grouping_strategy("field_grouped")
                    if group_name in cognitive_strategy["group_configs"]:
                        fields = cognitive_strategy["group_configs"][group_name][
                            "fields"
                        ]
                    else:
                        # Fallback to base group
                        group_config = self.get_group_config(group_name)
                        fields = group_config.get("fields", [])
                except Exception:
                    # Final fallback to base group
                    group_config = self.get_group_config(group_name)
                    fields = group_config.get("fields", [])
            else:
                fields = []

        field_count = len(fields)

        # Generate prompt based on strategy
        if strategy == "single_pass":
            return self._generate_single_pass_prompt(template, fields, field_count)
        elif strategy in ["field_grouped", "detailed_grouped"]:
            return self._generate_grouped_prompt(template, fields, field_count)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _generate_single_pass_prompt(
        self, template: dict, fields: list, field_count: int
    ) -> str:
        """Generate single-pass prompt from template."""
        prompt_parts = []

        # Opening
        if "opening_text" in template:
            prompt_parts.append(template["opening_text"])

        # Output instruction (InternVL3)
        if "output_instruction" in template:
            prompt_parts.append(template["output_instruction"])

        # Missing value instruction (InternVL3)
        if "missing_value_instruction" in template:
            prompt_parts.append(template["missing_value_instruction"])

        # Critical instructions for Llama
        if "critical_instructions" in template:
            prompt_parts.append("\nCRITICAL INSTRUCTIONS:")
            for instruction in template["critical_instructions"]:
                prompt_parts.append(f"- {instruction}")

        # Anti-assumption rules (InternVL3)
        if "anti_assumption_rules" in template:
            prompt_parts.append("\nKEY RULES:")
            for rule in template["anti_assumption_rules"]:
                prompt_parts.append(f"- {rule}")

        # Output format
        if "output_format" in template:
            format_text = template["output_format"].format(field_count=field_count)
            prompt_parts.append(f"\n{format_text}")
        elif "output_format_header" in template:
            header = template["output_format_header"].format(field_count=field_count)
            prompt_parts.append(f"\n{header}")

        # Field instructions
        prompt_parts.append("")
        for field in fields:
            field_data = self.get_field_metadata(field)
            instruction = field_data.get(
                "instruction", f"[{field.lower().replace('_', ' ')} or NOT_FOUND]"
            )
            prompt_parts.append(f"{field}: {instruction}")

        # Format rules
        if "format_rules" in template:
            prompt_parts.append("\nFORMAT RULES:")
            for rule in template["format_rules"]:
                formatted_rule = rule.format(field_count=field_count)
                prompt_parts.append(f"- {formatted_rule}")

        # Closing instruction
        if "closing_instruction" in template:
            prompt_parts.append(f"\n{template['closing_instruction']}")

        return "\n".join(prompt_parts)

    def _generate_grouped_prompt(
        self, template: dict, fields: list, field_count: int
    ) -> str:
        """Generate grouped extraction prompt from template."""
        prompt_parts = []

        # Task and expertise
        if "expertise_frame" in template:
            prompt_parts.append(f"TASK: {template['expertise_frame']}")

        # Document context
        prompt_parts.append(
            "\nDOCUMENT CONTEXT: You are analyzing a business document image. Consider the document type when extracting fields."
        )

        # Cognitive context
        if "cognitive_context" in template:
            prompt_parts.append(f"\n{template['cognitive_context']}")

        # Focus instruction
        if "focus_instruction" in template:
            prompt_parts.append(f"\n{template['focus_instruction']}")

        # Output format
        prompt_parts.append(f"\nOUTPUT FORMAT - EXACTLY {field_count} LINES:")

        # Field lines
        for field in fields:
            prompt_parts.append(f"{field}: [value or NOT_FOUND]")

        # Standard format rules
        prompt_parts.append(f"""
FORMAT RULES:
- Use exactly: KEY: value (colon and space)
- NEVER use: **KEY:** or **KEY** or *KEY* or KEY: or any formatting
- Plain text only - NO markdown, NO bold, NO italic
- Include ALL {field_count} fields even if NOT_FOUND
- Extract ONLY what you can see in the document
- Do NOT guess, calculate, or make up values
- Use NOT_FOUND if field is not visible or not applicable
- Output ONLY these {field_count} lines, nothing else
- For lists: use COMMA-SEPARATED values on ONE LINE per field
- DO NOT output the same field name multiple times

STOP after the last field. Do not add explanations or comments.""")

        return "\n".join(prompt_parts)

    def __repr__(self) -> str:
        """String representation of schema."""
        return (
            f"FieldSchema(fields={self.total_fields}, "
            f"groups={len(self.schema['groups'])}, "
            f"version={self.schema.get('schema_version', 'unknown')})"
        )


# Global schema instance for backwards compatibility
# This allows existing code to gradually migrate to explicit FieldSchema usage
_global_schema = None


def get_global_schema() -> FieldSchema:
    """Get or create global schema instance."""
    global _global_schema
    if _global_schema is None:
        _global_schema = FieldSchema()
    return _global_schema


# Convenience functions for common operations
def get_extraction_fields() -> List[str]:
    """Get ordered list of extraction field names."""
    return get_global_schema().field_names


def get_field_count() -> int:
    """Get total field count."""
    return get_global_schema().total_fields


def get_field_types() -> Dict[str, str]:
    """Get field name to type mapping."""
    return get_global_schema().generate_field_types_mapping()


def get_field_descriptions() -> Dict[str, str]:
    """Get field name to description mapping."""
    return get_global_schema().generate_field_descriptions_mapping()
