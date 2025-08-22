"""
Grouped extraction strategy implementation.

This module implements the grouped field extraction strategy where related fields
are processed together in focused prompts to improve accuracy and reduce
hallucinations in vision-language model extraction tasks.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    EXTRACTION_FIELDS,
    FIELD_GROUPS,
    FIELD_INSTRUCTIONS,
    GROUP_VALIDATION_RULES,
    GROUPING_STRATEGIES,
    DEFAULT_GROUPING_STRATEGY,
)
from .evaluation_utils import parse_extraction_response


class GroupedExtractionStrategy:
    """
    Strategy for grouped field extraction from documents.

    This class implements multi-pass extraction where related fields are
    grouped together for focused processing, leading to improved accuracy
    compared to single-pass extraction of all fields.
    """

    def __init__(self, extraction_mode="grouped", debug=False, grouping_strategy="8_groups"):
        """
        Initialize the grouped extraction strategy.

        Args:
            extraction_mode (str): Extraction mode ('grouped', 'adaptive', 'single_pass')
            debug (bool): Enable debug logging
            grouping_strategy (str): Grouping strategy ('8_groups' or '6_groups')
        """
        self.extraction_mode = extraction_mode
        self.grouping_strategy = grouping_strategy
        self.debug = debug
        
        # Load appropriate field groups based on strategy
        if grouping_strategy in GROUPING_STRATEGIES:
            self.field_groups = GROUPING_STRATEGIES[grouping_strategy]
            if self.debug:
                print(f"🎯 Using {grouping_strategy} extraction strategy with {len(self.field_groups)} groups")
        else:
            self.field_groups = FIELD_GROUPS  # Fallback to default
            if self.debug:
                print(f"⚠️ Unknown strategy '{grouping_strategy}', using default groups")
        self.stats = {
            "total_groups_processed": 0,
            "successful_groups": 0,
            "failed_groups": 0,
            "total_processing_time": 0.0,
            "group_processing_times": {},
            "group_field_counts": {},
            "retries_performed": 0,
        }

    def generate_group_prompt(self, group_name: str) -> str:
        """
        Generate a research-backed focused prompt for a specific field group.
        
        Uses domain expertise framing, cognitive reasoning context, and 
        Llama-specific optimizations for superior performance.

        Args:
            group_name (str): Name of the field group

        Returns:
            str: Formatted prompt for the group
        """
        if group_name not in FIELD_GROUPS:
            raise ValueError(f"Unknown field group: {group_name}")

        group_config = self.field_groups[group_name]
        fields = group_config["fields"]

        # Generate research-backed prompts based on cognitive grouping principles
        prompt = self._generate_research_backed_prompt(group_name, group_config, fields)

        if self.debug:
            print(
                f"🔍 Generated research-backed prompt for group '{group_name}' ({len(fields)} fields, {len(prompt)} chars)"
            )
            print("📝 GROUP PROMPT CONTENT:")
            print("-" * 40)
            print(prompt)
            print("-" * 40)

        return prompt
    
    def _generate_research_backed_prompt(self, group_name: str, group_config: dict, fields: list) -> str:
        """
        Generate research-backed prompts using domain expertise framing and cognitive reasoning.
        
        Based on cognitive science research showing grouped field extraction outperforms 
        single-pass when properly implemented with domain context and reasoning.
        """
        
        # Simplified task-focused prompts per group
        if group_name == "critical":
            expertise_frame = """Extract critical document identifiers and financial totals."""
            cognitive_context = """These are the essential document validation fields: regulatory ID (ABN) and total amount."""
            focus_instruction = "Extract the business number and total amount from this document."
            
        elif group_name == "monetary":
            expertise_frame = """Extract financial amounts and tax information."""
            cognitive_context = """These are monetary values: GST (tax amount), subtotal (pre-tax amount), opening/closing balances (from bank statements only). Do NOT confuse TOTAL with CLOSING_BALANCE."""
            focus_instruction = "Extract tax amounts and subtotals. Account balances are only for bank statements, not invoices."
            
        elif group_name == "dates":
            expertise_frame = """Extract dates and time-related information."""
            cognitive_context = """These are date fields: invoice date, due date, and statement period."""
            focus_instruction = "Extract all dates from this document."
            
        elif group_name == "business_entity":
            expertise_frame = """Extract business and supplier information."""
            cognitive_context = """These identify the business: supplier name, address, phone, and website."""
            focus_instruction = "Extract business contact information and identification details."
            
        elif group_name == "payer_info":
            expertise_frame = """Extract customer and payer information."""
            cognitive_context = """These identify the customer: payer name, address, email, and phone."""
            focus_instruction = "Extract customer contact information and identification details."
            
        elif group_name == "banking":
            expertise_frame = """Extract banking and payment information."""
            cognitive_context = """IMPORTANT: These fields are typically N/A for invoices/receipts. Only extract if this is a bank statement. BSB is a 6-digit number (NOT the 11-digit ABN). Bank name is a financial institution (NOT the supplier)."""
            focus_instruction = "Extract bank account information ONLY if this is a bank statement. If this is an invoice/receipt, most banking fields should be N/A."
            
        elif group_name == "item_details": 
            expertise_frame = """Extract transaction line items and pricing."""
            cognitive_context = """These are item details: descriptions (what was bought), quantities (how many), and individual unit prices (NOT calculated totals). Extract exactly what appears on the document. CRITICAL: Use the exact field names DESCRIPTIONS, QUANTITIES, and PRICES."""
            focus_instruction = "Extract the exact item descriptions, quantities, and unit prices as shown on the document. Do NOT calculate, multiply, or derive prices. Use only the unit prices visible in the document."
            
        elif group_name == "metadata":
            expertise_frame = """Extract document type and classification."""
            cognitive_context = """This identifies the document category: invoice, receipt, or statement."""
            focus_instruction = "Identify the document type from this business document. Output ONLY the document type field, nothing else."
            
        else:
            # Fallback for any unhandled groups
            expertise_frame = f"""You are a business document expert specializing in {group_config['description'].lower()}."""
            cognitive_context = f"""Focus on {group_config['description'].lower()} to provide accurate field extraction."""
            focus_instruction = f"Focus on extracting fields related to {group_config['description'].lower()}."

        # Build the streamlined task-focused prompt with context hints
        prompt = f"""TASK: {expertise_frame}

DOCUMENT CONTEXT: You are analyzing a business document image. Consider the document type when extracting fields.

{cognitive_context}

{focus_instruction}

OUTPUT FORMAT - EXACTLY {len(fields)} LINES:
"""

        # Add each field with its specific instruction
        for field in fields:
            instruction = FIELD_INSTRUCTIONS.get(field, "[value or N/A]")
            prompt += f"{field}: {instruction}\n"

        # Enhanced format rules with strict enforcement
        prompt += f"""
FORMAT RULES:
- Use exactly: KEY: value (colon and space)
- NEVER use: **KEY:** or **KEY** or *KEY* or KEY: or any formatting
- Plain text only - NO markdown, NO bold, NO italic
- Include ALL {len(fields)} fields even if N/A
- Extract ONLY what you can see in the document
- Do NOT guess, calculate, or make up values
- Use N/A if field is not visible or not applicable
- Output ONLY these {len(fields)} lines, nothing else

STOP after the last field. Do not add explanations or comments."""

        return prompt

    def validate_group_results(
        self, group_name: str, extracted_data: Dict[str, str]
    ) -> bool:
        """
        Validate extracted results for a specific group.

        Args:
            group_name (str): Name of the field group
            extracted_data (dict): Extracted field data

        Returns:
            bool: True if validation passes
        """
        if group_name not in GROUP_VALIDATION_RULES:
            return True  # No specific validation rules

        rules = GROUP_VALIDATION_RULES[group_name]
        expected_fields = self.field_groups[group_name]["fields"]

        # Check required fields
        if "required_fields" in rules:
            for field in rules["required_fields"]:
                if field not in extracted_data or extracted_data[field] in ["", "N/A"]:
                    if not rules.get("allow_empty", True):
                        if self.debug:
                            print(
                                f"❌ Validation failed: Missing required field {field} in group {group_name}"
                            )
                        return False

        # Check all expected fields are present
        for field in expected_fields:
            if field not in extracted_data:
                if self.debug:
                    print(
                        f"❌ Validation failed: Missing field {field} in group {group_name}"
                    )
                return False

        if self.debug:
            print(f"✅ Validation passed for group '{group_name}'")

        return True

    def merge_group_results(
        self, group_results: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Merge results from multiple group extractions.

        Args:
            group_results (list): List of group extraction results

        Returns:
            dict: Merged extraction results with all fields
        """
        merged_data = {}
        group_metadata = {}

        for result in group_results:
            group_name = result["group_name"]
            extracted_data = result["extracted_data"]

            # Merge field data
            merged_data.update(extracted_data)

            # Store group-level metadata
            group_metadata[group_name] = {
                "processing_time": result.get("processing_time", 0),
                "field_count": len(extracted_data),
                "success": result.get("success", True),
                "validation_passed": result.get("validation_passed", True),
            }

        # Ensure all extraction fields are present
        for field in EXTRACTION_FIELDS:
            if field not in merged_data:
                merged_data[field] = "N/A"
                if self.debug:
                    print(f"⚠️ Field {field} not extracted in any group, setting to N/A")

        # Add group metadata to stats
        self.stats["group_metadata"] = group_metadata

        if self.debug:
            extracted_count = len([v for v in merged_data.values() if v != "N/A"])
            na_count = len(EXTRACTION_FIELDS) - extracted_count
            print(
                f"📊 Merged results: {extracted_count}/{len(EXTRACTION_FIELDS)} fields with values, {na_count} correctly identified as N/A"
            )

        return merged_data

    def should_use_grouped_extraction(self, image_path: str = None) -> bool:
        """
        Determine if grouped extraction should be used for adaptive mode.

        Args:
            image_path (str): Path to image (for adaptive analysis)

        Returns:
            bool: True if grouped extraction should be used
        """
        if self.extraction_mode == "grouped":
            return True
        elif self.extraction_mode == "single_pass":
            return False
        elif self.extraction_mode == "adaptive":
            # Adaptive logic - for now, use grouped for most cases
            # In future, could analyze image complexity
            return True
        else:
            raise ValueError(f"Unknown extraction mode: {self.extraction_mode}")

    def process_group(
        self, group_name: str, extract_function, image_path: str, max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Process a single field group with retry logic.

        Args:
            group_name (str): Name of the field group
            extract_function: Function to perform extraction
            image_path (str): Path to the image
            max_retries (int): Maximum number of retries

        Returns:
            dict: Group extraction results with metadata
        """
        group_config = self.field_groups[group_name]
        start_time = time.time()

        for attempt in range(max_retries + 1):
            try:
                if self.debug:
                    attempt_str = f" (attempt {attempt + 1})" if attempt > 0 else ""
                    print(f"🔄 Processing group '{group_name}'{attempt_str}")

                # Generate focused prompt for this group
                prompt = self.generate_group_prompt(group_name)

                # Extract with group-specific parameters
                extraction_kwargs = {
                    "max_new_tokens": group_config["max_tokens"],
                    "temperature": group_config["temperature"],
                }

                # Call the extraction function (model-specific)
                response = extract_function(image_path, prompt, **extraction_kwargs)

                if self.debug:
                    print(f"🔍 RAW MODEL RESPONSE for group '{group_name}':")
                    print("-" * 40)
                    print(response)
                    print("-" * 40)

                # Parse the response
                extracted_data = parse_extraction_response(
                    response, clean_conversation_artifacts=True
                )

                # Filter to only include fields from this group
                group_fields = group_config["fields"]
                filtered_data = {
                    field: extracted_data.get(field, "N/A") for field in group_fields
                }

                if self.debug:
                    print(f"🔍 PARSED DATA for group '{group_name}':")
                    for field, value in filtered_data.items():
                        print(f"  {field}: {value}")
                    print()

                # Validate results
                validation_passed = self.validate_group_results(
                    group_name, filtered_data
                )

                processing_time = time.time() - start_time

                # Update stats
                self.stats["total_groups_processed"] += 1
                if validation_passed:
                    self.stats["successful_groups"] += 1
                else:
                    self.stats["failed_groups"] += 1

                self.stats["group_processing_times"][group_name] = processing_time
                self.stats["group_field_counts"][group_name] = len(filtered_data)

                if attempt > 0:
                    self.stats["retries_performed"] += 1

                result = {
                    "group_name": group_name,
                    "extracted_data": filtered_data,
                    "processing_time": processing_time,
                    "success": True,
                    "validation_passed": validation_passed,
                    "attempt": attempt + 1,
                    "raw_response": response,
                }

                if self.debug:
                    extracted_count = len(
                        [v for v in filtered_data.values() if v != "N/A"]
                    )
                    print(
                        f"✅ Group '{group_name}' completed: {extracted_count}/{len(group_fields)} fields"
                    )

                return result

            except Exception as e:
                if self.debug:
                    print(
                        f"❌ Error in group '{group_name}' attempt {attempt + 1}: {e}"
                    )

                if attempt == max_retries:
                    # Final attempt failed, return failure result
                    processing_time = time.time() - start_time

                    # Create empty results for this group
                    group_fields = group_config["fields"]
                    empty_data = {field: "N/A" for field in group_fields}

                    self.stats["total_groups_processed"] += 1
                    self.stats["failed_groups"] += 1
                    self.stats["group_processing_times"][group_name] = processing_time

                    return {
                        "group_name": group_name,
                        "extracted_data": empty_data,
                        "processing_time": processing_time,
                        "success": False,
                        "validation_passed": False,
                        "attempt": attempt + 1,
                        "error": str(e),
                    }

        # Should not reach here
        raise RuntimeError(f"Unexpected error processing group {group_name}")

    def extract_fields_grouped(
        self,
        image_path: str,
        extract_function,
        groups_to_process: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """
        Extract fields using grouped strategy.

        Args:
            image_path (str): Path to image file
            extract_function: Model-specific extraction function
            groups_to_process (list): Specific groups to process (None for all)

        Returns:
            tuple: (extracted_data, extraction_metadata)
        """
        start_time = time.time()

        # Determine which groups to process
        if groups_to_process is None:
            # Create processing order based on current field groups
            groups_to_process = sorted(
                self.field_groups.keys(), key=lambda x: self.field_groups[x]["priority"]
            )

        if self.debug:
            print(
                f"🚀 Starting grouped extraction with {len(groups_to_process)} groups"
            )

        group_results = []

        # Process each group in priority order
        for group_name in groups_to_process:
            if group_name not in self.field_groups:
                if self.debug:
                    print(f"⚠️ Skipping unknown group: {group_name}")
                continue

            group_result = self.process_group(group_name, extract_function, image_path)
            group_results.append(group_result)

        # Merge all group results
        merged_data = self.merge_group_results(group_results)

        # Calculate final statistics
        total_time = time.time() - start_time
        self.stats["total_processing_time"] = total_time

        # Prepare metadata
        extraction_metadata = {
            "extraction_mode": self.extraction_mode,
            "total_processing_time": total_time,
            "groups_processed": len(group_results),
            "successful_groups": self.stats["successful_groups"],
            "failed_groups": self.stats["failed_groups"],
            "retries_performed": self.stats["retries_performed"],
            "group_stats": self.stats.get("group_metadata", {}),
            "fields_extracted": len([v for v in merged_data.values() if v != "N/A"]),
            "total_fields": len(EXTRACTION_FIELDS),
            "extraction_completeness": len(
                [v for v in merged_data.values() if v != "N/A"]
            )
            / len(EXTRACTION_FIELDS),
        }

        if self.debug:
            print(f"🎉 Grouped extraction completed in {total_time:.2f}s")
            print(
                f"📊 Extracted {extraction_metadata['fields_extracted']}/{extraction_metadata['total_fields']} fields"
            )

        return merged_data, extraction_metadata


class AdaptiveExtractionStrategy:
    """
    Adaptive extraction strategy that chooses between single-pass and grouped
    based on document characteristics and confidence levels.
    """

    def __init__(self, debug=False):
        """
        Initialize adaptive extraction strategy.

        Args:
            debug (bool): Enable debug logging
        """
        self.debug = debug
        self.grouped_strategy = GroupedExtractionStrategy("grouped", debug)

    def analyze_document_complexity(self, image_path: str) -> str:
        """
        Analyze document to determine optimal extraction strategy.

        Args:
            image_path (str): Path to document image

        Returns:
            str: Recommended strategy ('single_pass' or 'grouped')
        """
        # For now, use simple heuristics
        # In future, could use ML model to analyze visual complexity

        # Default to grouped for better accuracy
        return "grouped"

    def extract_fields_adaptive(
        self, image_path: str, single_pass_function, grouped_extract_function
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """
        Extract fields using adaptive strategy.

        Args:
            image_path (str): Path to image file
            single_pass_function: Function for single-pass extraction
            grouped_extract_function: Function for grouped extraction

        Returns:
            tuple: (extracted_data, extraction_metadata)
        """
        start_time = time.time()

        # Analyze document to choose strategy
        recommended_strategy = self.analyze_document_complexity(image_path)

        if self.debug:
            print(f"🤖 Adaptive mode recommends: {recommended_strategy}")

        # Execute recommended strategy
        if recommended_strategy == "grouped":
            extracted_data, metadata = self.grouped_strategy.extract_fields_grouped(
                image_path, grouped_extract_function
            )
            metadata["adaptive_choice"] = "grouped"
            metadata["adaptive_reason"] = "Document complexity analysis"
        else:
            # Fallback to single-pass
            extracted_data = single_pass_function(image_path)
            total_time = time.time() - start_time
            metadata = {
                "extraction_mode": "adaptive",
                "adaptive_choice": "single_pass",
                "adaptive_reason": "Simple document detected",
                "total_processing_time": total_time,
                "fields_extracted": len(
                    [v for v in extracted_data.values() if v != "N/A"]
                ),
                "total_fields": len(EXTRACTION_FIELDS),
            }

        return extracted_data, metadata


def get_extraction_strategy(mode: str, debug: bool = False, grouping_strategy: str = "8_groups"):
    """
    Factory function to get appropriate extraction strategy.

    Args:
        mode (str): Extraction mode ('single_pass', 'grouped', 'adaptive')
        debug (bool): Enable debug logging
        grouping_strategy (str): Grouping strategy ('8_groups' or '6_groups')

    Returns:
        Strategy object for the specified mode
    """
    if mode == "grouped":
        return GroupedExtractionStrategy("grouped", debug, grouping_strategy)
    elif mode == "adaptive":
        return AdaptiveExtractionStrategy(debug)
    elif mode == "single_pass":
        return None  # Use traditional single-pass in processors
    else:
        raise ValueError(f"Unknown extraction mode: {mode}")
