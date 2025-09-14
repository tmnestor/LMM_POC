"""
Model output parsing and data cleaning utilities.

This module handles the critical task of converting raw model outputs (text responses)
into structured data dictionaries. It includes robust parsing logic to handle various
model output formats including markdown, plain text, and edge cases.
"""

import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .config import (
    EXTRACTION_FIELDS,
)
from .unified_schema import get_global_schema


def parse_extraction_response(
    response_text: str,
    clean_conversation_artifacts: bool = False,
    expected_fields: List[str] = None,
) -> Dict[str, str]:
    """
    Parse structured extraction response into dictionary.

    This function handles model responses that may contain conversation artifacts
    or formatting issues, extracting only the key-value pairs.

    Implements a two-pass parsing strategy:
    1. First pass: Standard line-by-line parsing (works for clean outputs like Llama)
    2. Second pass: Markdown handling fallback (handles problematic outputs like InternVL3)

    Args:
        response_text (str): Raw model response containing key-value pairs
        clean_conversation_artifacts (bool): Whether to clean Llama-style artifacts
        expected_fields (List[str]): Optional list of fields to parse (for filtered extraction)

    Returns:
        dict: Parsed key-value pairs with all expected fields
    """
    # Use provided fields or get from schema (supports filtered field extraction)
    if expected_fields is None:
        try:
            schema = get_global_schema()
            expected_fields = schema.field_names
        except Exception:
            # Fallback to config-based fields if schema fails
            expected_fields = EXTRACTION_FIELDS

    if not response_text:
        return {field: "NOT_FOUND" for field in expected_fields}

    # Clean Llama-specific conversation artifacts if requested
    if clean_conversation_artifacts:
        # Remove common Llama conversation patterns
        clean_patterns = [
            r"I'll extract.*?\n",
            r"I can extract.*?\n",
            r"Here (?:is|are) the.*?\n",
            r"Based on.*?\n",
            r"Looking at.*?\n",
            r"<\|start_header_id\|>.*?<\|end_header_id\|>",
            r"<image>",
            r"assistant\n\n",
            r"^\s*Extract.*?below\.\s*\n",
        ]

        for pattern in clean_patterns:
            response_text = re.sub(
                pattern, "", response_text, flags=re.IGNORECASE | re.MULTILINE
            )

    # Initialize with NOT_FOUND for all fields
    extracted_data = {field: "NOT_FOUND" for field in expected_fields}

    # Process each line looking for key-value pairs
    lines = response_text.strip().split("\n")

    # First pass: Try standard parsing (works for Llama and clean InternVL3 output)
    extracted_data_first = {}
    for line in lines:
        # Skip empty lines and non-key-value lines
        if not line.strip() or ":" not in line:
            continue

        # Clean the line from various formatting issues
        clean_line = line
        # Remove markdown formatting - handle bullet points and inline formatting
        # First remove bullet point asterisks at start: "*   **FIELD:**" -> "   **FIELD:**"
        clean_line = re.sub(r"^\s*\*+\s*", "", clean_line)
        # Then remove inline markdown: "**text**" -> "text"
        clean_line = re.sub(r"\*+([^*]+)\*+", r"\1", clean_line)
        # Finally remove any remaining asterisks
        clean_line = clean_line.replace("**", "").replace("*", "")
        # Fix InternVL3 "KEY:" prefix issues
        clean_line = re.sub(r"^KEY:\s*([A-Z_]+):", r"\1:", clean_line)
        clean_line = re.sub(r"^KEY\s+([A-Z_]+):", r"\1:", clean_line)
        # Fix field name variations
        clean_line = re.sub(r"^DESCRIPTION:", "DESCRIPTIONS:", clean_line)
        clean_line = re.sub(r"^DESCRIPTIONDESCRIPTION:", "DESCRIPTIONS:", clean_line)
        # Fix LINE_ITEM_DESCRIPTION -> LINE_ITEM_DESCRIPTIONS mismatch
        clean_line = re.sub(r"^LINE_ITEM_DESCRIPTION:", "LINE_ITEM_DESCRIPTIONS:", clean_line)

        # Extract key and value
        parts = clean_line.split(":", 1)
        if len(parts) == 2:
            key = parts[0].strip().upper()
            value = parts[1].strip()

            # Store if it's an expected field
            if key in expected_fields:
                extracted_data_first[key] = value if value else "NOT_FOUND"

    # If first pass got most fields with actual values, use it (this preserves Llama's performance)
    # Only count fields that actually have values (not "NOT_FOUND")
    first_pass_valid_fields = sum(
        1 for v in extracted_data_first.values() if v != "NOT_FOUND"
    )
    if (
        first_pass_valid_fields >= len(expected_fields) * 0.5
    ):  # Got at least 50% of fields with actual values
        extracted_data.update(extracted_data_first)
    else:
        # Second pass: Handle multi-line markdown format (fallback for problematic InternVL3 output)
        processed_lines = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            # Check if this is a markdown key line (e.g., "**SUPPLIER:**" or "**SUPPLIER:** value")
            # Handle both cases: value on same line or next line
            # Support both underscore and space patterns: "**SUPPLIER_NAME:**" OR "**SUPPLIER NAME:**"
            markdown_key_match = re.match(r"^\*\*([A-Z_]+):\*\*\s*(.*)?$|^\*\*([A-Z\s]+):\*\*\s*(.*)?$", line)
            if markdown_key_match:
                # Extract key from whichever pattern matched (group 1 or 3)
                key = markdown_key_match.group(1) or markdown_key_match.group(3)
                key = key.replace(" ", "_")  # Normalize spaces to underscores

                # Debug output for LINE_ITEM fields
                if "LINE_ITEM" in key:
                    print(f"DEBUG: Matched markdown key: '{key}' from line: '{line}'")
                # Extract value from whichever pattern matched (group 2 or 4)
                value = markdown_key_match.group(2) or markdown_key_match.group(4) or ""
                value = value.strip()

                # Debug output for LINE_ITEM fields
                if "LINE_ITEM" in key:
                    print(f"DEBUG: Initial value for {key}: '{value}' (empty={not value})")

                # If value is empty, collect multi-line value from subsequent lines
                if not value and i + 1 < len(lines):
                    value_lines = []
                    j = i + 1
                    # Collect all consecutive non-empty lines that don't look like keys
                    while j < len(lines):
                        next_line = lines[j].strip()
                        # Stop if we hit an empty line or another key (support both underscore and space patterns)
                        if not next_line or re.match(r"^\*\*[A-Z_]+:\*\*|^\*\*[A-Z\s]+:\*\*", next_line):
                            break
                        # Stop if line contains colon (might be another field)
                        if ":" in next_line and not any(
                            addr_word in next_line.lower()
                            for addr_word in [
                                "street",
                                "road",
                                "avenue",
                                "drive",
                                "lane",
                                "court",
                                "place",
                                "way",
                                "vic",
                                "nsw",
                                "qld",
                                "sa",
                                "wa",
                                "tas",
                                "nt",
                                "act",
                            ]
                        ):
                            break
                        value_lines.append(next_line)
                        j += 1

                    if value_lines:
                        # Handle list fields specially (LINE_ITEM_* fields)
                        if key.startswith("LINE_ITEM_"):
                            # Debug output
                            print(f"DEBUG: Found LINE_ITEM field: {key}")
                            print(f"DEBUG: Value lines: {value_lines}")

                            if all(line.strip().startswith("*") for line in value_lines):
                                # Remove bullet points and join with pipes for list fields
                                cleaned_items = [line.strip().lstrip("* ").strip() for line in value_lines if line.strip()]
                                value = " | ".join(cleaned_items)
                                print(f"DEBUG: Cleaned to: {value}")
                            else:
                                # Join with pipes even if no bullet points
                                value = " | ".join([line.strip() for line in value_lines if line.strip()])
                                print(f"DEBUG: No bullets, joined to: {value}")
                        else:
                            # Join multi-line values with space for regular fields
                            value = " ".join(value_lines)
                        i = j  # Skip to after the collected lines
                    else:
                        i += 1  # Just skip the key line
                else:
                    i += 1  # Just skip the current line

                processed_lines.append(
                    f"{key}: {value}" if value else f"{key}: NOT_FOUND"
                )
            else:
                processed_lines.append(line)
                i += 1

        for line in processed_lines:
            # Skip empty lines and non-key-value lines
            if not line.strip() or ":" not in line:
                continue

            # Clean the line from various formatting issues
            clean_line = line
            # Remove markdown formatting
            clean_line = re.sub(r"\*+([^*]+)\*+", r"\1", clean_line)
            # Fix InternVL3 "KEY:" prefix issues
            clean_line = re.sub(r"^KEY:\s*([A-Z_]+):", r"\1:", clean_line)
            clean_line = re.sub(r"^KEY\s+([A-Z_]+):", r"\1:", clean_line)
            # Fix field name variations
            clean_line = re.sub(r"^DESCRIPTION:", "DESCRIPTIONS:", clean_line)
            clean_line = re.sub(
                r"^DESCRIPTIONDESCRIPTION:", "DESCRIPTIONS:", clean_line
            )

            # Extract key and value
            parts = clean_line.split(":", 1)
            if len(parts) == 2:
                key = parts[0].strip().upper()
                value = parts[1].strip()

                # Store if it's an expected field - this filters out hallucinated content
                if key in extracted_data:
                    # Don't overwrite if we already have a non-NOT_FOUND value
                    if extracted_data[key] == "NOT_FOUND" or not extracted_data[key]:
                        extracted_data[key] = value if value else "NOT_FOUND"
                        if "LINE_ITEM" in key:
                            print(f"DEBUG: Stored {key} = '{value[:50]}...' (truncated)")
                else:
                    # Debug unexpected keys
                    if "LINE_ITEM" in key:
                        print(f"DEBUG: Key '{key}' not in expected_fields: {list(extracted_data.keys())}")
                # Silently ignore unexpected keys to prevent hallucination contamination

    return extracted_data


def validate_and_enhance_extraction(
    extracted_data: Dict[str, str], image_name: str = None
) -> Dict[str, Any]:
    """
    Validate extracted data and add validation metadata.

    Args:
        extracted_data: Raw extracted field data
        image_name: Name of processed image (for error reporting)

    Returns:
        Enhanced dictionary with validation results
    """
    from .field_validation import validate_extracted_fields

    # Run validation
    validation_result = validate_extracted_fields(extracted_data)

    # Create enhanced result
    enhanced_result = {
        "extracted_data": extracted_data,
        "validation": {
            "is_valid": validation_result.is_valid,
            "error_count": len(validation_result.errors),
            "warning_count": len(validation_result.warnings),
            "errors": validation_result.errors,
            "warnings": validation_result.warnings,
        },
    }

    # Add corrected values if available
    if validation_result.corrected_values:
        enhanced_result["corrected_values"] = validation_result.corrected_values

    # Add image context for debugging
    if image_name:
        enhanced_result["image_name"] = image_name

    return enhanced_result


def create_extraction_dataframe(results: List[Dict]) -> tuple:
    """
    Create structured DataFrames from extraction results.

    Args:
        results (list): List of extraction result dictionaries

    Returns:
        tuple: (main_df, metadata_df) - Main extraction data and metadata
    """
    if not results:
        return pd.DataFrame(), pd.DataFrame()

    # Main extraction DataFrame
    rows = []
    metadata_rows = []

    for result in results:
        # Main data row
        row = {"image_name": result["image_name"]}
        row.update(result["extracted_data"])
        rows.append(row)

        # Metadata row
        if "response_completeness" in result or "content_coverage" in result:
            metadata_row = {
                "image_name": result["image_name"],
                "response_completeness": result.get("response_completeness", 0),
                "content_coverage": result.get("content_coverage", 0),
                "extracted_fields_count": result.get("extracted_fields_count", 0),
                "processing_time": result.get("processing_time", 0),
            }
            metadata_rows.append(metadata_row)

    main_df = pd.DataFrame(rows)
    metadata_df = pd.DataFrame(metadata_rows)

    return main_df, metadata_df


def discover_images(directory_path: str) -> List[str]:
    """
    Discover all image files in the specified directory.

    Args:
        directory_path (str): Path to directory containing images

    Returns:
        list: List of image file paths
    """
    directory = Path(directory_path)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    image_files = []
    for ext in image_extensions:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))

    # Sort by filename for consistent ordering
    return sorted([str(img) for img in image_files])
