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

try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    import json
    HAS_ORJSON = False

from .config import (
    EXTRACTION_FIELDS,
)
from .unified_schema import get_global_schema


def _fast_json_detection(text: str) -> bool:
    """
    Ultra-fast JSON detection without full parsing overhead.
    Handles markdown code blocks and various JSON formats.

    Args:
        text: Text to check for JSON format

    Returns:
        bool: True if text appears to be JSON format
    """
    text = text.strip()

    # Handle markdown code blocks
    if text.startswith('```json'):
        text = text[7:].strip()
    elif text.startswith('```'):
        text = text[3:].strip()

    if text.endswith('```'):
        text = text[:-3].strip()

    # Check for JSON structure
    return (
        len(text) >= 2 and
        text[0] == '{' and
        text.count('"') >= 4 and  # Minimum for basic JSON object
        (text[-1] == '}' or text.find('"') > 0)  # Either properly closed or has JSON fields
    )


def _try_parse_json(text: str, expected_fields: List[str]) -> Dict[str, str] | None:
    """
    Attempt to parse response as JSON using fastest available parser.
    Includes repair for common truncation issues.

    Args:
        text: Response text to parse
        expected_fields: Expected field names for extraction

    Returns:
        dict: Parsed fields if JSON, None if not JSON or parsing failed
    """
    if not _fast_json_detection(text):
        return None

    # Try to repair common JSON truncation issues
    repaired_text = _repair_truncated_json(text, expected_fields)

    try:
        if HAS_ORJSON:
            # Use orjson for maximum performance (3-5x faster than stdlib)
            json_data = orjson.loads(repaired_text)
        else:
            # Fallback to standard library json
            json_data = json.loads(repaired_text)

        if not isinstance(json_data, dict):
            return None

        # Convert JSON to expected format
        extracted_data = {field: "NOT_FOUND" for field in expected_fields}
        for field in expected_fields:
            if field in json_data:
                value = json_data[field]
                # Convert to string, handling various types
                if value is None or value == "":
                    extracted_data[field] = "NOT_FOUND"
                else:
                    extracted_data[field] = str(value)

        return extracted_data

    except (ValueError, TypeError) as e:
        # orjson raises ValueError, json raises JSONDecodeError (which inherits from ValueError)
        return None


def _repair_truncated_json(text: str, expected_fields: List[str]) -> str:
    """
    Attempt to repair common JSON truncation and formatting issues.

    Args:
        text: Potentially truncated JSON text
        expected_fields: Expected field names for validation

    Returns:
        str: Repaired JSON text
    """
    text = text.strip()

    # Remove markdown code blocks if present
    if text.startswith('```json'):
        text = text[7:]  # Remove ```json
    if text.startswith('```'):
        text = text[3:]  # Remove ```
    if text.endswith('```'):
        text = text[:-3]  # Remove trailing ```

    text = text.strip()

    # If JSON doesn't end with }, try to close it properly
    if not text.endswith('}'):
        # Find the last field that was being written
        lines = text.split('\n')

        # Look for incomplete field (missing closing quote or value)
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()

            # Handle incomplete string value (missing closing quote)
            if line.count('"') % 2 == 1 and ':' in line:
                # Find the last quote and close the string
                last_quote = line.rfind('"')
                if last_quote > 0 and line[last_quote-1] != '\\':
                    # Add closing quote if not escaped
                    lines[i] = line + '"'
                break

            # Handle incomplete field assignment (ends with |, comma, etc.)
            elif line.endswith(('|', ',', '| ')):
                # Complete the truncated field with closing quote
                lines[i] = line.rstrip('| ,') + '"'
                break

        # Reconstruct text and ensure proper JSON closure
        text = '\n'.join(lines)

        # Remove trailing commas and incomplete entries
        text = re.sub(r',\s*$', '', text, flags=re.MULTILINE)

        # Ensure JSON closes properly
        if not text.endswith('}'):
            text += '\n}'

    # Fix standalone commas and malformed JSON structure
    lines = text.split('\n')
    fixed_lines = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip standalone comma lines (malformed JSON pattern)
        if line == '",' or line == ',':
            # If we skipped a comma line, ensure previous line has comma (if it's a field)
            if (fixed_lines and
                '":' in fixed_lines[-1] and
                not fixed_lines[-1].endswith(',') and
                not fixed_lines[-1].endswith('}')):
                fixed_lines[-1] += ','
            i += 1
            continue

        # Remove trailing comma followed by quote if present
        if line.endswith('",'):
            line = line[:-2] + '"'

        fixed_lines.append(line)
        i += 1

    # Now add missing commas between fields
    final_lines = []
    for i, line in enumerate(fixed_lines):
        # If this line has a field and the next line also has a field
        # but current line doesn't end with comma or closing brace, add comma
        next_line = fixed_lines[i + 1] if i < len(fixed_lines) - 1 else ""

        if (i < len(fixed_lines) - 1 and
            '":' in line and
            not line.endswith(',') and
            not line.endswith('}') and
            '":' in next_line):
            line += ','
        final_lines.append(line)

    text = '\n'.join(final_lines)

    # Fix common formatting issues
    text = re.sub(r',\s*"', ',\n  "', text)  # Fix line breaks after commas
    text = re.sub(r'",\s*,', '",', text)  # Remove double commas

    return text


def hybrid_parse_response(response_text: str, expected_fields: List[str] = None) -> Dict[str, str]:
    """
    Hybrid parser that handles both JSON and plain text formats automatically.

    This is the main entry point for parsing model responses. It tries JSON first
    (optimized for complex documents like bank statements) and falls back to
    plain text parsing (for simple documents).

    Args:
        response_text: Raw model response
        expected_fields: Expected field names (optional, uses schema if None)

    Returns:
        dict: Parsed fields in consistent format
    """
    # Use provided fields or get from schema
    if expected_fields is None:
        try:
            schema = get_global_schema()
            expected_fields = schema.field_names
        except Exception:
            # Fallback to config-based fields if schema fails
            expected_fields = EXTRACTION_FIELDS

    if not response_text:
        return {field: "NOT_FOUND" for field in expected_fields}

    # Step 1: Try JSON parsing first (fast path for complex documents)
    json_result = _try_parse_json(response_text.strip(), expected_fields)
    if json_result is not None:
        return json_result

    # Step 2: Fallback to existing plain text parser
    return parse_extraction_response(
        response_text=response_text,
        clean_conversation_artifacts=False,
        expected_fields=expected_fields
    )


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

    # HYBRID PARSING: Try JSON first (fast path for complex documents like bank statements)
    json_result = _try_parse_json(response_text.strip(), expected_fields)
    if json_result is not None:
        return json_result

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

                # Extract value from whichever pattern matched (group 2 or 4)
                value = markdown_key_match.group(2) or markdown_key_match.group(4) or ""
                value = value.strip()


                # If value is empty, collect multi-line value from subsequent lines
                if not value and i + 1 < len(lines):
                    value_lines = []
                    j = i + 1
                    # Skip initial empty lines to find content (especially for LINE_ITEM fields)
                    while j < len(lines) and not lines[j].strip():
                        j += 1

                    # Collect all consecutive content lines that don't look like keys
                    while j < len(lines):
                        next_line = lines[j].strip()
                        # Stop if we hit another key (support both underscore and space patterns)
                        if re.match(r"^\*\*[A-Z_]+:\*\*|^\*\*[A-Z\s]+:\*\*", next_line):
                            break
                        # Stop if line contains colon (might be another field) - but allow address patterns
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
                        # Skip empty lines within the content but don't break
                        if not next_line:
                            j += 1
                            continue
                        value_lines.append(next_line)
                        j += 1

                    if value_lines:
                        # Handle list fields specially (LINE_ITEM_* fields)
                        if key.startswith("LINE_ITEM_"):
                            if all(line.strip().startswith("*") for line in value_lines):
                                # Remove bullet points and join with pipes for list fields
                                cleaned_items = [line.strip().lstrip("* ").strip() for line in value_lines if line.strip()]
                                value = " | ".join(cleaned_items)
                            else:
                                # Join with pipes even if no bullet points
                                value = " | ".join([line.strip() for line in value_lines if line.strip()])
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
