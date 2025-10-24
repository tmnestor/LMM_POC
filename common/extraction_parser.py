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
from dateutil import parser as date_parser

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


def _normalize_date(date_str: str) -> str:
    """
    Normalize various date formats to DD/MM/YYYY format.

    Handles formats like:
    - "26 Apr 2023" → "26/04/2023"
    - "2023-04-14 11:22 AM (UTC+10:00)" → "14/04/2023"
    - "Wednesday, 24th August 2022" → "24/08/2022"

    Args:
        date_str: Date string in any common format

    Returns:
        str: Date in DD/MM/YYYY format, or original string if parsing fails
    """
    if not date_str or date_str == "NOT_FOUND":
        return date_str

    try:
        # Remove timezone info and extra content for cleaner parsing
        # Strip anything after ( like "(UTC+10:00)"
        clean_str = date_str.split("(")[0].strip()

        # Parse with dayfirst=True for Australian DD/MM/YYYY preference
        parsed_date = date_parser.parse(clean_str, dayfirst=True)

        # Format as DD/MM/YYYY
        return parsed_date.strftime("%d/%m/%Y")
    except (ValueError, TypeError, date_parser.ParserError):
        # If parsing fails, return original string
        return date_str


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
    if text.startswith("```json"):
        text = text[7:].strip()
    elif text.startswith("```"):
        text = text[3:].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    # Check for JSON structure
    return (
        len(text) >= 2
        and text[0] == "{"
        and text.count('"') >= 4  # Minimum for basic JSON object
        and (
            text[-1] == "}" or text.find('"') > 0
        )  # Either properly closed or has JSON fields
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
    if text.startswith("```json"):
        text = text[7:]  # Remove ```json
    if text.startswith("```"):
        text = text[3:]  # Remove ```
    if text.endswith("```"):
        text = text[:-3]  # Remove trailing ```

    text = text.strip()

    # If JSON doesn't end with }, try to close it properly
    if not text.endswith("}"):
        # Find the last field that was being written
        lines = text.split("\n")

        # Look for incomplete field (missing closing quote or value)
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()

            # Handle incomplete string value (missing closing quote)
            if line.count('"') % 2 == 1 and ":" in line:
                # Find the last quote and close the string
                last_quote = line.rfind('"')
                if last_quote > 0 and line[last_quote - 1] != "\\":
                    # Add closing quote if not escaped
                    lines[i] = line + '"'
                break

            # Handle incomplete field assignment (ends with |, comma, etc.)
            elif line.endswith(("|", ",", "| ")):
                # Complete the truncated field with closing quote
                lines[i] = line.rstrip("| ,") + '"'
                break

        # Reconstruct text and ensure proper JSON closure
        text = "\n".join(lines)

        # Remove trailing commas and incomplete entries
        text = re.sub(r",\s*$", "", text, flags=re.MULTILINE)

        # Ensure JSON closes properly
        if not text.endswith("}"):
            text += "\n}"

    # Fix standalone commas and malformed JSON structure
    lines = text.split("\n")
    fixed_lines = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip standalone comma lines (malformed JSON pattern)
        if line == '",' or line == ",":
            # If we skipped a comma line, ensure previous line has comma (if it's a field)
            if (
                fixed_lines
                and '":' in fixed_lines[-1]
                and not fixed_lines[-1].endswith(",")
                and not fixed_lines[-1].endswith("}")
            ):
                fixed_lines[-1] += ","
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

        if (
            i < len(fixed_lines) - 1
            and '":' in line
            and not line.endswith(",")
            and not line.endswith("}")
            and '":' in next_line
        ):
            line += ","
        final_lines.append(line)

    text = "\n".join(final_lines)

    # Fix common formatting issues
    text = re.sub(r',\s*"', ',\n  "', text)  # Fix line breaks after commas
    text = re.sub(r'",\s*,', '",', text)  # Remove double commas

    return text


def hybrid_parse_response(
    response_text: str, expected_fields: List[str] = None
) -> Dict[str, str]:
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
        expected_fields=expected_fields,
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
    i = 0
    while i < len(lines):
        line = lines[i]
        # Skip empty lines and non-key-value lines
        if not line.strip() or ":" not in line:
            i += 1
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
        clean_line = re.sub(
            r"^LINE_ITEM_DESCRIPTION:", "LINE_ITEM_DESCRIPTIONS:", clean_line
        )

        # Extract key and value
        parts = clean_line.split(":", 1)
        if len(parts) == 2:
            key = parts[0].strip().upper()
            # Convert to string first to handle boolean/numeric values
            value = str(parts[1]).strip()

            # Store if it's an expected field
            if key in expected_fields:
                # If value is empty, look ahead for bullet list on next lines (list fields only)
                # List fields: LINE_ITEM_*, TRANSACTION_*, ACCOUNT_BALANCE
                is_list_field = key.startswith(
                    ("LINE_ITEM_", "TRANSACTION_", "ACCOUNT_BALANCE")
                )
                if not value and is_list_field and i + 1 < len(lines):
                    # Collect subsequent bullet point lines or plain text lines
                    value_lines = []
                    j = i + 1
                    # Skip initial empty lines
                    while j < len(lines) and not lines[j].strip():
                        j += 1

                    # Now collect value lines
                    while j < len(lines):
                        next_line = lines[j].strip()
                        # Stop if we hit another field (non-bullet line with colon)
                        if ":" in next_line and not next_line.startswith(("*", "-")):
                            # Check if this looks like a field name (all caps before colon)
                            before_colon = next_line.split(":")[0].strip()
                            if before_colon.isupper() and "_" in before_colon:
                                break
                        # Stop at empty line after we've started collecting
                        if not next_line and value_lines:
                            break
                        # Skip empty lines between bullets
                        if not next_line:
                            j += 1
                            continue
                        # Collect the line (bullet or plain text)
                        if next_line.startswith(("*", "-")):
                            value_lines.append(next_line)
                            j += 1
                        else:
                            # Not a bullet point, might be next field
                            break

                    # If we found lines, process them
                    if value_lines:
                        # Remove bullet points and join
                        items = [line.lstrip("* -").strip() for line in value_lines]
                        value = " | ".join(items)
                        i = j - 1  # Skip the lines we consumed

                extracted_data_first[key] = value if value else "NOT_FOUND"

        i += 1

    # ALWAYS use first pass results as the starting point (includes look-ahead parsing for LINE_ITEM fields)
    extracted_data.update(extracted_data_first)

    # If first pass got most fields with actual values, skip second pass (preserves Llama's performance)
    # Only count fields that actually have values (not "NOT_FOUND")
    first_pass_valid_fields = sum(
        1 for v in extracted_data_first.values() if v != "NOT_FOUND"
    )
    if first_pass_valid_fields < len(expected_fields) * 0.5:
        # First pass didn't get enough fields, run second pass to fill gaps
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
            markdown_key_match = re.match(
                r"^\*\*([A-Z_]+):\*\*\s*(.*)?$|^\*\*([A-Z\s]+):\*\*\s*(.*)?$", line
            )
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
                            if all(
                                line.strip().startswith("*") for line in value_lines
                            ):
                                # Remove bullet points and join with pipes for list fields
                                cleaned_items = [
                                    line.strip().lstrip("* ").strip()
                                    for line in value_lines
                                    if line.strip()
                                ]
                                value = " | ".join(cleaned_items)
                            else:
                                # Join with pipes even if no bullet points
                                value = " | ".join(
                                    [
                                        line.strip()
                                        for line in value_lines
                                        if line.strip()
                                    ]
                                )
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
                # Convert to string first to handle boolean/numeric values
                value = str(parts[1]).strip()

                # Store if it's an expected field - this filters out hallucinated content
                if key in extracted_data:
                    # Don't overwrite if we already have a non-NOT_FOUND value
                    if extracted_data[key] == "NOT_FOUND" or not extracted_data[key]:
                        extracted_data[key] = value if value else "NOT_FOUND"
                # Silently ignore unexpected keys to prevent hallucination contamination

    # POST-PROCESSING: Clean field values
    # 1. Document type normalization
    # 2. List fields: Convert commas/markdown/spaces to " | " separator
    # 3. Address fields: Remove commas entirely
    # 4. Quantity fields: Remove " EACH" suffix
    # 5. Date fields: Normalize to DD/MM/YYYY format
    list_field_prefixes = ("LINE_ITEM_", "TRANSACTION_", "ACCOUNT_BALANCE")
    address_fields = ("BUSINESS_ADDRESS", "PAYER_ADDRESS")
    date_fields = ("INVOICE_DATE", "TRANSACTION_DATES")

    for field_name, field_value in extracted_data.items():
        if field_value == "NOT_FOUND":
            continue

        # Normalize DOCUMENT_TYPE values to canonical forms
        if field_name == "DOCUMENT_TYPE":
            doc_type_lower = field_value.lower().strip()
            if doc_type_lower in ("statement", "bank statement"):
                extracted_data[field_name] = "BANK_STATEMENT"
            elif doc_type_lower in ("invoice", "bill"):
                extracted_data[field_name] = "INVOICE"
            elif doc_type_lower == "receipt":
                extracted_data[field_name] = "RECEIPT"
            continue

        # Handle list fields: convert commas/markdown/spaces to pipes
        if field_name.startswith(list_field_prefixes):
            # Check if value contains markdown bullet points or commas instead of pipes
            if "," in field_value and " | " not in field_value:
                # Convert comma-separated to pipe-separated
                items = [
                    item.strip() for item in field_value.split(",") if item.strip()
                ]
                extracted_data[field_name] = " | ".join(items)
            elif "*" in field_value and " | " not in field_value:
                # Convert markdown list to pipe-separated
                # Split by newlines and clean bullet points
                lines = field_value.split("\n")
                items = [
                    line.strip().lstrip("* ").strip() for line in lines if line.strip()
                ]
                extracted_data[field_name] = " | ".join(items)
            elif re.search(r"\s{2,}", field_value) and " | " not in field_value:
                # Convert space-separated to pipe-separated (2+ consecutive spaces)
                items = [
                    item.strip()
                    for item in re.split(r"\s{2,}", field_value)
                    if item.strip()
                ]
                extracted_data[field_name] = " | ".join(items)

            # Special handling for LINE_ITEM_QUANTITIES: remove " EACH" suffix
            if field_name == "LINE_ITEM_QUANTITIES":
                # Remove " EACH" from each quantity item
                items = [
                    item.strip() for item in extracted_data[field_name].split(" | ")
                ]
                cleaned_items = [
                    re.sub(r"\s+EACH$", "", item, flags=re.IGNORECASE).strip()
                    for item in items
                ]
                extracted_data[field_name] = " | ".join(cleaned_items)

        # Handle address fields: remove commas entirely
        elif field_name in address_fields and "," in field_value:
            # Remove commas and normalize spaces
            extracted_data[field_name] = " ".join(field_value.split(",")).strip()

        # Handle date fields: normalize to DD/MM/YYYY format
        # Note: Use 'if' not 'elif' because TRANSACTION_DATES is also a list field
        if field_name in date_fields:
            if field_name == "TRANSACTION_DATES":
                # Handle list of dates (pipe-separated)
                # Use extracted_data[field_name] to get the updated value after pipe conversion
                current_value = extracted_data[field_name]
                dates = [d.strip() for d in current_value.split(" | ")]
                normalized_dates = [_normalize_date(d) for d in dates]
                extracted_data[field_name] = " | ".join(normalized_dates)
            else:
                # Handle single date (INVOICE_DATE)
                extracted_data[field_name] = _normalize_date(extracted_data[field_name])

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
