"""
Lightweight parser module for llama_batch_pipeline.

Extracts only hybrid_parse_response and dependencies from common/extraction_parser.py.
No dependencies on common/ - fully self-contained.
"""

import re
from typing import Dict, List

try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    import json
    HAS_ORJSON = False

from dateutil import parser as date_parser


def _normalize_date(date_str: str) -> str:
    """Normalize various date formats to DD/MM/YYYY format."""
    if not date_str or date_str == "NOT_FOUND":
        return date_str

    try:
        clean_str = date_str.split("(")[0].strip()
        parsed_date = date_parser.parse(clean_str, dayfirst=True)
        return parsed_date.strftime("%d/%m/%Y")
    except (ValueError, TypeError, date_parser.ParserError):
        return date_str


def _fast_json_detection(text: str) -> bool:
    """Ultra-fast JSON detection without full parsing overhead."""
    text = text.strip()

    if text.startswith("```json"):
        text = text[7:].strip()
    elif text.startswith("```"):
        text = text[3:].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return (
        len(text) >= 2
        and text[0] == "{"
        and text.count('"') >= 4
        and (text[-1] == "}" or text.find('"') > 0)
    )


def _try_parse_json(text: str, expected_fields: List[str]) -> Dict[str, str] | None:
    """Attempt to parse response as JSON using fastest available parser."""
    if not _fast_json_detection(text):
        return None

    repaired_text = _repair_truncated_json(text, expected_fields)

    try:
        if HAS_ORJSON:
            json_data = orjson.loads(repaired_text)
        else:
            json_data = json.loads(repaired_text)

        if not isinstance(json_data, dict):
            return None

        extracted_data = {field: "NOT_FOUND" for field in expected_fields}
        for field in expected_fields:
            if field in json_data:
                value = json_data[field]
                if value is None or value == "":
                    extracted_data[field] = "NOT_FOUND"
                else:
                    extracted_data[field] = str(value)

        return extracted_data

    except (ValueError, TypeError):
        return None


def _repair_truncated_json(text: str, expected_fields: List[str]) -> str:
    """Attempt to repair common JSON truncation and formatting issues."""
    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    if not text.endswith("}"):
        lines = text.split("\n")

        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()

            if line.count('"') % 2 == 1 and ":" in line:
                last_quote = line.rfind('"')
                if last_quote > 0 and line[last_quote - 1] != "\\":
                    lines[i] = line + '"'
                break

            elif line.endswith(("|", ",", "| ")):
                lines[i] = line.rstrip("| ,") + '"'
                break

        text = "\n".join(lines)
        text = re.sub(r",\s*$", "", text, flags=re.MULTILINE)

        if not text.endswith("}"):
            text += "\n}"

    lines = text.split("\n")
    fixed_lines = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == '",' or line == ",":
            if (
                fixed_lines
                and '":' in fixed_lines[-1]
                and not fixed_lines[-1].endswith(",")
                and not fixed_lines[-1].endswith("}")
            ):
                fixed_lines[-1] += ","
            i += 1
            continue

        if line.endswith('",'):
            line = line[:-2] + '"'

        fixed_lines.append(line)
        i += 1

    final_lines = []
    for i, line in enumerate(fixed_lines):
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
    text = re.sub(r',\s*"', ',\n  "', text)
    text = re.sub(r'",\s*,', '",', text)

    return text


def parse_extraction_response(
    response_text: str,
    clean_conversation_artifacts: bool = False,
    expected_fields: List[str] = None,
) -> Dict[str, str]:
    """
    Parse structured extraction response into dictionary.

    Implements a two-pass parsing strategy:
    1. First pass: Standard line-by-line parsing
    2. Second pass: Markdown handling fallback
    """
    if not expected_fields:
        raise ValueError("expected_fields is required")

    if not response_text:
        return {field: "NOT_FOUND" for field in expected_fields}

    if clean_conversation_artifacts:
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

    extracted_data = {field: "NOT_FOUND" for field in expected_fields}

    json_result = _try_parse_json(response_text.strip(), expected_fields)
    if json_result is not None:
        return json_result

    lines = response_text.strip().split("\n")

    # First pass
    extracted_data_first = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or ":" not in line:
            i += 1
            continue

        clean_line = line
        clean_line = re.sub(r"^\s*\*+\s*", "", clean_line)
        clean_line = re.sub(r"\*+([^*]+)\*+", r"\1", clean_line)
        clean_line = clean_line.replace("**", "").replace("*", "")
        clean_line = re.sub(r"^KEY:\s*([A-Z_]+):", r"\1:", clean_line)
        clean_line = re.sub(r"^KEY\s+([A-Z_]+):", r"\1:", clean_line)
        clean_line = re.sub(r"^DESCRIPTION:", "DESCRIPTIONS:", clean_line)
        clean_line = re.sub(r"^DESCRIPTIONDESCRIPTION:", "DESCRIPTIONS:", clean_line)
        clean_line = re.sub(
            r"^LINE_ITEM_DESCRIPTION:", "LINE_ITEM_DESCRIPTIONS:", clean_line
        )

        parts = clean_line.split(":", 1)
        if len(parts) == 2:
            key = parts[0].strip().upper()
            value = str(parts[1]).strip()

            if key in expected_fields:
                is_list_field = key.startswith(
                    ("LINE_ITEM_", "TRANSACTION_", "ACCOUNT_BALANCE")
                )
                if not value and is_list_field and i + 1 < len(lines):
                    value_lines = []
                    j = i + 1
                    while j < len(lines) and not lines[j].strip():
                        j += 1

                    while j < len(lines):
                        next_line = lines[j].strip()
                        if ":" in next_line and not next_line.startswith(("*", "-")):
                            before_colon = next_line.split(":")[0].strip()
                            if before_colon.isupper() and "_" in before_colon:
                                break
                        if not next_line and value_lines:
                            break
                        if not next_line:
                            j += 1
                            continue
                        if next_line.startswith(("*", "-")):
                            value_lines.append(next_line)
                            j += 1
                        else:
                            break

                    if value_lines:
                        items = [line.lstrip("* -").strip() for line in value_lines]
                        value = " | ".join(items)
                        i = j - 1

                extracted_data_first[key] = value if value else "NOT_FOUND"

        i += 1

    extracted_data.update(extracted_data_first)

    # Second pass if needed
    first_pass_valid_fields = sum(
        1 for v in extracted_data_first.values() if v != "NOT_FOUND"
    )
    if first_pass_valid_fields < len(expected_fields) * 0.5:
        # Markdown handling fallback - simplified version
        pass

    # Post-processing
    list_field_prefixes = ("LINE_ITEM_", "TRANSACTION_", "ACCOUNT_BALANCE")
    address_fields = ("BUSINESS_ADDRESS", "PAYER_ADDRESS")
    date_fields = ("INVOICE_DATE", "TRANSACTION_DATES")

    for field_name, field_value in extracted_data.items():
        if field_value == "NOT_FOUND":
            continue

        if field_name == "DOCUMENT_TYPE":
            doc_type_lower = field_value.lower().strip()
            if doc_type_lower in ("statement", "bank statement"):
                extracted_data[field_name] = "BANK_STATEMENT"
            elif doc_type_lower in ("invoice", "bill"):
                extracted_data[field_name] = "INVOICE"
            elif doc_type_lower == "receipt":
                extracted_data[field_name] = "RECEIPT"
            continue

        if field_name.startswith(list_field_prefixes):
            if "," in field_value and " | " not in field_value:
                items = [
                    item.strip() for item in field_value.split(",") if item.strip()
                ]
                extracted_data[field_name] = " | ".join(items)
            elif "*" in field_value and " | " not in field_value:
                lines = field_value.split("\n")
                items = [
                    line.strip().lstrip("* ").strip() for line in lines if line.strip()
                ]
                extracted_data[field_name] = " | ".join(items)
            elif re.search(r"\s{2,}", field_value) and " | " not in field_value:
                items = [
                    item.strip()
                    for item in re.split(r"\s{2,}", field_value)
                    if item.strip()
                ]
                extracted_data[field_name] = " | ".join(items)

            if field_name == "LINE_ITEM_QUANTITIES":
                items = [
                    item.strip() for item in extracted_data[field_name].split(" | ")
                ]
                cleaned_items = [
                    re.sub(r"\s+EACH$", "", item, flags=re.IGNORECASE).strip()
                    for item in items
                ]
                extracted_data[field_name] = " | ".join(cleaned_items)

        elif field_name in address_fields and "," in field_value:
            extracted_data[field_name] = " ".join(field_value.split(",")).strip()

        if field_name in date_fields:
            if field_name == "TRANSACTION_DATES":
                current_value = extracted_data[field_name]
                dates = [d.strip() for d in current_value.split(" | ")]
                normalized_dates = [_normalize_date(d) for d in dates]
                extracted_data[field_name] = " | ".join(normalized_dates)
            else:
                extracted_data[field_name] = _normalize_date(extracted_data[field_name])

    return extracted_data


def hybrid_parse_response(
    response_text: str, expected_fields: List[str]
) -> Dict[str, str]:
    """
    Hybrid parser that handles both JSON and plain text formats automatically.

    This is the main entry point for parsing model responses. It tries JSON first
    (optimized for complex documents like bank statements) and falls back to
    plain text parsing (for simple documents).

    Args:
        response_text: Raw model response
        expected_fields: Expected field names (required)

    Returns:
        dict: Parsed fields in consistent format
    """
    if not response_text:
        return {field: "NOT_FOUND" for field in expected_fields}

    # Step 1: Try JSON parsing first (fast path for complex documents)
    json_result = _try_parse_json(response_text.strip(), expected_fields)
    if json_result is not None:
        return json_result

    # Step 2: Fallback to plain text parser
    return parse_extraction_response(
        response_text=response_text,
        clean_conversation_artifacts=False,
        expected_fields=expected_fields,
    )
