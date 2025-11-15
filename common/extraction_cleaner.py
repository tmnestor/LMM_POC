#!/usr/bin/env python3
"""
Extraction Value Cleaner - Centralized field value cleaning for document processors.

This module provides standardized cleaning and normalization of extracted field values
for both Llama and InternVL3 document-aware processors, ensuring consistent output
formatting and improved accuracy against ground truth data.
"""

import re
from typing import Any, Dict


def sanitize_for_rich(content: str, max_length: int = 200) -> str:
    """
    Sanitize content for safe Rich console rendering.

    Rich console interprets '[' and ']' as markup syntax, which can cause
    RecursionError with certain content patterns. This function escapes
    those characters and truncates long content.

    Args:
        content: Content to sanitize
        max_length: Maximum length before truncation

    Returns:
        Sanitized content safe for Rich console output
    """
    if not content:
        return content

    # Escape Rich markup characters that cause recursion
    safe_content = str(content).replace("[", "\\[").replace("]", "\\]")

    # Truncate if too long to prevent overwhelming output
    if len(safe_content) > max_length:
        safe_content = safe_content[:max_length] + "..."

    return safe_content


class ExtractionCleaner:
    """
    Centralized field value cleaning for all document processors.

    Provides standardized cleaning methods for different field types to improve
    accuracy and consistency across different vision-language models.
    """

    def __init__(self, debug: bool = False):
        """
        Initialize extraction cleaner.

        Args:
            debug (bool): Enable debug output for cleaning operations
        """
        self.debug = debug

        # Field type patterns for automatic detection
        self.monetary_patterns = [
            "AMOUNT",
            "PRICE",
            "COST",
            "FEE",
            "TOTAL",
            "SUBTOTAL",
            "GST",
        ]
        self.list_patterns = [
            "LINE_ITEM",
            "ITEMS",
            "DESCRIPTIONS",
            "QUANTITIES",
            "PRICES",
            "TRANSACTION_DATES",  # Bank statement transaction dates
            "TRANSACTION_AMOUNTS",  # Bank statement amounts (paid/received)
            "ACCOUNT_BALANCE",  # Bank statement running balances
        ]
        self.date_patterns = ["DATE", "TIME", "PERIOD", "RANGE"]
        self.id_patterns = ["NUMBER", "ID", "ABN", "BSB", "ACCOUNT"]
        self.address_patterns = ["ADDRESS", "LOCATION", "STREET"]

        # Common cleaning patterns
        self.price_suffix_patterns = [
            r"\s+(each|per\s+item|per\s+unit|per\s+piece)$",
            r"\s+(ea\.?|@|apiece)$",
        ]

        # Currency symbol standardization
        self.currency_patterns = {
            r"AUD?\s*": "$",
            r"USD?\s*": "$",
            r"\$\$+": "$",  # Multiple dollar signs
        }

    def clean_field_value(self, field_name: str, raw_value: Any) -> str:
        """
        Clean extracted value based on field type and name.

        Args:
            field_name (str): Name of the field being cleaned
            raw_value (Any): Raw extracted value from model

        Returns:
            str: Cleaned and normalized value
        """
        # Handle None, empty, or NOT_FOUND cases (including model variations)
        missing_values = [
            "NOT_FOUND",
            "Not Found",
            "not found",
            "None",
            "null",
            "",
            "N/A",
            "n/a",
        ]
        if not raw_value or raw_value in missing_values:
            return "NOT_FOUND"

        # Convert to string and basic cleaning
        value = str(raw_value).strip()
        if not value:
            return "NOT_FOUND"

        if self.debug:
            safe_raw_value = sanitize_for_rich(str(raw_value), max_length=100)
            print(f"🧹 CLEANER CALLED: {field_name}: '{safe_raw_value}' -> ", end="")

        # Pre-process lists: convert comma-separated to pipe-separated format
        if self._is_list_field(field_name) and "," in value and "|" not in value:
            value = " | ".join(item.strip() for item in value.split(","))

        # Route to appropriate cleaning method based on field type using pattern matching
        field_type = self._get_field_type(field_name)

        match field_type:
            case "monetary":
                cleaned_value = self._clean_monetary_field(value)
            case "list":
                cleaned_value = self._clean_list_field(field_name, value)
            case "date":
                cleaned_value = self._clean_date_field(value)
            case "id":
                cleaned_value = self._clean_id_field(value)
            case "address":
                cleaned_value = self._clean_address_field(value)
            case "text":
                cleaned_value = self._clean_text_field(value)
            case _:
                cleaned_value = self._clean_text_field(value)

        # Special field-specific post-processing
        if field_name == "DOCUMENT_TYPE":
            cleaned_value = self._normalize_document_type(cleaned_value)

        if self.debug:
            safe_cleaned_value = sanitize_for_rich(str(cleaned_value), max_length=100)
            print(f"'{safe_cleaned_value}'")

        return cleaned_value

    def _is_monetary_field(self, field_name: str) -> bool:
        """Check if field is monetary type."""
        return any(pattern in field_name.upper() for pattern in self.monetary_patterns)

    def _is_list_field(self, field_name: str) -> bool:
        """Check if field is list type."""
        return any(pattern in field_name.upper() for pattern in self.list_patterns)

    def _is_date_field(self, field_name: str) -> bool:
        """Check if field is date type."""
        return any(pattern in field_name.upper() for pattern in self.date_patterns)

    def _is_id_field(self, field_name: str) -> bool:
        """Check if field is ID/number type."""
        return any(pattern in field_name.upper() for pattern in self.id_patterns)

    def _is_address_field(self, field_name: str) -> bool:
        """Check if field is address type."""
        return any(pattern in field_name.upper() for pattern in self.address_patterns)

    def _get_field_type(self, field_name: str) -> str:
        """
        Determine the field type for pattern matching.

        Returns:
            str: Field type category for routing to appropriate cleaning method
        """
        match field_name.upper():
            case name if any(pattern in name for pattern in self.list_patterns):
                return "list"
            case name if any(pattern in name for pattern in self.monetary_patterns):
                return "monetary"
            case name if any(pattern in name for pattern in self.date_patterns):
                return "date"
            case name if any(pattern in name for pattern in self.id_patterns):
                return "id"
            case name if any(pattern in name for pattern in self.address_patterns):
                return "address"
            case _:
                return "text"

    def _clean_monetary_field(self, value: str) -> str:
        """
        Clean monetary values (AMOUNT, PRICE fields).

        Handles:
        - Remove markdown artifacts
        - Remove "each", "per item", etc. suffixes
        - Standardize currency symbols
        - Remove commas from large amounts to match ground truth format
        - Clean up spacing and formatting
        """
        if not value or value == "NOT_FOUND":
            return "NOT_FOUND"

        cleaned = value

        # Remove markdown artifacts first
        cleaned = re.sub(
            r"^\s*\*+\s*", "", cleaned
        )  # Remove leading whitespace + asterisks + spaces
        cleaned = cleaned.replace("**", "")  # Remove any double asterisks

        # Remove common price suffixes
        for pattern in self.price_suffix_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        # Standardize currency symbols
        for pattern, replacement in self.currency_patterns.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

        # CRITICAL: Remove commas from monetary values to match ground truth format
        # Examples: "$4,834.03" -> "$4834.03", "$1,234.56" -> "$1234.56"
        cleaned = cleaned.replace(",", "")

        # Clean up multiple spaces and trim
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Ensure dollar sign is at the beginning if present
        if "$" in cleaned and not cleaned.startswith("$"):
            cleaned = re.sub(r"(.*?)\$(.+)", r"$\2", cleaned)

        return cleaned if cleaned else "NOT_FOUND"

    def _clean_list_field(self, field_name: str, value: str) -> str:
        """
        Clean list fields (LINE_ITEM_*) and convert to pipe-separated format.

        Handles:
        - Remove markdown artifacts from the whole value and each item
        - Split by comma or pipe, clean each item
        - Remove price suffixes from price lists
        - Convert to standard pipe-separated format
        - Normalize whitespace and separators
        - Issue #5: Fix missing pipe separators in amount lists
        """
        if not value or value == "NOT_FOUND":
            return "NOT_FOUND"

        # Remove markdown artifacts from the whole value first
        value = re.sub(
            r"^\s*\*+\s*", "", value
        )  # Remove leading whitespace + asterisks + spaces
        value = value.replace("**", "")  # Remove any double asterisks

        # Issue #5: Fix missing pipe separators between currency amounts
        # Pattern: "$XX.XX $YY.YY" → "$XX.XX | $YY.YY"
        # Only apply to monetary list fields (TRANSACTION_AMOUNTS, LINE_ITEM_PRICES, etc.)
        if any(
            pattern in field_name.upper()
            for pattern in ["AMOUNT", "PRICE", "COST", "FEE"]
        ):
            # Fix concatenated amounts: "$251.33 $98.53" → "$251.33 | $98.53"
            value = re.sub(r"(\$\d+\.\d{2})\s+(\$\d+\.\d{2})", r"\1 | \2", value)

        # Split by comma or pipe and clean each item
        if "|" in value:
            items = [item.strip() for item in value.split("|")]
        else:
            items = [item.strip() for item in value.split(",")]

        cleaned_items = []

        for item in items:
            if not item:
                continue

            cleaned_item = item

            # For price lists, apply monetary cleaning to each item
            if "PRICE" in field_name.upper():
                cleaned_item = self._clean_monetary_field(item)
            else:
                # General text cleaning for descriptions and quantities
                cleaned_item = self._clean_text_field(item)

            # PRESERVE POSITIONAL ARRAY STRUCTURE: Keep NOT_FOUND values for bank statement arrays
            # This is critical for TRANSACTION_AMOUNTS_* fields where position matters
            if (
                field_name.upper().startswith("TRANSACTION_AMOUNTS")
                or field_name.upper() == "ACCOUNT_BALANCE"
            ):
                # For bank statement transaction arrays, preserve ALL positions including NOT_FOUND
                cleaned_items.append(cleaned_item if cleaned_item else "NOT_FOUND")
            else:
                # For other list fields, filter out NOT_FOUND (original behavior)
                if cleaned_item and cleaned_item != "NOT_FOUND":
                    cleaned_items.append(cleaned_item)

        # Always return pipe-separated format for consistency
        return " | ".join(cleaned_items) if cleaned_items else "NOT_FOUND"

    def _clean_date_field(self, value: str) -> str:
        """
        Standardize date formats.

        Handles:
        - Remove markdown artifacts
        - Normalize common date separators
        - Trim whitespace
        - Standardize format consistency
        """
        if not value or value == "NOT_FOUND":
            return "NOT_FOUND"

        cleaned = value.strip()

        # Remove markdown artifacts
        cleaned = re.sub(
            r"^\s*\*+\s*", "", cleaned
        )  # Remove leading whitespace + asterisks + spaces
        cleaned = cleaned.replace("**", "")  # Remove any double asterisks

        # Normalize date separators (keep original format but clean spacing)
        cleaned = re.sub(r"\s*[-/\.]\s*", lambda m: m.group(0).strip(), cleaned)

        # Clean up extra spaces
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned if cleaned else "NOT_FOUND"

    def _clean_id_field(self, value: str) -> str:
        """
        Clean ID fields (ABN, account numbers, etc.).

        Handles:
        - Remove field label prefixes (ABN, BSB, GST, etc.)
        - Normalize spacing in formatted numbers
        - Clean up extra characters
        - Preserve number grouping where appropriate
        """
        if not value or value == "NOT_FOUND":
            return "NOT_FOUND"

        cleaned = value.strip()

        # CRITICAL: Remove field label prefixes that VLMs often include
        # Common patterns: "ABN 12345", "ABN: 12345", "GST 12345", "BSB 123-456"
        id_label_pattern = r"^(ABN|BSB|ACN|GST|TAX|ID|NUMBER|NO\.?|#)\s*:?\s*"
        cleaned = re.sub(id_label_pattern, "", cleaned, flags=re.IGNORECASE).strip()

        # For ABN: Remove ALL spaces to match ground truth format
        # Ground truth stores ABN as: "96723184640" (no spaces)
        # VLMs extract as: "ABN 9672 3184 640" or "96 723 184 640"
        # After prefix removal: "9672 3184 640" -> "96723184640"
        if re.match(r"^\d+(\s+\d+)+$", cleaned) and len(cleaned.replace(" ", "")) == 11:
            # This is an 11-digit ABN with spaces - remove all spaces
            cleaned = cleaned.replace(" ", "")
        elif re.match(r"^\d{3}-\d{3}$", cleaned):
            # BSB format: XXX-XXX - keep the dash
            pass
        else:
            # General ID cleaning - normalize whitespace
            cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned if cleaned else "NOT_FOUND"

    def _clean_text_field(self, value: str) -> str:
        """
        Clean general text fields.

        Handles:
        - Remove markdown artifacts (**, *, etc.)
        - Normalize whitespace
        - Trim leading/trailing spaces
        - Remove redundant punctuation
        - Document type normalization (STATEMENT → BANK_STATEMENT)
        """
        if not value or value == "NOT_FOUND":
            return "NOT_FOUND"

        cleaned = value.strip()
        original_cleaned = cleaned

        # Remove markdown artifacts that models sometimes generate
        # Handle " ** STATEMENT", "**STATEMENT", "* value", etc.
        cleaned = re.sub(
            r"^\s*\*+\s*", "", cleaned
        )  # Remove leading whitespace + asterisks + spaces
        cleaned = re.sub(
            r"\s*\*+\s*$", "", cleaned
        )  # Remove trailing whitespace + asterisks
        cleaned = cleaned.replace("**", "")  # Remove any remaining double asterisks

        # Debug logging for markdown cleaning
        if self.debug and original_cleaned != cleaned and "*" in original_cleaned:
            print(f"🧹 CLEANER DEBUG: '{original_cleaned}' → '{cleaned}'")

        # Normalize whitespace - collapse multiple spaces to single space
        cleaned = re.sub(r"\s+", " ", cleaned)

        # Remove trailing punctuation that might be artifacts
        cleaned = re.sub(r"\s*[,;]\s*$", "", cleaned)

        # Final trim to remove any leading/trailing whitespace
        cleaned = cleaned.strip()

        return cleaned if cleaned else "NOT_FOUND"

    def _normalize_document_type(self, value: str) -> str:
        """
        Normalize document type values to match ground truth expectations.

        Handles:
        - STATEMENT → BANK_STATEMENT (common model output vs ground truth mismatch)
        - BILL → INVOICE (models sometimes classify invoices as bills)
        - TAX INVOICE → INVOICE (Issue #4: Llama 4 outputs "Tax Invoice")
        """
        if not value or value == "NOT_FOUND":
            return "NOT_FOUND"

        # Normalize document type to match ground truth format
        normalized = value.strip().upper()

        # Map model outputs to ground truth format
        document_type_mappings = {
            "STATEMENT": "BANK_STATEMENT",
            "BILL": "INVOICE",
            "TAX INVOICE": "INVOICE",  # Issue #4: Handle "Tax Invoice" variant
        }

        return document_type_mappings.get(normalized, normalized)

    def _clean_address_field(self, value: str) -> str:
        """
        Clean address fields by removing embedded contact information and commas.

        Business Knowledge:
        - Addresses often contain embedded phone numbers, emails, or other contact info
        - These should be cleaned to extract pure address only
        - Remove commas to match ground truth format (e.g., "123 Main St, City" -> "123 Main St City")
        """
        if not value or value == "NOT_FOUND":
            return "NOT_FOUND"

        cleaned = value.strip()

        # Remove common address suffixes with contact info
        # Pattern: " - P: (phone)" or " - Phone: (phone)" or " - Tel: (phone)"
        cleaned = re.sub(
            r"\s*-\s*(?:P|Phone|Tel|T):\s*\(?[0-9\s\)\(-]+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        # Remove email addresses from address fields
        cleaned = re.sub(
            r"\s*-?\s*[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "", cleaned
        )

        # Remove standalone phone numbers at end of address
        # Pattern: phone numbers with common formats
        # IMPORTANT: Avoid matching ABN numbers (11 digits: XX XXX XXX XXX)
        # Only match if it looks like a phone number, not an ABN
        phone_pattern = r"\s*-?\s*\(?[0-9]{2,4}\)?\s*[0-9\s\-]{6,}$"
        # Check if this looks like an ABN number (XX XXX XXX XXX format)
        abn_pattern = r"\d{2}\s+\d{3}\s+\d{3}\s+\d{3}$"
        if not re.search(abn_pattern, cleaned):
            cleaned = re.sub(phone_pattern, "", cleaned)

        # CRITICAL: Remove commas from addresses to match ground truth format
        # Examples: "1/92 Watt Road, Mornington, VIC 3931" -> "1/92 Watt Road Mornington VIC 3931"
        cleaned = cleaned.replace(",", "")

        # Clean up any trailing punctuation left after removals
        cleaned = re.sub(r"\s*[;\-]\s*$", "", cleaned)

        # Remove spaces around slashes (for unit numbers like "1 / 92" -> "1/92")
        cleaned = re.sub(r"\s*/\s*", "/", cleaned)

        # CRITICAL: Aggressive whitespace normalization - MUST be last step
        # Replace ANY sequence of whitespace (spaces, tabs, non-breaking spaces, Unicode spaces) with single space
        # This handles double spaces, triple spaces, tabs, etc.
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        if self.debug and cleaned != value.strip():
            print(f"🏠 Address cleaned: '{value.strip()}' -> '{cleaned}'")

        return cleaned if cleaned else "NOT_FOUND"

    def clean_extraction_dict(self, field_dict: Dict[str, Any]) -> Dict[str, str]:
        """
        Clean all fields in an extraction dictionary with business knowledge validation.

        Args:
            field_dict (Dict[str, Any]): Dictionary of extracted field values

        Returns:
            Dict[str, str]: Dictionary with cleaned field values
        """
        cleaned_dict = {}

        # Step 1: Clean individual fields
        for field_name, raw_value in field_dict.items():
            cleaned_dict[field_name] = self.clean_field_value(field_name, raw_value)

        # Step 2: Apply business knowledge validation
        cleaned_dict = self._apply_business_knowledge(cleaned_dict)

        if self.debug:
            cleaned_count = sum(1 for v in cleaned_dict.values() if v != "NOT_FOUND")
            total_count = len(cleaned_dict)
            print(f"🧹 Cleaned {cleaned_count}/{total_count} fields")

        return cleaned_dict

    def _apply_business_knowledge(self, cleaned_dict: Dict[str, str]) -> Dict[str, str]:
        """
        Apply business knowledge and cross-field validation using pattern matching.

        Key business rules:
        - LINE_ITEM_PRICES = unit prices (price per individual item)
        - LINE_ITEM_TOTAL_PRICES = quantity × unit price
        - GST calculations must be consistent
        - Issue #6: Bank statement transaction counts must match
        """
        validated_dict = cleaned_dict.copy()

        # Apply business rules based on data characteristics
        match (self._has_line_items(cleaned_dict), self._has_gst_fields(cleaned_dict)):
            case (True, True):
                # Document has both line items and GST - validate both
                validated_dict = self._validate_line_item_pricing(validated_dict)
                validated_dict = self._validate_gst_consistency(validated_dict)
            case (True, False):
                # Document has line items only - validate pricing
                validated_dict = self._validate_line_item_pricing(validated_dict)
            case (False, True):
                # Document has GST only - validate GST consistency
                validated_dict = self._validate_gst_consistency(validated_dict)
            case (False, False):
                # No special validation needed
                pass

        # Issue #6: Validate bank statement transaction count consistency
        if self._is_bank_statement(cleaned_dict):
            validated_dict = self._validate_transaction_counts(validated_dict)

        return validated_dict

    def _has_line_items(self, data: Dict[str, str]) -> bool:
        """Check if document has line item data."""
        required_fields = [
            "LINE_ITEM_DESCRIPTIONS",
            "LINE_ITEM_QUANTITIES",
            "LINE_ITEM_PRICES",
        ]
        return any(
            data.get(field, "NOT_FOUND") != "NOT_FOUND" for field in required_fields
        )

    def _has_gst_fields(self, data: Dict[str, str]) -> bool:
        """Check if document has GST-related fields."""
        gst_fields = ["GST_AMOUNT", "IS_GST_INCLUDED", "LINE_ITEM_GST_AMOUNTS"]
        return any(data.get(field, "NOT_FOUND") != "NOT_FOUND" for field in gst_fields)

    def _is_bank_statement(self, data: Dict[str, str]) -> bool:
        """Check if document is a bank statement."""
        doc_type = data.get("DOCUMENT_TYPE", "").upper()
        return "BANK" in doc_type or "STATEMENT" in doc_type

    def _validate_line_item_pricing(self, data: Dict[str, str]) -> Dict[str, str]:
        """
        Validate LINE_ITEM_PRICES represents unit prices, not totals.

        Business Knowledge:
        - LINE_ITEM_PRICES = price per unit (unit price)
        - LINE_ITEM_TOTAL_PRICES = quantity × unit price (often labeled 'Amount' column)
        """
        prices = data.get("LINE_ITEM_PRICES", "NOT_FOUND")
        total_prices = data.get("LINE_ITEM_TOTAL_PRICES", "NOT_FOUND")
        quantities = data.get("LINE_ITEM_QUANTITIES", "NOT_FOUND")

        if all(field != "NOT_FOUND" for field in [prices, total_prices, quantities]):
            try:
                price_list = [
                    self._parse_monetary_value(p.strip()) for p in prices.split("|")
                ]
                total_list = [
                    self._parse_monetary_value(p.strip())
                    for p in total_prices.split("|")
                ]
                qty_list = [float(q.strip()) for q in quantities.split("|")]

                if len(price_list) == len(total_list) == len(qty_list):
                    issues_found = []

                    for i, (unit_price, total_price, qty) in enumerate(
                        zip(price_list, total_list, qty_list, strict=False)
                    ):
                        if unit_price and total_price and qty:
                            expected_total = unit_price * qty
                            # Allow 1% tolerance for rounding
                            if abs(total_price - expected_total) > (
                                expected_total * 0.01
                            ):
                                issues_found.append(
                                    f"Item {i + 1}: Unit=${unit_price:.2f} × {qty} ≠ Total=${total_price:.2f}"
                                )

                    if issues_found and self.debug:
                        print("⚠️  LINE_ITEM_PRICES validation issues:")
                        for issue in issues_found:
                            print(f"   {issue}")

            except (ValueError, AttributeError) as e:
                if self.debug:
                    print(f"⚠️  Could not validate line item pricing: {e}")

        return data

    def _validate_gst_consistency(self, data: Dict[str, str]) -> Dict[str, str]:
        """Validate GST calculations are internally consistent using pattern matching."""
        gst_amount = data.get("GST_AMOUNT", "NOT_FOUND")
        subtotal = data.get("SUBTOTAL_AMOUNT", "NOT_FOUND")
        total = data.get("TOTAL_AMOUNT", "NOT_FOUND")
        is_gst_included = data.get("IS_GST_INCLUDED", "NOT_FOUND")

        # Use pattern matching for GST validation logic
        match (gst_amount, subtotal, total, is_gst_included):
            case (gst, sub, tot, inc) if all(
                field != "NOT_FOUND" for field in [gst, sub, tot]
            ):
                try:
                    gst_val = self._parse_monetary_value(gst)
                    subtotal_val = self._parse_monetary_value(sub)
                    total_val = self._parse_monetary_value(tot)

                    if all(
                        val is not None for val in [gst_val, subtotal_val, total_val]
                    ):
                        match inc.lower():
                            case "true":
                                # GST included: total = subtotal, gst = total/11 (for 10% GST)
                                expected_gst = total_val / 11
                                if abs(gst_val - expected_gst) > 0.02 and self.debug:
                                    print(
                                        f"⚠️  GST included calculation: expected ${expected_gst:.2f}, got ${gst_val:.2f}"
                                    )
                            case "false":
                                # GST excluded: total = subtotal + gst
                                expected_total = subtotal_val + gst_val
                                if (
                                    abs(total_val - expected_total) > 0.02
                                    and self.debug
                                ):
                                    print(
                                        f"⚠️  GST excluded calculation: ${subtotal_val:.2f} + ${gst_val:.2f} ≠ ${total_val:.2f}"
                                    )
                            case _:
                                if self.debug:
                                    print(f"⚠️  Unknown GST inclusion status: '{inc}'")

                except (ValueError, AttributeError) as e:
                    if self.debug:
                        print(f"⚠️  Could not validate GST consistency: {e}")
            case _:
                # Insufficient data for GST validation
                pass

        return data

    def _validate_transaction_counts(self, data: Dict[str, str]) -> Dict[str, str]:
        """
        Validate that bank statement transaction field counts match.

        Issue #6: Detects when transaction descriptions, dates, and amounts
        have mismatched counts, which indicates merged transactions.

        Args:
            data: Cleaned field dictionary

        Returns:
            Same dictionary (validation only, no modifications)
        """
        # Get transaction fields
        descriptions = data.get("LINE_ITEM_DESCRIPTIONS", "NOT_FOUND")
        dates = data.get("TRANSACTION_DATES", "NOT_FOUND")
        amounts_paid = data.get("TRANSACTION_AMOUNTS_PAID", "NOT_FOUND")

        # Count items in each field (split by pipe)
        counts = {}
        for field_name, value in [
            ("LINE_ITEM_DESCRIPTIONS", descriptions),
            ("TRANSACTION_DATES", dates),
            ("TRANSACTION_AMOUNTS_PAID", amounts_paid),
        ]:
            if value != "NOT_FOUND":
                counts[field_name] = len(
                    [x.strip() for x in value.split("|") if x.strip()]
                )

        # Check if all counts match
        if counts and len(set(counts.values())) > 1:
            if self.debug:
                print(
                    "⚠️  Issue #6: Bank statement transaction count mismatch detected:"
                )
                for field_name, count in counts.items():
                    print(f"   {field_name}: {count} items")
                print("   → This may indicate merged transactions in extraction")

        return data

    def _parse_monetary_value(self, value: str) -> float | None:
        """Parse monetary string to float value."""
        if not value or value == "NOT_FOUND":
            return None
        try:
            # Remove currency symbols, commas, and whitespace
            cleaned = re.sub(r"[^\d.-]", "", value.replace(",", ""))
            return float(cleaned) if cleaned else None
        except ValueError:
            return None

    def get_cleaning_stats(
        self, original_dict: Dict[str, Any], cleaned_dict: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Generate statistics about cleaning operations.

        Args:
            original_dict (Dict[str, Any]): Original extracted values
            cleaned_dict (Dict[str, str]): Cleaned extracted values

        Returns:
            Dict[str, Any]: Cleaning statistics
        """
        stats = {
            "total_fields": len(original_dict),
            "fields_cleaned": 0,
            "fields_unchanged": 0,
            "fields_normalized": [],
            "cleaning_operations": [],
        }

        for field_name in original_dict:
            original = str(original_dict[field_name]).strip()
            cleaned = cleaned_dict.get(field_name, "NOT_FOUND")

            if original != cleaned:
                stats["fields_cleaned"] += 1
                stats["fields_normalized"].append(field_name)
                stats["cleaning_operations"].append(
                    {
                        "field": field_name,
                        "original": original,
                        "cleaned": cleaned,
                        "operation": self._identify_cleaning_operation(
                            field_name, original, cleaned
                        ),
                    }
                )
            else:
                stats["fields_unchanged"] += 1

        return stats

    def _identify_cleaning_operation(
        self, field_name: str, original: str, cleaned: str
    ) -> str:
        """Identify what cleaning operation was performed using pattern matching."""
        match (original, cleaned):
            case (orig, clean) if orig == clean:
                return "none"
            case (orig, clean) if (
                "each" in orig.lower() and "each" not in clean.lower()
            ):
                return "price_suffix_removal"
            case (orig, clean) if len(orig.split()) != len(clean.split()):
                return "whitespace_normalization"
            case (orig, clean) if "$" in orig and "$" in clean and orig != clean:
                return "currency_formatting"
            case (orig, clean) if "," in orig and "|" in clean:
                return "list_formatting"
            case (orig, clean) if "P:" in orig and "P:" not in clean:
                return "address_contact_removal"
            case _:
                return "general_cleaning"


# Convenience functions for direct use
def clean_field_value(field_name: str, value: Any, debug: bool = False) -> str:
    """Convenience function to clean a single field value."""
    cleaner = ExtractionCleaner(debug=debug)
    return cleaner.clean_field_value(field_name, value)


def clean_extraction_dict(
    field_dict: Dict[str, Any], debug: bool = False
) -> Dict[str, str]:
    """Convenience function to clean all fields in a dictionary."""
    cleaner = ExtractionCleaner(debug=debug)
    return cleaner.clean_extraction_dict(field_dict)
