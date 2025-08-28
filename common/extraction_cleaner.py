#!/usr/bin/env python3
"""
Extraction Value Cleaner - Centralized field value cleaning for document processors.

This module provides standardized cleaning and normalization of extracted field values
for both Llama and InternVL3 document-aware processors, ensuring consistent output
formatting and improved accuracy against ground truth data.
"""

import re
from typing import Any, Dict


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
        self.monetary_patterns = ["AMOUNT", "PRICE", "COST", "FEE", "TOTAL", "SUBTOTAL", "GST"]
        self.list_patterns = ["LINE_ITEM", "ITEMS", "DESCRIPTIONS", "QUANTITIES", "PRICES"]
        self.date_patterns = ["DATE", "TIME", "PERIOD", "RANGE"]
        self.id_patterns = ["NUMBER", "ID", "ABN", "BSB", "ACCOUNT"]
        
        # Common cleaning patterns
        self.price_suffix_patterns = [
            r'\s+(each|per\s+item|per\s+unit|per\s+piece)',
            r'\s+(ea\.?|@|apiece)',
        ]
        
        # Currency symbol standardization
        self.currency_patterns = {
            r'AUD?\s*': '$',
            r'USD?\s*': '$', 
            r'\$\$+': '$',  # Multiple dollar signs
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
        missing_values = ["NOT_FOUND", "Not Found", "not found", "None", "null", "", "N/A", "n/a"]
        if not raw_value or raw_value in missing_values:
            return "NOT_FOUND"
            
        # Convert to string and basic cleaning
        value = str(raw_value).strip()
        if not value:
            return "NOT_FOUND"
            
        if self.debug:
            print(f"🧹 Cleaning {field_name}: '{raw_value}' -> ", end="")
        
        # Pre-process lists: convert comma-separated to pipe-separated format
        if self._is_list_field(field_name) and ',' in value and '|' not in value:
            value = ' | '.join(item.strip() for item in value.split(','))
        
        # Route to appropriate cleaning method based on field type
        cleaned_value = value
        
        if self._is_monetary_field(field_name):
            cleaned_value = self._clean_monetary_field(value)
        elif self._is_list_field(field_name):
            cleaned_value = self._clean_list_field(field_name, value)
        elif self._is_date_field(field_name):
            cleaned_value = self._clean_date_field(value)
        elif self._is_id_field(field_name):
            cleaned_value = self._clean_id_field(value)
        else:
            cleaned_value = self._clean_text_field(value)
            
        if self.debug:
            print(f"'{cleaned_value}'")
            
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
    
    def _clean_monetary_field(self, value: str) -> str:
        """
        Clean monetary values (AMOUNT, PRICE fields).
        
        Handles:
        - Remove "each", "per item", etc. suffixes
        - Standardize currency symbols
        - Clean up spacing and formatting
        """
        if not value or value == "NOT_FOUND":
            return "NOT_FOUND"
            
        cleaned = value
        
        # Remove common price suffixes
        for pattern in self.price_suffix_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Standardize currency symbols
        for pattern, replacement in self.currency_patterns.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        
        # Clean up multiple spaces and trim
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Ensure dollar sign is at the beginning if present
        if '$' in cleaned and not cleaned.startswith('$'):
            cleaned = re.sub(r'(.*?)\$(.+)', r'$\2', cleaned)
        
        return cleaned if cleaned else "NOT_FOUND"
    
    def _clean_list_field(self, field_name: str, value: str) -> str:
        """
        Clean list fields (LINE_ITEM_*) and convert to pipe-separated format.
        
        Handles:
        - Split by comma or pipe, clean each item
        - Remove price suffixes from price lists
        - Convert to standard pipe-separated format
        - Normalize whitespace and separators
        """
        if not value or value == "NOT_FOUND":
            return "NOT_FOUND"
        
        # Split by comma or pipe and clean each item
        if '|' in value:
            items = [item.strip() for item in value.split('|')]
        else:
            items = [item.strip() for item in value.split(',')]
        
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
            
            if cleaned_item and cleaned_item != "NOT_FOUND":
                cleaned_items.append(cleaned_item)
        
        # Always return pipe-separated format for consistency
        return ' | '.join(cleaned_items) if cleaned_items else "NOT_FOUND"
    
    def _clean_date_field(self, value: str) -> str:
        """
        Standardize date formats.
        
        Handles:
        - Normalize common date separators
        - Trim whitespace
        - Standardize format consistency
        """
        if not value or value == "NOT_FOUND":
            return "NOT_FOUND"
        
        cleaned = value.strip()
        
        # Normalize date separators (keep original format but clean spacing)
        cleaned = re.sub(r'\s*[-/\.]\s*', lambda m: m.group(0).strip(), cleaned)
        
        # Clean up extra spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned if cleaned else "NOT_FOUND"
    
    def _clean_id_field(self, value: str) -> str:
        """
        Clean ID fields (ABN, account numbers, etc.).
        
        Handles:
        - Normalize spacing in formatted numbers
        - Clean up extra characters
        - Preserve number grouping where appropriate
        """
        if not value or value == "NOT_FOUND":
            return "NOT_FOUND"
        
        cleaned = value.strip()
        
        # For ABN and similar formatted numbers, normalize spacing
        if re.match(r'^\d{2}\s+\d{3}\s+\d{3}\s+\d{3}$', cleaned):
            # ABN format: XX XXX XXX XXX
            parts = cleaned.split()
            cleaned = ' '.join(parts)
        elif re.match(r'^\d{3}-\d{3}$', cleaned):
            # BSB format: XXX-XXX
            cleaned = cleaned.replace(' ', '')
        else:
            # General ID cleaning - remove extra spaces but preserve structure
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned if cleaned else "NOT_FOUND"
    
    def _clean_text_field(self, value: str) -> str:
        """
        Clean general text fields.
        
        Handles:
        - Normalize whitespace
        - Trim leading/trailing spaces
        - Remove redundant punctuation
        """
        if not value or value == "NOT_FOUND":
            return "NOT_FOUND"
        
        cleaned = value.strip()
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove trailing punctuation that might be artifacts
        cleaned = re.sub(r'\s*[,;]\s*$', '', cleaned)
        
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
        Apply business knowledge and cross-field validation.
        
        Key business rules:
        - LINE_ITEM_PRICES = unit prices (price per individual item)
        - LINE_ITEM_TOTAL_PRICES = quantity × unit price
        - GST calculations must be consistent
        """
        validated_dict = cleaned_dict.copy()
        
        # Business Rule: LINE_ITEM_PRICES validation
        if self._has_line_items(cleaned_dict):
            validated_dict = self._validate_line_item_pricing(validated_dict)
        
        # Business Rule: GST consistency validation  
        if self._has_gst_fields(cleaned_dict):
            validated_dict = self._validate_gst_consistency(validated_dict)
            
        return validated_dict
    
    def _has_line_items(self, data: Dict[str, str]) -> bool:
        """Check if document has line item data."""
        required_fields = ["LINE_ITEM_DESCRIPTIONS", "LINE_ITEM_QUANTITIES", "LINE_ITEM_PRICES"]
        return any(data.get(field, "NOT_FOUND") != "NOT_FOUND" for field in required_fields)
    
    def _has_gst_fields(self, data: Dict[str, str]) -> bool:
        """Check if document has GST-related fields."""
        gst_fields = ["GST_AMOUNT", "IS_GST_INCLUDED", "LINE_ITEM_GST_AMOUNTS"]
        return any(data.get(field, "NOT_FOUND") != "NOT_FOUND" for field in gst_fields)
    
    def _validate_line_item_pricing(self, data: Dict[str, str]) -> Dict[str, str]:
        """
        Validate LINE_ITEM_PRICES represents unit prices, not totals.
        
        Business Knowledge:
        - LINE_ITEM_PRICES = price per unit (unit price)
        - LINE_ITEM_TOTAL_PRICES = quantity × unit price (line total)
        """
        prices = data.get("LINE_ITEM_PRICES", "NOT_FOUND")
        total_prices = data.get("LINE_ITEM_TOTAL_PRICES", "NOT_FOUND")
        quantities = data.get("LINE_ITEM_QUANTITIES", "NOT_FOUND")
        
        if all(field != "NOT_FOUND" for field in [prices, total_prices, quantities]):
            try:
                price_list = [self._parse_monetary_value(p.strip()) for p in prices.split('|')]
                total_list = [self._parse_monetary_value(p.strip()) for p in total_prices.split('|')]
                qty_list = [float(q.strip()) for q in quantities.split('|')]
                
                if len(price_list) == len(total_list) == len(qty_list):
                    issues_found = []
                    
                    for i, (unit_price, total_price, qty) in enumerate(zip(price_list, total_list, qty_list)):
                        if unit_price and total_price and qty:
                            expected_total = unit_price * qty
                            # Allow 1% tolerance for rounding
                            if abs(total_price - expected_total) > (expected_total * 0.01):
                                issues_found.append(f"Item {i+1}: Unit=${unit_price:.2f} × {qty} ≠ Total=${total_price:.2f}")
                    
                    if issues_found and self.debug:
                        print(f"⚠️  LINE_ITEM_PRICES validation issues:")
                        for issue in issues_found:
                            print(f"   {issue}")
                            
            except (ValueError, AttributeError) as e:
                if self.debug:
                    print(f"⚠️  Could not validate line item pricing: {e}")
        
        return data
    
    def _validate_gst_consistency(self, data: Dict[str, str]) -> Dict[str, str]:
        """Validate GST calculations are internally consistent."""
        gst_amount = data.get("GST_AMOUNT", "NOT_FOUND")
        subtotal = data.get("SUBTOTAL_AMOUNT", "NOT_FOUND") 
        total = data.get("TOTAL_AMOUNT", "NOT_FOUND")
        is_gst_included = data.get("IS_GST_INCLUDED", "NOT_FOUND")
        
        if all(field != "NOT_FOUND" for field in [gst_amount, subtotal, total]):
            try:
                gst_val = self._parse_monetary_value(gst_amount)
                subtotal_val = self._parse_monetary_value(subtotal)
                total_val = self._parse_monetary_value(total)
                
                if gst_val and subtotal_val and total_val:
                    if is_gst_included.lower() == "true":
                        # GST included: total = subtotal, gst = total/11 (for 10% GST)
                        expected_gst = total_val / 11
                    else:
                        # GST excluded: total = subtotal + gst
                        expected_total = subtotal_val + gst_val
                        if abs(total_val - expected_total) > 0.02 and self.debug:
                            print(f"⚠️  GST calculation: ${subtotal_val} + ${gst_val} ≠ ${total_val}")
                            
            except (ValueError, AttributeError) as e:
                if self.debug:
                    print(f"⚠️  Could not validate GST consistency: {e}")
        
        return data
    
    def _parse_monetary_value(self, value: str) -> float | None:
        """Parse monetary string to float value."""
        if not value or value == "NOT_FOUND":
            return None
        try:
            # Remove currency symbols, commas, and whitespace
            cleaned = re.sub(r'[^\d.-]', '', value.replace(',', ''))
            return float(cleaned) if cleaned else None
        except ValueError:
            return None
    
    def get_cleaning_stats(self, original_dict: Dict[str, Any], cleaned_dict: Dict[str, str]) -> Dict[str, Any]:
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
            "cleaning_operations": []
        }
        
        for field_name in original_dict:
            original = str(original_dict[field_name]).strip()
            cleaned = cleaned_dict.get(field_name, "NOT_FOUND")
            
            if original != cleaned:
                stats["fields_cleaned"] += 1
                stats["fields_normalized"].append(field_name)
                stats["cleaning_operations"].append({
                    "field": field_name,
                    "original": original,
                    "cleaned": cleaned,
                    "operation": self._identify_cleaning_operation(field_name, original, cleaned)
                })
            else:
                stats["fields_unchanged"] += 1
        
        return stats
    
    def _identify_cleaning_operation(self, field_name: str, original: str, cleaned: str) -> str:
        """Identify what cleaning operation was performed."""
        if original == cleaned:
            return "none"
        elif "each" in original.lower() and "each" not in cleaned.lower():
            return "price_suffix_removal"
        elif len(original.split()) != len(cleaned.split()):
            return "whitespace_normalization" 
        elif '$' in original and '$' in cleaned and original != cleaned:
            return "currency_formatting"
        elif ',' in original and ',' in cleaned:
            return "list_formatting"
        else:
            return "general_cleaning"


# Convenience functions for direct use
def clean_field_value(field_name: str, value: Any, debug: bool = False) -> str:
    """Convenience function to clean a single field value."""
    cleaner = ExtractionCleaner(debug=debug)
    return cleaner.clean_field_value(field_name, value)


def clean_extraction_dict(field_dict: Dict[str, Any], debug: bool = False) -> Dict[str, str]:
    """Convenience function to clean all fields in a dictionary."""
    cleaner = ExtractionCleaner(debug=debug)
    return cleaner.clean_extraction_dict(field_dict)