"""
Field validation and interdependency checking for extracted document data.

This module implements business rule validation for extracted fields,
including mathematical relationships and data consistency checks.
"""

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional


@dataclass
class ValidationResult:
    """Result of field validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    corrected_values: Dict[str, str] = None

class FieldValidator:
    """Validates extracted field data according to business rules."""
    
    def __init__(self):
        self.validation_rules = self._load_validation_rules()
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules from schema."""
        from .schema_loader import get_global_schema
        schema = get_global_schema()
        return schema.schema.get("validation_rules", {})
    
    def validate_all_fields(self, extracted_data: Dict[str, str]) -> ValidationResult:
        """
        Validate all extracted fields including interdependencies.
        
        Args:
            extracted_data: Dictionary of field_name -> extracted_value
            
        Returns:
            ValidationResult with validation status and any errors/warnings
        """
        errors = []
        warnings = []
        corrected_values = {}
        
        # 1. Individual field validation
        field_errors = self._validate_individual_fields(extracted_data)
        errors.extend(field_errors)
        
        # 2. Field interdependency validation
        interdep_errors, interdep_warnings = self._validate_interdependencies(extracted_data)
        errors.extend(interdep_errors)
        warnings.extend(interdep_warnings)
        
        # 3. Business logic validation
        business_errors = self._validate_business_rules(extracted_data)
        errors.extend(business_errors)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            corrected_values=corrected_values if corrected_values else None
        )
    
    def _validate_individual_fields(self, extracted_data: Dict[str, str]) -> List[str]:
        """Validate individual field formats and types."""
        errors = []
        
        # ABN validation - must be exactly 11 digits
        abn = extracted_data.get('BUSINESS_ABN', '')
        if abn and abn != 'NOT_FOUND':
            # Remove spaces and check if 11 digits
            abn_digits = re.sub(r'\s+', '', abn)
            if not re.match(r'^\d{11}$', abn_digits):
                errors.append(f"BUSINESS_ABN must be 11 digits, got: {abn}")
        
        # Phone number validation - Australian format
        for phone_field in ['BUSINESS_PHONE', 'PAYER_PHONE']:
            phone = extracted_data.get(phone_field, '')
            if phone and phone != 'NOT_FOUND':
                # Should be 10 digits with area code
                phone_digits = re.sub(r'[\s\(\)\-]', '', phone)
                if not re.match(r'^\d{10}$', phone_digits):
                    errors.append(f"{phone_field} should be 10 digits, got: {phone}")
        
        # Date validation - basic format check
        for date_field in ['INVOICE_DATE', 'DUE_DATE']:
            date_val = extracted_data.get(date_field, '')
            if date_val and date_val != 'NOT_FOUND':
                # Check for common date formats
                if not re.match(r'\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4}', date_val):
                    errors.append(f"{date_field} invalid format: {date_val}")
        
        return errors
    
    def _validate_interdependencies(self, extracted_data: Dict[str, str]) -> tuple[List[str], List[str]]:
        """Validate field interdependencies and mathematical relationships."""
        errors = []
        warnings = []
        
        # Financial calculation validation: SUBTOTAL + GST = TOTAL
        subtotal_raw = extracted_data.get('SUBTOTAL_AMOUNT', 'NOT_FOUND')
        gst_raw = extracted_data.get('GST_AMOUNT', 'NOT_FOUND')
        total_raw = extracted_data.get('TOTAL_AMOUNT', 'NOT_FOUND')
        
        try:
            if all(val != 'NOT_FOUND' for val in [subtotal_raw, gst_raw, total_raw]):
                subtotal = self._parse_currency(subtotal_raw)
                gst = self._parse_currency(gst_raw)
                total = self._parse_currency(total_raw)
                
                if subtotal is not None and gst is not None and total is not None:
                    calculated_total = subtotal + gst
                    difference = abs(calculated_total - total)
                    
                    if difference > Decimal('0.02'):  # 2 cent tolerance
                        errors.append(
                            f"Financial inconsistency: SUBTOTAL ({subtotal}) + GST ({gst}) = "
                            f"{calculated_total}, but TOTAL is {total} (difference: ${difference})"
                        )
                    elif difference > Decimal('0.001'):  # 0.1 cent warning
                        warnings.append(
                            f"Minor financial discrepancy: ${difference} difference in total calculation"
                        )
        except Exception as e:
            warnings.append(f"Could not validate financial calculations: {e}")
        
        # Line item consistency - quantities vs descriptions vs prices
        descriptions = extracted_data.get('LINE_ITEM_DESCRIPTIONS', 'NOT_FOUND')
        quantities = extracted_data.get('LINE_ITEM_QUANTITIES', 'NOT_FOUND')  
        prices = extracted_data.get('LINE_ITEM_PRICES', 'NOT_FOUND')
        
        if all(val != 'NOT_FOUND' for val in [descriptions, quantities, prices]):
            try:
                desc_count = len(descriptions.split(',')) if descriptions else 0
                qty_count = len(quantities.split(',')) if quantities else 0  
                price_count = len(prices.split(',')) if prices else 0
                
                if desc_count != qty_count or desc_count != price_count:
                    errors.append(
                        f"Line item count mismatch: {desc_count} descriptions, "
                        f"{qty_count} quantities, {price_count} prices"
                    )
            except Exception as e:
                warnings.append(f"Could not validate line item consistency: {e}")
        
        # Date logic validation - due date should be after invoice date
        invoice_date = extracted_data.get('INVOICE_DATE', 'NOT_FOUND')
        due_date = extracted_data.get('DUE_DATE', 'NOT_FOUND')
        
        if all(val != 'NOT_FOUND' for val in [invoice_date, due_date]):
            try:
                # Simple validation - due date string should be >= invoice date string
                # (This is basic - could be enhanced with proper date parsing)
                if due_date < invoice_date:  # Basic string comparison
                    warnings.append(f"DUE_DATE ({due_date}) appears before INVOICE_DATE ({invoice_date})")
            except Exception as e:
                warnings.append(f"Could not validate date logic: {e}")
        
        return errors, warnings
    
    def _validate_business_rules(self, extracted_data: Dict[str, str]) -> List[str]:
        """Validate business-specific rules."""
        errors = []
        
        # Document type consistency
        doc_type = extracted_data.get('DOCUMENT_TYPE', 'NOT_FOUND')
        
        # For invoices, certain fields should be present
        if doc_type and 'INVOICE' in doc_type.upper():
            required_invoice_fields = ['SUPPLIER_NAME', 'TOTAL_AMOUNT', 'INVOICE_DATE']
            missing_fields = []
            
            for field in required_invoice_fields:
                if extracted_data.get(field, 'NOT_FOUND') == 'NOT_FOUND':
                    missing_fields.append(field)
            
            if missing_fields:
                errors.append(f"Invoice missing required fields: {missing_fields}")
        
        # GST percentage check (Australian standard is 10%)
        subtotal_raw = extracted_data.get('SUBTOTAL_AMOUNT', 'NOT_FOUND')
        gst_raw = extracted_data.get('GST_AMOUNT', 'NOT_FOUND')
        
        if all(val != 'NOT_FOUND' for val in [subtotal_raw, gst_raw]):
            try:
                subtotal = self._parse_currency(subtotal_raw)
                gst = self._parse_currency(gst_raw)
                
                if subtotal and gst and subtotal > 0:
                    gst_percentage = (gst / subtotal) * 100
                    # Australian GST should be 10% (allow 9.5%-10.5% range)
                    if not (9.5 <= gst_percentage <= 10.5):
                        errors.append(
                            f"GST percentage unusual: {gst_percentage:.1f}% "
                            f"(expected ~10% for Australian documents)"
                        )
            except Exception:
                pass  # Already caught in interdependency validation
        
        return errors
    
    def _parse_currency(self, currency_str: str) -> Optional[Decimal]:
        """Parse currency string to Decimal."""
        if not currency_str or currency_str == 'NOT_FOUND':
            return None
            
        try:
            # Remove currency symbols and whitespace
            cleaned = re.sub(r'[$,\s]', '', currency_str)
            return Decimal(cleaned)
        except (InvalidOperation, ValueError):
            return None

# Convenience function for quick validation
def validate_extracted_fields(extracted_data: Dict[str, str]) -> ValidationResult:
    """
    Quick validation function for extracted field data.
    
    Args:
        extracted_data: Dictionary of field_name -> extracted_value
        
    Returns:
        ValidationResult with validation status and details
    """
    validator = FieldValidator()
    return validator.validate_all_fields(extracted_data)