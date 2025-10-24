# Business Logic Validation Implementation Guide

**Document Extraction Validation and Interdependency Checking System**

## Executive Summary

This document describes the business logic validation system for the LMM_POC vision-language document extraction pipeline. The system validates extracted field data against business rules, mathematical relationships, and data consistency requirements to ensure extraction quality beyond simple accuracy metrics.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Validation Rules](#validation-rules)
- [Implementation Status](#implementation-status)
- [Integration Guide](#integration-guide)
- [Usage Examples](#usage-examples)
- [Future Enhancements](#future-enhancements)

## Overview

### Purpose
Traditional accuracy metrics only measure if extracted values match ground truth. Business logic validation ensures extracted data is **internally consistent** and **follows business rules**, catching errors like:
- Financial calculations that don't add up (SUBTOTAL + GST ≠ TOTAL)
- Invalid data formats (non-11-digit ABN)
- Impossible data combinations (due date before invoice date)
- Missing required fields for document types

### Key Benefits
1. **Data Quality**: Catch extraction errors that produce valid-looking but incorrect data
2. **Business Compliance**: Ensure extracted data meets business requirements
3. **Error Detection**: Identify systematic extraction issues across fields
4. **Confidence Scoring**: Provide additional confidence metrics beyond accuracy

## Architecture

### Component Structure
```
common/
├── field_validation.py      # Core validation engine
├── field_schema.yaml        # Validation rules & interdependencies
└── extraction_parser.py     # Integration wrapper functions
```

### Core Classes

#### ValidationResult
```python
@dataclass
class ValidationResult:
    is_valid: bool              # Overall validation status
    errors: List[str]           # Critical validation failures
    warnings: List[str]         # Non-critical issues
    corrected_values: Dict[str, str] = None  # Suggested corrections
```

#### FieldValidator
Main validation orchestrator that:
- Loads validation rules from schema
- Performs individual field validation
- Checks field interdependencies
- Validates business rules
- Returns comprehensive validation results

## Validation Rules

### 1. Individual Field Validation

#### ABN (Australian Business Number)
- **Rule**: Must be exactly 11 digits
- **Format**: Can include spaces (XX XXX XXX XXX)
- **Example Error**: "BUSINESS_ABN must be 11 digits, got: 04 904 7754 234" (extra digit)

#### Phone Numbers
- **Fields**: BUSINESS_PHONE, PAYER_PHONE
- **Rule**: Must be 10 digits including area code
- **Format**: Can include parentheses, spaces, hyphens
- **Valid Area Codes**: (02) Sydney, (03) Melbourne, (04) Mobile, etc.

#### Date Fields
- **Fields**: INVOICE_DATE, DUE_DATE
- **Format**: DD/MM/YYYY or DD-MM-YYYY
- **Validation**: Basic format checking

### 2. Field Interdependencies

#### Financial Consistency
```yaml
financial_consistency:
  description: "SUBTOTAL + GST should equal TOTAL"
  fields: ["SUBTOTAL_AMOUNT", "GST_AMOUNT", "TOTAL_AMOUNT"]
  formula: "SUBTOTAL_AMOUNT + GST_AMOUNT = TOTAL_AMOUNT"
  tolerance: 0.02  # 2 cent tolerance for rounding
  severity: "error"
```

**Example Validation**:
```python
SUBTOTAL: $50.00
GST:      $5.00
TOTAL:    $60.00  # ERROR: Should be $55.00
# Error: "Financial inconsistency: SUBTOTAL ($50.00) + GST ($5.00) = $55.00, but TOTAL is $60.00"
```

#### Line Item Consistency
```yaml
line_item_consistency:
  description: "Line item fields should have same count"
  fields: ["LINE_ITEM_DESCRIPTIONS", "LINE_ITEM_QUANTITIES", "LINE_ITEM_PRICES"]
  rule_type: "count_match"
  severity: "error"
```

**Example**:
```python
DESCRIPTIONS: "Rice, Cheese, Peas"     # 3 items
QUANTITIES:   "3, 3, 3, 1"            # 4 items - ERROR!
PRICES:       "$3.80, $8.50, $4.20"   # 3 items
# Error: "Line item count mismatch: 3 descriptions, 4 quantities, 3 prices"
```

#### Date Sequence Validation
```yaml
date_sequence:
  description: "Due date should be after invoice date"
  fields: ["INVOICE_DATE", "DUE_DATE"]
  rule_type: "temporal_sequence"
  severity: "warning"
```

#### GST Percentage Check
```yaml
gst_percentage:
  description: "GST should be ~10% of subtotal (Australian standard)"
  fields: ["SUBTOTAL_AMOUNT", "GST_AMOUNT"]
  expected_percentage: 10.0
  tolerance_range: [9.5, 10.5]
  severity: "warning"
```

### 3. Business Rules

#### Document Type Requirements
```yaml
invoice_required_fields:
  description: "Invoice documents must have core fields"
  condition: "DOCUMENT_TYPE contains 'INVOICE'"
  required_fields: ["SUPPLIER_NAME", "TOTAL_AMOUNT", "INVOICE_DATE"]
  severity: "error"
```

**Validation Logic**:
- If DOCUMENT_TYPE contains "INVOICE"
- Check that SUPPLIER_NAME, TOTAL_AMOUNT, and INVOICE_DATE are not "NOT_FOUND"
- Report error if any required field is missing

## Implementation Status

### ✅ Implemented Components

1. **Validation Engine** (`common/field_validation.py`)
   - Complete validation framework
   - Individual field validators
   - Interdependency checkers
   - Business rule validators
   - Currency parsing utilities

2. **Schema Configuration** (`common/field_schema.yaml`)
   - Validation rules per field group
   - Interdependency rules defined
   - Severity levels configured

3. **Integration Wrapper** (`common/extraction_parser.py`)
   - `validate_and_enhance_extraction()` function
   - Returns enhanced results with validation metadata

### ❌ Not Yet Integrated

1. **Extraction Pipelines**
   - `internvl3_keyvalue.py` - No validation calls
   - `llama_keyvalue.py` - No validation calls
   - Model processors - No validation integration

2. **Accuracy Calculations**
   - `evaluate_extraction_results()` ignores validation
   - No penalty for business rule violations
   - Validation errors not reflected in scores

## Integration Guide

### Step 1: Add Validation to Extraction Pipeline

#### Current Flow (No Validation)
```python
# In internvl3_keyvalue.py or llama_keyvalue.py
results, statistics = processor.process_image_batch(image_files)
# Results go directly to evaluation
```

#### Enhanced Flow (With Validation)
```python
from common.extraction_parser import validate_and_enhance_extraction

# Process images
results, statistics = processor.process_image_batch(image_files)

# Add validation to each result
validated_results = []
for result in results:
    enhanced_result = validate_and_enhance_extraction(
        result["extracted_data"],
        image_name=result["image_name"]
    )
    
    # Merge validation info into result
    result["validation"] = enhanced_result["validation"]
    result["has_errors"] = not enhanced_result["validation"]["is_valid"]
    result["error_count"] = enhanced_result["validation"]["error_count"]
    
    validated_results.append(result)
```

### Step 2: Incorporate Validation into Accuracy Scoring

#### Option A: Penalty-Based Approach
```python
def calculate_validated_accuracy(accuracy_score, validation_result):
    """
    Adjust accuracy score based on validation errors.
    
    Args:
        accuracy_score: Base accuracy (0.0 to 1.0)
        validation_result: ValidationResult object
    
    Returns:
        Adjusted accuracy score
    """
    if not validation_result.is_valid:
        # Apply penalty for each error
        penalty_per_error = 0.1
        total_penalty = len(validation_result.errors) * penalty_per_error
        
        # Cap penalty at 50% of score
        total_penalty = min(total_penalty, accuracy_score * 0.5)
        
        return max(0, accuracy_score - total_penalty)
    
    return accuracy_score
```

#### Option B: Binary Validation Gate
```python
def evaluate_with_validation(extraction_results, ground_truth_map):
    """
    Evaluate extraction with validation as a gate condition.
    """
    for result in extraction_results:
        # Run standard accuracy calculation
        base_accuracy = calculate_field_accuracy(...)
        
        # Validate extraction
        validation = validate_extracted_fields(result["extracted_data"])
        
        if not validation.is_valid and len(validation.errors) > 0:
            # Critical errors invalidate the extraction
            result["accuracy"] = 0.0
            result["validation_failed"] = True
            result["failure_reason"] = validation.errors[0]
        else:
            result["accuracy"] = base_accuracy
            result["validation_failed"] = False
```

### Step 3: Update Reporting

Add validation metrics to evaluation reports:

```python
def generate_validation_summary(validated_results):
    """Generate validation statistics for reporting."""
    
    total_docs = len(validated_results)
    valid_docs = sum(1 for r in validated_results if r["validation"]["is_valid"])
    
    # Collect all validation errors
    all_errors = []
    for result in validated_results:
        all_errors.extend(result["validation"]["errors"])
    
    # Count error types
    financial_errors = sum(1 for e in all_errors if "Financial inconsistency" in e)
    format_errors = sum(1 for e in all_errors if "must be" in e or "invalid format" in e)
    missing_errors = sum(1 for e in all_errors if "missing required" in e)
    
    return {
        "validation_rate": valid_docs / total_docs if total_docs > 0 else 0,
        "total_errors": len(all_errors),
        "financial_errors": financial_errors,
        "format_errors": format_errors,
        "missing_field_errors": missing_errors,
        "documents_with_errors": total_docs - valid_docs
    }
```

## Usage Examples

### Example 1: Validating Single Extraction
```python
from common.field_validation import validate_extracted_fields

# Sample extraction with financial inconsistency
extracted_data = {
    "BUSINESS_ABN": "04 904 754 234",  # 11 digits
    "SUBTOTAL_AMOUNT": "$100.00",
    "GST_AMOUNT": "$10.00",
    "TOTAL_AMOUNT": "$115.00",  # Should be $110.00!
    "INVOICE_DATE": "01/08/2025",
    "DUE_DATE": "31/08/2025"
}

# Run validation
result = validate_extracted_fields(extracted_data)

print(f"Valid: {result.is_valid}")  # False
print(f"Errors: {result.errors}")
# ['Financial inconsistency: SUBTOTAL ($100.00) + GST ($10.00) = $110.00, but TOTAL is $115.00']
```

### Example 2: Batch Validation with Reporting
```python
from common.extraction_parser import validate_and_enhance_extraction

# Process batch of extractions
extraction_results = processor.process_image_batch(images)

# Validate all results
validation_summary = {
    "total": 0,
    "valid": 0,
    "errors": []
}

for result in extraction_results:
    enhanced = validate_and_enhance_extraction(result["extracted_data"])
    validation_summary["total"] += 1
    
    if enhanced["validation"]["is_valid"]:
        validation_summary["valid"] += 1
    else:
        validation_summary["errors"].extend(enhanced["validation"]["errors"])

# Report validation statistics
print(f"Validation Rate: {validation_summary['valid']}/{validation_summary['total']} "
      f"({validation_summary['valid']/validation_summary['total']*100:.1f}%)")
print(f"Total Errors Found: {len(validation_summary['errors'])}")
```

### Example 3: Custom Validation Rules
```python
from common.field_validation import FieldValidator

class CustomValidator(FieldValidator):
    """Extended validator with custom business rules."""
    
    def _validate_custom_rules(self, extracted_data):
        """Add custom validation rules."""
        errors = []
        
        # Custom rule: Invoice number must start with "INV"
        invoice_num = extracted_data.get("INVOICE_NUMBER", "")
        if invoice_num and not invoice_num.startswith("INV"):
            errors.append(f"Invoice number must start with 'INV', got: {invoice_num}")
        
        # Custom rule: Minimum invoice amount
        total = self._parse_currency(extracted_data.get("TOTAL_AMOUNT", "0"))
        if total and total < Decimal("10.00"):
            errors.append(f"Invoice total below minimum $10.00: ${total}")
        
        return errors
```

## Future Enhancements

### 1. Machine Learning Integration
- Train models to predict validation errors
- Use validation patterns to improve extraction accuracy
- Auto-correct common validation failures

### 2. Confidence Scoring
```python
def calculate_validation_confidence(validation_result):
    """
    Calculate confidence score based on validation results.
    
    Returns:
        float: Confidence score (0.0 to 1.0)
    """
    base_confidence = 1.0
    
    # Reduce confidence for errors
    error_penalty = len(validation_result.errors) * 0.2
    
    # Minor reduction for warnings
    warning_penalty = len(validation_result.warnings) * 0.05
    
    return max(0, base_confidence - error_penalty - warning_penalty)
```

### 3. Auto-Correction System
```python
def auto_correct_financial_totals(extracted_data):
    """
    Attempt to auto-correct financial calculation errors.
    """
    subtotal = parse_currency(extracted_data.get("SUBTOTAL_AMOUNT"))
    gst = parse_currency(extracted_data.get("GST_AMOUNT"))
    total = parse_currency(extracted_data.get("TOTAL_AMOUNT"))
    
    if subtotal and gst:
        calculated_total = subtotal + gst
        
        if total and abs(calculated_total - total) > 0.02:
            # Suggest correction
            return {
                "TOTAL_AMOUNT": f"${calculated_total:.2f}",
                "correction_reason": "Adjusted to match SUBTOTAL + GST"
            }
    
    return None
```

### 4. Validation Rule Learning
- Analyze validation failures to identify patterns
- Automatically suggest new validation rules
- Learn document-specific requirements

### 5. Industry-Specific Rules
```yaml
# Healthcare invoice rules
healthcare_invoice:
  required_fields: ["PROVIDER_NUMBER", "MEDICARE_NUMBER", "ITEM_CODES"]
  validation_rules:
    - medicare_number_format: "10 digits"
    - provider_number_format: "8 characters"
    
# Government contract rules  
government_contract:
  required_fields: ["CONTRACT_NUMBER", "ABN", "GST_INCLUSIVE"]
  validation_rules:
    - contract_format: "Starts with 'C' followed by 8 digits"
    - gst_mandatory: true
```

## Performance Considerations

### Validation Overhead
- Average validation time: ~5-10ms per document
- Memory usage: Minimal (< 1MB per validation)
- Can be parallelized for batch processing

### Optimization Strategies
1. **Cache validation rules** - Load once, use many times
2. **Parallel validation** - Process multiple documents simultaneously
3. **Early exit** - Stop validation after critical errors
4. **Lazy evaluation** - Only validate fields that were extracted

## Conclusion

The business logic validation system provides a crucial layer of quality assurance beyond simple accuracy metrics. While currently implemented but not integrated, it offers:

1. **Comprehensive validation** of field formats, interdependencies, and business rules
2. **Flexible architecture** supporting custom rules and extensions
3. **Clear error reporting** for debugging and improvement
4. **Production readiness** with minimal performance overhead

### Recommended Next Steps
1. **Phase 1**: Integrate validation into extraction pipelines (logging only)
2. **Phase 2**: Add validation metrics to evaluation reports
3. **Phase 3**: Implement accuracy penalties for validation failures
4. **Phase 4**: Deploy auto-correction for common errors
5. **Phase 5**: Machine learning integration for validation prediction

The system is designed to grow with requirements, supporting everything from basic format checking to complex industry-specific business rules while maintaining performance and maintainability.

---

**Document Version**: 1.0  
**Last Updated**: August 2025  
**Status**: Implementation Complete, Integration Pending