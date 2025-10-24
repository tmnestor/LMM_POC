# Prompt Customization Guide for Llama and InternVL3 Models

## Overview
This guide explains how to customize prompts for both Llama-3.2-Vision and InternVL3 models in the document-aware extraction system. All prompt customization is done through the unified YAML configuration file (`config/unified_schema.yaml`) following the YAML-first architecture.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Understanding the Unified Schema Structure](#understanding-the-unified-schema-structure)
3. [Changing Field Names and Numbers](#changing-field-names-and-numbers)
4. [Customizing Extraction Prompts](#customizing-extraction-prompts)
5. [Customizing Document Detection Prompts](#customizing-document-detection-prompts)
6. [Advanced Customization](#advanced-customization)
7. [Testing Your Changes](#testing-your-changes)
8. [Troubleshooting](#troubleshooting)

## Quick Start

All prompt customization happens in one file:
```
/Users/tod/Desktop/LMM_POC/config/unified_schema.yaml
```

**Key Principle**: The YAML-first architecture means you NEVER need to modify Python code to change prompts or fields. Everything is configured in the unified schema.

## Understanding the Unified Schema Structure

The unified schema has several key sections:

```yaml
# 1. Schema metadata
schema_version: "5.0-unified"
schema_type: "essential_field_reduction"

# 2. Field definitions
essential_fields:
  FIELD_NAME:
    order: 1
    instruction: "[extraction instruction]"
    description: "Field description"
    applicable_documents: [invoice, receipt]
    required: true
    category: "field_category"

# 3. Document type configurations
document_types:
  invoice:
    required_fields: [...]
    critical_fields: [...]
    accuracy_threshold: 0.95

# 4. Prompt templates
prompt_templates:
  expertise_frame: "..."
  critical_instructions: [...]
  format_rules: [...]

# 5. Document detection prompts
document_type_detection:
  prompts:
    llama: {...}
    internvl3: {...}
```

## Changing Field Names and Numbers

### Adding a New Field

1. **Add the field definition** to `essential_fields`:

```yaml
essential_fields:
  # Existing fields...
  
  YOUR_NEW_FIELD:
    order: 16  # Next sequential number
    instruction: "[extraction instruction or NOT_FOUND]"
    description: "What this field captures"
    applicable_documents: [invoice, receipt, bank_statement]
    required: true
    category: "appropriate_category"
```

2. **Update document type configurations** to include the new field:

```yaml
document_types:
  invoice:
    required_fields:
      # Existing fields...
      - YOUR_NEW_FIELD
    field_count: 13  # Increment the count
```

3. **Update field categories** if needed:

```yaml
field_categories:
  your_category:
    - YOUR_NEW_FIELD
```

### Removing a Field

1. **Comment out or delete** the field from `essential_fields`
2. **Remove from all document type** `required_fields` lists
3. **Update field counts** in document type configurations
4. **Remove from field categories**

### Renaming a Field

1. **Change the field key** in `essential_fields`:

```yaml
essential_fields:
  # OLD_FIELD_NAME:  # Comment out old name
  NEW_FIELD_NAME:    # Add new name
    order: 7
    instruction: "[updated instruction]"
    # Rest of configuration...
```

2. **Update all references** in:
   - `document_types` → `required_fields`
   - `document_types` → `critical_fields` 
   - `document_types` → `ato_compliance_fields`
   - `field_categories`

### Changing the Number of Fields

To change from 12 fields to 8 fields for invoices:

1. **Select which fields to keep** in `essential_fields`
2. **Update the document type configuration**:

```yaml
document_types:
  invoice:
    document_type: "invoice"
    accuracy_threshold: 0.95
    field_count: 8  # Changed from 12
    required_fields:
      - DOCUMENT_TYPE
      - BUSINESS_ABN
      - SUPPLIER_NAME
      - PAYER_NAME
      - INVOICE_DATE
      - GST_AMOUNT
      - TOTAL_AMOUNT
      - LINE_ITEM_DESCRIPTIONS
      # Removed 4 fields
```

## Customizing Extraction Prompts

### Main Extraction Prompt Template

The extraction prompt is built from templates in the `prompt_templates` section:

```yaml
prompt_templates:
  # The main expertise/role instruction
  expertise_frame: "Extract key information from this business document."
  
  # Critical instructions header
  critical_instructions_header: "IMPORTANT RULES:"
  
  # List of critical instructions
  critical_instructions:
    - "Extract only the requested fields"
    - "Use NOT_FOUND for missing information"
    - "Be precise and accurate"
    
  # Output format header
  output_format: "EXTRACT THESE {field_count} FIELDS:"
  
  # Format rules header
  format_rules_header: "FORMATTING REQUIREMENTS:"
  
  # Formatting rules
  format_rules:
    - "Use KEY: value format"
    - "No markdown or formatting"
    - "Include all fields even if NOT_FOUND"
```

### Field-Specific Instructions

Each field can have custom extraction instructions:

```yaml
essential_fields:
  TOTAL_AMOUNT:
    instruction: "[extract total including tax in format $X,XXX.XX or NOT_FOUND]"
    
  INVOICE_DATE:
    instruction: "[date in DD/MM/YYYY format or NOT_FOUND]"
    
  LINE_ITEM_DESCRIPTIONS:
    instruction: "[pipe-separated list of items (item1 | item2 | item3) or NOT_FOUND]"
```

### Model-Specific Prompt Variations

While the main templates are shared, you can create model-specific behavior through the document detection prompts:

```yaml
document_type_detection:
  prompts:
    llama:
      system_prompt: |
        You are a document classification expert.
        Analyze documents with high precision.
      user_prompt: |
        Classify this document into one of these types:
        invoice, receipt, bank_statement
        
    internvl3:
      user_prompt: |
        Document type? Choose: invoice/receipt/bank_statement
        Answer:
```

## Customizing Document Detection Prompts

### For Llama Models

```yaml
document_type_detection:
  prompts:
    llama:
      system_prompt: |
        You are an expert document classifier.
        Your task is to identify document types accurately.
        Focus on key identifying features.
      
      user_prompt: |
        Examine this document image carefully.
        
        Identify if it is:
        - invoice (includes tax invoice, bill, estimate, quote)
        - receipt (includes sales receipt, payment receipt)
        - bank_statement (includes credit card statement)
        
        Respond with only the document type.
      
      max_tokens: 50
      temperature: 0.0
```

### For InternVL3 Models

```yaml
document_type_detection:
  prompts:
    internvl3:
      user_prompt: |
        What type of business document is shown?
        
        Options:
        • invoice (bills, estimates, quotes)
        • receipt (proof of purchase)
        • bank_statement (account statements)
        
        Type:
      
      max_tokens: 20
      temperature: 0.0
```

## Advanced Customization

### Creating Document-Specific Field Sets

Different document types can have different required fields:

```yaml
document_types:
  invoice:
    required_fields:
      - DOCUMENT_TYPE
      - BUSINESS_ABN
      - SUPPLIER_NAME
      - TOTAL_AMOUNT
      # ... invoice-specific fields
    
  receipt:
    required_fields:
      - DOCUMENT_TYPE
      - SUPPLIER_NAME
      - TOTAL_AMOUNT
      - TRANSACTION_DATE
      # ... receipt-specific fields
    
  bank_statement:
    required_fields:
      - DOCUMENT_TYPE
      - STATEMENT_DATE_RANGE
      - TRANSACTION_DATES
      - TRANSACTION_AMOUNTS
      # ... bank statement-specific fields
```

### Adding New Document Types

1. **Define the document type**:

```yaml
document_types:
  purchase_order:
    document_type: "purchase_order"
    accuracy_threshold: 0.85
    field_count: 10
    required_fields:
      - DOCUMENT_TYPE
      - PO_NUMBER
      - VENDOR_NAME
      - ORDER_DATE
      - DELIVERY_DATE
      - ITEM_DESCRIPTIONS
      - QUANTITIES
      - UNIT_PRICES
      - SUBTOTAL
      - TOTAL_AMOUNT
    critical_fields:
      - PO_NUMBER
      - TOTAL_AMOUNT
    aliases:
      - "purchase_order"
      - "po"
      - "purchase order"
```

2. **Update supported document types**:

```yaml
supported_document_types:
  - invoice
  - receipt
  - bank_statement
  - purchase_order  # Add new type
```

3. **Add detection prompt for the new type**:

```yaml
document_type_detection:
  prompts:
    llama:
      user_prompt: |
        Identify document type:
        invoice, receipt, bank_statement, purchase_order
```

### Custom Validation Rules

Add field-specific validation in the field definition:

```yaml
essential_fields:
  BUSINESS_ABN:
    instruction: "[11-digit ABN number or NOT_FOUND]"
    validation_regex: "^\\d{2}\\s\\d{3}\\s\\d{3}\\s\\d{3}$"
    validation_message: "ABN must be 11 digits (XX XXX XXX XXX)"
    
  EMAIL:
    instruction: "[email address or NOT_FOUND]"
    validation_regex: "^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$"
    validation_message: "Must be valid email format"
```

## Testing Your Changes

### 1. Validate YAML Syntax

```bash
# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('config/unified_schema.yaml'))"
```

### 2. Test Schema Loading

```bash
python -c "
from common.unified_schema import DocumentTypeFieldSchema
schema = DocumentTypeFieldSchema()
print(f'Loaded {schema.total_fields} fields')
print(f'Invoice fields: {len(schema.get_document_fields(\"invoice\"))}')
"
```

### 3. Test with Debug Mode

```bash
# Test Llama model
python llama_document_aware.py --image-path test.png --debug

# Test InternVL3 model  
python internvl3_document_aware.py --image-path test.png --debug
```

### 4. Verify Prompt Generation

The debug output will show:
- Document type detection prompts
- Field extraction prompts
- Number of fields being extracted
- Actual prompt sent to the model

## Troubleshooting

### Common Issues and Solutions

#### Issue: Fields not appearing in extraction
**Solution**: Check that the field is:
1. Defined in `essential_fields`
2. Listed in the appropriate document type's `required_fields`
3. Has correct `applicable_documents` list

#### Issue: Wrong number of fields extracted
**Solution**: Update the `field_count` in document type configuration:
```yaml
document_types:
  invoice:
    field_count: 15  # Must match length of required_fields
```

#### Issue: Prompt too long for model
**Solution**: Reduce the number of fields or shorten instructions:
```yaml
essential_fields:
  FIELD_NAME:
    instruction: "[value or N/F]"  # Shortened from "NOT_FOUND"
```

#### Issue: Model not following format
**Solution**: Strengthen format rules:
```yaml
prompt_templates:
  format_rules:
    - "CRITICAL: Use exact format FIELD_NAME: value"
    - "CRITICAL: Output exactly {field_count} lines"
    - "CRITICAL: No additional text or explanation"
```

### Validation Checklist

Before deploying changes:
- [ ] YAML syntax is valid
- [ ] Field counts match actual field lists
- [ ] All document types have required fields defined
- [ ] Field categories are updated
- [ ] Detection prompts include all document types
- [ ] Test with sample images for each document type

## Best Practices

1. **Keep Instructions Concise**: Shorter prompts often work better
2. **Be Specific**: Use examples in instructions (e.g., "format: $1,234.56")
3. **Test Incrementally**: Change one thing at a time
4. **Document Changes**: Comment your customizations in the YAML
5. **Maintain Consistency**: Use similar instruction styles across fields
6. **Version Control**: Commit changes before major modifications

## Example: Complete Custom Configuration

Here's an example of changing from 12 invoice fields to 6 critical fields:

```yaml
# Simplified invoice configuration - 6 critical fields only
essential_fields:
  DOCUMENT_TYPE:
    order: 1
    instruction: "[document type: invoice/receipt/statement or NOT_FOUND]"
    applicable_documents: [invoice, receipt, bank_statement]
    required: true
    
  INVOICE_NUMBER:
    order: 2
    instruction: "[invoice/receipt number or NOT_FOUND]"
    applicable_documents: [invoice, receipt]
    required: true
    
  VENDOR:
    order: 3
    instruction: "[vendor/supplier name or NOT_FOUND]"
    applicable_documents: [invoice, receipt]
    required: true
    
  DATE:
    order: 4  
    instruction: "[document date (DD/MM/YYYY) or NOT_FOUND]"
    applicable_documents: [invoice, receipt, bank_statement]
    required: true
    
  TOTAL:
    order: 5
    instruction: "[total amount with currency symbol or NOT_FOUND]"
    applicable_documents: [invoice, receipt]
    required: true
    
  TAX:
    order: 6
    instruction: "[tax/GST amount or NOT_FOUND]"
    applicable_documents: [invoice, receipt]
    required: true

document_types:
  invoice:
    document_type: "invoice"
    field_count: 6
    required_fields:
      - DOCUMENT_TYPE
      - INVOICE_NUMBER
      - VENDOR
      - DATE
      - TOTAL
      - TAX
    critical_fields:
      - INVOICE_NUMBER
      - TOTAL
    accuracy_threshold: 0.90

prompt_templates:
  expertise_frame: "Extract 6 key fields from this document."
  critical_instructions:
    - "Extract ONLY these 6 fields"
    - "Use NOT_FOUND if missing"
  format_rules:
    - "Format: FIELD: value"
    - "Exactly 6 lines of output"
```

## Conclusion

The YAML-first architecture provides complete control over prompts and field extraction without touching any Python code. All customization happens in `config/unified_schema.yaml`, making the system highly maintainable and adaptable to different document processing needs.