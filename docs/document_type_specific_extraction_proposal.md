# Document Type Specific Information Extraction Proposal

**Git Hash at Proposal Creation**: `4b866e0f3dca8e298d47290b5bbce1d803dc3297`  
*Use this hash to rollback if needed: `git checkout 4b866e0f3dca8e298d47290b5bbce1d803dc3297`*

## Executive Summary
The business client requires document-type-specific information extraction because different document types (invoices, bank statements, receipts) contain fundamentally different information. This proposal outlines modifications to the current unified schema approach to support targeted extraction based on document type.

## Current State Analysis

### Existing Unified Schema Issues
1. **25 fields for ALL document types** - wasteful and confusing
2. **Many NOT_FOUND values** - fields irrelevant to document type
3. **Mixed field relevance** - invoice fields on bank statements, etc.
4. **Reduced accuracy** - models searching for non-existent fields
5. **Poor cognitive focus** - diluted attention across irrelevant fields

### Current Architecture Strengths
- Centralized configuration via `field_schema.yaml`
- Dynamic schema loading via `FieldSchema` class
- Consistent evaluation logic and validation
- Model-specific prompt templates

## Proposed Solution: Hierarchical Document-Type Schema

### Core Design Principles
1. **Document type detection first** - Identify document type before field extraction
2. **Type-specific schemas** - Only extract relevant fields per document type
3. **Shared common fields** - Reuse fields across document types where applicable
4. **Backward compatibility** - Minimal changes to existing pipeline

## Implementation Architecture

### 1. Schema Structure Modification

```yaml
# Modified field_schema.yaml structure
schema_version: "2.0"
extraction_mode: "document_type_specific"  # or "unified" for backward compat

# Common fields across all document types
common_fields:
  - name: "DOCUMENT_TYPE"
    type: "text"
    evaluation_logic: "exact_match"
    group: "metadata"
    description: "Type of business document"
    
  - name: "BUSINESS_ABN"
    type: "numeric_id"
    evaluation_logic: "exact_numeric_match"
    group: "critical"
    description: "11-digit Australian Business Number"
    
  - name: "SUPPLIER_NAME"
    type: "text"
    evaluation_logic: "fuzzy_text_match"
    group: "business_entity"
    description: "Supplier/business name"

# Document type specific schemas
document_schemas:
  invoice:
    document_type_values: ["invoice", "tax invoice", "bill"]
    total_fields: 18
    inherits: ["common_fields"]
    specific_fields:
      - name: "INVOICE_NUMBER"
        type: "text"
        group: "metadata"
        description: "Invoice reference number"
        
      - name: "INVOICE_DATE"
        type: "date"
        group: "dates"
        description: "Invoice issue date"
        
      - name: "DUE_DATE"
        type: "date"
        group: "dates"
        description: "Payment due date"
        
      - name: "LINE_ITEM_DESCRIPTIONS"
        type: "list"
        group: "item_details"
        
      - name: "LINE_ITEM_QUANTITIES"
        type: "list"
        group: "item_details"
        
      - name: "LINE_ITEM_PRICES"
        type: "list"
        group: "item_details"
        
      - name: "SUBTOTAL_AMOUNT"
        type: "monetary"
        group: "monetary"
        
      - name: "GST_AMOUNT"
        type: "monetary"
        group: "monetary"
        
      - name: "TOTAL_AMOUNT"
        type: "monetary"
        group: "critical"
    
    excluded_fields:
      - "OPENING_BALANCE"  # Bank statement only
      - "CLOSING_BALANCE"  # Bank statement only
      - "STATEMENT_DATE_RANGE"  # Bank statement only
      - "TRANSACTION_DATES"  # Bank statement only
      
  bank_statement:
    document_type_values: ["bank statement", "account statement", "statement"]
    total_fields: 15
    inherits: ["common_fields"]
    specific_fields:
      - name: "BANK_NAME"
        type: "text"
        group: "banking"
        description: "Financial institution name"
        
      - name: "BANK_BSB_NUMBER"
        type: "numeric_id"
        group: "banking"
        description: "6-digit BSB number"
        
      - name: "BANK_ACCOUNT_NUMBER"
        type: "numeric_id"
        group: "banking"
        
      - name: "ACCOUNT_HOLDER"
        type: "text"
        group: "banking"
        
      - name: "STATEMENT_DATE_RANGE"
        type: "date_range"
        group: "dates"
        description: "Statement period (e.g., May 2025)"
        
      - name: "OPENING_BALANCE"
        type: "monetary"
        group: "monetary"
        description: "Starting balance"
        
      - name: "CLOSING_BALANCE"
        type: "monetary"
        group: "monetary"
        description: "Ending balance"
        
      - name: "TOTAL_CREDITS"
        type: "monetary"
        group: "monetary"
        
      - name: "TOTAL_DEBITS"
        type: "monetary"
        group: "monetary"
        
      - name: "TRANSACTION_DESCRIPTIONS"
        type: "list"
        group: "transactions"
        
      - name: "TRANSACTION_AMOUNTS"
        type: "list"
        group: "transactions"
        
      - name: "TRANSACTION_DATES"
        type: "list"
        group: "transactions"
    
    excluded_fields:
      - "INVOICE_DATE"  # Invoice only
      - "DUE_DATE"  # Invoice only
      - "LINE_ITEM_QUANTITIES"  # Invoice/receipt only
      - "GST_AMOUNT"  # Invoice/receipt only
      
  receipt:
    document_type_values: ["receipt", "purchase receipt", "payment receipt"]
    total_fields: 12
    inherits: ["common_fields"]
    specific_fields:
      - name: "RECEIPT_NUMBER"
        type: "text"
        group: "metadata"
        
      - name: "TRANSACTION_DATE"
        type: "date"
        group: "dates"
        description: "Date of purchase"
        
      - name: "PAYMENT_METHOD"
        type: "text"
        group: "payment"
        description: "Cash, card, etc."
        
      - name: "LINE_ITEM_DESCRIPTIONS"
        type: "list"
        group: "item_details"
        
      - name: "LINE_ITEM_QUANTITIES"
        type: "list"
        group: "item_details"
        
      - name: "LINE_ITEM_PRICES"
        type: "list"
        group: "item_details"
        
      - name: "SUBTOTAL_AMOUNT"
        type: "monetary"
        group: "monetary"
        
      - name: "GST_AMOUNT"
        type: "monetary"
        group: "monetary"
        
      - name: "TOTAL_AMOUNT"
        type: "monetary"
        group: "critical"
    
    excluded_fields:
      - "DUE_DATE"  # Not applicable
      - "OPENING_BALANCE"  # Bank statement only
      - "CLOSING_BALANCE"  # Bank statement only
```

### 2. Modified FieldSchema Class

```python
class DocumentTypeFieldSchema(FieldSchema):
    """Extended schema loader supporting document-type-specific extraction."""
    
    def __init__(self, schema_file: str = "field_schema.yaml"):
        super().__init__(schema_file)
        self.extraction_mode = self.schema.get("extraction_mode", "unified")
        self.document_schemas = self.schema.get("document_schemas", {})
        
    def detect_document_type(self, image_path: str) -> str:
        """First-pass extraction to identify document type."""
        # Quick extraction focused only on DOCUMENT_TYPE field
        # Uses lightweight prompt for fast classification
        pass
        
    def get_document_schema(self, document_type: str) -> dict:
        """Get schema specific to document type."""
        # Normalize document type
        doc_type_normalized = self._normalize_document_type(document_type)
        
        if doc_type_normalized in self.document_schemas:
            return self._build_complete_schema(doc_type_normalized)
        else:
            # Fallback to unified schema
            return self.get_unified_schema()
            
    def _build_complete_schema(self, doc_type: str) -> dict:
        """Build complete field list for document type."""
        schema = self.document_schemas[doc_type]
        complete_fields = []
        
        # Add inherited common fields
        if "common_fields" in schema.get("inherits", []):
            complete_fields.extend(self.schema["common_fields"])
            
        # Add document-specific fields
        complete_fields.extend(schema["specific_fields"])
        
        return {
            "fields": complete_fields,
            "total_fields": len(complete_fields),
            "document_type": doc_type,
            "excluded_fields": schema.get("excluded_fields", [])
        }
```

### 3. Modified Processing Pipeline

```python
# In llama_processor.py and internvl3_processor.py

def process_single_image(self, image_path: str) -> dict:
    """Process image with document-type-specific extraction."""
    
    # Step 1: Detect and classify document type (modern approach)
    classification_info = self.schema_loader.detect_and_classify_document(image_path)
    doc_type = classification_info['document_type']
    
    # Step 2: Get appropriate schema
    doc_schema = self.schema_loader.get_document_schema(doc_type)
    
    # Step 3: Generate targeted prompt with only relevant fields
    prompt = self._generate_document_specific_prompt(doc_schema)
    
    # Step 4: Extract only relevant fields
    extraction_result = self._extract_with_schema(image_path, prompt, doc_schema)
    
    # Step 5: Validate based on document type rules
    validated_result = self._validate_by_document_type(extraction_result, doc_type)
    
    return validated_result
```

## Implementation Phases

### Phase 1: Document Type Detection (Week 1)
1. Create lightweight document classification prompt
2. Test classification accuracy on existing test data
3. Build fallback mechanisms for uncertain classification

### Phase 2: Schema Refactoring (Week 2)
1. Create new `field_schema_v2.yaml` with hierarchical structure
2. Extend `FieldSchema` class to support both v1 and v2
3. Maintain backward compatibility flag

### Phase 3: Document-Specific Schemas (Week 3)
1. Define invoice-specific schema (18 fields)
2. Define bank statement schema (15 fields)
3. Define receipt schema (12 fields)
4. Create schema validation tests

### Phase 4: Pipeline Integration (Week 4)
1. Modify processors to use document-type routing
2. Update prompt generation for targeted extraction
3. Adjust evaluation metrics per document type
4. Performance testing and optimization

## Benefits

### Accuracy Improvements
- **Focused extraction**: 30-40% fewer fields to extract per document
- **Reduced false positives**: No searching for irrelevant fields
- **Better prompts**: Document-type-specific instructions

### Performance Gains
- **Faster processing**: Fewer fields = less computation
- **Reduced token usage**: Smaller prompts and responses
- **Better memory usage**: Smaller field sets in memory

### User Experience
- **Cleaner outputs**: No unnecessary NOT_FOUND values
- **Better validation**: Document-specific business rules
- **Clearer reporting**: Type-specific accuracy metrics

## Risk Mitigation

### Backward Compatibility
- Keep unified schema as fallback option
- Configuration flag for extraction mode
- Gradual migration path for existing users

### Document Type Misclassification
- Confidence threshold for classification
- Manual override option
- Fallback to unified schema when uncertain

### Schema Maintenance
- Centralized common fields to avoid duplication
- Inheritance mechanism for shared fields
- Version control for schema evolution

## Success Metrics

### Primary Metrics
1. **Extraction Accuracy**: Target 95%+ for relevant fields
2. **Processing Speed**: 30% reduction in extraction time
3. **Token Usage**: 40% reduction in prompt tokens

### Secondary Metrics
1. **Document Classification Accuracy**: 98%+ correct classification
2. **User Satisfaction**: Reduced NOT_FOUND noise
3. **Maintenance Effort**: Single-point schema updates

## Migration Strategy

### Step 1: Parallel Testing
- Run both unified and type-specific extraction
- Compare results and measure improvements
- Identify edge cases and refine schemas

### Step 2: Gradual Rollout
- Enable for new users first
- A/B testing with existing users
- Monitor metrics and gather feedback

### Step 3: Full Migration
- Set type-specific as default mode
- Maintain unified as legacy option
- Document migration guide for users

## Alternative Approaches Considered

### 1. Multiple Separate Schemas
- **Pros**: Complete independence
- **Cons**: Maintenance overhead, code duplication
- **Decision**: Rejected - too much duplication

### 2. Dynamic Field Detection
- **Pros**: No predefined schemas needed
- **Cons**: Unpredictable, harder to validate
- **Decision**: Rejected - lack of structure

### 3. ML-Based Field Selection
- **Pros**: Adaptive to new document types
- **Cons**: Requires training data, complexity
- **Decision**: Future consideration

## Recommendation

Proceed with the hierarchical document-type schema approach because it:
1. **Solves the immediate business need** for type-specific extraction
2. **Maintains system architecture** with minimal disruption
3. **Provides clear improvement path** with measurable benefits
4. **Enables future extensions** for new document types

## Next Steps

1. **Stakeholder Review**: Present proposal to business client
2. **Technical Proof of Concept**: Implement Phase 1 (classification)
3. **Schema Design Workshop**: Collaborate on field definitions
4. **Implementation Planning**: Detailed sprint planning
5. **Testing Strategy**: Define test cases per document type

## Appendix: Field Mapping

### Current Unified Schema (25 fields) → Type-Specific Schemas

| Field Name | Invoice | Bank Statement | Receipt |
|------------|---------|----------------|---------|
| DOCUMENT_TYPE | ✓ | ✓ | ✓ |
| BUSINESS_ABN | ✓ | ✓ | ✓ |
| SUPPLIER_NAME | ✓ | ✗ | ✓ |
| INVOICE_DATE | ✓ | ✗ | ✗ |
| DUE_DATE | ✓ | ✗ | ✗ |
| STATEMENT_DATE_RANGE | ✗ | ✓ | ✗ |
| OPENING_BALANCE | ✗ | ✓ | ✗ |
| CLOSING_BALANCE | ✗ | ✓ | ✗ |
| LINE_ITEM_DESCRIPTIONS | ✓ | ✗ | ✓ |
| LINE_ITEM_QUANTITIES | ✓ | ✗ | ✓ |
| LINE_ITEM_PRICES | ✓ | ✗ | ✓ |
| TRANSACTION_DESCRIPTIONS | ✗ | ✓ | ✗ |
| TRANSACTION_AMOUNTS | ✗ | ✓ | ✗ |
| TRANSACTION_DATES | ✗ | ✓ | ✗ |
| BANK_NAME | ✗ | ✓ | ✗ |
| BANK_BSB_NUMBER | ✗ | ✓ | ✗ |
| BANK_ACCOUNT_NUMBER | ✗ | ✓ | ✗ |
| PAYMENT_METHOD | ✗ | ✗ | ✓ |
| RECEIPT_NUMBER | ✗ | ✗ | ✓ |
| GST_AMOUNT | ✓ | ✗ | ✓ |
| TOTAL_AMOUNT | ✓ | ✗ | ✓ |

### Efficiency Gains
- **Invoice**: 18 fields (28% reduction)
- **Bank Statement**: 15 fields (40% reduction)  
- **Receipt**: 12 fields (52% reduction)