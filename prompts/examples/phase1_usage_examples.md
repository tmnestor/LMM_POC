# Phase 1: Document Type Detection - Usage Examples

## Quick Start

### Command Line Testing
```bash
# Test with Llama processor
python test_document_classification.py --model llama

# Test with InternVL3 processor  
python test_document_classification.py --model internvl3

# Test single image
python test_document_classification.py --model llama --image evaluation_data/synthetic_invoice_001.png

# Test specific directory
python test_document_classification.py --model llama --directory evaluation_data/
```

### Interactive Notebook
```bash
# Launch Jupyter and open the classification test notebook
jupyter notebook notebooks/document_classification_test.ipynb
```

### Python API Usage
```python
from models.llama_processor import LlamaProcessor
from common.document_type_detector import DocumentTypeDetector

# Initialize
processor = LlamaProcessor()
detector = DocumentTypeDetector(processor)

# Classify single document
result = detector.detect_document_type("path/to/document.png")
print(f"Type: {result['type']}, Confidence: {result['confidence']:.2f}")

# Batch classify directory
results = detector.batch_classify_images("evaluation_data/")
report = detector.generate_classification_report(results)
print(report)
```

## Expected Output Examples

### Single Image Classification
```
🔍 Classifying document type: synthetic_invoice_001.png
✅ Classification: invoice (confidence: 0.92)

📄 Image: synthetic_invoice_001.png
🏷️  Type: invoice
🎯 Confidence: 0.920
⏱️  Time: 2.34s
💭 Reasoning: Document contains invoice number, due date, and line items
```

### Batch Classification Report
```
📊 DOCUMENT TYPE CLASSIFICATION REPORT
==================================================

📈 OVERVIEW:
   Total Images: 10
   Successfully Classified: 10
   Errors: 0
   Fallbacks Used: 2
   Manual Review Needed: 0

📋 CLASSIFICATION RESULTS:
   INVOICE: 4 documents (40.0%) - Avg confidence: 0.89
   BANK_STATEMENT: 3 documents (30.0%) - Avg confidence: 0.85
   RECEIPT: 3 documents (30.0%) - Avg confidence: 0.91

🎯 QUALITY METRICS:
   High Confidence Classifications: 8/10 (80.0%)
   Confidence Threshold: 0.85
   Overall Success Rate: 100.0%
```

## Performance Expectations

### Processing Times
- **Llama-3.2-11B**: 1.5-3.0 seconds per image
- **InternVL3-8B**: 0.8-2.0 seconds per image

### Accuracy Targets
- **Primary Goal**: 90%+ classification accuracy
- **Confidence Threshold**: 0.85 (adjustable)
- **Fallback Rate**: <20% of classifications

### Supported Document Types
1. **invoice** - Sales invoices, tax invoices, bills
2. **bank_statement** - Account statements, transaction histories
3. **receipt** - Purchase receipts, payment proofs

## Integration with Existing System

### Processor Integration
The detector works with existing processor classes:

```python
# Works with LlamaProcessor
from models.llama_processor import LlamaProcessor
processor = LlamaProcessor()
detector = DocumentTypeDetector(processor)

# Works with InternVL3Processor
from models.internvl3_processor import InternVL3Processor
processor = InternVL3Processor()
detector = DocumentTypeDetector(processor)
```

### Custom Prompt Method
Uses existing `_extract_with_custom_prompt()` method in processors:
```python
# Detector automatically uses the appropriate method
response = self.processor._extract_with_custom_prompt(
    image_path, 
    classification_prompt,
    max_new_tokens=150,
    temperature=0.0,
    do_sample=False
)
```

## Error Handling

### Common Issues and Solutions

**1. Import Errors**
```bash
❌ Import error: No module named 'models.llama_processor'
💡 Make sure you're running from the project root directory
```

**2. Model Access Issues**
```bash
❌ Failed to initialize processor: CUDA out of memory
💡 Make sure you're running on a machine with GPU and model access
```

**3. Image Not Found**
```bash
❌ Image not found: evaluation_data/missing_image.png
💡 Check image path and file existence
```

### Fallback Mechanisms

**Confidence Too Low**
```python
# Automatic fallback to keyword-based classification
result = {
    'type': 'invoice',
    'confidence': 0.65,
    'reasoning': 'Keyword-based classification (score: 3)',
    'fallback_used': True
}
```

**Classification Failure**
```python
# Ultimate fallback for errors
result = {
    'type': 'unknown',
    'confidence': 0.0,
    'reasoning': 'Classification error: Connection timeout',
    'fallback_used': True,
    'manual_review_needed': True
}
```

## Tuning and Optimization

### Confidence Threshold Adjustment
```python
# Adjust based on your accuracy requirements
detector.confidence_threshold = 0.75  # More lenient
detector.confidence_threshold = 0.90  # More strict
```

### Model-Specific Prompts
The detector uses optimized prompts for each model:

**Llama (Detailed)**
```
Identify the document type of this business document.

DOCUMENT TYPES:
- invoice: Sales invoice, tax invoice, bill for goods/services
- bank_statement: Bank account statement, transaction history
- receipt: Purchase receipt, payment receipt, proof of payment

Look for key indicators:
- Invoice: Invoice number, due date, line items with quantities
- Bank statement: Account number, statement period, transaction list
- Receipt: Receipt number, payment method, store location

Output format:
DOCUMENT_TYPE: [invoice|bank_statement|receipt]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation]
```

**InternVL3 (Concise)**
```
Document type classification:

Types: invoice, bank_statement, receipt

Key features:
- invoice: has due date, line items, customer details
- bank_statement: has account number, transaction history
- receipt: has payment method, store details

Output only:
TYPE: [invoice|bank_statement|receipt]
CONFIDENCE: [0.0-1.0]
```

## Testing Workflows

### Development Testing
1. **Single Image**: Test classification on known document types
2. **Batch Processing**: Process evaluation dataset
3. **Performance Analysis**: Check speed and accuracy metrics
4. **Edge Cases**: Test with unclear or damaged documents

### Production Readiness Checklist
- [ ] 90%+ classification accuracy on test dataset
- [ ] Average processing time <3 seconds
- [ ] Fallback rate <20%
- [ ] Error handling for all failure modes
- [ ] Confidence threshold tuned for use case

## Next Steps to Phase 2

Once Phase 1 achieves target performance:

1. **Schema Creation** - Define document-specific field schemas
2. **Routing Logic** - Implement type-based schema selection
3. **Integration** - Modify processors for type-specific extraction
4. **Testing** - Validate end-to-end document-specific pipeline

## Troubleshooting

### Debug Mode
```python
# Enable detailed logging in detector
detector = DocumentTypeDetector(processor)

# Check raw responses
result = detector.detect_document_type("test.png")
print("Raw response:", result.get('raw_response'))
```

### Performance Issues
```python
# Reduce token limits for faster processing
detector.classification_prompts["llama"]["max_tokens"] = 100
detector.classification_prompts["internvl3"]["max_tokens"] = 30
```

### Classification Accuracy Issues
```python
# Lower confidence threshold temporarily
detector.confidence_threshold = 0.70

# Check fallback classifications
fallback_results = [r for r in results if r.get('fallback_used')]
print(f"Fallback count: {len(fallback_results)}")
```

This completes Phase 1 implementation - ready for production testing!