# Production Field Changes Guide

## 📋 How to Add New EXTRACTION_FIELDS

The refactored architecture makes it easy to add new fields. Here's exactly what to change:

### 1. **Update Field Configuration** (`common/config.py`)

```python
# Add new fields to EXTRACTION_FIELDS list (keep alphabetical order)
EXTRACTION_FIELDS = [
    'ABN',
    'ACCOUNT_HOLDER',
    # ... existing fields ...
    'TOTAL',
    
    # ✅ Add new fields here in alphabetical order:
    'BUSINESS_EMAIL',           # New field example
    'CUSTOMER_ID',             # New field example
    'PAYMENT_TERMS',           # New field example
    'TAX_RATE',                # New field example
]
```

**That's it!** The `FIELD_COUNT` is automatically calculated.

### 2. **Add Field-Specific Instructions (Optional)**

For Llama processor, add field-specific instructions in `models/llama_processor.py`:

```python
field_instructions = {
    # ... existing instructions ...
    'BUSINESS_EMAIL': '[business email address or N/A]',
    'CUSTOMER_ID': '[customer identifier or N/A]',
    'PAYMENT_TERMS': '[payment terms (e.g., Net 30) or N/A]',
    'TAX_RATE': '[tax rate percentage or N/A]',
}
```

### 3. **Update Ground Truth CSV**

Add new field columns to your ground truth CSV file with actual values.

### 4. **Test the Changes**

Run the test script to verify everything works:
```bash
python test_refactoring.py
```

## 🔄 What Updates Automatically

The refactored code automatically handles:

- ✅ **Prompt generation** - Fields added to extraction prompts
- ✅ **Token allocation** - Generation tokens scale with field count  
- ✅ **Progress reporting** - Shows correct field counts (e.g., "25/30 fields")
- ✅ **CSV columns** - New fields appear in output CSV
- ✅ **Evaluation metrics** - Accuracy calculated for all fields
- ✅ **Reports** - Executive summaries include new field counts

## 📊 Field Count Scaling

The system automatically scales:

| Component | Scaling Formula |
|-----------|----------------|
| InternVL3 Tokens | `max(1000, FIELD_COUNT * 50)` |
| Llama Tokens | `max(800, FIELD_COUNT * 40)` |
| Performance Threshold | `max(15, FIELD_COUNT * 0.6)` fields at 90% |

## 🧪 Example: Adding 5 New Fields

```python
# Before (25 fields)
EXTRACTION_FIELDS = [
    'ABN', 'ACCOUNT_HOLDER', ..., 'TOTAL'
]

# After (30 fields) 
EXTRACTION_FIELDS = [
    'ABN', 'ACCOUNT_HOLDER', ..., 'TOTAL',
    'BUSINESS_EMAIL',
    'CUSTOMER_ID', 
    'DISCOUNT_AMOUNT',
    'PAYMENT_TERMS',
    'TAX_RATE'
]
```

**Automatic Updates:**
- ✅ Prompts: "OUTPUT FORMAT (30 required fields)"
- ✅ Tokens: InternVL3 uses 1,500 tokens, Llama uses 1,200 tokens
- ✅ Reporting: "Successfully extracts X out of 30 fields"
- ✅ Thresholds: "At least 18 fields with ≥90% accuracy"

## 🚀 Production Deployment

1. **Update Configuration**
   ```bash
   # Edit common/config.py
   vim common/config.py
   ```

2. **Update Ground Truth**
   ```bash
   # Add new columns to ground truth CSV
   vim evaluation_ground_truth.csv
   ```

3. **Test Locally**
   ```bash
   python test_refactoring.py
   ```

4. **Deploy to Remote System**
   ```bash
   # Copy updated files
   scp -r common/ models/ remote-system:/path/
   
   # Run evaluation
   python internvl3_keyvalue_refactored.py
   python llama_keyvalue_refactored.py
   ```

## ⚠️ Important Notes

- **Keep alphabetical order** in EXTRACTION_FIELDS for consistent CSV column ordering
- **Update ground truth CSV** to include new field columns
- **Field names must match exactly** between config, ground truth, and extraction results
- **Test with sample documents** to ensure new fields are being extracted properly

## 🔍 Validation Checklist

Before deploying new fields to production:

- [ ] Added new fields to `EXTRACTION_FIELDS` in alphabetical order
- [ ] Updated ground truth CSV with new field columns
- [ ] Added field-specific instructions for complex fields (optional)
- [ ] Ran `test_refactoring.py` successfully
- [ ] Tested extraction on sample documents
- [ ] Verified CSV output includes new fields
- [ ] Checked evaluation reports show correct field counts

## 📈 Performance Impact

Adding fields will increase:
- **Token usage** (automatically scaled)
- **Processing time** (more fields to extract)
- **Output file size** (more columns in CSV)
- **Evaluation complexity** (more comparisons)

Monitor performance and adjust token limits if needed for very large field counts (50+ fields).