# Troubleshooting and FAQ

Common issues, solutions, and frequently asked questions for the LMM_POC vision model evaluation framework.

## Table of Contents

1. [Quick Diagnostic Commands](#quick-diagnostic-commands)
2. [Configuration Issues](#configuration-issues)
3. [Model Loading Problems](#model-loading-problems)
4. [Evaluation Issues](#evaluation-issues)
5. [Ground Truth Problems](#ground-truth-problems)
6. [Performance Issues](#performance-issues)
7. [Output and Reporting Issues](#output-and-reporting-issues)
8. [Environment and Dependencies](#environment-and-dependencies)
9. [Frequently Asked Questions](#frequently-asked-questions)
10. [Getting Help](#getting-help)

---

## Quick Diagnostic Commands

### System Health Check
```bash
# Check configuration is valid
python -c "
from common.config import FIELD_DEFINITIONS, validate_field_definitions
try:
    validate_field_definitions()
    print(f'✅ Configuration valid: {len(FIELD_DEFINITIONS)} fields')
except Exception as e:
    print(f'❌ Configuration error: {e}')
"

# Check imports work
python -c "
try:
    from models.llama_processor import LlamaProcessor
    from models.internvl3_processor import InternVL3Processor  
    from common.evaluation_utils import calculate_field_accuracy
    print('✅ All imports successful')
except Exception as e:
    print(f'❌ Import error: {e}')
"

# Check GPU availability
python -c "
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU count: {torch.cuda.device_count()}')
        print(f'Current GPU: {torch.cuda.current_device()}')
except ImportError:
    print('❌ PyTorch not installed')
"
```

### Environment Status
```bash
# Check conda environment
conda info --envs | grep '*'

# Check key dependencies
pip list | grep -E "(torch|transformers|pandas|pillow)"

# Check available memory
free -h  # Linux
# or
vm_stat | head -10  # macOS
```

### File Structure Check
```bash
# Verify all required files exist
ls -la common/config.py
ls -la models/llama_processor.py models/internvl3_processor.py
ls -la evaluation_data/evaluation_ground_truth.csv
ls -la common/evaluation_utils.py common/reporting.py
```

---

## Configuration Issues

### Issue: ValidationError on Configuration Load

**Symptoms**:
```
ValueError: Field 'TOTAL' missing required key: 'instruction'
```

**Diagnosis**:
```python
# Check specific field definition
from common.config import FIELD_DEFINITIONS
field_name = 'TOTAL'  # Replace with problematic field
definition = FIELD_DEFINITIONS.get(field_name, {})
required_keys = ['type', 'instruction', 'evaluation_logic', 'description', 'required']

print(f"Field '{field_name}' definition:")
for key in required_keys:
    status = '✅' if key in definition else '❌ MISSING'
    value = definition.get(key, 'NOT SET')
    print(f"  {key}: {status} {value}")
```

**Solutions**:
1. **Add missing keys**:
```python
'TOTAL': {
    'type': 'monetary',                           # ✅ Add if missing
    'instruction': '[total amount in dollars or N/A]',  # ✅ Add if missing
    'evaluation_logic': 'monetary_with_tolerance',       # ✅ Add if missing
    'description': 'Total amount of invoice/expense',    # ✅ Add if missing
    'required': True                                     # ✅ Add if missing
}
```

2. **Fix formatting issues**:
```python
# ❌ Wrong format
'instruction': 'total amount'

# ✅ Correct format  
'instruction': '[total amount in dollars or N/A]'
```

### Issue: Invalid Field Type Error

**Symptoms**:
```
ValueError: Field 'CUSTOM_FIELD' has invalid type: 'percentage'
```

**Diagnosis**:
```python
# Check valid types
from common.config import validate_field_definitions
# Look at the error message for valid_types list
```

**Solutions**:
1. **Use existing field type**:
```python
# Change to supported type
'CLAIM_PERCENTAGE': {
    'type': 'text',  # Use 'text' instead of 'percentage'
    'instruction': '[percentage claimable or N/A]',
    'evaluation_logic': 'fuzzy_text_match',
    'description': 'Percentage of expense claimable',
    'required': False
}
```

2. **Add support for new type** (advanced):
```python
# In validate_field_definitions() function
valid_types = ['numeric_id', 'monetary', 'date', 'list', 'text', 'percentage']
```

### Issue: Field Instructions Not Applied

**Symptoms**: Models not using field-specific instructions

**Diagnosis**:
```python
# Check if instructions are being imported correctly
from common.config import FIELD_INSTRUCTIONS
print("Sample instructions:")
for field in list(FIELD_INSTRUCTIONS.keys())[:3]:
    print(f"  {field}: {FIELD_INSTRUCTIONS[field]}")
```

**Solution**: Verify processors are importing correctly:
```python
# In models/llama_processor.py and models/internvl3_processor.py
from common.config import FIELD_INSTRUCTIONS  # ✅ Should be imported

# In get_extraction_prompt() method
instruction = FIELD_INSTRUCTIONS.get(field, '[value or N/A]')  # ✅ Should use this
```

---

## Model Loading Problems

### Issue: CUDA Out of Memory

**Symptoms**:
```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 22.00 GiB
```

**Diagnosis**:
```bash
# Check GPU memory
nvidia-smi

# Check what's using GPU memory
python -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_properties(i)}')
        print(f'Memory allocated: {torch.cuda.memory_allocated(i)/1024**3:.2f} GB')
        print(f'Memory cached: {torch.cuda.memory_reserved(i)/1024**3:.2f} GB')
"
```

**Solutions**:
1. **Use 8-bit quantization** (already enabled by default):
```python
# In model processors, verify this is set:
load_in_8bit=True  # Should be enabled
```

2. **Clear GPU memory**:
```bash
# Kill other processes using GPU
nvidia-smi  # Find process IDs
kill -9 <PID>

# Or restart system if needed
```

3. **Use CPU fallback**:
```python
# In processor initialization
device = 'cpu'  # Force CPU usage
```

4. **Process smaller batches**:
```python
# Process images one at a time instead of batch loading
```

### Issue: Model Not Found

**Symptoms**:
```
OSError: Model weights not found at /path/to/Llama-3.2-11B-Vision-Instruct
```

**Diagnosis**:
```python
# Check model paths in config
from common.config import LLAMA_MODEL_PATH, INTERNVL3_MODEL_PATH
import os

print(f"Llama path exists: {os.path.exists(LLAMA_MODEL_PATH)}")
print(f"InternVL3 path exists: {os.path.exists(INTERNVL3_MODEL_PATH)}")
print(f"Llama path: {LLAMA_MODEL_PATH}")
print(f"InternVL3 path: {INTERNVL3_MODEL_PATH}")
```

**Solutions**:
1. **Update model paths** in `common/config.py`:
```python
LLAMA_MODEL_PATH = "/actual/path/to/Llama-3.2-11B-Vision-Instruct"
INTERNVL3_MODEL_PATH = "/actual/path/to/InternVL3-2B"
```

2. **Download models** if missing:
```bash
# Using Hugging Face transformers
python -c "
from transformers import AutoModel, AutoProcessor
model = AutoModel.from_pretrained('OpenGVLab/InternVL3-2B', cache_dir='/path/to/save')
"
```

### Issue: Transformers Version Incompatibility

**Symptoms**:
```
ImportError: cannot import name 'MllamaForConditionalGeneration'
```

**Diagnosis**:
```python
import transformers
print(f"Transformers version: {transformers.__version__}")
# Should be 4.45.2 for Llama compatibility
```

**Solution**:
```bash
# Install correct version
pip install transformers==4.45.2
```

---

## Evaluation Issues

### Issue: Low Accuracy Despite Correct Results

**Symptoms**: Evaluation shows low accuracy but extracted values look correct

**Diagnosis**:
```python
# Debug specific field evaluation
from common.evaluation_utils import calculate_field_accuracy

# Test problematic case
extracted = "$123.45"
ground_truth = "123.45"
field_name = "TOTAL"

accuracy = calculate_field_accuracy(extracted, ground_truth, field_name)
print(f"Accuracy: {accuracy}")

# Check field type
from common.config import FIELD_TYPES
print(f"Field type: {FIELD_TYPES.get(field_name)}")
```

**Common Causes & Solutions**:

1. **Format Mismatch**:
```python
# Problem: Different currency formatting
extracted = "$123.45"
ground_truth = "123.45"  # Missing $ sign

# Solution: Ground truth should match expected format
ground_truth = "$123.45"  # ✅ Better
```

2. **Field Type Mismatch**:
```python
# Problem: Monetary field marked as text
'TOTAL': {'type': 'text'}  # Won't get numeric tolerance

# Solution: Use correct field type
'TOTAL': {'type': 'monetary'}  # ✅ Will get 1% tolerance
```

3. **N/A Handling Issues**:
```python
# Check N/A variants
na_variants = ['N/A', 'NA', '', 'NAN', 'NULL', 'NONE', 'NIL']
extracted_normalized = str(extracted).upper().strip()
gt_normalized = str(ground_truth).upper().strip()

print(f"Extracted normalized: '{extracted_normalized}'")
print(f"Ground truth normalized: '{gt_normalized}'")
print(f"Both are N/A variants: {extracted_normalized in na_variants and gt_normalized in na_variants}")
```

### Issue: Field Type Logic Not Working

**Symptoms**: Fields not getting expected evaluation behavior

**Diagnosis**:
```python
# Check field type groupings
from common.config import MONETARY_FIELDS, DATE_FIELDS, NUMERIC_ID_FIELDS
print(f"Monetary fields: {MONETARY_FIELDS}")
print(f"Date fields: {DATE_FIELDS}")
print(f"Numeric ID fields: {NUMERIC_ID_FIELDS}")

# Check if problem field is in expected group
problem_field = "TOTAL"
print(f"{problem_field} in monetary fields: {problem_field in MONETARY_FIELDS}")
```

**Solution**: Verify field type groupings are correct:
```python
# Check that derivation is working
from common.config import FIELD_DEFINITIONS
expected_monetary = [k for k, v in FIELD_DEFINITIONS.items() if v['type'] == 'monetary']
print(f"Expected monetary fields: {expected_monetary}")
```

### Issue: Missing Ground Truth Matches

**Symptoms**: Many fields showing as "no ground truth available"

**Diagnosis**:
```python
# Check ground truth loading
from common.evaluation_utils import load_ground_truth
from common.config import GROUND_TRUTH_PATH

ground_truth = load_ground_truth(GROUND_TRUTH_PATH)
print(f"Ground truth entries: {len(ground_truth)}")
print("Sample entries:")
for i, (image_name, values) in enumerate(list(ground_truth.items())[:3]):
    print(f"  {image_name}: {len(values)} fields")
```

**Solutions**:
1. **Check image name matching**:
```python
# Ensure image names in ground truth match processed images
processed_images = ["invoice_001.png", "invoice_002.png"]  # Example
gt_images = list(ground_truth.keys())
missing = set(processed_images) - set(gt_images)
if missing:
    print(f"Missing from ground truth: {missing}")
```

2. **Verify CSV structure**:
```bash
head -1 evaluation_data/evaluation_ground_truth.csv
# Should have image_file column plus all EXTRACTION_FIELDS
```

---

## Ground Truth Problems

### Issue: Ground Truth CSV Loading Error

**Symptoms**:
```
Error loading ground truth: No columns to parse from file
```

**Diagnosis**:
```bash
# Check CSV file structure
head -5 evaluation_data/evaluation_ground_truth.csv
wc -l evaluation_data/evaluation_ground_truth.csv
file evaluation_data/evaluation_ground_truth.csv
```

**Solutions**:
1. **Fix CSV format**:
```csv
# Ensure proper CSV structure with header row
image_file,ABN,ACCOUNT_HOLDER,BANK_ACCOUNT_NUMBER,...,TOTAL
invoice_001.png,04904754234,N/A,N/A,...,$156.90
```

2. **Check encoding**:
```bash
file -i evaluation_data/evaluation_ground_truth.csv
# Should be UTF-8 or ASCII
```

3. **Verify no empty file**:
```bash
ls -la evaluation_data/evaluation_ground_truth.csv
# Should have size > 0
```

### Issue: Column Count Mismatch

**Symptoms**: Ground truth has wrong number of columns

**Diagnosis**:
```python
import pandas as pd
from common.config import EXTRACTION_FIELDS

# Load and check CSV
df = pd.read_csv("evaluation_data/evaluation_ground_truth.csv")
expected_columns = ["image_file"] + EXTRACTION_FIELDS
actual_columns = list(df.columns)

print(f"Expected columns: {len(expected_columns)}")
print(f"Actual columns: {len(actual_columns)}")
print("Missing columns:", set(expected_columns) - set(actual_columns))
print("Extra columns:", set(actual_columns) - set(expected_columns))
```

**Solution**: Update ground truth CSV to match field configuration:
```bash
# Add missing columns with N/A values
# Remove extra columns not in EXTRACTION_FIELDS
```

### Issue: Ground Truth Data Quality

**Symptoms**: Inconsistent or poor quality reference data

**Diagnosis**:
```python
# Analyze ground truth data quality
import pandas as pd
df = pd.read_csv("evaluation_data/evaluation_ground_truth.csv")

# Check for missing values
print("Missing values per field:")
print(df.isnull().sum())

# Check for inconsistent formats
for field in ['TOTAL', 'GST', 'INVOICE_DATE']:  # Example fields
    if field in df.columns:
        values = df[field].dropna().unique()[:10]
        print(f"{field} sample values: {values}")
```

**Solutions**:
1. **Standardize formats**:
```csv
# Ensure consistent formatting
TOTAL,$123.45  # Not: 123.45, $123, 123.45 USD
INVOICE_DATE,15/03/2024  # Not: March 15 2024, 2024-03-15
```

2. **Use N/A consistently**:
```csv
# Use N/A for missing values, not empty, null, etc.
BANK_NAME,N/A  # Not: empty cell, "null", "-"
```

---

## Performance Issues

### Issue: Slow Processing Speed

**Symptoms**: Taking too long to process documents

**Diagnosis**:
```python
# Profile processing time
import time

start_time = time.time()
# Run processing
end_time = time.time()

print(f"Total time: {end_time - start_time:.2f} seconds")
print(f"Time per image: {(end_time - start_time)/num_images:.2f} seconds")
```

**Solutions**:
1. **Check GPU utilization**:
```bash
nvidia-smi
# Should show GPU usage during processing
```

2. **Optimize batch size**:
```python
# Process images in smaller batches if memory limited
# Use GPU if available, CPU as fallback
```

3. **Check model quantization**:
```python
# Ensure 8-bit quantization is enabled
load_in_8bit=True
```

### Issue: High Memory Usage

**Symptoms**: System running out of memory

**Diagnosis**:
```bash
# Monitor memory usage during processing
top -p $(pgrep python)  # Linux
# or
ps aux | grep python    # macOS/Linux
```

**Solutions**:
1. **Enable memory optimization**:
```python
# Use low_cpu_mem_usage option
low_cpu_mem_usage=True
```

2. **Clear cache between images**:
```python
import torch
torch.cuda.empty_cache()  # Clear GPU cache
```

3. **Process smaller batches**:
```python
# Process images one at a time
for image in image_files:
    result = process_single_image(image)
    # Clear memory before next image
```

---

## Output and Reporting Issues

### Issue: CSV Output Missing Columns

**Symptoms**: Generated CSV files missing expected fields

**Diagnosis**:
```python
# Check CSV structure
import pandas as pd
df = pd.read_csv("output/llama_batch_extraction_*.csv")
from common.config import EXTRACTION_FIELDS

expected_columns = ["image_name"] + EXTRACTION_FIELDS
actual_columns = list(df.columns)
missing = set(expected_columns) - set(actual_columns)
print(f"Missing columns: {missing}")
```

**Solutions**:
1. **Check field configuration**:
```python
# Verify all fields are defined properly
from common.config import FIELD_DEFINITIONS
print(f"Defined fields: {len(FIELD_DEFINITIONS)}")
```

2. **Verify extraction results**:
```python
# Check that extraction includes all fields
for result in extraction_results[:1]:  # Check first result
    extracted_fields = set(result.keys())
    missing = set(EXTRACTION_FIELDS) - extracted_fields
    print(f"Missing from extraction: {missing}")
```

### Issue: Report Generation Errors

**Symptoms**: Errors when generating reports

**Diagnosis**:
```python
# Check evaluation data structure
evaluation_summary = {...}  # Your evaluation results
required_keys = ['total_images', 'overall_accuracy', 'field_accuracies']
for key in required_keys:
    if key not in evaluation_summary:
        print(f"Missing key: {key}")
```

**Solution**: Ensure evaluation summary has required structure:
```python
evaluation_summary = {
    'total_images': 20,
    'overall_accuracy': 0.85,
    'field_accuracies': {'TOTAL': 0.90, ...},
    'evaluation_data': [...]  # Document-level data
}
```

---

## Environment and Dependencies

### Issue: Import Errors

**Symptoms**:
```
ModuleNotFoundError: No module named 'torch'
ImportError: No module named 'common.config'
```

**Diagnosis**:
```bash
# Check conda environment
conda info --envs
echo "Current environment: $CONDA_DEFAULT_ENV"

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Check if we're in project directory
ls -la common/config.py
```

**Solutions**:
1. **Activate correct environment**:
```bash
conda activate vision_notebooks
```

2. **Install missing dependencies**:
```bash
pip install torch transformers pandas pillow
```

3. **Fix Python path**:
```bash
# Ensure you're in project directory
cd /path/to/LMM_POC
export PYTHONPATH=/path/to/LMM_POC:$PYTHONPATH
```

### Issue: Version Conflicts

**Symptoms**: Incompatible package versions

**Diagnosis**:
```bash
pip list | grep -E "(torch|transformers|numpy|pandas)"
```

**Solution**:
```bash
# Install specific versions as required
pip install transformers==4.45.2 torch>=2.0.0
```

---

## Frequently Asked Questions

### Q: How do I add a completely new field?

**A**: Add to `FIELD_DEFINITIONS` in `common/config.py`:
```python
'NEW_FIELD': {
    'type': 'text',  # or appropriate type
    'instruction': '[field instruction or N/A]',
    'evaluation_logic': 'fuzzy_text_match',  # or appropriate logic
    'description': 'Description of field purpose',
    'required': False
}
```

Then update ground truth CSV with new column.

### Q: How do I change field instructions for better accuracy?

**A**: Edit the `instruction` key in `FIELD_DEFINITIONS`:
```python
'SUPPLIER': {
    'instruction': '[complete supplier name including company type (Pty Ltd, Inc, etc.) or N/A]',
    # More specific instruction for better results
}
```

### Q: Can I use different field sets for different document types?

**A**: Currently, the system uses one field set. For different document types:
1. Create separate configurations by modifying `FIELD_DEFINITIONS`
2. Use different ground truth files
3. Run separate evaluation pipelines

### Q: How do I improve accuracy for specific fields?

**A**: 
1. **Improve instructions**: Be more specific about format and content
2. **Check field type**: Ensure correct type for proper evaluation logic
3. **Analyze failures**: Look at specific cases where the field fails
4. **Update ground truth**: Ensure reference data is accurate and consistent

### Q: What's the difference between field types?

**A**:
- `numeric_id`: Exact matching (ABN, account numbers)
- `monetary`: Numeric with 1% tolerance (amounts, prices)  
- `date`: Flexible date format matching
- `list`: Multiple items with partial credit
- `text`: Fuzzy text matching with partial credit

### Q: How do I handle fields that don't exist in some documents?

**A**: Use `N/A` values consistently:
- In ground truth CSV: `BANK_NAME,N/A`
- In model responses: Models should output `N/A` for missing fields
- Evaluation system rewards correct `N/A` identification

### Q: Can I run evaluation without GPU?

**A**: Yes, set `device='cpu'` in processor initialization. Processing will be slower but functional.

### Q: How do I back up my configuration before making changes?

**A**:
```bash
cp common/config.py common/config.py.backup.$(date +%Y%m%d_%H%M%S)
cp evaluation_data/evaluation_ground_truth.csv evaluation_data/ground_truth.backup.$(date +%Y%m%d_%H%M%S)
```

### Q: What file formats are supported for images?

**A**: PNG, JPG, JPEG (case insensitive). See `IMAGE_EXTENSIONS` in config.py.

### Q: How do I process images from a different directory?

**A**: Update `DATA_DIR` in `common/config.py` to point to your image directory.

---

## Getting Help

### Debug Information to Collect

When seeking help, provide:

1. **Configuration status**:
```bash
python -c "
from common.config import FIELD_DEFINITIONS
print(f'Field count: {len(FIELD_DEFINITIONS)}')
print(f'Sample fields: {list(FIELD_DEFINITIONS.keys())[:5]}')
"
```

2. **Environment information**:
```bash
conda info --envs
python --version
pip list | grep -E "(torch|transformers|pandas)"
```

3. **Error messages** (full traceback)

4. **System specifications**:
```bash
nvidia-smi  # If using GPU
free -h     # Memory info
```

### Escalation Path

1. **Check this FAQ** for common issues
2. **Review documentation** in `docs/` directory
3. **Check configuration** with diagnostic commands
4. **Test with minimal example** to isolate issue
5. **Collect debug information** as listed above

### Self-Help Resources

- **Field Configuration Guide**: `docs/FIELD_CONFIGURATION_GUIDE.md`
- **Evaluation System Guide**: `docs/EVALUATION_SYSTEM_GUIDE.md`  
- **Field Management Workflows**: `docs/FIELD_MANAGEMENT_WORKFLOWS.md`
- **Ground Truth Evaluation System**: `ground_truth_evaluation_system.md`
- **Refactoring Summary**: `REFACTORING_SUMMARY.md`

---

Remember: Most issues are configuration-related and can be resolved by carefully checking field definitions, ground truth data format, and environment setup. Use the diagnostic commands liberally to understand what's happening before making changes.