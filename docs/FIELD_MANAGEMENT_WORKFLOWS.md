# Field Management Workflows

Practical workflows and procedures for managing fields in the LMM_POC vision model evaluation framework.

## Table of Contents

1. [Overview](#overview)
2. [Basic Field Operations](#basic-field-operations)
3. [Common Workflows](#common-workflows)
4. [Field Migration Procedures](#field-migration-procedures)
5. [Quality Assurance](#quality-assurance)
6. [Production Deployment](#production-deployment)
7. [Rollback Procedures](#rollback-procedures)
8. [Monitoring and Maintenance](#monitoring-and-maintenance)

---

## Overview

This guide provides step-by-step workflows for common field management tasks in the LMM_POC framework. All procedures follow the centralized configuration system where `common/config.py` serves as the single source of truth.

### Prerequisites

Before following these workflows, ensure:
- ✅ Understanding of [Field Configuration Guide](FIELD_CONFIGURATION_GUIDE.md)
- ✅ Understanding of [Evaluation System Guide](EVALUATION_SYSTEM_GUIDE.md)  
- ✅ Access to modify `common/config.py`
- ✅ Ground truth data management capabilities
- ✅ Testing environment setup

---

## Basic Field Operations

### Adding a Single New Field

**Use Case**: Adding one new field to the existing field set.

#### Step 1: Define the Field
```python
# Edit common/config.py - Add to FIELD_DEFINITIONS
'BUSINESS_PURPOSE': {
    'type': 'text',
    'instruction': '[business reason for expense or N/A]',
    'evaluation_logic': 'fuzzy_text_match',
    'description': 'Justification for business expense deduction',
    'required': False
}
```

#### Step 2: Validate Configuration
```bash
python -c "
from common.config import FIELD_DEFINITIONS, EXTRACTION_FIELDS
print(f'✅ Total fields: {len(EXTRACTION_FIELDS)}')
print(f'✅ New field added: {\"BUSINESS_PURPOSE\" in EXTRACTION_FIELDS}')
"
```

#### Step 3: Update Ground Truth
```csv
# Add column to evaluation_ground_truth.csv
image_file,ABN,ACCOUNT_HOLDER,...,BUSINESS_PURPOSE,TOTAL
invoice_001.png,04904754234,N/A,...,Client meeting,$156.90
```

#### Step 4: Test with Sample
```bash
# Test with small dataset
python llama_keyvalue.py
# Check CSV output includes new field
```

#### Step 5: Full Validation
```bash
# Run complete evaluation
python llama_keyvalue.py
python internvl3_keyvalue.py
# Verify reports include new field statistics
```

### Modifying an Existing Field

**Use Case**: Changing field instructions, type, or requirements.

#### Step 1: Backup Current Configuration
```bash
cp common/config.py common/config.py.backup.$(date +%Y%m%d_%H%M%S)
```

#### Step 2: Make Changes
```python
# Example: Making GST field required and updating instruction
'GST': {
    'type': 'monetary',
    'instruction': '[GST amount in dollars (required for Australian invoices) or N/A]',
    'evaluation_logic': 'monetary_with_tolerance',
    'description': 'Goods and Services Tax amount',
    'required': True  # Changed from False
}
```

#### Step 3: Validate Impact
```python
# Check required fields list
from common.config import REQUIRED_FIELDS
print(f"Required fields: {REQUIRED_FIELDS}")
```

#### Step 4: Update Ground Truth (if needed)
```bash
# If field type changed, may need to update ground truth format
# If field became required, ensure no N/A values in ground truth
```

#### Step 5: Test and Deploy
```bash
python llama_keyvalue.py
# Check that field behaves as expected
```

### Removing a Field

**Use Case**: Eliminating a field that is no longer needed.

#### Step 1: Backup Configuration
```bash
cp common/config.py common/config.py.backup.$(date +%Y%m%d_%H%M%S)
```

#### Step 2: Remove from FIELD_DEFINITIONS
```python
# Delete the field entry completely
FIELD_DEFINITIONS = {
    'ABN': { ... },
    # Remove this line:
    # 'DEPRECATED_FIELD': { ... },
    'TOTAL': { ... }
}
```

#### Step 3: Update Ground Truth CSV
```bash
# Remove corresponding column from evaluation_ground_truth.csv
# Can keep column for historical analysis but not required
```

#### Step 4: Validate Removal
```python
from common.config import EXTRACTION_FIELDS
print(f"Remaining fields: {len(EXTRACTION_FIELDS)}")
print(f"Deprecated field removed: {'DEPRECATED_FIELD' not in EXTRACTION_FIELDS}")
```

#### Step 5: Test Complete Pipeline
```bash
python llama_keyvalue.py
python internvl3_keyvalue.py
# Verify CSV outputs have correct column count
```

---

## Common Workflows

### Workflow 1: Converting from Business to Tax Document Fields

**Scenario**: Switching entire field set from business documents to tax document focus.

#### Planning Phase
1. **Document Current State**
```bash
python -c "
from common.config import EXTRACTION_FIELDS
print('Current fields:', len(EXTRACTION_FIELDS))
for field in EXTRACTION_FIELDS[:10]:
    print(f'  {field}')
print('  ...')
"
```

2. **Design New Field Set**
```python
# Design tax-specific fields
tax_fields = [
    'BUSINESS_PURPOSE',
    'CLAIMANT_NAME', 
    'DATE_OF_EXPENSE',
    'EXPENSE_AMOUNT',
    'EXPENSE_CATEGORY',
    'GST_AMOUNT',
    'PAYMENT_METHOD',
    'RECEIPT_NUMBER',
    'SUPPLIER_NAME',
    'TAX_DEDUCTIBLE_AMOUNT'
]
```

#### Implementation Phase
1. **Backup Current System**
```bash
cp common/config.py common/config.py.business_docs.backup
cp evaluation_data/evaluation_ground_truth.csv evaluation_data/business_ground_truth.backup.csv
```

2. **Replace FIELD_DEFINITIONS**
```python
# In common/config.py - completely replace FIELD_DEFINITIONS
FIELD_DEFINITIONS = {
    'BUSINESS_PURPOSE': {
        'type': 'text',
        'instruction': '[business reason for this expense or N/A]',
        'evaluation_logic': 'fuzzy_text_match',
        'description': 'Business justification for expense deduction',
        'required': True
    },
    'CLAIMANT_NAME': {
        'type': 'text',
        'instruction': '[name of person claiming expense or N/A]',
        'evaluation_logic': 'fuzzy_text_match',
        'description': 'Person or entity claiming the expense',
        'required': True
    },
    # ... continue with all tax fields
}
```

3. **Create New Ground Truth**
```csv
# Create new evaluation_ground_truth.csv with tax fields
image_file,BUSINESS_PURPOSE,CLAIMANT_NAME,DATE_OF_EXPENSE,EXPENSE_AMOUNT,EXPENSE_CATEGORY,GST_AMOUNT,PAYMENT_METHOD,RECEIPT_NUMBER,SUPPLIER_NAME,TAX_DEDUCTIBLE_AMOUNT
receipt_001.png,Client meeting,John Smith,15/03/2024,$89.50,Travel,$8.95,Credit Card,R12345,Uber,$89.50
```

#### Validation Phase
1. **Configuration Validation**
```bash
python -c "
from common.config import EXTRACTION_FIELDS, FIELD_COUNT
print(f'New field count: {FIELD_COUNT}')
print('Tax fields loaded successfully')
"
```

2. **Pipeline Testing**
```bash
# Test with small sample
python llama_keyvalue.py
python internvl3_keyvalue.py
```

3. **Output Verification**
```bash
# Check CSV structure
head -1 output/llama_batch_extraction_*.csv
# Should show tax field column headers
```

### Workflow 2: Adding Field Type with Custom Evaluation

**Scenario**: Adding a new field type that requires custom evaluation logic.

#### Step 1: Define New Field Type
```python
# Add percentage field type
'CLAIM_PERCENTAGE': {
    'type': 'percentage',  # New type
    'instruction': '[percentage claimable (e.g., 50%, 100%) or N/A]',
    'evaluation_logic': 'percentage_match',  # New logic
    'description': 'Percentage of expense that is tax deductible',
    'required': False
}
```

#### Step 2: Update Validation Rules
```python
# In validate_field_definitions() function
valid_types = ['numeric_id', 'monetary', 'date', 'list', 'text', 'percentage']
valid_evaluation_logic = [
    'exact_numeric_match', 'monetary_with_tolerance', 'flexible_date_match',
    'list_overlap_match', 'fuzzy_text_match', 'percentage_match'
]
```

#### Step 3: Add Field Type Grouping
```python
# In derived configurations section
PERCENTAGE_FIELDS = [k for k, v in FIELD_DEFINITIONS.items() if v['type'] == 'percentage']
```

#### Step 4: Implement Evaluation Logic
```python
# In common/evaluation_utils.py - add import
from .config import (
    # ... existing imports ...
    PERCENTAGE_FIELDS,
)

# In calculate_field_accuracy() function - add new case
elif field_name in PERCENTAGE_FIELDS:
    # Custom percentage evaluation logic
    try:
        # Extract numeric percentage (50% -> 50)
        extracted_pct = float(re.sub(r'[^\d.]', '', extracted))
        ground_truth_pct = float(re.sub(r'[^\d.]', '', ground_truth))
        
        # Allow 5% tolerance for percentages
        tolerance = 5.0
        return 1.0 if abs(extracted_pct - ground_truth_pct) <= tolerance else 0.0
    except (ValueError, AttributeError):
        return 0.0
```

#### Step 5: Test New Field Type
```python
# Test percentage evaluation
from common.evaluation_utils import calculate_field_accuracy
accuracy = calculate_field_accuracy("50%", "50%", "CLAIM_PERCENTAGE")
print(f"Percentage field accuracy: {accuracy}")
```

### Workflow 3: Field Instruction Optimization

**Scenario**: Improving field instructions based on model performance analysis.

#### Step 1: Analyze Current Performance
```bash
# Run evaluation and check field accuracies
python llama_keyvalue.py
# Look for fields with low accuracy in reports
```

#### Step 2: Identify Problematic Instructions
```python
# Check current instructions for low-performing fields
from common.config import FIELD_INSTRUCTIONS
problem_fields = ['DESCRIPTIONS', 'BUSINESS_ADDRESS']  # Example low performers
for field in problem_fields:
    print(f"{field}: {FIELD_INSTRUCTIONS[field]}")
```

#### Step 3: Improve Instructions
```python
# Make instructions more specific and actionable
'DESCRIPTIONS': {
    'type': 'list',
    'instruction': '[list of item descriptions separated by commas or N/A]',  # Was generic
    'evaluation_logic': 'list_overlap_match',
    'description': 'Transaction or item descriptions',
    'required': False
},
'BUSINESS_ADDRESS': {
    'type': 'text', 
    'instruction': '[complete business address including street, suburb, postcode or N/A]',  # More specific
    'evaluation_logic': 'fuzzy_text_match',
    'description': 'Physical address of business',
    'required': False
}
```

#### Step 4: A/B Testing
```bash
# Test improved instructions
python llama_keyvalue.py > results_improved.txt
python internvl3_keyvalue.py >> results_improved.txt

# Compare with baseline (from backup results)
# Look for improvements in problematic field accuracies
```

#### Step 5: Iterative Refinement
```python
# Continue refining based on results
# Document what works and what doesn't
# Keep track of instruction evolution
```

---

## Field Migration Procedures

### Migration Planning

#### 1. Impact Assessment
```bash
# Assess scope of changes
echo "Current field count: $(python -c 'from common.config import FIELD_COUNT; print(FIELD_COUNT)')"
echo "Fields being changed: [list here]"
echo "Ground truth rows affected: [estimate]"
echo "Evaluation history to preserve: [check]"
```

#### 2. Rollback Preparation
```bash
# Create comprehensive backup
mkdir -p backups/$(date +%Y%m%d_%H%M%S)
cp common/config.py backups/$(date +%Y%m%d_%H%M%S)/
cp evaluation_data/evaluation_ground_truth.csv backups/$(date +%Y%m%d_%H%M%S)/
cp -r output/ backups/$(date +%Y%m%d_%H%M%S)/output_backup/
```

#### 3. Testing Strategy
```python
# Define test cases
test_cases = [
    "Single field addition",
    "Field type change", 
    "Required field modification",
    "Field removal",
    "Complete field set replacement"
]
```

### Staged Migration Process

#### Stage 1: Development Testing
```bash
# Test on small sample
# Use subset of images (3-5 samples)
# Verify basic functionality
python -c "
# Test with minimal data
sample_images = ['invoice_001.png', 'invoice_002.png']
print(f'Testing with {len(sample_images)} images')
"
```

#### Stage 2: Integration Testing
```bash
# Test complete pipeline
# Use full dataset
# Check all output formats
python llama_keyvalue.py
python internvl3_keyvalue.py

# Verify outputs
ls -la output/
echo "Checking CSV structure..."
head -1 output/*_batch_extraction_*.csv
```

#### Stage 3: Production Deployment
```bash
# Deploy to production environment
# Monitor for issues
# Have rollback ready
```

### Migration Validation Checklist

#### Pre-Migration
- [ ] Current configuration backed up
- [ ] Ground truth data backed up
- [ ] Test plan documented
- [ ] Rollback procedure prepared
- [ ] Stakeholders notified

#### During Migration
- [ ] Configuration validation passes
- [ ] Test samples process correctly
- [ ] Output formats correct
- [ ] Field count matches expectations
- [ ] Evaluation logic works properly

#### Post-Migration
- [ ] Full pipeline runs successfully
- [ ] Output quality maintained or improved
- [ ] All stakeholders updated
- [ ] Documentation updated
- [ ] Monitoring confirms stability

---

## Quality Assurance

### Automated Testing

#### Configuration Testing
```python
# Create test script: test_field_config.py
def test_field_definitions():
    """Test that field configuration is valid."""
    from common.config import FIELD_DEFINITIONS, validate_field_definitions
    
    # This will raise exception if invalid
    validate_field_definitions()
    print("✅ Field definitions valid")

def test_field_derivations():
    """Test that derived configurations are correct."""
    from common.config import (
        EXTRACTION_FIELDS, FIELD_INSTRUCTIONS, FIELD_TYPES,
        MONETARY_FIELDS, DATE_FIELDS, FIELD_DEFINITIONS
    )
    
    # Test counts match
    assert len(EXTRACTION_FIELDS) == len(FIELD_DEFINITIONS)
    assert len(FIELD_INSTRUCTIONS) == len(FIELD_DEFINITIONS)
    assert len(FIELD_TYPES) == len(FIELD_DEFINITIONS)
    
    # Test field type groupings
    expected_monetary = len([f for f, d in FIELD_DEFINITIONS.items() if d['type'] == 'monetary'])
    assert len(MONETARY_FIELDS) == expected_monetary
    
    print("✅ Field derivations correct")

if __name__ == "__main__":
    test_field_definitions()
    test_field_derivations()
```

#### Evaluation Testing
```python
# Create test script: test_evaluation.py
def test_evaluation_logic():
    """Test evaluation logic with known cases."""
    from common.evaluation_utils import calculate_field_accuracy
    
    test_cases = [
        # (extracted, ground_truth, field_name, expected_accuracy)
        ("$123.45", "$123.45", "TOTAL", 1.0),
        ("$123.44", "$123.45", "TOTAL", 1.0),  # Within tolerance
        ("15/03/2024", "15-03-2024", "INVOICE_DATE", 1.0),  # Date format
        ("N/A", "N/A", "ABN", 1.0),  # N/A handling
        ("12345678901", "12-345-678-901", "ABN", 1.0),  # Numeric ID
    ]
    
    for extracted, ground_truth, field_name, expected in test_cases:
        accuracy = calculate_field_accuracy(extracted, ground_truth, field_name)
        assert abs(accuracy - expected) < 0.01, f"Failed: {field_name}"
        print(f"✅ {field_name}: {accuracy}")
    
    print("✅ Evaluation logic tests passed")

if __name__ == "__main__":
    test_evaluation_logic()
```

### Manual Testing Procedures

#### 1. Spot Checking
```bash
# Randomly select samples for manual verification
python -c "
import random
from common.config import EXTRACTION_FIELDS

sample_fields = random.sample(EXTRACTION_FIELDS, 3)
print('Manually verify these fields:')
for field in sample_fields:
    print(f'  {field}')
"
```

#### 2. Edge Case Testing
```python
# Test edge cases for each field type
edge_cases = {
    'monetary': ['$0.00', '$999,999.99', '0', 'N/A'],
    'date': ['01/01/2000', '31/12/2099', 'N/A'],
    'text': ['', 'Very long business name with special chars !@#', 'N/A'],
    'numeric_id': ['00000000000', '99999999999', 'N/A'],
    'list': ['Single item', 'Item 1, Item 2, Item 3', 'N/A']
}
```

#### 3. Cross-Model Consistency
```bash
# Compare outputs between models
python llama_keyvalue.py > llama_results.txt 2>&1
python internvl3_keyvalue.py > internvl3_results.txt 2>&1

# Check for major inconsistencies in structure
diff llama_results.txt internvl3_results.txt | head -20
```

### Performance Monitoring

#### Accuracy Tracking
```python
def track_field_performance():
    """Monitor field accuracy trends."""
    import json
    from datetime import datetime
    
    # Load latest evaluation results
    # Compare with historical performance
    # Alert if significant degradation
    pass
```

#### Processing Performance
```bash
# Monitor processing times
time python llama_keyvalue.py
time python internvl3_keyvalue.py

# Track memory usage
/usr/bin/time -v python llama_keyvalue.py 2>&1 | grep "Maximum resident set size"
```

---

## Production Deployment

### Deployment Checklist

#### Pre-Deployment
- [ ] All tests passing
- [ ] Configuration validated
- [ ] Ground truth updated
- [ ] Backup procedures in place
- [ ] Rollback plan ready
- [ ] Stakeholder approval obtained

#### Deployment Steps
1. **Backup Production Environment**
```bash
# Backup production config and data
scp user@prod-server:/path/to/config.py config.py.prod.backup
scp user@prod-server:/path/to/ground_truth.csv ground_truth.prod.backup.csv
```

2. **Deploy Configuration**
```bash
# Copy new configuration
scp common/config.py user@prod-server:/path/to/common/config.py
scp evaluation_data/evaluation_ground_truth.csv user@prod-server:/path/to/evaluation_data/
```

3. **Validation Testing**
```bash
# Run validation on production server
ssh user@prod-server "cd /path/to/project && python -c 'from common.config import validate_field_definitions; validate_field_definitions()'"
```

4. **Production Testing**
```bash
# Run limited production test
ssh user@prod-server "cd /path/to/project && python llama_keyvalue.py | head -20"
```

#### Post-Deployment
- [ ] Production tests passing
- [ ] Output quality verified
- [ ] Monitoring alerts configured
- [ ] Documentation updated
- [ ] Team notified

### Monitoring Setup

#### Health Checks
```python
def production_health_check():
    """Verify production system health."""
    try:
        from common.config import FIELD_DEFINITIONS, EXTRACTION_FIELDS
        
        # Basic configuration checks
        assert len(EXTRACTION_FIELDS) > 0
        assert len(FIELD_DEFINITIONS) == len(EXTRACTION_FIELDS)
        
        # Test evaluation logic
        from common.evaluation_utils import calculate_field_accuracy
        test_accuracy = calculate_field_accuracy("$100", "$100", "TOTAL")
        assert test_accuracy == 1.0
        
        return {"status": "healthy", "field_count": len(EXTRACTION_FIELDS)}
    
    except Exception as e:
        return {"status": "error", "error": str(e)}
```

#### Performance Alerts
```bash
# Set up monitoring for:
# - Processing time increases
# - Accuracy degradation  
# - Error rate spikes
# - Memory usage growth
```

---

## Rollback Procedures

### When to Rollback

**Immediate Rollback Triggers**:
- Configuration validation fails
- Evaluation pipeline crashes
- Major accuracy degradation (>10%)
- Missing required fields in output
- Ground truth loading errors

### Rollback Execution

#### 1. Identify Issue
```bash
# Check logs for errors
tail -50 /path/to/logs/evaluation.log

# Check configuration status
python -c "
try:
    from common.config import validate_field_definitions
    validate_field_definitions()
    print('Config OK')
except Exception as e:
    print(f'Config Error: {e}')
"
```

#### 2. Restore Configuration
```bash
# Restore from backup
cp common/config.py.backup common/config.py
cp evaluation_data/evaluation_ground_truth.csv.backup evaluation_data/evaluation_ground_truth.csv
```

#### 3. Validate Rollback
```bash
# Test restored configuration
python -c "
from common.config import FIELD_DEFINITIONS
print(f'Restored field count: {len(FIELD_DEFINITIONS)}')
"

# Run quick test
python llama_keyvalue.py | head -10
```

#### 4. Verify System Health
```bash
# Full pipeline test
python llama_keyvalue.py
python internvl3_keyvalue.py

# Check outputs
ls -la output/
echo "System restored successfully"
```

### Post-Rollback Analysis

#### 1. Root Cause Analysis
```python
# Analyze what went wrong
failure_analysis = {
    "failure_time": "timestamp",
    "symptoms": ["list of issues observed"],
    "root_cause": "identified cause",
    "fix_required": "what needs to be changed"
}
```

#### 2. Prevention Measures
- Improved testing procedures
- Enhanced validation checks
- Better rollback automation
- Additional monitoring

---

## Monitoring and Maintenance

### Regular Maintenance Tasks

#### Weekly Tasks
- [ ] Review field accuracy trends
- [ ] Check for processing errors
- [ ] Validate ground truth integrity
- [ ] Monitor resource usage

#### Monthly Tasks  
- [ ] Analyze field performance patterns
- [ ] Review and update field instructions
- [ ] Backup configuration and data
- [ ] Test disaster recovery procedures

#### Quarterly Tasks
- [ ] Comprehensive accuracy review
- [ ] Field set optimization analysis
- [ ] Documentation updates
- [ ] Team training updates

### Maintenance Scripts

#### Configuration Audit
```python
def audit_configuration():
    """Audit current configuration for issues."""
    from common.config import FIELD_DEFINITIONS, EXTRACTION_FIELDS
    
    issues = []
    
    # Check for unused field types
    used_types = set(d['type'] for d in FIELD_DEFINITIONS.values())
    print(f"Field types in use: {used_types}")
    
    # Check instruction consistency
    for field, definition in FIELD_DEFINITIONS.items():
        instruction = definition['instruction']
        if not instruction.startswith('[') or not instruction.endswith(']'):
            issues.append(f"Bad instruction format: {field}")
        if 'N/A' not in instruction:
            issues.append(f"Missing N/A in instruction: {field}")
    
    return issues
```

#### Performance Monitoring
```python
def monitor_performance():
    """Monitor system performance trends."""
    import glob
    import json
    
    # Collect recent evaluation results
    result_files = glob.glob("output/*_evaluation_results_*.json")
    result_files.sort()
    
    # Track accuracy trends over time
    accuracies = []
    for file in result_files[-10:]:  # Last 10 results
        with open(file) as f:
            data = json.load(f)
            accuracies.append(data.get('overall_accuracy', 0))
    
    # Alert if declining trend
    if len(accuracies) > 3:
        recent_avg = sum(accuracies[-3:]) / 3
        earlier_avg = sum(accuracies[-6:-3]) / 3 if len(accuracies) >= 6 else recent_avg
        
        if recent_avg < earlier_avg - 0.05:  # 5% degradation
            print(f"⚠️ Performance degradation detected: {recent_avg:.3f} vs {earlier_avg:.3f}")
    
    return accuracies
```

---

## Conclusion

Effective field management requires systematic procedures, thorough testing, and careful monitoring. Key principles:

- **Plan Thoroughly**: Always assess impact before making changes
- **Test Extensively**: Validate changes at multiple stages
- **Backup Everything**: Always have rollback options ready  
- **Monitor Continuously**: Track performance and health metrics
- **Document Changes**: Maintain clear records of what changed and why
- **Automate When Possible**: Reduce manual error with automation

Following these workflows ensures reliable, maintainable field management that supports the overall goals of the LMM_POC evaluation framework.