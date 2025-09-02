# Risk Assessment: llama_keyvalue.py vs llama_document_aware.py

## Current Situation

You have **two parallel implementations**:
1. **llama_document_aware.py** (currently working)
2. **llama_keyvalue.py** (moved from legacy/, status unknown)

## Dependency Analysis

### llama_keyvalue.py Dependencies
```python
✅ models.llama_processor (exists)
✅ common.config (exists)
✅ common.evaluation_metrics (exists)
✅ common.extraction_parser (exists)
❌ common.reporting (MISSING - likely in legacy/)
```

### llama_document_aware.py Dependencies
```python
✅ models.document_aware_llama_processor (exists, working)
✅ common.unified_schema (exists)
✅ common.document_type_metrics (exists)
✅ common.evaluation_metrics (exists)
✅ common.extraction_parser (exists)
```

## Critical Risks

### 🔴 HIGH RISK: Missing Dependencies
- `common.reporting` module is missing (needed by llama_keyvalue.py)
- This module is likely in `legacy/common/`
- Without it, llama_keyvalue.py will fail immediately on import

### 🔴 HIGH RISK: Configuration Mismatch
- `llama_keyvalue.py` uses `models.llama_processor` → `prompts/llama_single_pass_v4.yaml`
- `llama_document_aware.py` uses `document_aware_llama_processor` → `config/unified_schema.yaml`
- **Different prompt configurations = Different results**

### 🟡 MEDIUM RISK: Field Schema Differences
- llama_keyvalue.py: Fixed 47-49 fields (V4 schema)
- llama_document_aware.py: Dynamic 5-11 fields (BOSS reduction)
- **Performance and accuracy implications**

### 🟡 MEDIUM RISK: Untested State
- llama_keyvalue.py was moved to legacy for a reason
- No guarantee it works with current model paths or data
- May have breaking changes since it was archived

## Safe Testing Plan

### Option 1: Minimal Risk Test (RECOMMENDED)
1. **DO NOT move reporting.py back yet**
2. Create a test script to check imports:
```python
# test_llama_keyvalue.py
try:
    from models.llama_processor import LlamaProcessor
    print("✅ llama_processor imports successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")

try:
    from common.reporting import generate_comprehensive_reports
    print("✅ reporting module found")
except ImportError as e:
    print(f"❌ Missing reporting module: {e}")
```

3. If imports fail, you know it won't work without more migration

### Option 2: Dry Run Test
1. Copy (don't move) `reporting.py` from legacy to common temporarily
2. Run llama_keyvalue.py with `--limit-images 1` flag
3. Compare output with llama_document_aware.py on same image
4. Delete the copied reporting.py if test fails

### Option 3: Safe Parallel Testing
1. Keep both systems running
2. Create a comparison script that runs both on same data
3. Only migrate after confirming identical/better results

## Recommendation: STAY WITH llama_document_aware.py

### Why:
1. **It's currently working** - Don't fix what isn't broken
2. **Better architecture** - Document-aware, dynamic fields
3. **Better performance** - 50-75% faster with field reduction
4. **Cleaner dependencies** - No missing modules

### If You Must Use llama_keyvalue.py:
1. First recover ALL dependencies from legacy/:
   - `common/reporting.py`
   - Any other missing imports
2. Test thoroughly on small dataset first
3. Keep llama_document_aware.py as backup
4. Document exactly which version works

## The Safest Path Forward

### Immediate Actions:
1. **KEEP using llama_document_aware.ipynb** - it works!
2. **Document the working configuration** before any changes
3. **Create backups** of working code

### Long-term Solution:
1. **Consolidate to ONE implementation** (as per LLAMA_PROCESSOR_COMPARISON.md)
2. **Use unified_schema.yaml** as single source of truth
3. **Migrate best features** from both implementations
4. **Delete legacy code** once consolidated version is proven

## Bottom Line

**DO NOT SWITCH** from llama_document_aware.py to llama_keyvalue.py unless you:
1. Have a compelling reason (what's broken that needs fixing?)
2. Have fully tested llama_keyvalue.py works
3. Have benchmarked both for accuracy/performance
4. Have a rollback plan

The risk of breaking a working system is too high for uncertain benefits.