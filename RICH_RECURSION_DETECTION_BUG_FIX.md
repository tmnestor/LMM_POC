# Rich Console Recursion Bug - Document Detection Fix

## Issue Summary

**Duration**: 3 weeks of debugging
**Impact**: Complete failure of document type detection, causing all documents to be misclassified as "INVOICE"
**Root Cause**: Rich console recursion error triggered by specific content during document detection
**Status**: ✅ RESOLVED

## Problem Description

### Symptoms
- All documents (receipts, bank statements, invoices) were being classified as "INVOICE"
- No visible errors in standard output
- Detection appeared to work but always returned fallback classification
- Debugging output was missing critical detection responses

### User Reports
```
"NO it is detecting INVOICE for all documents"
"we need to use the 'detection' key. The 'detection_simple' causes repeated responses."
"spreading config across the codebase is a 'code-smell'!!!!"
```

### Investigation Timeline
1. **Week 1**: Suspected prompt configuration issues (`detection_simple` vs `detection`)
2. **Week 2**: Investigated configuration management and single source of truth
3. **Week 3**: Discovered actual root cause was console rendering bug

## Root Cause Analysis

### The Hidden Error
The issue was a **Rich console RecursionError** that was silently failing the detection method:

```python
RecursionError: maximum recursion depth exceeded while calling a Python object

File "/home/jovyan/nfs_share/tod/LMM_POC/models/document_aware_internvl3_processor.py", line 430, in detect_and_classify_document
    print(f"📝 Prompt: {detection_prompt[:100]}...")
```

### Why It Was Hidden
- Exception was caught and fallback logic returned "INVOICE"
- Rich console error didn't surface to user
- Made it appear like a logic/configuration issue rather than infrastructure bug

### Technical Details
Rich console hit infinite recursion in regex compilation when processing content containing special markup characters (`[` and `]` brackets) in the detection prompt content.

## Solution Implementation

### Approach: Content Sanitization
Rather than abandoning Rich (used throughout codebase), we sanitized problematic content:

```python
# BEFORE (caused recursion)
if verbose:
    print(f"📝 Prompt: {detection_prompt[:100]}...")

# AFTER (sanitized for Rich)
if verbose:
    safe_prompt = detection_prompt[:100].replace('[', '\\[').replace(']', '\\]')
    print(f"📝 Prompt: {safe_prompt}...")
```

### Files Modified
- `/models/document_aware_internvl3_processor.py`
  - `detect_and_classify_document()` method
  - `_parse_document_type_response()` method
  - Error handling sections

### Changes Made
1. **Escape Rich markup characters**: Replace `[` with `\\[` and `]` with `\\]`
2. **Truncate long content**: Limit response display to prevent overwhelming output
3. **Preserve Rich functionality**: Keep using `print()` throughout codebase
4. **Enhanced error visibility**: Ensure detection errors are always shown

## Results

### Before Fix
```
Processing [1/3]: image_003.png
🔧 CONFIG DEBUG - Using prompt_config: detection_key='detection'
🔍 Using InternVL3 document detection prompt: detection
[CRASH - RecursionError]
✅ Detected document type: INVOICE  # Fallback
```

### After Fix
```
Processing [1/3]: image_003.png
🔧 CONFIG DEBUG - Using prompt_config: detection_key='detection'
🔍 Using InternVL3 document detection prompt: detection
📝 Prompt: What type of business document is this?...
🤖 Model response: BANK_STATEMENT
✅ Detected document type: BANK_STATEMENT
```

### Impact
- ✅ Documents now correctly classified (RECEIPT, BANK_STATEMENT, INVOICE)
- ✅ Full debug visibility restored
- ✅ Rich formatting preserved across codebase
- ✅ No more silent detection failures

## Technical Lessons Learned

### 1. Infrastructure Bugs Can Mask Logic Issues
- Spent weeks debugging prompt configuration and logic
- Real issue was console rendering library bug
- Always check for hidden exceptions in fallback paths

### 2. Rich Console Content Safety
- Rich interprets `[` and `]` as markup syntax
- Content from external sources (YAML files, model responses) can trigger rendering bugs
- Always sanitize dynamic content before Rich rendering

### 3. Exception Handling Visibility
- Silent fallbacks can hide critical errors
- Always log exceptions, even in fallback scenarios
- Consider fail-fast vs graceful degradation tradeoffs

### 4. Multi-Layer Debugging
- Surface-level symptoms (wrong classifications)
- Mid-level investigation (prompt configuration)
- Deep-level root cause (console rendering)

## Prevention Strategies

### 1. Content Sanitization Helper
```python
def sanitize_for_rich(content: str, max_length: int = 200) -> str:
    """Sanitize content for safe Rich console rendering."""
    if not content:
        return content

    # Escape Rich markup characters
    safe_content = content.replace('[', '\\[').replace(']', '\\]')

    # Truncate if too long
    if len(safe_content) > max_length:
        safe_content = safe_content[:max_length] + "..."

    return safe_content
```

### 2. Enhanced Error Visibility
```python
except Exception as e:
    # ALWAYS show detection errors - critical for debugging
    print(f"❌ DETECTION ERROR: {e}")
    if self.debug:
        import traceback
        print("❌ DETECTION ERROR TRACEBACK:")
        traceback.print_exc()
    # ... fallback logic
```

### 3. Rich Console Testing
- Test Rich console with realistic content from YAML files
- Include special characters in test cases
- Monitor for recursion patterns in CI/CD

## Extended Investigation - Additional Rich Recursion Sources

### Secondary Issue: Model Response Content Recursion
After resolving the detection prompt issue, **additional Rich recursion sources were discovered**:

#### Model Response Patterns Causing Recursion
- **Repetitive content**: Model generating hundreds of repeated date entries (`27/08/2025` repeated 100+ times)
- **Large content blocks**: Extremely long field values (e.g., `TRANSACTION_DATES` with excessive entries)
- **Combined content size**: Multiple large fields in single response overwhelming Rich console

#### Example Problematic Model Output
```
TRANSACTION_DATES: 07/09/2025 | 08/09/2025 | 09/03/2025 | 03/09/2025 | 02/09/2025 | 01/09/2025 | 01/08/2025 |
01/08/2025 | 01/08/2025 | 20/08/2025 | 20/08/2025 | 20/08/2025 | 20/08/2025 | 27/08/2025 | 27/08/2025 | 27/08/2025
| 27/08/2025 | 27/08/2025 | 27/08/2025 | [... 50+ more repeated 27/08/2025 entries]
```

### Additional Fixes Applied

#### 1. Universal Rich Sanitization Function
Added comprehensive sanitization helper in `common/extraction_cleaner.py`:

```python
def sanitize_for_rich(content: str, max_length: int = 200) -> str:
    """
    Sanitize content for safe Rich console rendering.

    Rich console interprets '[' and ']' as markup syntax, which can cause
    RecursionError with certain content patterns. This function escapes
    those characters and truncates long content.
    """
    if not content:
        return content

    # Escape Rich markup characters that cause recursion
    safe_content = str(content).replace('[', '\\[').replace(']', '\\]')

    # Truncate if too long to prevent overwhelming output
    if len(safe_content) > max_length:
        safe_content = safe_content[:max_length] + "..."

    return safe_content
```

#### 2. Extraction Cleaner Debug Output
**Fixed**: `/common/extraction_cleaner.py` debug prints now use sanitization:

```python
# BEFORE - caused recursion with long content
if self.debug:
    print(f"🧹 CLEANER CALLED: {field_name}: '{raw_value}' -> ", end="")
    print(f"'{cleaned_value}'")

# AFTER - sanitized for Rich safety
if self.debug:
    safe_raw_value = sanitize_for_rich(str(raw_value), max_length=100)
    print(f"🧹 CLEANER CALLED: {field_name}: '{safe_raw_value}' -> ", end="")
    safe_cleaned_value = sanitize_for_rich(str(cleaned_value), max_length=100)
    print(f"'{safe_cleaned_value}'")
```

#### 3. Batch Processor Model Response Display
**Fixed**: `/common/batch_processor.py` verbose response output:

```python
# BEFORE - could cause recursion with large responses
if verbose:
    rprint(f"[yellow]Model Response:[/yellow] {response}")

# AFTER - sanitized and truncated
if verbose:
    safe_response = sanitize_for_rich(response, max_length=500)
    rprint(f"[yellow]Model Response:[/yellow] {safe_response}")
```

### Root Cause Analysis: Two-Layer Problem

#### Layer 1: Model Generation Issue
- **InternVL3 repetition**: Model generating excessive repeated content
- **Content length**: Single fields exceeding thousands of characters
- **Pattern recognition failure**: Model losing context and repeating patterns

#### Layer 2: Rich Console Rendering Limits
- **Markup interpretation**: Rich parsing `[` and `]` characters in content
- **Recursion threshold**: Large content blocks hitting internal regex limits
- **Memory exhaustion**: Console rendering running out of stack space

### Impact Summary

#### Before Complete Fix
- ❌ Detection failures (all documents classified as INVOICE)
- ❌ Cleaner recursion on long field values
- ❌ Batch processor recursion on verbose model responses
- ❌ Silent failures masking underlying issues

#### After Complete Fix
- ✅ Document detection working correctly
- ✅ Safe handling of large model responses
- ✅ Cleaner debug output functioning reliably
- ✅ All Rich console outputs sanitized and limited
- ✅ Comprehensive error visibility maintained

## Related Issues

### Similar Patterns to Watch For
- Any dynamic content from YAML files displayed via Rich
- Model responses containing markdown-like syntax
- File paths or URLs with bracket characters
- JSON content displayed for debugging
- **NEW**: Repetitive model-generated content (dates, transactions, etc.)
- **NEW**: Large field values from document extraction
- **NEW**: Batch processing verbose outputs

### Code Locations Using Rich
- Batch processing progress output ✅ **FIXED**
- Debug logging throughout pipeline ✅ **FIXED**
- Error reporting and status messages ✅ **PROTECTED**
- Configuration display and validation ✅ **PROTECTED**
- Extraction cleaner debug output ✅ **FIXED**
- Model response display in verbose mode ✅ **FIXED**

## Summary

A 3-week debugging journey revealed that document detection failures were not due to prompt configuration or logic errors, but rather a Rich console recursion bug triggered by specific content. The fix preserves Rich functionality while ensuring reliable document classification through content sanitization.

**Key Takeaway**: Infrastructure bugs can present as domain logic issues. Always verify the execution path completes successfully before debugging the logic itself.