# InternVL3-8B Debug Fixes

## Issue Analysis
The H200 environment error shows `embedding(): argument 'indices' (position 2) must be Tensor, not NoneType`, indicating the tokenizer is returning None instead of proper token indices when processing the input text.

## Root Cause
The ResilientGenerator integration with InternVL3's chat interface had compatibility issues where the tokenizer state or input formatting was causing tokenization to fail.

## Fixes Applied

### 1. Input Validation & Tokenizer Testing
```python
# Test tokenizer functionality before use
try:
    test_tokens = self.tokenizer("Test", return_tensors="pt")
    if test_tokens["input_ids"] is None:
        raise ValueError("Tokenizer returning None for input_ids")
except Exception as tokenizer_error:
    print(f"⚠️ Tokenizer validation failed: {tokenizer_error}")
```

### 2. ResilientGenerator Fallback
```python
# Generate with resilient fallback strategies
try:
    response = self.resilient_generator.generate(inputs, **self.generation_config)
except Exception as resilient_error:
    print(f"⚠️ ResilientGenerator failed: {resilient_error}")
    print("🔄 Falling back to direct chat method...")
    
    # Fallback to direct chat method
    response = self.model.chat(
        self.tokenizer, pixel_values, question, 
        self.generation_config, history=None, return_history=False
    )
```

### 3. Emergency Reload Reference Updates
```python
# Update our instance references to the new model and tokenizer
self.model = model
self.tokenizer = tokenizer

# Update the ResilientGenerator's references as well
if self.resilient_generator:
    self.resilient_generator.model = model
    self.resilient_generator.processor = tokenizer
```

### 4. Enhanced Debug Information
- Added input validation before ResilientGenerator calls
- Added tokenizer functionality testing
- Improved error messages and fallback reporting

## Testing
Run the test script to verify fixes:
```bash
python test_internvl3_fix.py
```

This will test:
1. Tokenizer functionality
2. Direct chat method
3. ResilientGenerator with fallback

## Expected Behavior
1. **If ResilientGenerator works**: Normal processing with all optimizations
2. **If ResilientGenerator fails**: Automatic fallback to direct chat method
3. **If both fail**: Clear error messages for further debugging

## Files Modified
- `models/internvl3_processor.py`: Main fixes and fallbacks
- `common/gpu_optimization.py`: Enhanced error handling
- `test_internvl3_fix.py`: Test script for validation

## Next Steps
1. Run test script on H200 to validate fixes
2. If tests pass, run full evaluation with `python internvl3_keyvalue.py`
3. Monitor for any remaining tokenizer or memory issues
4. Remove debug prints once stable