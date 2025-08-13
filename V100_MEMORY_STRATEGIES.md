# V100 Memory Management Strategies

This document outlines non-quantization strategies to prevent V100 VRAM overload on the second image processing.

## Problem Analysis
The "second image overload" pattern suggests memory accumulation rather than insufficient capacity. The first image processes successfully, but accumulated memory causes the second to fail.

## Strategy 1: Disable KV Caching (Highest Impact)

**Issue**: KV cache grows with each generation and persists between images
**Solution**: Disable caching entirely for V100

```python
# In common/config.py - Modify LLAMA_GENERATION_CONFIG
LLAMA_GENERATION_CONFIG = {
    "max_new_tokens_base": 800,
    "max_new_tokens_per_field": 40,
    "temperature": 0.1,
    "do_sample": True,
    "top_p": 0.95,
    "use_cache": False,  # ← CHANGE: Disable KV caching
}
```

**Expected Impact**: 30-50% VRAM reduction during generation
**Trade-off**: Slightly slower generation per image

## Strategy 2: Aggressive Memory Cleanup Between Images

**Issue**: Tensors and intermediate results accumulating in VRAM
**Solution**: Explicit cleanup after each image

```python
# In models/llama_processor.py - Modify process_single_image method
def process_single_image(self, image_path):
    try:
        # PRE-PROCESSING CLEANUP
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # ... existing processing code ...
        
        # POST-PROCESSING CLEANUP
        # Delete all large objects explicitly
        del inputs, output, image
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    except Exception as e:
        # Cleanup on error too
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
```

**Expected Impact**: Prevents memory accumulation between images
**Trade-off**: Small overhead from cleanup operations

## Strategy 3: Reduce Image Resolution

**Issue**: Large images consume significant VRAM during preprocessing
**Solution**: Resize images before processing

```python
# In models/llama_processor.py - Modify load_document_image method
def load_document_image(self, image_path):
    try:
        image = Image.open(image_path)
        
        # Resize to reduce VRAM usage
        max_size = 1024  # Adjust based on testing
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            print(f"📏 Resized image to {image.size} for memory efficiency")
            
        return image
    except Exception as e:
        print(f"❌ Error loading image {image_path}: {e}")
        raise
```

**Expected Impact**: 20-40% reduction in preprocessing VRAM
**Trade-off**: Potential loss of fine detail in documents

## Strategy 4: Reset Processor State Between Images

**Issue**: Processor may cache tokenization or other state
**Solution**: Clear processor caches between images

```python
# In models/llama_processor.py - Add to process_single_image method
def process_single_image(self, image_path):
    try:
        # CLEAR PROCESSOR STATE
        if hasattr(self.processor, 'tokenizer'):
            # Clear tokenizer cache if it exists
            if hasattr(self.processor.tokenizer, 'clear_cache'):
                self.processor.tokenizer.clear_cache()
        
        # ... existing processing code ...
        
    except Exception as e:
        # ... error handling ...
```

**Expected Impact**: Prevents processor state accumulation
**Trade-off**: Minimal - mostly safety measure

## Strategy 5: Reduce Generation Tokens (Emergency Option)

**Issue**: Large token generation consumes significant VRAM
**Solution**: Drastically reduce token limits for V100

```python
# In common/config.py - Emergency V100 configuration
LLAMA_GENERATION_CONFIG_V100_EMERGENCY = {
    "max_new_tokens_base": 400,  # Half the normal tokens
    "max_new_tokens_per_field": 20,  # Half per field  
    "temperature": 0.1,
    "do_sample": True,
    "top_p": 0.95,
    "use_cache": False,  # No caching
}

# Replace LLAMA_GENERATION_CONFIG with LLAMA_GENERATION_CONFIG_V100_EMERGENCY
```

**Expected Impact**: 40-60% reduction in generation VRAM
**Trade-off**: Shorter responses, may truncate field extractions

## Strategy 6: Batch Size Enforcement

**Issue**: Batch processing attempting multiple images simultaneously
**Solution**: Force batch size of 1 for V100

```python
# In models/llama_processor.py - Modify _configure_batch_processing
def _configure_batch_processing(self, batch_size: Optional[int]):
    # Force batch size of 1 for V100 memory constraints
    self.batch_size = 1
    print(f"🎯 V100 Memory Mode: Forced batch size = 1")
```

**Expected Impact**: Eliminates multi-image memory accumulation
**Trade-off**: No batch processing speedup (but may not be working anyway)

## Implementation Order (Recommended)

### Phase 1: Low-Risk, High-Impact
1. **Strategy 1**: Disable KV caching (`use_cache: False`)
2. **Strategy 6**: Force batch size = 1

### Phase 2: Memory Cleanup  
3. **Strategy 2**: Add aggressive cleanup between images
4. **Strategy 4**: Reset processor state

### Phase 3: Quality Trade-offs (If Still Needed)
5. **Strategy 3**: Reduce image resolution  
6. **Strategy 5**: Reduce generation tokens (emergency)

## Testing Protocol

For each strategy:

1. **Apply the change** to the appropriate file
2. **Test with 2-3 images** in sequence
3. **Monitor VRAM usage** with `nvidia-smi`
4. **Check output quality** - ensure field extraction still works
5. **Record results** before moving to next strategy

## Success Criteria

- ✅ Process at least 5 images in sequence without OOM
- ✅ Maintain reasonable field extraction quality (>80% of baseline)
- ✅ VRAM usage stays below 15GB throughout processing

## Rollback Plan

If any strategy breaks functionality:
```bash
git checkout HEAD -- <modified_file>
```

Each strategy can be implemented and tested independently.