# EVALUATION_SYSTEM_GUIDE.md - Version 1.1 Changelog

**Date**: 2025-10-08
**Previous Version**: 1.0 (2025-10-06)

---

## Summary of Changes

Version 1.1 adds comprehensive documentation for pipe normalization, word-based matching, and format handling based on recent implementation updates to the evaluation system.

---

## Major Additions

### 1. Multi-Line Fields and Pipe Normalization (NEW SECTION)

**Location**: After "Field Types and Custom Comparison Logic"

**What was added**:
- Explanation of how multi-line document fields are handled
- Pipe normalization for text fields (addresses, names)
- Clarification that pipes remain delimiters for list/transaction fields
- Benefits of this approach for ground truth accuracy

**Key Points**:
```
Ground truth: "123 Main St | Sydney NSW 2000"
Model extracts: "123 Main St Sydney NSW 2000"
Result: ✅ Perfect match (1.0)
```

**Why it matters**: Users can now accurately represent multi-line source documents in ground truth without breaking evaluation.

---

### 2. Word-Based Matching Explained (NEW SECTION)

**Location**: In "Custom Comparison Logic: Design Rationale"

**What was added**:
- Complete explanation of word-based matching algorithm
- Step-by-step breakdown with code examples
- Comparison table showing different scenarios
- Text field matching cascade (exact → substring → word-based → no match)
- Key characteristics (order doesn't matter, case insensitive, etc.)

**Example breakdown**:
```python
extracted = "Aussie Office Supplies Pty Ltd"
ground_truth = "Aussie Office Supplies Corporation Pty Ltd"

# 5 out of 6 words match = 83.3% overlap
# Score: 0.833 (above 80% threshold)
```

**Why it matters**: Users can now understand exactly why certain text matches score the way they do.

---

### 3. Numeric ID Fields (ABN, Tax IDs) (NEW SECTION)

**Location**: In "Custom Comparison Logic: Design Rationale"

**What was added**:
- Detailed explanation of format-agnostic digit matching
- Examples showing different formats all matching
- Rationale for zero tolerance on digit errors
- Implementation code reference

**Key Point**: Format doesn't matter, only digits:
```
"06 082 698 025" = "06-082-698-025" = "06082698025" ✅ (1.0)
```

**Why it matters**: Clarifies that ABN formatting variations don't affect scoring.

---

## Documentation Updates

### Enhanced Field Types Table

**Added column**: "Scoring Rules" with specific thresholds
**Updated**: Text field examples to include BUSINESS_ADDRESS
**Added row**: Document Type field with canonical mapping explanation

### Updated Best Practices

**Added**:
- Guidance on using pipes for multi-line text fields
- Note about automatic pipe normalization (v1.1+)
- Examples of correct vs incorrect ground truth formatting

### Enhanced Troubleshooting

**Added common issues**:
- Pipes in addresses (normalized automatically in v1.1+)
- ABN format variations (digits-only comparison)

**Added to "Common surprises"**:
- ABN format example
- Pipe normalization example

### Updated Summary Section

**Added to "What Makes This Evaluation System Unique"**:
- #6: Pipe normalization for multi-line fields
- #7: Word-based matching with 80% threshold
- #8: Format-agnostic numeric IDs

---

## Version Header Updates

**Changed**:
```
Version: 1.0 → 1.1
Last Updated: 2025-10-06 → 2025-10-08
```

**Added version notes**:
- Pipe normalization for multi-line text fields
- Word-based matching explanation
- ABN/numeric ID format handling
- Enhanced troubleshooting

---

## Implementation References

All new documentation corresponds to actual code in:

**File**: `common/evaluation_metrics.py`

**Key changes**:
- Lines 149-157: Pipe normalization for text fields
- Lines 159-165: Separated normalized variables
- Lines 198-207: Numeric ID digit-only comparison
- Lines 368-375: Word-based matching implementation

---

## User Impact

### What Users Can Now Do

1. ✅ **Use pipes in ground truth** for multi-line addresses/names without worrying about matching
2. ✅ **Understand word-based scoring** and why certain text fields score the way they do
3. ✅ **Use any ABN format** (spaces, dashes, or none) without affecting accuracy
4. ✅ **Debug scoring decisions** with better understanding of the cascade logic

### What Users Should Know

1. **Pipes in text fields**: Automatically normalized to spaces (v1.1+)
2. **Pipes in list fields**: Still used as delimiters (unchanged)
3. **Word matching threshold**: Need 80%+ word overlap for partial credit
4. **ABN formatting**: Only digits matter, format ignored

### Migration Notes

**No breaking changes** - all existing ground truth CSVs remain compatible:
- Addresses without pipes: Still work perfectly
- Addresses with pipes: Now work correctly (previously may have failed)
- List fields: Unchanged behavior
- ABN fields: More forgiving (different formats now match)

---

## Testing

All changes validated with test suite:
- ✅ Pipe normalization for text fields
- ✅ Pipe preservation for list fields
- ✅ Word-based matching thresholds
- ✅ ABN format variations

**Test file**: `test_pipe_normalization.py`

---

## Related Files Changed

1. **EVALUATION_SYSTEM_GUIDE.md** - Documentation (this file)
2. **common/evaluation_metrics.py** - Implementation
3. **test_pipe_normalization.py** - Test suite (new)
4. **clean_address_pipes.py** - Utility script (created but not needed)

---

## Quick Reference: What Changed

| Aspect | v1.0 | v1.1 |
|--------|------|------|
| **Pipes in addresses** | ❌ Break matching | ✅ Normalized to spaces |
| **ABN format** | ⚠️ Unclear | ✅ Documented (digits only) |
| **Word matching** | ⚠️ Mentioned | ✅ Fully explained |
| **Multi-line fields** | ❌ Not documented | ✅ Clear guidance |
| **Troubleshooting** | Basic | Enhanced with new issues |

---

## Future Considerations

**Potential v1.2 enhancements**:
- Configurable word overlap threshold (currently hardcoded 80%)
- Configurable monetary tolerance (currently hardcoded 1%)
- Support for fuzzy string matching (Levenshtein distance) for typos
- Custom field-specific thresholds via configuration

---

**Questions or Issues?**

See the updated troubleshooting section in EVALUATION_SYSTEM_GUIDE.md or the implementation in `common/evaluation_metrics.py:145-175`.
