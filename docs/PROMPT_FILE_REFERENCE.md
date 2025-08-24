# Prompt File Reference Guide

This document clarifies which prompt files are used by which scripts and extraction modes.

## 🗂️ Prompt File Mapping

| Model | Script | Mode | Prompt File | Description |
|-------|--------|------|-------------|-------------|
| **Llama** | `llama_keyvalue.py` | `single_pass` | `llama_single_pass_prompts.yaml` | Single-pass extraction with semantic field ordering |
| **Llama** | `llama_keyvalue.py` | `grouped` | `llama_prompts.yaml` | Grouped extraction with 6 field groups |
| **Llama** | `llama_keyvalue.py` | `adaptive` | `llama_prompts.yaml` | Adaptive extraction (uses grouped approach) |
| **InternVL3** | `internvl3_keyvalue.py` | `single_pass` | `internvl3_prompts.yaml` (single_pass section) | Single-pass extraction with semantic ordering |
| **InternVL3** | `internvl3_keyvalue.py` | `grouped` | `internvl3_prompts.yaml` (group sections) | Grouped extraction with 6 field groups |

## 📝 Prompt File Details

### `llama_single_pass_prompts.yaml` ✅ **Most Important for Performance Recovery**
- **Used by**: Llama processor in single_pass mode
- **Contains**: Semantic field ordering (DOCUMENT_TYPE → TOTAL_AMOUNT)  
- **Structure**: Single `single_pass` section with `field_instructions`
- **Key fix**: Changed from alphabetical to semantic field order

### `llama_prompts.yaml` 
- **Used by**: Llama processor in grouped/adaptive modes
- **Contains**: 6 grouped extraction sections
- **Groups**: `regulatory_financial`, `entity_contacts`, `line_item_transactions`, `temporal_data`, `banking_payment`, `document_balances`

### `internvl3_prompts.yaml`
- **Used by**: InternVL3 processor for all modes
- **Contains**: Both single_pass section AND grouped sections in same file
- **Structure**: 
  - `single_pass` section for single-pass mode
  - Individual group sections for grouped mode

## 🚨 Recent Changes (Semantic Ordering Fix)

**Files Modified for Performance Recovery**:
- ✅ `llama_single_pass_prompts.yaml` - **CRITICAL**: Reordered to semantic sequence
- ✅ `internvl3_prompts.yaml` - **CRITICAL**: Reordered single_pass section
- ✅ `llama_prompts.yaml` - Uses standardized field names (unchanged order)

**Key Change**: Field order changed from alphabetical to semantic:
```
OLD: ACCOUNT_CLOSING_BALANCE → ... → TOTAL_AMOUNT (alphabetical)
NEW: DOCUMENT_TYPE → BUSINESS_ABN → ... → TOTAL_AMOUNT (semantic)
```

## 🔧 Debug Command Usage

To see which prompt file and configuration is being used:

```bash
# Llama single-pass (uses llama_single_pass_prompts.yaml)
python llama_keyvalue.py --extraction-mode single_pass --debug

# Llama grouped (uses llama_prompts.yaml)  
python llama_keyvalue.py --extraction-mode grouped --debug

# InternVL3 single-pass (uses internvl3_prompts.yaml single_pass section)
python internvl3_keyvalue.py --extraction-mode single_pass --debug

# InternVL3 grouped (uses internvl3_prompts.yaml group sections)
python internvl3_keyvalue.py --extraction-mode grouped --debug
```

The `--debug` flag now shows:
- ✅ Exact prompt file being used
- ✅ Prompt method and YAML sections
- ✅ Field sequence (first → last)
- ✅ Prompt preview (first 200 characters)
- ✅ Complete configuration details

## 💡 Recommendation for Testing

**For performance recovery validation**, focus on:
1. `python llama_keyvalue.py --extraction-mode single_pass --debug`
2. `python internvl3_keyvalue.py --extraction-mode single_pass --debug`

These modes use the semantically reordered prompts that should restore ~84% accuracy.