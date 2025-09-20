# Migration Plan - Simplified Document-Aware Architecture

## Current Status: ✅ YAML-First Simplification Complete

### Completed:
- ✅ Created simplified `config/fields.yaml` (130 lines vs 625)
- ✅ Created unified `common/unified_schema.py` (290 lines vs 1400+)
- ✅ Moved legacy files to `legacy/` directory
- ✅ Updated document-aware scripts to use simplified schema
- ✅ Removed all backward compatibility aliases
- ✅ 80% code reduction while maintaining functionality

## Next Phase: Legacy Script Cleanup

### 🗑️ **Files Planned for Deletion** (once document-aware versions are validated):

#### Legacy Keyvalue Scripts:
- `llama_keyvalue.py` → **DELETE** (replaced by `llama_document_aware.py`)
- `internvl3_keyvalue.py` → **DELETE** (replaced by `internvl3_document_aware.py`)

#### Rationale:
- Document-aware versions provide **superior functionality**:
  - 📊 Document-specific field filtering (40-67% field reduction)
  - 🎯 Higher accuracy through targeted extraction
  - 📋 Document type detection and routing
  - ⚡ Better performance through reduced complexity

### 🧪 **Validation Checklist** (before deletion):

#### Document-Aware Scripts Must:
- [ ] ✅ Run successfully on H200 testing environment
- [ ] ✅ Process all document types (invoice, receipt, bank_statement)
- [ ] ✅ Maintain accuracy parity with legacy scripts
- [ ] ✅ Handle edge cases and error conditions
- [ ] ✅ Work with simplified YAML schema
- [ ] ✅ Pass all existing test cases

#### Performance Requirements:
- [ ] ✅ Processing time ≤ legacy keyvalue scripts
- [ ] ✅ Memory usage acceptable on V100 production
- [ ] ✅ Accuracy ≥ 90% for each document type
- [ ] ✅ No regressions in extraction quality

### 📋 **Migration Steps**:

1. **Testing Phase**:
   - Run document-aware scripts on full evaluation dataset
   - Compare results with legacy keyvalue outputs
   - Validate accuracy and performance metrics

2. **Documentation Phase**:
   - Update README.md to reference document-aware scripts
   - Update CLAUDE.md project instructions
   - Update setup.sh completion messages

3. **Cleanup Phase** (only after validation):
   ```bash
   # Move legacy scripts to archive
   mv llama_keyvalue.py legacy/
   mv internvl3_keyvalue.py legacy/
   
   # Update notebooks to use document-aware versions
   # Update any remaining references in docs/
   ```

### 🎯 **Expected Benefits After Cleanup**:

- **📦 Simplified codebase**: No duplicate keyvalue/document-aware scripts
- **🎯 Single extraction paradigm**: Document-aware only
- **📚 Easier maintenance**: One script per model
- **🔧 Cleaner setup**: Fewer scripts to explain/maintain
- **📈 Better performance**: Document-specific optimization

### ⚠️ **Important Notes**:

- **DO NOT delete legacy scripts until document-aware versions are fully validated**
- Keep legacy scripts as backup during transition period
- Ensure all notebook references are updated before deletion
- Document any breaking changes in migration notes

## Current Architecture Post-Simplification:

### Active Scripts:
- `llama_document_aware.py` - Modern document-aware extraction
- `internvl3_document_aware.py` - Modern document-aware extraction

### Legacy Scripts (to be deleted):
- `llama_keyvalue.py` - Universal field extraction (legacy)
- `internvl3_keyvalue.py` - Universal field extraction (legacy)

### Schema Files:
- `config/fields.yaml` - Simplified YAML-first schema
- `legacy/field_schema_v4.yaml` - Original complex schema (archived)

---

**Status**: Ready for document-aware validation on H200 machine
**Timeline**: Delete legacy scripts after successful H200/V100 validation
**Risk**: Low (legacy scripts preserved until validation complete)