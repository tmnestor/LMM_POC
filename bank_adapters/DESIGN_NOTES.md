# Bank Statement Adapter Design Notes

## Problem Statement

VLM-based bank statement extraction faces three types of variation across Australia's Big Four banks:

1. **Spatial variation**: Where data appears on the page
   - ANZ: Dense multi-column layouts
   - CommBank: Cleaner single-flow statements
   - NAB/Westpac: Variable header/footer structures

2. **Syntactic variation**: How data is formatted
   - Date formats: DD/MM/YYYY vs DD MMM vs mixed
   - Amount formats: $1,234.56 vs 1234.56 vs (1234.56) vs 1234.56 CR
   - Transaction structure: Payee first vs reference first

3. **Semantic variation**: What data means
   - "Withdrawal" vs "Debit" vs "DR" (same meaning, different terms)
   - "Credit" vs "Deposit" vs "CR" (same meaning, different terms)
   - Balance labeling: "Opening Balance" vs "Balance B/F" vs "Previous Balance"
   - Transaction categorization differences across banks

## Key Finding

**2-turn balance extraction outperforms 3-turn table extraction.**

The VLM's inductive bias is better suited to targeted extraction than structured table parsing. Table extraction requires the model to:
- Detect table boundaries
- Understand column semantics
- Maintain row alignment
- Handle merged cells and wrapped text

Each step compounds error. Balance extraction is simpler with clearer visual anchors.

## Architecture Decision

Trade unpredictable model failures for predictable code complexity.

| Approach | VLM burden | Python burden | Failure mode |
|----------|-----------|---------------|--------------|
| 3-turn table | High | Low | Hallucinated structure, row misalignment |
| 2-turn balance | Lower | High | Bank-specific edge cases in post-processing |

**Deterministic bugs are easier to fix than stochastic ones.**

## Design Patterns Used

### Strategy Pattern

Defines a common interface (`StatementAdapter`) with bank-specific implementations. Each strategy is interchangeable from the caller's perspective.

**Benefits:**
- Bank-specific logic is isolated
- Adding a new bank = add one file
- Each adapter is independently testable
- Main pipeline stays generic

### Registry Pattern

Lookup mechanism that returns the right strategy based on bank key.

**Benefits:**
- Clean retrieval: `registry.get(BankType.ANZ)`
- Centralized registration
- Easy to enumerate supported banks
- Decouples adapter creation from usage

## Processing Pipeline

```
VLM extraction (bank-agnostic prompts)
        ↓
Bank detector (classifier or rule-based on header/logo)
        ↓
Bank-specific adapter
    ├── spatial normalizer (expected regions → canonical positions)
    ├── syntactic parser (date/amount formats → standard formats)
    └── semantic mapper (bank terminology → canonical terms)
        ↓
Unified output schema (BalanceResult)
```

## Implementation Priority

1. **Bank detector** - Critical for routing to correct adapter
2. **One complete adapter** - Start with cleanest bank format as template
3. **Remaining adapters** - Use first adapter as reference implementation

## Future Enhancements

- **Confidence scoring**: Route uncertain extractions to manual review
- **Schema validation**: Pydantic models per bank for structural expectations
- **Template versioning**: Handle format changes within a bank over time
- **Fallback chain**: Try multiple adapters if bank detection uncertain

## Related Concepts

### Inductive Bias

VLMs pre-trained on natural images have biases that don't match structured documents:
- Natural images: photographic content, continuous gradients
- Bank statements: sparse text, tabular structure, semantic position

This mismatch is why prompt engineering and post-processing matter more than model scale for this task.

### No-Free-Lunch Theorem

No algorithm dominates across all problems. Choosing a model means choosing biases. For bank statements, the right architecture is:
- VLM for flexibility and OCR-like extraction
- Rule-based adapters for structural reasoning
- Hybrid approach outperforms pure end-to-end VLM
