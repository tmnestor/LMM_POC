# Bank Statement Extraction Cluster — Hybrid A+D Refactor

Branch: `refactor/bank-extraction-decompose` (from `refactor/batch-processor-decompose`)

Behavior-preserving refactor. Verification = identical F1 scores before/after.

## Summary

Absorb the adapter into `UnifiedBankExtractor`, extract correction/filtering into
a testable `bank_corrector.py`, extract value objects into `bank_types.py`,
deduplicate the response parser, and remove the `skip_math_enhancement` flag.

**Files created:**
- `common/bank_types.py` — pure dataclasses (`ColumnMapping`, `ExtractionResult`, `ExtractionStrategy`, `CorrectionStats`)
- `common/bank_corrector.py` — `BalanceCorrector`, `TransactionFilter` (zero VLM/torch deps)

**Files modified:**
- `common/unified_bank_extractor.py` — absorb adapter duties, import types from `bank_types.py`, import correction from `bank_corrector.py`, deduplicate parser
- `common/batch_types.py` — remove `skip_math_enhancement` field from `ExtractionOutput`
- `common/extraction_evaluator.py` — remove `skip_math_enhancement` guard
- `common/batch_processor.py` — stop setting `skip_math_enhancement`, update bank adapter call
- `cli.py` — import `UnifiedBankExtractor` instead of `BankStatementAdapter`

**Files deleted:**
- `common/bank_statement_adapter.py`

**Files unchanged:**
- `common/bank_statement_calculator.py` — evaluation-only path, untouched

---

## Phase 1: Create `common/bank_types.py`

Extract these from `unified_bank_extractor.py` into a new pure-dataclass file:

```python
# common/bank_types.py
"""Value objects for bank statement extraction pipeline.

Pure dataclasses with zero project dependencies — safe to import anywhere.
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class ExtractionStrategy(Enum):
    BALANCE_DESCRIPTION = auto()
    AMOUNT_DESCRIPTION = auto()
    DEBIT_CREDIT_DESCRIPTION = auto()
    TABLE_EXTRACTION = auto()


@dataclass
class ColumnMapping:
    date: str | None = None
    description: str | None = None
    debit: str | None = None
    credit: str | None = None
    balance: str | None = None
    amount: str | None = None

    @property
    def has_balance(self) -> bool:
        return self.balance is not None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "date": self.date, "description": self.description,
            "debit": self.debit, "credit": self.credit,
            "balance": self.balance, "amount": self.amount,
        }


@dataclass
class ExtractionResult:
    """Standardized extraction result from UnifiedBankExtractor."""
    document_type: str = "BANK_STATEMENT"
    statement_date_range: str = "NOT_FOUND"
    transaction_dates: list[str] = field(default_factory=list)
    line_item_descriptions: list[str] = field(default_factory=list)
    transaction_amounts_paid: list[str] = field(default_factory=list)
    account_balances: list[str] = field(default_factory=list)

    strategy_used: str = ""
    turns_executed: int = 0
    headers_detected: list[str] = field(default_factory=list)
    column_mapping: ColumnMapping | None = None
    raw_responses: dict[str, str] = field(default_factory=dict)
    correction_stats: Any = None

    def to_schema_dict(self) -> dict[str, str]:
        return {
            "DOCUMENT_TYPE": self.document_type,
            "STATEMENT_DATE_RANGE": self.statement_date_range,
            "TRANSACTION_DATES": " | ".join(self.transaction_dates) if self.transaction_dates else "NOT_FOUND",
            "LINE_ITEM_DESCRIPTIONS": " | ".join(self.line_item_descriptions) if self.line_item_descriptions else "NOT_FOUND",
            "TRANSACTION_AMOUNTS_PAID": " | ".join(self.transaction_amounts_paid) if self.transaction_amounts_paid else "NOT_FOUND",
            "ACCOUNT_BALANCE": " | ".join(self.account_balances) if self.account_balances else "NOT_FOUND",
        }

    def to_metadata_dict(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "strategy_used": self.strategy_used,
            "turns_executed": self.turns_executed,
            "headers_detected": self.headers_detected,
            "column_mapping": self.column_mapping.to_dict() if self.column_mapping else {},
            "raw_responses": self.raw_responses,
        }
        if self.correction_stats:
            meta["correction_stats"] = str(self.correction_stats)
        return meta
```

Update `unified_bank_extractor.py` to import these instead of defining them locally.

---

## Phase 2: Create `common/bank_corrector.py`

Extract from `unified_bank_extractor.py`:
- `CorrectionStats` dataclass (lines 704-720)
- `BalanceCorrector` class (lines 723-1001)
- `TransactionFilter` class (lines 577-701)

These have **zero VLM/torch/PIL dependencies** — they operate purely on
`list[dict[str, str]]`. Moving them out makes them unit-testable.

```python
# common/bank_corrector.py
"""Balance-arithmetic correction and transaction filtering.

Pure computation — no GPU, no model, no PIL dependencies.
Independently testable with synthetic transaction data.
"""
from dataclasses import dataclass
# ... CorrectionStats, BalanceCorrector, TransactionFilter moved here
```

Update `unified_bank_extractor.py` to import from `bank_corrector`.

---

## Phase 3: Deduplicate the response parser

`ResponseParser.parse_balance_description()` (lines 224-416, 193 LOC) and
`parse_amount_description()` (lines 418-574, 157 LOC) share ~80% identical code.

### 3a: Extract shared date patterns

```python
# Inside ResponseParser (unified_bank_extractor.py)

_DATE_PATTERNS: list[re.Pattern] = [
    # Pattern 1: "1. **Date:** 03/05/2025"
    re.compile(r"^\d*\.?\s*\*?\*?Date:?\*?\*?\s*(.+)$", re.IGNORECASE),
    # Pattern 2: Bold date "**03/05/2025**"
    re.compile(r"^\*\*(\d{1,2}/\d{1,2}/\d{4})\*\*$"),
    # Pattern 3: "1. **Thu 04 Sep 2025**"
    re.compile(r"^\d+\.\s*\*?\*?([A-Za-z]{3}\s+\d{1,2}\s+[A-Za-z]{3}\s+\d{4})\*?\*?"),
    # Pattern 4: "1. **04 Sep 2025**"
    re.compile(r"^\d+\.\s*\*?\*?(\d{1,2}\s+[A-Za-z]{3}\s+\d{4})\*?\*?"),
    # Pattern 5: "1. **03/05/2025**"
    re.compile(r"^\d+\.\s*\*?\*?(\d{1,2}/\d{1,2}/\d{4})\*?\*?"),
    # Pattern 6: "1. **[20 May]**" (bracketed date)
    re.compile(r"^\d+\.\s*\*?\*?\[(\d{1,2}\s+[A-Za-z]{3,9})\]\*?\*?"),
    # Pattern 7: "1. **20 May**" (date without year)
    re.compile(r"^\d+\.\s*\*?\*?(\d{1,2}\s+[A-Za-z]{3,9})\*?\*?$"),
    # Pattern 8: "1. **06 Aug 24**" (2-digit year)
    re.compile(r"^\d+\.\s*\*?\*?(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2})\*?\*?$"),
]

@staticmethod
def _detect_date(line: str) -> str | None:
    """Try all date patterns on a line. Returns date string or None."""
    for pattern in _DATE_PATTERNS:
        m = pattern.match(line)
        if m:
            return m.group(1).strip().strip("*").strip()
    return None
```

### 3b: Parameterize the parse loop

Create a `ColumnSpec` that tells the parser which columns to extract:

```python
@dataclass(frozen=True)
class ColumnSpec:
    date_col: str
    desc_col: str
    amount_cols: list[str]       # debit col(s) or amount col
    credit_col: str | None       # only for BALANCE_DESCRIPTION
    balance_col: str | None

@staticmethod
def parse_transactions(response: str, spec: ColumnSpec) -> list[dict[str, str]]:
    """Unified parser replacing parse_balance_description and parse_amount_description.

    Same state machine, parameterized by which columns to look for.
    """
    ...
```

The two old methods become thin wrappers that build the right `ColumnSpec` and call
`parse_transactions()`. This preserves backward compatibility while eliminating
the 350 lines of duplication.

**Estimated reduction**: ~350 lines removed from `unified_bank_extractor.py`.

---

## Phase 4: Absorb the adapter into `UnifiedBankExtractor`

### 4a: Move adapter logic into extractor

The adapter does three things:
1. Load PIL image from path
2. Call `self.extractor.extract(image)` with Rich stdout bypass
3. Convert `ExtractionResult` → `(schema_dict, metadata)` via `to_schema_dict()` + `_serialize_column_mapping()`

Move all three into `UnifiedBankExtractor`:

```python
class UnifiedBankExtractor:
    def extract(self, image, force_strategy=None) -> ExtractionResult:
        """Existing method — accepts PIL Image or path. Unchanged internally."""
        ...

    def extract_bank_statement(
        self, image_path: str | Path,
        force_strategy: ExtractionStrategy | None = None,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """Pipeline-compatible entry point (replaces BankStatementAdapter).

        Loads image, runs extraction with Rich stdout bypass, returns
        (schema_fields, metadata) tuple.
        """
        from PIL import Image

        image = Image.open(image_path).convert("RGB")

        with _bypass_rich_stdout():
            result = self.extract(image=image, force_strategy=force_strategy)

        schema_fields = result.to_schema_dict()
        metadata = result.to_metadata_dict()

        if self.verbose and result.raw_responses.get("turn1"):
            _safe_print(f"[UBE] Raw Turn 1 response:\n{result.raw_responses['turn1']}")

        return schema_fields, metadata
```

### 4b: Move `_bypass_rich_stdout()` and `_safe_print()` into `unified_bank_extractor.py`

These are the only non-trivial pieces from the adapter.

### 4c: Delete `common/bank_statement_adapter.py`

---

## Phase 5: Remove `skip_math_enhancement`

### 5a: `common/batch_types.py`

Remove the `skip_math_enhancement: bool = False` field from `ExtractionOutput`.

### 5b: `common/extraction_evaluator.py`

In `_evaluate_extraction()`, remove the `skip_math_enhancement` parameter and the
`not skip_math_enhancement` guard:

```python
# Before
if (
    document_type.upper() == "BANK_STATEMENT"
    and self.enable_math_enhancement
    and not skip_math_enhancement
):
    ...

# After
if (
    document_type.upper() == "BANK_STATEMENT"
    and self.enable_math_enhancement
):
    ...
```

The calculator is idempotent on data that `BalanceCorrector` has already corrected:
when amounts and balance deltas already agree, it makes zero corrections.

### 5c: `common/batch_processor.py`

Stop setting `skip_math_enhancement` on `ExtractionOutput` for bank statements.
Search for any references and remove them.

### 5d: `common/extraction_evaluator.py` `evaluate()` method

The `evaluate()` method passes `extraction.skip_math_enhancement` to
`_evaluate_extraction()`. Remove this parameter from the call.

---

## Phase 6: Update callers

### 6a: `cli.py`

```python
# Before
from common.bank_statement_adapter import BankStatementAdapter
bank_adapter = BankStatementAdapter(
    generate_fn=processor.generate,
    verbose=config.verbose,
    use_balance_correction=config.balance_correction,
)

# After
from common.unified_bank_extractor import UnifiedBankExtractor
bank_adapter = UnifiedBankExtractor(
    generate_fn=processor.generate,
    verbose=config.verbose,
    use_balance_correction=config.balance_correction,
)
```

### 6b: `common/batch_processor.py`

The call site `self._bank_adapter.extract_bank_statement(image_path)` stays the
same — `UnifiedBankExtractor` now has that method. Type annotation changes from
`BankStatementAdapter | None` to `UnifiedBankExtractor | None` (or keep duck-typed).

---

## Phase 7: Lint and type-check

```bash
# Local macOS
conda run -n du ruff check --fix common/bank_types.py common/bank_corrector.py \
  common/unified_bank_extractor.py common/batch_types.py common/extraction_evaluator.py \
  common/batch_processor.py cli.py --ignore ARG001,ARG002,F841
conda run -n du ruff format common/bank_types.py common/bank_corrector.py \
  common/unified_bank_extractor.py common/batch_types.py common/extraction_evaluator.py \
  common/batch_processor.py cli.py
conda run -n du mypy common/bank_types.py common/bank_corrector.py \
  common/unified_bank_extractor.py common/batch_types.py common/extraction_evaluator.py \
  common/batch_processor.py cli.py --ignore-missing-imports 2>&1 | \
  grep -E "^(common/bank_types|common/bank_corrector|common/unified_bank_extractor|common/batch_types|common/extraction_evaluator|common/batch_processor|cli)\.py:"

# Import smoke tests
conda run -n du python -c "from common.bank_types import ExtractionStrategy, ColumnMapping, ExtractionResult; print('bank_types OK')"
conda run -n du python -c "from common.bank_corrector import BalanceCorrector, TransactionFilter, CorrectionStats; print('bank_corrector OK')"
conda run -n du python -c "from common.unified_bank_extractor import UnifiedBankExtractor; print('unified_bank_extractor OK')"
conda run -n du python -c "from common.batch_processor import create_batch_pipeline; print('batch_processor OK')"
```

---

## Verification (remote GPU)

### Baseline (on current branch)

```bash
git checkout refactor/batch-processor-decompose
python cli.py -c config/run_config.yml --max-images 5 2>&1 | tee /tmp/baseline_bank_5.log
python cli.py -c config/run_config.yml 2>&1 | tee /tmp/baseline_bank_full.log
```

### Refactored branch

```bash
git checkout refactor/bank-extraction-decompose
python cli.py -c config/run_config.yml --max-images 5 2>&1 | tee /tmp/refactor_bank_5.log
python cli.py -c config/run_config.yml 2>&1 | tee /tmp/refactor_bank_full.log
```

### Compare

```bash
diff <(grep -E "Median F1|Mean F1|Weighted" /tmp/baseline_bank_5.log) \
     <(grep -E "Median F1|Mean F1|Weighted" /tmp/refactor_bank_5.log)
# Expected: identical

diff <(grep -E "Median F1|Mean F1|Weighted" /tmp/baseline_bank_full.log) \
     <(grep -E "Median F1|Mean F1|Weighted" /tmp/refactor_bank_full.log)
# Expected: identical
```

### Edge cases

```bash
# Sequential mode
python cli.py -c config/run_config.yml --batch-size 1 --max-images 3

# Inference-only (no ground truth)
python cli.py -c config/run_config.yml --max-images 3 --ground-truth ""
```

---

## What to look for

- **F1 scores must match** between baseline and refactor
- **No ImportError or AttributeError** — especially around the moved types
- **Bank statements route correctly** — look for `[UBE]` log lines (not `[BSA]`)
- **Balance correction runs** when `use_balance_correction=True`
- **`skip_math_enhancement` is completely gone** — `grep -r skip_math` should return nothing

---

## Risk assessment

- **Low risk**: Phases 1-2 (extracting types/corrector) are purely additive — existing code works until the old definitions are removed
- **Medium risk**: Phase 3 (parser dedup) touches the most complex parsing logic — must verify every strategy produces identical rows
- **Low risk**: Phase 4 (absorb adapter) is a simple inline of passthrough code
- **Low risk**: Phase 5 (remove flag) relies on calculator idempotency — verified by F1 match
- **Low risk**: Phase 6 (caller updates) is ~10 lines changed across 2 files

Implementation order: 1 → 2 → 4 → 5 → 6 → 7, then 3 (parser dedup) last since
it has the highest risk and can be deferred to a follow-up if needed.
