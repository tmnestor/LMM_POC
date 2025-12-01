# Unified Bank Statement Extraction Architecture

## Executive Summary

This proposal outlines a unified extraction system that automatically selects the optimal extraction strategy based on document characteristics, eliminating the need for 8 separate notebooks while improving maintainability and enabling strategy optimization.

---

## 1. Current Problem: 8 Duplicate Notebooks

| Model | 2-Turn Balance | 3-Turn Table |
|-------|----------------|--------------|
| Llama | `llama_bank_statement_2turn_balance_prompt.ipynb` | `llama_bank_statement_3turn_table.ipynb` |
| InternVL3-2B | `ivl3_2b_bank_statement_2turn_balance_prompt.ipynb` | `ivl3_2b_bank_statement_3turn_table.ipynb` |
| InternVL3-8B | `ivl3_8b_bank_statement_2turn_balance_prompt.ipynb` | `ivl3_8b_bank_statement_3turn_table.ipynb` |
| InternVL3.5-8B | `ivl3_5_8b_bank_statement_2turn_balance_prompt.ipynb` | `ivl3_5_8b_bank_statement_3turn_table.ipynb` |

**Problems:**
- 90%+ code duplication across notebooks
- Bug fixes must be applied 8 times
- No automatic strategy selection
- Hard to A/B test strategies
- Prompt changes require editing multiple files

---

## 2. Strategy Comparison

### 2-Turn Balance-Description
```
Turn 0: Image → Headers
    ↓
Python: Pattern match → Check for Balance column
    ↓
Turn 1: Image + Balance-anchored prompt → Hierarchical list
    ↓
Python: Parse list → Filter debits → Schema fields
```

**Strengths:**
- Simpler pipeline (2 turns vs 3)
- No format classification needed
- Balance-anchored extraction reduces debit/credit confusion
- Universal date handling (works for both date-per-row and date-grouped)

**Requirements:**
- Document MUST have a Balance column

### 3-Turn Table Extraction
```
Turn 0: Image → Headers
    ↓
Python: Pattern match
    ↓
Turn 0.5: Image → Format classification (date-per-row vs date-grouped)
    ↓
Turn 1: Image + Format-specific prompt → Markdown table
    ↓
Python: Parse table → Balance validation → Filter debits → Schema fields
```

**Strengths:**
- Works without Balance column
- Format-specific examples improve accuracy
- Balance validation catches misalignments

**Requirements:**
- More LLM calls
- Requires format classification

---

## 3. Unified Architecture

### 3.1 Core Design: Strategy Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│                    UnifiedBankStatementExtractor                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐                                                │
│  │   Turn 0    │  Header Detection (SHARED)                     │
│  │  (shared)   │  - Same prompt for all strategies              │
│  └──────┬──────┘  - Same parser                                 │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────┐                                                │
│  │  Strategy   │  Document Analysis                              │
│  │  Selector   │  - Has Balance column?                         │
│  └──────┬──────┘  - Historical performance?                     │
│         │         - Model capabilities?                          │
│         │                                                        │
│    ┌────┴────┐                                                  │
│    │         │                                                  │
│    ▼         ▼                                                  │
│ ┌──────┐ ┌──────┐                                               │
│ │2-Turn│ │3-Turn│  Strategy Implementations                      │
│ │Balance│ │Table │  - Each strategy is a separate class         │
│ └───┬──┘ └───┬──┘  - Implements common interface                │
│     │        │                                                   │
│     └────┬───┘                                                  │
│          ▼                                                       │
│  ┌─────────────┐                                                │
│  │   Output    │  Schema Fields (SHARED)                         │
│  │  (shared)   │  - TRANSACTION_DATES                           │
│  └─────────────┘  - LINE_ITEM_DESCRIPTIONS                      │
│                   - TRANSACTION_AMOUNTS_PAID                     │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Directory Structure

```
bank_statement/
├── extractors/
│   ├── __init__.py
│   ├── base.py                      # Abstract base class
│   ├── unified_extractor.py         # Main entry point
│   ├── strategy_selector.py         # Strategy selection logic
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── balance_description.py   # 2-turn strategy
│   │   └── table_extraction.py      # 3-turn strategy
│   └── parsers/
│       ├── __init__.py
│       ├── header_parser.py         # Turn 0 response parsing
│       ├── balance_parser.py        # Balance-description parsing
│       └── table_parser.py          # Markdown table parsing
├── prompts/
│   ├── turn0_headers.yaml           # Shared header detection
│   ├── balance_extraction.yaml      # 2-turn prompts
│   ├── format_classification.yaml   # 3-turn: Turn 0.5
│   └── table_extraction.yaml        # 3-turn: Turn 1
├── config/
│   ├── column_patterns.yaml         # Column matching patterns
│   ├── strategy_rules.yaml          # Strategy selection rules
│   └── model_configs.yaml           # Model-specific settings
└── notebooks/
    ├── unified_extraction.ipynb     # Single notebook for all models
    └── strategy_comparison.ipynb    # A/B testing notebook
```

### 3.3 Strategy Selection Logic

```python
# bank_statement/extractors/strategy_selector.py

from dataclasses import dataclass
from enum import Enum, auto

class ExtractionStrategy(Enum):
    BALANCE_DESCRIPTION = auto()  # 2-turn
    TABLE_EXTRACTION = auto()     # 3-turn

@dataclass
class DocumentCharacteristics:
    """Characteristics detected from Turn 0."""
    has_balance_column: bool
    column_count: int
    detected_headers: list[str]
    date_format: str | None = None  # Detected later if needed

@dataclass
class StrategyDecision:
    """Result of strategy selection."""
    strategy: ExtractionStrategy
    confidence: float
    reason: str

class StrategySelector:
    """
    Selects optimal extraction strategy based on document characteristics.

    Decision tree:
    1. If no Balance column → TABLE_EXTRACTION (only option)
    2. If Balance column exists → BALANCE_DESCRIPTION (simpler, reliable)
    3. Future: Historical performance data could influence selection
    """

    def __init__(self, config_path: str = "config/strategy_rules.yaml"):
        self.config = self._load_config(config_path)

    def select(self, characteristics: DocumentCharacteristics) -> StrategyDecision:
        """Select the best strategy for this document."""

        # Rule 1: No Balance column → must use table extraction
        if not characteristics.has_balance_column:
            return StrategyDecision(
                strategy=ExtractionStrategy.TABLE_EXTRACTION,
                confidence=1.0,
                reason="No Balance column detected - table extraction required"
            )

        # Rule 2: Balance column exists → prefer balance-description
        # (simpler, no format classification needed)
        return StrategyDecision(
            strategy=ExtractionStrategy.BALANCE_DESCRIPTION,
            confidence=0.9,
            reason="Balance column detected - using balance-description (simpler pipeline)"
        )

    def select_with_override(
        self,
        characteristics: DocumentCharacteristics,
        force_strategy: ExtractionStrategy | None = None
    ) -> StrategyDecision:
        """Select strategy with optional manual override."""
        if force_strategy:
            return StrategyDecision(
                strategy=force_strategy,
                confidence=1.0,
                reason=f"Manual override: {force_strategy.name}"
            )
        return self.select(characteristics)
```

### 3.4 Base Strategy Interface

```python
# bank_statement/extractors/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

@dataclass
class ExtractionResult:
    """Standardized extraction result."""
    document_type: str = "BANK_STATEMENT"
    statement_date_range: str = "NOT_FOUND"
    transaction_dates: list[str] = None
    line_item_descriptions: list[str] = None
    transaction_amounts_paid: list[str] = None

    # Metadata
    strategy_used: str = ""
    turns_executed: int = 0
    raw_responses: dict[str, str] = None

    def to_schema_dict(self) -> dict[str, str]:
        """Convert to schema format with pipe-delimited fields."""
        return {
            "DOCUMENT_TYPE": self.document_type,
            "STATEMENT_DATE_RANGE": self.statement_date_range,
            "TRANSACTION_DATES": " | ".join(self.transaction_dates or []),
            "LINE_ITEM_DESCRIPTIONS": " | ".join(self.line_item_descriptions or []),
            "TRANSACTION_AMOUNTS_PAID": " | ".join(self.transaction_amounts_paid or []),
        }

class BaseExtractionStrategy(ABC):
    """Abstract base class for extraction strategies."""

    def __init__(self, model, tokenizer, processor=None, config: dict = None):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor  # For Llama
        self.config = config or {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging."""
        pass

    @property
    @abstractmethod
    def num_turns(self) -> int:
        """Number of LLM turns this strategy uses."""
        pass

    @abstractmethod
    def extract(
        self,
        image,
        headers: list[str],
        column_mapping: dict[str, str]
    ) -> ExtractionResult:
        """
        Execute the extraction strategy.

        Args:
            image: PIL Image or path
            headers: Detected column headers from Turn 0
            column_mapping: Mapped columns {DATE, DESCRIPTION, DEBIT, CREDIT, BALANCE}

        Returns:
            Standardized ExtractionResult
        """
        pass

    def _generate_response(self, image, prompt: str, max_tokens: int = 4096) -> str:
        """Generate model response - to be implemented per model type."""
        raise NotImplementedError("Subclass must implement _generate_response")
```

### 3.5 Strategy Implementations

```python
# bank_statement/extractors/strategies/balance_description.py

from ..base import BaseExtractionStrategy, ExtractionResult
from ..parsers.balance_parser import parse_balance_description_response

class BalanceDescriptionStrategy(BaseExtractionStrategy):
    """2-Turn Balance-Description extraction strategy."""

    @property
    def name(self) -> str:
        return "balance_description_2turn"

    @property
    def num_turns(self) -> int:
        return 1  # Turn 0 (headers) is shared, this is Turn 1 only

    def extract(
        self,
        image,
        headers: list[str],
        column_mapping: dict[str, str]
    ) -> ExtractionResult:
        """Execute balance-description extraction."""

        # Build dynamic prompt using actual column names
        prompt = self._build_prompt(column_mapping)

        # Execute Turn 1
        response = self._generate_response(image, prompt)

        # Parse response
        rows = parse_balance_description_response(
            response,
            date_col=column_mapping['DATE'],
            desc_col=column_mapping['DESCRIPTION'],
            debit_col=column_mapping['DEBIT'],
            credit_col=column_mapping['CREDIT'],
            balance_col=column_mapping['BALANCE']
        )

        # Filter for debit transactions
        debit_rows = self._filter_debits(rows, column_mapping['DEBIT'])

        # Build result
        return ExtractionResult(
            transaction_dates=[r[column_mapping['DATE']] for r in debit_rows],
            line_item_descriptions=[r[column_mapping['DESCRIPTION']] for r in debit_rows],
            transaction_amounts_paid=[r[column_mapping['DEBIT']] for r in debit_rows],
            strategy_used=self.name,
            turns_executed=1,
            raw_responses={'turn1': response}
        )

    def _build_prompt(self, column_mapping: dict[str, str]) -> str:
        """Build extraction prompt with actual column names."""
        balance_col = column_mapping['BALANCE']
        desc_col = column_mapping['DESCRIPTION']
        debit_col = column_mapping['DEBIT']
        credit_col = column_mapping['CREDIT']

        return f"""List all the balances in the {balance_col} column, including:
- Date from the Date Header of the balance
- {desc_col}
- {debit_col} Amount or "NOT_FOUND"
- {credit_col} Amount or "NOT_FOUND"

Format each balance entry like this:
1. **[Date]**
   - {desc_col}: [description text]
   - {debit_col}: [amount or NOT_FOUND]
   - {credit_col}: [amount or NOT_FOUND]
   - {balance_col}: [balance amount]

CRITICAL RULES:
1. List EVERY balance entry in order from top to bottom
2. EVERY balance entry has a date, either on the same row, or above
3. Include the FULL description text, not abbreviated
4. If amount is in {debit_col} column, put it there and use NOT_FOUND for {credit_col}
5. If amount is in {credit_col} column, put it there and use NOT_FOUND for {debit_col}
6. Do NOT skip any transactions
"""
```

```python
# bank_statement/extractors/strategies/table_extraction.py

from ..base import BaseExtractionStrategy, ExtractionResult
from ..parsers.table_parser import parse_markdown_table

class TableExtractionStrategy(BaseExtractionStrategy):
    """3-Turn Table extraction strategy."""

    @property
    def name(self) -> str:
        return "table_extraction_3turn"

    @property
    def num_turns(self) -> int:
        return 2  # Turn 0 shared, this does Turn 0.5 + Turn 1

    def extract(
        self,
        image,
        headers: list[str],
        column_mapping: dict[str, str]
    ) -> ExtractionResult:
        """Execute table extraction with format classification."""

        # Turn 0.5: Classify date format
        date_format = self._classify_date_format(image, headers)

        # Turn 1: Extract table with format-specific prompt
        prompt = self._build_prompt(headers, column_mapping, date_format)
        response = self._generate_response(image, prompt)

        # Parse markdown table
        rows = parse_markdown_table(response)

        # Filter for debit transactions
        debit_col_idx = self._get_column_index(headers, column_mapping['DEBIT'])
        debit_rows = self._filter_debits(rows, debit_col_idx)

        # Build result
        date_idx = self._get_column_index(headers, column_mapping['DATE'])
        desc_idx = self._get_column_index(headers, column_mapping['DESCRIPTION'])

        return ExtractionResult(
            transaction_dates=[r[date_idx] for r in debit_rows if len(r) > date_idx],
            line_item_descriptions=[r[desc_idx] for r in debit_rows if len(r) > desc_idx],
            transaction_amounts_paid=[r[debit_col_idx] for r in debit_rows if len(r) > debit_col_idx],
            strategy_used=self.name,
            turns_executed=2,
            raw_responses={'turn0.5': date_format, 'turn1': response}
        )

    def _classify_date_format(self, image, headers: list[str]) -> str:
        """Turn 0.5: Classify date format."""
        prompt = self._build_classification_prompt(headers)
        response = self._generate_response(image, prompt, max_tokens=100)

        if "date-grouped" in response.lower():
            return "date_grouped"
        return "date_per_row"

    def _build_prompt(
        self,
        headers: list[str],
        column_mapping: dict[str, str],
        date_format: str
    ) -> str:
        """Build format-specific extraction prompt."""
        if date_format == "date_grouped":
            return self._build_date_grouped_prompt(headers, column_mapping)
        return self._build_date_per_row_prompt(headers, column_mapping)
```

### 3.6 Unified Extractor (Main Entry Point)

```python
# bank_statement/extractors/unified_extractor.py

from .base import ExtractionResult
from .strategy_selector import StrategySelector, DocumentCharacteristics, ExtractionStrategy
from .strategies.balance_description import BalanceDescriptionStrategy
from .strategies.table_extraction import TableExtractionStrategy
from common.header_mapping import map_headers_smart

class UnifiedBankStatementExtractor:
    """
    Unified extractor that automatically selects the best strategy.

    Usage:
        extractor = UnifiedBankStatementExtractor(model, tokenizer)
        result = extractor.extract(image)
        schema = result.to_schema_dict()
    """

    def __init__(
        self,
        model,
        tokenizer,
        processor=None,
        model_type: str = "internvl3",
        config: dict = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.model_type = model_type
        self.config = config or {}

        self.selector = StrategySelector()

        # Initialize strategies
        self.strategies = {
            ExtractionStrategy.BALANCE_DESCRIPTION: BalanceDescriptionStrategy(
                model, tokenizer, processor, config
            ),
            ExtractionStrategy.TABLE_EXTRACTION: TableExtractionStrategy(
                model, tokenizer, processor, config
            ),
        }

    def extract(
        self,
        image,
        force_strategy: ExtractionStrategy | None = None
    ) -> ExtractionResult:
        """
        Extract bank statement data using optimal strategy.

        Args:
            image: PIL Image or path
            force_strategy: Optional manual strategy override

        Returns:
            ExtractionResult with extracted data
        """
        # Turn 0: Header detection (shared across all strategies)
        headers = self._detect_headers(image)

        # Map headers to semantic fields
        headers_pipe = " | ".join(headers)
        column_mapping = map_headers_smart(headers_pipe)

        # Analyze document characteristics
        characteristics = DocumentCharacteristics(
            has_balance_column=column_mapping.get('BALANCE') is not None,
            column_count=len(headers),
            detected_headers=headers
        )

        # Select strategy
        decision = self.selector.select_with_override(characteristics, force_strategy)
        print(f"📋 Strategy selected: {decision.strategy.name}")
        print(f"   Reason: {decision.reason}")

        # Execute selected strategy
        strategy = self.strategies[decision.strategy]
        result = strategy.extract(image, headers, column_mapping)

        # Add metadata
        result.strategy_used = decision.strategy.name

        return result

    def _detect_headers(self, image) -> list[str]:
        """Turn 0: Detect column headers."""
        prompt = """Look at the transaction table in this bank statement image.

What are the exact column header names used in the transaction table?

List each column header exactly as it appears, in order from left to right.
Do not interpret or rename them - use the EXACT text from the image."""

        response = self._generate_response(image, prompt, max_tokens=500)
        return self._parse_headers(response)

    def _generate_response(self, image, prompt: str, max_tokens: int) -> str:
        """Model-specific response generation."""
        if self.model_type == "llama":
            return self._generate_llama(image, prompt, max_tokens)
        return self._generate_internvl3(image, prompt, max_tokens)
```

### 3.7 Notebook Usage (Simplified)

```python
# bank_statement/notebooks/unified_extraction.ipynb

# Cell 1: Setup
import sys
sys.path.insert(0, '..')

from extractors.unified_extractor import UnifiedBankStatementExtractor
from extractors.strategy_selector import ExtractionStrategy
from common.reproducibility import set_seed, configure_deterministic_mode

# Deterministic setup
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
set_seed(42)
configure_deterministic_mode(True)

# Cell 2: Load model (model-specific, unchanged)
# ... model loading code ...

# Cell 3: Create extractor
extractor = UnifiedBankStatementExtractor(
    model=model,
    tokenizer=tokenizer,
    processor=processor,  # For Llama
    model_type="llama"  # or "internvl3"
)

# Cell 4: Extract (automatic strategy selection)
image_path = "evaluation_data/bank/minimal/image_008.png"
result = extractor.extract(image_path)

print(f"Strategy used: {result.strategy_used}")
print(f"Turns executed: {result.turns_executed}")

# Cell 5: View results
schema = result.to_schema_dict()
for field, value in schema.items():
    print(f"{field}: {value[:80]}...")

# Cell 6: Force specific strategy (for comparison/debugging)
result_2turn = extractor.extract(image_path, force_strategy=ExtractionStrategy.BALANCE_DESCRIPTION)
result_3turn = extractor.extract(image_path, force_strategy=ExtractionStrategy.TABLE_EXTRACTION)

# Compare results
print(f"2-Turn dates: {len(result_2turn.transaction_dates)}")
print(f"3-Turn dates: {len(result_3turn.transaction_dates)}")
```

---

## 4. Strategy Selection Rules (YAML Config)

```yaml
# bank_statement/config/strategy_rules.yaml

version: "1.0"
description: "Rules for automatic strategy selection"

# Primary decision: Based on document characteristics
primary_rules:
  - name: "no_balance_column"
    condition: "not has_balance_column"
    strategy: "TABLE_EXTRACTION"
    confidence: 1.0
    reason: "No Balance column - table extraction required"

  - name: "has_balance_column"
    condition: "has_balance_column"
    strategy: "BALANCE_DESCRIPTION"
    confidence: 0.9
    reason: "Balance column detected - using simpler balance-description"

# Future: Model-specific preferences
model_preferences:
  llama:
    preferred_strategy: null  # No preference, use rules
  internvl3_2b:
    preferred_strategy: "BALANCE_DESCRIPTION"  # Better for smaller model
  internvl3_8b:
    preferred_strategy: null

# Future: Historical performance overrides
# (Would be populated from evaluation results)
performance_overrides:
  # image_003.png:
  #   strategy: "TABLE_EXTRACTION"
  #   reason: "Historical: 3-turn performs better on this document"
```

---

## 5. Migration Path

### Phase 1: Create Core Infrastructure (Week 1)
1. Create `bank_statement/extractors/` package
2. Implement `BaseExtractionStrategy` and `ExtractionResult`
3. Implement `StrategySelector`
4. Port parsing logic to `parsers/` module

### Phase 2: Implement Strategies (Week 2)
1. Implement `BalanceDescriptionStrategy` (port from 2-turn notebooks)
2. Implement `TableExtractionStrategy` (port from 3-turn notebooks)
3. Create `UnifiedBankStatementExtractor`

### Phase 3: Model Adapters (Week 3)
1. Create Llama-specific generation methods
2. Create InternVL3-specific generation methods
3. Test with all 4 model variants

### Phase 4: Notebook Consolidation (Week 4)
1. Create `unified_extraction.ipynb`
2. Create `strategy_comparison.ipynb` for A/B testing
3. Archive old notebooks (don't delete immediately)
4. Update CLAUDE.md documentation

---

## 6. Benefits Summary

| Aspect | Before (8 Notebooks) | After (Unified) |
|--------|---------------------|-----------------|
| **Code duplication** | ~90% duplicated | ~0% (shared infrastructure) |
| **Bug fixes** | Apply 8 times | Apply once |
| **Strategy selection** | Manual notebook choice | Automatic per-document |
| **A/B testing** | Run separate notebooks | Single notebook, force_strategy param |
| **New model support** | Create 2 new notebooks | Add one model adapter |
| **Prompt changes** | Edit 8 files | Edit 1 YAML file |
| **Strategy comparison** | Manual process | Built-in reporting |

---

## 7. Future Enhancements

### 7.1 Learning-Based Strategy Selection
```python
# After running evaluations, learn which strategy works best per document
class LearningStrategySelector(StrategySelector):
    def __init__(self):
        self.performance_db = load_performance_history()

    def select(self, characteristics, image_hash=None):
        # Check if we have historical data for this document
        if image_hash and image_hash in self.performance_db:
            best = self.performance_db[image_hash]['best_strategy']
            return StrategyDecision(strategy=best, confidence=0.95,
                                   reason="Historical performance data")

        # Fall back to rules
        return super().select(characteristics)
```

### 7.2 Ensemble Mode
```python
# Run both strategies and merge results
def extract_ensemble(self, image) -> ExtractionResult:
    result_2turn = self.strategies[BALANCE_DESCRIPTION].extract(...)
    result_3turn = self.strategies[TABLE_EXTRACTION].extract(...)

    # Merge results using voting/confidence
    return self._merge_results(result_2turn, result_3turn)
```

### 7.3 Confidence-Based Fallback
```python
# If primary strategy has low confidence, try alternative
def extract_with_fallback(self, image) -> ExtractionResult:
    result = self.extract(image)

    if result.confidence < 0.7:
        # Try alternative strategy
        alt_strategy = self._get_alternative_strategy()
        alt_result = alt_strategy.extract(...)

        if alt_result.confidence > result.confidence:
            return alt_result

    return result
```
