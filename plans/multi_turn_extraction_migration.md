# Multi-Turn Extraction Migration Plan

## Status Quo

- **Bank statements**: Already multi-turn via `UnifiedBankExtractor` (turn 0: header detection, turn 1: strategy-specific extraction). Hardwired 2-turn flow with strategy enum.
- **Standard docs** (invoice, receipt): Single-turn via `processor.process_document_aware()`. Prompt sent, raw response returned.
- **Pipeline contract**: Extract stage produces `raw_response` string → clean stage parses it via `ResponseHandler.handle()`.

## Goal

Generalize multi-turn extraction so **any document type** can declare a turn sequence in YAML, without hardcoding strategy selection in Python. Bank statements become one instance of a generic multi-turn framework.

---

## Data Structures

### A. Prompt Definition (YAML)

Each document type declares an ordered turn sequence. Turns can depend on earlier turns via `inject` mappings.

```yaml
# prompts/internvl3_prompts.yaml
prompts:
  invoice:
    turns:
      - key: extract
        template: |
          Extract ALL data from this invoice image.
          For each field, write FIELD_NAME: value ...

  bank_statement:
    turns:
      - key: detect_headers
        template: |
          Look at the transaction table in this bank statement.
          List the exact column header names, one per line.
        max_tokens: 500
        parser: header_list          # named parser for this turn's output

      - key: select_strategy
        type: router                 # special turn type: no model call
        rules:
          - when: "{has_balance} and {has_debit}"
            select: extract_balance
          - when: "{has_amount}"
            select: extract_amount
          - default: extract_debit_credit

      - key: extract_balance
        template: |
          List all balances in the {balance_col} column ...
        inject:
          balance_col: "detect_headers.column_mapping.balance"
          desc_col: "detect_headers.column_mapping.description"
          debit_col: "detect_headers.column_mapping.debit"
          credit_col: "detect_headers.column_mapping.credit"
        max_tokens: 4096
        parser: balance_description

      - key: extract_amount
        template: |
          Extract all transactions using the {amount_col} column ...
        inject:
          amount_col: "detect_headers.column_mapping.amount"
        max_tokens: 4096
        parser: amount_description

      - key: extract_debit_credit
        template: |
          Extract debit and credit transactions ...
        inject:
          debit_col: "detect_headers.column_mapping.debit"
          credit_col: "detect_headers.column_mapping.credit"
        max_tokens: 4096
        parser: debit_credit_description
```

**Key points:**
- Single-turn docs (invoice, receipt) have `turns: [one item]` — backward compatible.
- `inject` resolves values from earlier turns' parsed output using dot-path references.
- `type: router` turns evaluate conditions against accumulated state — no model call.
- Each turn declares its own `parser` for structured output extraction.
- `max_tokens` per turn (defaults to generation config if omitted).

### B. Runtime State (Dataclasses)

```python
# common/extraction_types.py  (new file, zero project deps like bank_types.py)

@dataclass
class TurnResult:
    """Result of a single extraction turn."""
    key: str                        # matches YAML turn key
    prompt_sent: str                # after template injection
    raw_response: str               # raw model output
    parsed: dict[str, Any]          # structured parse of this turn
    elapsed: float                  # seconds

@dataclass
class ExtractionSession:
    """Complete multi-turn extraction for one image."""
    image_path: str
    image_name: str
    document_type: str
    turns: list[TurnResult]
    final_fields: dict[str, str]    # schema-format output for clean stage
    strategy: str                   # "extract_balance", "extract", etc.

    @property
    def raw_response(self) -> str:
        """Flat FIELD: value string for the clean stage contract."""
        return "\n".join(
            f"{k}: {v}" for k, v in self.final_fields.items()
        )

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def to_record(self) -> dict[str, Any]:
        """Serialize for raw_extractions.jsonl."""
        return {
            "image_name": self.image_name,
            "image_path": self.image_path,
            "document_type": self.document_type,
            "raw_response": self.raw_response,
            "turns": [
                {
                    "key": t.key,
                    "raw_response": t.raw_response,
                    "elapsed": t.elapsed,
                }
                for t in self.turns
            ],
            "strategy": self.strategy,
            "processing_time": sum(t.elapsed for t in self.turns),
            "error": None,
        }
```

**Key points:**
- `final_fields` is the schema-format dict (same shape as `ExtractionResult.to_schema_dict()`).
- `raw_response` property serializes to flat text — clean stage contract preserved.
- `turns` list in the JSONL record is the debug artifact (replaces `metadata["raw_responses"]`).
- `ExtractionSession` replaces the current `(schema_fields, metadata)` tuple return.

### C. Inter-Stage Serialization (JSONL)

```jsonc
// raw_extractions.jsonl
{
  "image_name": "bank_001.png",
  "image_path": "/data/bank_001.png",
  "document_type": "BANK_STATEMENT",
  "raw_response": "DOCUMENT_TYPE: BANK_STATEMENT\nTRANSACTION_DATES: 03/05 | 04/05",
  "turns": [
    {"key": "detect_headers", "raw_response": "Date\nDescription\nDebit\nCredit\nBalance", "elapsed": 1.2},
    {"key": "extract_balance", "raw_response": "1. **03/05/2025** ...", "elapsed": 3.4}
  ],
  "strategy": "extract_balance",
  "processing_time": 4.6,
  "error": null
}
```

- `raw_response` is the parseable flat text (clean stage reads this).
- `turns` is debug-only (clean stage ignores it).
- Standard single-turn docs have `turns: [{key: "extract", ...}]`.

---

## Execution Engine

### `common/turn_executor.py` (new file)

The execution loop uses **structural pattern matching** (PEP 634) throughout for
clean, exhaustive dispatch on turn definitions and parsed state.

```python
class TurnExecutor:
    """Execute a YAML-defined turn sequence against a model."""

    def __init__(
        self,
        generate_fn: Callable[[Image, str, int], str],
        catalog: PromptCatalog,
        parsers: dict[str, TurnParser],     # registry of named parsers
    ) -> None: ...

    def execute(
        self,
        image_path: str,
        document_type: str,
        turn_defs: list[dict],              # from YAML
    ) -> ExtractionSession: ...
```

#### Turn Dispatch via `match`

The core loop replaces `if/elif` chains with exhaustive pattern matching on
turn definition dicts loaded from YAML:

```python
for turn_def in turn_defs:
    match turn_def:
        case {"type": "router", "key": str(key), "rules": list(rules)}:
            # No model call — evaluate conditions against accumulated state
            next_key = self._evaluate_router(rules, state)
            active_turns = {t["key"]: t for t in turn_defs}
            turn_def = active_turns[next_key]
            # Fall through to execute the selected turn

        case {"key": str(key), "template": str(template), "parser": str(parser_name),
              "inject": dict(injections), **rest}:
            # Full turn: template + injection + named parser
            prompt = self._resolve_inject(template, injections, state)
            max_tokens = rest.get("max_tokens", default_max_tokens)
            raw = generate_fn(image, prompt, max_tokens)
            parsed = self._parsers[parser_name].parse(raw, state)
            turns.append(TurnResult(key=key, prompt_sent=prompt,
                                    raw_response=raw, parsed=parsed, elapsed=elapsed))
            state[key] = parsed

        case {"key": str(key), "template": str(template), **rest}:
            # Simple turn: template only, default field_value parser
            prompt = self._resolve_inject(template, rest.get("inject", {}), state)
            max_tokens = rest.get("max_tokens", default_max_tokens)
            raw = generate_fn(image, prompt, max_tokens)
            parsed = self._parsers["field_value"].parse(raw, state)
            turns.append(TurnResult(key=key, prompt_sent=prompt,
                                    raw_response=raw, parsed=parsed, elapsed=elapsed))
            state[key] = parsed

        case unknown:
            raise ValueError(f"Unrecognized turn definition: {unknown!r}")
```

**Why match/case here:**
- YAML turn dicts have variable shapes (router vs full vs simple). Pattern matching
  destructures and validates shape in one expression.
- The `case unknown` arm guarantees exhaustive handling — bad YAML fails fast with
  a clear diagnostic instead of silently skipping.
- `**rest` capture cleanly handles optional keys (`max_tokens`, `inject`) without
  nested `.get()` chains.

#### Strategy Selection via Dataclass Matching

The bank `type: router` turn evaluates column mappings. Instead of boolean
`if/elif`, pattern match on the `ColumnMapping` dataclass attributes:

```python
def _select_bank_strategy(self, mapping: ColumnMapping) -> str:
    """Select extraction strategy from detected column structure."""
    match mapping:
        case ColumnMapping(balance=str(), debit=str()):
            return "extract_balance"
        case ColumnMapping(balance=str(), amount=str()):
            return "extract_amount"
        case ColumnMapping(debit=str()) | ColumnMapping(credit=str()):
            return "extract_debit_credit"
        case ColumnMapping():
            return "table_fallback"
```

**Why match/case here:**
- `ColumnMapping(balance=str())` matches when `balance` is a non-None string —
  replaces the `mapping.has_balance` property and boolean combinations.
- Union patterns (`|`) express "debit OR credit present" directly.
- Adding a new strategy (e.g., `ColumnMapping(balance=str(), amount=str(), debit=str())`)
  is one line, not a nested conditional.
- The final `case ColumnMapping()` is an exhaustive catch-all — guaranteed no
  unhandled column combinations.

#### Response Structure Detection

`hybrid_parse_response` currently uses `isinstance` + try/except to detect JSON vs
plain text. Pattern matching on the parsed structure is cleaner:

```python
def _detect_response_format(self, text: str, expected_fields: list[str]) -> dict[str, str]:
    """Detect and parse response format."""
    try:
        data = json.loads(self._strip_markdown_fences(text))
    except json.JSONDecodeError:
        return self._parse_plain_text(text, expected_fields)

    match data:
        case {field: str()} if field in expected_fields:
            # Direct field dict: {"DOCUMENT_TYPE": "INVOICE", ...}
            return {f: str(data.get(f, "NOT_FOUND")) for f in expected_fields}
        case [{"DOCUMENT_TYPE": str(), **fields}, *_]:
            # Array of records — take first
            return {f: str(fields.get(f, "NOT_FOUND")) for f in expected_fields}
        case {"turn0": str(), **_}:
            # Multi-turn raw responses — not parseable as fields
            return {f: "NOT_FOUND" for f in expected_fields}
        case _:
            return self._parse_plain_text(text, expected_fields)
```

**Why match/case here:**
- JSON can arrive in multiple shapes (dict, list-of-dicts, multi-turn keyed by
  turn index). Each shape needs different extraction logic.
- Guards (`if field in expected_fields`) validate that the dict actually contains
  extraction fields, not arbitrary JSON.
- The `{"turn0": str(), **_}` arm catches the old multi-turn format explicitly
  instead of silently falling through to plain-text parsing.

### Turn Parsers

Register named parsers that match YAML `parser:` keys:

```python
# common/turn_parsers.py
class TurnParser(Protocol):
    def parse(self, raw_response: str, context: dict[str, Any]) -> dict[str, Any]: ...

# Concrete implementations (refactored from existing code):
class HeaderListParser(TurnParser):     # from UnifiedBankExtractor._parse_headers
class BalanceDescriptionParser(TurnParser):  # from ResponseParser.parse_balance_description
class AmountDescriptionParser(TurnParser):   # from ResponseParser.parse_amount_description
class FieldValueParser(TurnParser):     # from hybrid_parse_response (standard docs)
```

Existing parsing logic moves into these classes — no new algorithms, just reorganization.

#### Parser Result Normalization via `match`

Turn parsers return heterogeneous structures. Normalize before storing in state:

```python
def _normalize_parser_result(
    self, result: dict[str, Any], turn_key: str,
) -> dict[str, Any]:
    """Normalize parser output for state accumulation."""
    match result:
        case {"column_mapping": ColumnMapping() as mapping, "headers": list(hdrs)}:
            # Header detection turn — expose mapping attributes for injection
            return {
                "column_mapping": mapping,
                "headers": hdrs,
                "has_balance": mapping.has_balance,
                "has_debit": mapping.debit is not None,
                "has_credit": mapping.credit is not None,
                "has_amount": mapping.amount is not None,
            }
        case {"rows": list(rows), "date_range": str(dr)}:
            # Extraction turn with structured rows — convert to schema format
            return {"rows": rows, "date_range": dr}
        case {str(): str(), **_}:
            # Simple field dict — pass through
            return result
        case _:
            raise ValueError(f"Unexpected parser result for turn '{turn_key}': {result!r}")
```

---

## Migration Path (3 phases)

### Phase 1: Introduce Data Structures (no behavior change)

1. Create `common/extraction_types.py` with `TurnResult` and `ExtractionSession`.
2. Have `UnifiedBankExtractor.extract_bank_statement()` return `ExtractionSession` instead of `(schema_fields, metadata)` tuple.
   - Internal: build `TurnResult` for each turn it already executes.
   - `ExtractionSession.final_fields` = current `ExtractionResult.to_schema_dict()`.
   - Keep `ExtractionResult` as internal implementation detail.
3. Update `stages/extract.py` `_extract_bank_with_adapter()` to call `session.to_record()`.
4. Update `stages/extract.py` `_extract_standard()` to wrap single-turn result in `ExtractionSession` too.
5. **Clean stage unchanged** — still reads `raw_response`, calls `handler.handle()`.

**Validation:** Pipeline produces identical output. Diff `raw_extractions.jsonl` before/after (ignoring new `turns` field).

### Phase 2: YAML Turn Definitions + TurnExecutor

1. Restructure prompt YAMLs to use `turns:` format (backward-compat: single-turn docs get `turns: [one entry]`).
2. Create `common/turn_executor.py` with the generic `match`-based execution loop (see Execution Engine above).
3. Extract bank-specific parsers from `UnifiedBankExtractor` into `common/turn_parsers.py`.
4. Replace `hybrid_parse_response` JSON detection with `match`-based `_detect_response_format`.
5. Replace `ColumnMapping` boolean checks with dataclass pattern matching in `_select_bank_strategy`.
6. Wire `TurnExecutor` into `stages/extract.py` — replaces both `_extract_bank_with_adapter()` and `_extract_standard()` with a single code path.
7. `UnifiedBankExtractor` becomes a thin wrapper or is retired (its strategy selection logic moves to YAML `type: router` turns + `match` on `ColumnMapping`).

**Structural pattern matching touchpoints in this phase:**
- `TurnExecutor.execute()` — turn dispatch (router vs full vs simple)
- `_select_bank_strategy()` — `ColumnMapping` dataclass matching
- `_detect_response_format()` — JSON shape detection
- `_normalize_parser_result()` — heterogeneous parser output normalization

**Validation:** Same pipeline output. Bank extraction F1 unchanged.

### Phase 3: Multi-Turn Standard Docs

With the framework in place, add multi-turn sequences for standard docs where it helps:

- **Invoice**: Turn 1 = extract line items, Turn 2 = extract header fields (split focus for better accuracy).
- **Receipt**: Turn 1 = extract all, Turn 2 = verify totals match line items.
- New doc types can define arbitrary turn sequences in YAML without Python changes.

---

## What Stays the Same

- **Clean stage** — always `handler.handle(raw_response, expected_fields)`. Zero changes.
- **ResponseHandler** — parse → clean → validate chain untouched.
- **PromptCatalog** — same API, just reads richer YAML structure.
- **Model backends** — `generate_fn` signature unchanged.
- **Evaluation** — reads `cleaned_extractions.jsonl`, format unchanged.

## What Changes

| Component | Current | After |
|-----------|---------|-------|
| Bank extraction return | `(schema_fields, metadata)` tuple | `ExtractionSession` dataclass |
| Standard extraction return | `dict` with `raw_response` key | `ExtractionSession` dataclass |
| Extract stage dispatch | `if bank: ... else: ...` | `TurnExecutor.execute()` for all |
| Strategy selection | Python `if/elif` in `UnifiedBankExtractor` | YAML `type: router` turns |
| Per-turn parsing | Methods on `ResponseParser` class | Named `TurnParser` implementations |
| Prompt YAML | Flat `key: {prompt: ...}` | `key: {turns: [...]}` |
| Debug artifact | `metadata["raw_responses"]` dict | `turns` list in JSONL |

## Structural Pattern Matching Summary

Four `match` sites replace `if/elif` chains and `isinstance` checks:

| Site | Matches on | Replaces | Benefit |
|------|-----------|----------|---------|
| Turn dispatch | YAML dict shape (`{"type": "router", ...}` vs `{"template": ..., "parser": ...}`) | `if turn_def.get("type") == "router"` chains | Exhaustive — bad YAML fails fast via `case unknown` |
| Strategy selection | `ColumnMapping` dataclass attributes | `if mapping.has_balance and mapping.debit` booleans | One-line per strategy, union patterns for OR logic |
| Response format | JSON structure shape (dict vs list vs multi-turn) | `isinstance` + nested `if` in `hybrid_parse_response` | Guards validate field presence inline |
| Parser normalization | Heterogeneous result dicts | `if "column_mapping" in result` key checks | Destructures + type-narrows in one expression |

**Where match/case is NOT used** (and why):
- **Clean stage** — single `handler.handle()` call, no branching.
- **Per-field cleaning** — field type routing is string-prefix lookup (`"AMOUNT" in name`), better as dict dispatch.
- **YAML loading** — already clean dict access via `PromptCatalog`.

## Risk Mitigation

- **Phase 1 is zero-risk** — just wrapping existing returns in dataclasses. No logic changes.
- **Phase 2 parser extraction** — each parser already exists as a method. Moving to a class is mechanical.
- **YAML router** — bank strategy selection logic is well-tested. Translating `if/elif` to declarative rules is straightforward.
- **Rollback** — each phase is independently deployable. Phase 1 can ship without 2 or 3.
