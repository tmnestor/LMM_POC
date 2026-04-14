"""Value objects for bank statement extraction pipeline.

Pure dataclasses with zero project dependencies — safe to import anywhere.
Extracted from unified_bank_extractor.py to enable independent testing
of correction and filtering logic without VLM/torch/PIL dependencies.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class ExtractionStrategy(Enum):
    """Available extraction strategies."""

    BALANCE_DESCRIPTION = auto()  # 2-turn: header detection + balance extraction
    AMOUNT_DESCRIPTION = auto()  # 2-turn: header + amount extraction (negative = debit)
    DEBIT_CREDIT_DESCRIPTION = (
        auto()
    )  # 2-turn: header + debit/credit extraction (no balance)
    TABLE_EXTRACTION = auto()  # 3-turn: header + format classify + table


@dataclass
class ColumnMapping:
    """Mapped column names from detected headers."""

    date: str | None = None
    description: str | None = None
    debit: str | None = None
    credit: str | None = None
    balance: str | None = None
    amount: str | None = None

    @property
    def has_balance(self) -> bool:
        """Check if balance column was detected."""
        return self.balance is not None

    def to_dict(self) -> dict[str, str | None]:
        """Serialize to dict for JSON compatibility."""
        return {
            "date": self.date,
            "description": self.description,
            "debit": self.debit,
            "credit": self.credit,
            "balance": self.balance,
            "amount": self.amount,
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

    # Metadata
    strategy_used: str = ""
    turns_executed: int = 0
    headers_detected: list[str] = field(default_factory=list)
    column_mapping: ColumnMapping | None = None
    raw_responses: dict[str, str] = field(default_factory=dict)
    correction_stats: Any = None

    def to_schema_dict(self) -> dict[str, str]:
        """Convert to schema format with pipe-delimited fields."""
        return {
            "DOCUMENT_TYPE": self.document_type,
            "STATEMENT_DATE_RANGE": self.statement_date_range,
            "TRANSACTION_DATES": " | ".join(self.transaction_dates)
            if self.transaction_dates
            else "NOT_FOUND",
            "LINE_ITEM_DESCRIPTIONS": " | ".join(self.line_item_descriptions)
            if self.line_item_descriptions
            else "NOT_FOUND",
            "TRANSACTION_AMOUNTS_PAID": " | ".join(self.transaction_amounts_paid)
            if self.transaction_amounts_paid
            else "NOT_FOUND",
            "ACCOUNT_BALANCE": " | ".join(self.account_balances)
            if self.account_balances
            else "NOT_FOUND",
        }

    def to_metadata_dict(self) -> dict[str, Any]:
        """Build metadata dict for pipeline compatibility."""
        meta: dict[str, Any] = {
            "strategy_used": self.strategy_used,
            "turns_executed": self.turns_executed,
            "headers_detected": self.headers_detected,
            "column_mapping": self.column_mapping.to_dict()
            if self.column_mapping
            else {},
            "raw_responses": self.raw_responses,
        }
        if self.correction_stats:
            meta["correction_stats"] = str(self.correction_stats)
        return meta
