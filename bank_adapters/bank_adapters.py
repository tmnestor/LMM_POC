"""
Bank Statement Adapter System

Strategy Pattern + Registry Pattern implementation for handling
bank-specific variations in VLM-based statement extraction.

Architecture:
    VLM extraction (bank-agnostic prompts)
            ↓
    Bank detector (classifier or rule-based)
            ↓
    Bank-specific adapter
        ├── spatial normalizer (expected regions → canonical positions)
        └── semantic parser (date/amount/description patterns)
            ↓
    Unified output schema
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum


class BankType(str, Enum):
    """Supported Australian banks."""

    ANZ = "anz"
    COMMBANK = "commbank"
    NAB = "nab"
    WESTPAC = "westpac"


@dataclass
class BalanceResult:
    """Unified output schema for extracted balance data."""

    opening: Decimal
    closing: Decimal
    statement_date: date
    raw_transactions: list[dict] = field(default_factory=list)
    confidence: float = 1.0
    bank_type: BankType | None = None


class StatementAdapter(ABC):
    """
    Base class for bank-specific statement adapters.

    Each adapter implements the same interface but encodes
    that bank's spatial and semantic quirks.
    """

    DATE_PATTERNS: list[str] = []

    @abstractmethod
    def normalize_spatial(self, vlm_output: dict) -> dict:
        """
        Map bank-specific regions to canonical positions.

        Args:
            vlm_output: Raw output from VLM extraction

        Returns:
            Normalized dict with standardized field positions
        """
        ...

    @abstractmethod
    def parse_balances(self, normalized: dict) -> BalanceResult:
        """
        Extract balances using bank-specific semantics.

        Args:
            normalized: Output from normalize_spatial()

        Returns:
            BalanceResult with extracted data
        """
        ...

    def parse_date(self, raw_date: str) -> date:
        """
        Handle bank-specific date formats.

        Args:
            raw_date: Raw date string from statement

        Returns:
            Parsed date object

        Raises:
            ValueError: If date format not recognized
        """
        for pattern in self.DATE_PATTERNS:
            try:
                return datetime.strptime(raw_date.strip(), pattern).date()
            except ValueError:
                continue
        raise ValueError(f"Unknown date format: {raw_date}")

    def parse_amount(self, raw_amount: str) -> Decimal:
        """
        Parse monetary amount string to Decimal.

        Handles common variations: $1,234.56, 1234.56, (1234.56) for negative
        """
        cleaned = raw_amount.replace("$", "").replace(",", "").strip()

        # Handle parentheses for negative (accounting notation)
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = "-" + cleaned[1:-1]

        # Handle CR/DR suffixes
        if cleaned.endswith("CR"):
            cleaned = cleaned[:-2].strip()
        elif cleaned.endswith("DR"):
            cleaned = "-" + cleaned[:-2].strip()

        return Decimal(cleaned)


class ANZAdapter(StatementAdapter):
    """
    ANZ-specific statement adapter.

    Characteristics:
    - Dense multi-column layouts
    - Date formats: DD/MM/YYYY, DD MMM YYYY
    - Balance typically in right column
    """

    DATE_PATTERNS = ["%d/%m/%Y", "%d %b %Y", "%d %B %Y"]

    def normalize_spatial(self, vlm_output: dict) -> dict:
        """ANZ-specific: balance often in right column, dense layout."""
        # TODO: Implement based on actual ANZ layout analysis
        raise NotImplementedError("ANZ spatial normalization pending")

    def parse_balances(self, normalized: dict) -> BalanceResult:
        """ANZ-specific balance extraction."""
        # TODO: Implement based on actual ANZ statement structure
        raise NotImplementedError("ANZ balance parsing pending")


class CommBankAdapter(StatementAdapter):
    """
    Commonwealth Bank specific adapter.

    Characteristics:
    - Cleaner single-flow statements
    - Date format: DD MMM YYYY
    - Generally more consistent layout
    """

    DATE_PATTERNS = ["%d %b %Y", "%d %B %Y", "%d/%m/%Y"]

    def normalize_spatial(self, vlm_output: dict) -> dict:
        """CommBank-specific: cleaner layout, single flow."""
        # TODO: Implement based on actual CommBank layout analysis
        raise NotImplementedError("CommBank spatial normalization pending")

    def parse_balances(self, normalized: dict) -> BalanceResult:
        """CommBank-specific balance extraction."""
        # TODO: Implement based on actual CommBank statement structure
        raise NotImplementedError("CommBank balance parsing pending")


class NABAdapter(StatementAdapter):
    """
    NAB-specific statement adapter.

    Characteristics:
    - Mixed layout complexity
    - Variable header/footer structures
    """

    DATE_PATTERNS = ["%d/%m/%Y", "%d %b %Y", "%d-%m-%Y"]

    def normalize_spatial(self, vlm_output: dict) -> dict:
        """NAB-specific spatial normalization."""
        # TODO: Implement based on actual NAB layout analysis
        raise NotImplementedError("NAB spatial normalization pending")

    def parse_balances(self, normalized: dict) -> BalanceResult:
        """NAB-specific balance extraction."""
        # TODO: Implement based on actual NAB statement structure
        raise NotImplementedError("NAB balance parsing pending")


class WestpacAdapter(StatementAdapter):
    """
    Westpac-specific statement adapter.

    Characteristics:
    - Variable layout complexity
    - Multiple statement formats in use
    """

    DATE_PATTERNS = ["%d/%m/%Y", "%d %b %Y", "%d/%m/%y"]

    def normalize_spatial(self, vlm_output: dict) -> dict:
        """Westpac-specific spatial normalization."""
        # TODO: Implement based on actual Westpac layout analysis
        raise NotImplementedError("Westpac spatial normalization pending")

    def parse_balances(self, normalized: dict) -> BalanceResult:
        """Westpac-specific balance extraction."""
        # TODO: Implement based on actual Westpac statement structure
        raise NotImplementedError("Westpac balance parsing pending")


class AdapterRegistry:
    """
    Registry for bank-specific adapters.

    Provides lookup mechanism to retrieve the right strategy
    based on bank type.
    """

    def __init__(self):
        self._adapters: dict[BankType, StatementAdapter] = {}

    def register(self, bank: BankType, adapter: StatementAdapter) -> None:
        """Register an adapter for a bank type."""
        self._adapters[bank] = adapter

    def get(self, bank: BankType) -> StatementAdapter:
        """
        Get adapter for a bank type.

        Raises:
            KeyError: If no adapter registered for bank
        """
        if bank not in self._adapters:
            raise KeyError(f"No adapter registered for {bank}")
        return self._adapters[bank]

    def list_banks(self) -> list[BankType]:
        """List all registered bank types."""
        return list(self._adapters.keys())


# Module-level singleton registry
registry = AdapterRegistry()
registry.register(BankType.ANZ, ANZAdapter())
registry.register(BankType.COMMBANK, CommBankAdapter())
registry.register(BankType.NAB, NABAdapter())
registry.register(BankType.WESTPAC, WestpacAdapter())


def process_statement(vlm_output: dict, bank_key: str) -> BalanceResult:
    """
    Main entry point for statement processing.

    Args:
        vlm_output: Raw output from VLM extraction
        bank_key: Bank identifier (e.g., "anz", "commbank")

    Returns:
        BalanceResult with extracted and normalized data
    """
    bank = BankType(bank_key)
    adapter = registry.get(bank)

    normalized = adapter.normalize_spatial(vlm_output)
    result = adapter.parse_balances(normalized)
    result.bank_type = bank

    return result


# Optional: Decorator-based registration for cleaner syntax
def register_adapter(bank: BankType):
    """
    Decorator for registering adapters.

    Usage:
        @register_adapter(BankType.ANZ)
        class ANZAdapter(StatementAdapter):
            ...
    """

    def decorator(cls):
        registry.register(bank, cls())
        return cls

    return decorator
