"""
Bank Statement Adapter System

Strategy + Registry pattern for handling bank-specific variations
in VLM-based statement extraction.
"""

from .bank_adapters import (
    AdapterRegistry,
    ANZAdapter,
    BalanceResult,
    BankType,
    CommBankAdapter,
    NABAdapter,
    StatementAdapter,
    WestpacAdapter,
    process_statement,
    register_adapter,
    registry,
)

__all__ = [
    "AdapterRegistry",
    "ANZAdapter",
    "BalanceResult",
    "BankType",
    "CommBankAdapter",
    "NABAdapter",
    "StatementAdapter",
    "WestpacAdapter",
    "process_statement",
    "register_adapter",
    "registry",
]
