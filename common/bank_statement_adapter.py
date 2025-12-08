"""Bank Statement Processing Adapter.

Bridges BatchDocumentProcessor with UnifiedBankExtractor for sophisticated
multi-turn bank statement extraction.

Usage:
    from common.bank_statement_adapter import BankStatementAdapter

    adapter = BankStatementAdapter(model, processor)
    schema_fields, metadata = adapter.extract_bank_statement(image_path)
"""

import sys
from pathlib import Path
from typing import Any

from PIL import Image

from .unified_bank_extractor import (
    ExtractionResult,
    ExtractionStrategy,
    UnifiedBankExtractor,
)


def _safe_print(msg: str) -> None:
    """Print without triggering Rich console recursion in Jupyter.

    Writes directly to sys.__stdout__ to bypass Rich's file proxy
    which can cause RecursionError in Jupyter notebooks.
    """
    try:
        # Use original stdout to bypass Rich's file proxy
        sys.__stdout__.write(msg + "\n")
        sys.__stdout__.flush()
    except Exception:
        # Fallback: silently ignore if even this fails
        pass


class BankStatementAdapter:
    """Adapter that provides BatchDocumentProcessor-compatible interface to UnifiedBankExtractor.

    This adapter bridges the batch processing notebook with the sophisticated
    multi-turn bank statement extraction in UnifiedBankExtractor.

    Features:
        - Turn 0: Header detection (identifies actual column names)
        - Turn 1: Adaptive extraction with structure-dynamic prompts
        - Automatic strategy selection: BALANCE_DESCRIPTION, AMOUNT_DESCRIPTION, etc.
        - Optional balance-based mathematical correction
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        config_dir: str | Path | None = None,
        verbose: bool = True,
        use_balance_correction: bool = False,
    ):
        """Initialize adapter with Llama model components.

        Args:
            model: Loaded Llama model
            processor: Loaded Llama processor
            config_dir: Path to config directory (default: project config/)
            verbose: Enable verbose output during extraction
            use_balance_correction: Enable mathematical balance correction
        """
        self.model = model
        self.processor = processor
        self.verbose = verbose

        # Initialize UnifiedBankExtractor for Llama
        self.extractor = UnifiedBankExtractor(
            model=model,
            tokenizer=processor,  # Llama processor serves as tokenizer
            processor=processor,
            model_type="llama",
            config_dir=config_dir,
            use_balance_correction=use_balance_correction,
        )

    def extract_bank_statement(
        self,
        image_path: str | Path,
        force_strategy: ExtractionStrategy | None = None,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """Extract bank statement using multi-turn strategy.

        Args:
            image_path: Path to bank statement image
            force_strategy: Optional strategy override (default: auto-select)

        Returns:
            Tuple of (schema_fields, metadata)
            - schema_fields: Dict with DOCUMENT_TYPE, STATEMENT_DATE_RANGE,
              TRANSACTION_DATES, LINE_ITEM_DESCRIPTIONS, TRANSACTION_AMOUNTS_PAID
            - metadata: Dict with strategy_used, turns_executed, headers_detected,
              column_mapping, raw_responses, correction_stats
        """
        if self.verbose:
            _safe_print(f"\n[BankStatementAdapter] Processing: {Path(image_path).name}")

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Execute extraction
        result: ExtractionResult = self.extractor.extract(
            image=image,
            force_strategy=force_strategy,
        )

        # Convert to schema dict
        schema_fields = result.to_schema_dict()

        # Build metadata for batch processor compatibility
        metadata = {
            "strategy_used": result.strategy_used,
            "turns_executed": result.turns_executed,
            "headers_detected": result.headers_detected,
            "column_mapping": self._serialize_column_mapping(result.column_mapping),
            "raw_responses": result.raw_responses,
        }

        if result.correction_stats:
            metadata["correction_stats"] = str(result.correction_stats)

        if self.verbose:
            _safe_print(f"  Strategy: {result.strategy_used}")
            _safe_print(f"  Turns: {result.turns_executed}")
            _safe_print(f"  Transactions extracted: {len(result.transaction_dates)}")

        return schema_fields, metadata

    @staticmethod
    def _serialize_column_mapping(column_mapping) -> dict[str, str | None]:
        """Serialize ColumnMapping to dict for JSON compatibility."""
        if column_mapping is None:
            return {}
        return {
            "date": column_mapping.date,
            "description": column_mapping.description,
            "debit": column_mapping.debit,
            "credit": column_mapping.credit,
            "balance": column_mapping.balance,
            "amount": column_mapping.amount,
        }


def create_bank_adapter(
    model: Any,
    processor: Any,
    verbose: bool = True,
    use_balance_correction: bool = False,
) -> BankStatementAdapter:
    """Factory function to create BankStatementAdapter.

    Args:
        model: Loaded Llama model
        processor: Loaded Llama processor
        verbose: Enable verbose output
        use_balance_correction: Enable balance-based correction

    Returns:
        Configured BankStatementAdapter instance
    """
    return BankStatementAdapter(
        model=model,
        processor=processor,
        verbose=verbose,
        use_balance_correction=use_balance_correction,
    )
