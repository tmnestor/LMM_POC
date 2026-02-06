"""Bank Statement Processing Adapter.

Bridges BatchDocumentProcessor with UnifiedBankExtractor for sophisticated
multi-turn bank statement extraction using InternVL3.

Usage:
    from common.bank_statement_adapter import BankStatementAdapter

    # Pass the hybrid processor - adapter extracts model/tokenizer automatically
    adapter = BankStatementAdapter(hybrid_processor)
    schema_fields, metadata = adapter.extract_bank_statement(image_path)
"""

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
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


@contextmanager
def _bypass_rich_stdout():
    """Context manager to temporarily bypass Rich's stdout proxy.

    UnifiedBankExtractor uses print() statements internally which can
    cause RecursionError when Rich's console file proxy intercepts them.
    This context manager redirects stdout to the original sys.__stdout__
    during extraction to prevent the recursion.
    """
    original_stdout = sys.stdout
    try:
        # Flush Rich's buffer BEFORE switching to prevent output interleaving
        if hasattr(original_stdout, "flush"):
            original_stdout.flush()
        # Redirect stdout to original (bypasses Rich's file proxy)
        sys.stdout = sys.__stdout__
        yield
    finally:
        # Flush raw stdout before restoring
        sys.__stdout__.flush()
        # Restore original stdout (Rich's proxy)
        sys.stdout = original_stdout
        # Flush Rich's buffer after restoration
        if hasattr(sys.stdout, "flush"):
            sys.stdout.flush()


_BSA_INSTANCE_COUNTER = 0


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
        processor: Any = None,
        config_dir: str | Path | None = None,
        verbose: bool = True,
        use_balance_correction: bool = False,
        model_dtype: Any = None,
    ):
        """Initialize adapter with InternVL3 model components.

        Args:
            model: InternVL3 hybrid processor (with model and tokenizer attributes)
            processor: Not used, kept for API compatibility
            config_dir: Path to config directory (default: project config/)
            verbose: Enable verbose output during extraction
            use_balance_correction: Enable mathematical balance correction
            model_dtype: Optional dtype override (e.g., torch.bfloat16, torch.float32)
        """
        global _BSA_INSTANCE_COUNTER
        _BSA_INSTANCE_COUNTER += 1
        self._instance_id = _BSA_INSTANCE_COUNTER

        self.verbose = verbose

        if self.verbose:
            _safe_print(f"[BSA] Created BankStatementAdapter instance #{self._instance_id}")

        # Extract model components from hybrid processor
        if hasattr(model, "model") and hasattr(model, "tokenizer"):
            actual_model = model.model
            actual_tokenizer = model.tokenizer
        else:
            # Direct model/tokenizer passed
            actual_model = model
            actual_tokenizer = processor

        # Auto-detect dtype from model parameters if not specified
        if model_dtype is None:
            try:
                model_dtype = next(actual_model.parameters()).dtype
            except (StopIteration, AttributeError):
                model_dtype = torch.bfloat16  # Default for InternVL3

        self.model = actual_model
        self.tokenizer = actual_tokenizer
        self.processor = None
        self.model_dtype = model_dtype

        # Initialize UnifiedBankExtractor for InternVL3
        self.extractor = UnifiedBankExtractor(
            model=actual_model,
            tokenizer=actual_tokenizer,
            config_dir=config_dir,
            model_dtype=model_dtype,
            use_balance_correction=use_balance_correction,
            verbose=verbose,
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
            _safe_print(f"\n[BSA#{self._instance_id}] >>> START extract_bank_statement({Path(image_path).name})")
            sys.__stdout__.flush()  # Ensure START message appears before extraction

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Execute extraction with stdout bypass to prevent Rich console recursion
        # UnifiedBankExtractor uses print() internally which conflicts with Rich
        with _bypass_rich_stdout():
            result: ExtractionResult = self.extractor.extract(
                image=image,
                force_strategy=force_strategy,
            )

        # Print raw Turn 1 response AFTER bypass context (visible in notebook)
        if self.verbose and result.raw_responses.get("turn1"):
            _safe_print(f"[UBE] Raw Turn 1 response:\n{result.raw_responses['turn1']}\n[UBE] === End raw response ===")

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
            _safe_print(f"[BSA#{self._instance_id}]   Strategy: {result.strategy_used}")
            _safe_print(f"[BSA#{self._instance_id}]   Turns: {result.turns_executed}")
            _safe_print(f"[BSA#{self._instance_id}]   Transactions extracted: {len(result.transaction_dates)}")
            _safe_print(f"[BSA#{self._instance_id}] <<< END extract_bank_statement")
            # Ensure all output is flushed before returning
            sys.__stdout__.flush()

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
    processor: Any = None,
    verbose: bool = True,
    use_balance_correction: bool = False,
    model_dtype: Any = None,
) -> BankStatementAdapter:
    """Factory function to create BankStatementAdapter.

    Args:
        model: InternVL3 hybrid processor (with model and tokenizer attributes)
        processor: Not used, kept for API compatibility
        verbose: Enable verbose output
        use_balance_correction: Enable balance-based correction
        model_dtype: Optional dtype override

    Returns:
        Configured BankStatementAdapter instance
    """
    return BankStatementAdapter(
        model=model,
        processor=processor,
        verbose=verbose,
        use_balance_correction=use_balance_correction,
        model_dtype=model_dtype,
    )
