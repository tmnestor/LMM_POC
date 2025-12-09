"""Bank Statement Processing Adapter.

Bridges BatchDocumentProcessor with UnifiedBankExtractor for sophisticated
multi-turn bank statement extraction.

Supports both Llama and InternVL3 models.

Usage (Llama):
    from common.bank_statement_adapter import BankStatementAdapter

    adapter = BankStatementAdapter(model, processor, model_type="llama")
    schema_fields, metadata = adapter.extract_bank_statement(image_path)

Usage (InternVL3):
    from common.bank_statement_adapter import BankStatementAdapter

    # Pass the hybrid processor - adapter extracts model/tokenizer automatically
    adapter = BankStatementAdapter(hybrid_processor, None, model_type="internvl3")
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


class BankStatementAdapter:
    """Adapter that provides BatchDocumentProcessor-compatible interface to UnifiedBankExtractor.

    This adapter bridges the batch processing notebook with the sophisticated
    multi-turn bank statement extraction in UnifiedBankExtractor.

    Supports both Llama and InternVL3 models.

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
        model_type: str = "llama",
        model_dtype: Any = None,
    ):
        """Initialize adapter with model components.

        Args:
            model: Loaded model. For Llama: the model object. For InternVL3: the hybrid processor.
            processor: For Llama: the processor. For InternVL3: can be None.
            config_dir: Path to config directory (default: project config/)
            verbose: Enable verbose output during extraction
            use_balance_correction: Enable mathematical balance correction
            model_type: "llama" or "internvl3"
            model_dtype: Optional dtype override (e.g., torch.bfloat16, torch.float32)
        """
        self.verbose = verbose
        self.model_type = model_type.lower()

        # Extract model components based on model type
        if self.model_type == "internvl3":
            # For InternVL3, model parameter is the hybrid processor
            # Extract the actual model and tokenizer from it
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
                processor=None,
                model_type="internvl3",
                config_dir=config_dir,
                model_dtype=model_dtype,
                use_balance_correction=use_balance_correction,
            )
        else:
            # Llama model
            self.model = model
            self.processor = processor
            self.tokenizer = processor  # Llama processor serves as tokenizer
            self.model_dtype = model_dtype

            # Initialize UnifiedBankExtractor for Llama
            self.extractor = UnifiedBankExtractor(
                model=model,
                tokenizer=processor,
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
            _safe_print(f"\n[BSA] >>> START BankStatementAdapter.extract_bank_statement({Path(image_path).name})")

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Execute extraction with stdout bypass to prevent Rich console recursion
        # UnifiedBankExtractor uses print() internally which conflicts with Rich
        with _bypass_rich_stdout():
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
            _safe_print(f"[BSA]   Strategy: {result.strategy_used}")
            _safe_print(f"[BSA]   Turns: {result.turns_executed}")
            _safe_print(f"[BSA]   Transactions extracted: {len(result.transaction_dates)}")
            _safe_print("[BSA] <<< END BankStatementAdapter.extract_bank_statement")

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
    model_type: str = "llama",
    model_dtype: Any = None,
) -> BankStatementAdapter:
    """Factory function to create BankStatementAdapter.

    Args:
        model: For Llama: model object. For InternVL3: hybrid processor or model.
        processor: For Llama: processor. For InternVL3: can be None.
        verbose: Enable verbose output
        use_balance_correction: Enable balance-based correction
        model_type: "llama" or "internvl3"
        model_dtype: Optional dtype override

    Returns:
        Configured BankStatementAdapter instance
    """
    return BankStatementAdapter(
        model=model,
        processor=processor,
        verbose=verbose,
        use_balance_correction=use_balance_correction,
        model_type=model_type,
        model_dtype=model_dtype,
    )
