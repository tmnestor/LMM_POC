"""Unified Bank Statement Extractor.

Automatically selects the optimal extraction strategy based on document characteristics.
Follows the 2-turn balance-description approach when Balance column is detected.

Also serves as the pipeline entry point (absorbs the former BankStatementAdapter).

Usage:
    from common.unified_bank_extractor import UnifiedBankExtractor

    extractor = UnifiedBankExtractor(generate_fn=processor.generate)
    schema, metadata = extractor.extract_bank_statement(image_path)
"""

import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

from .bank_corrector import BalanceCorrector, TransactionFilter
from .bank_types import ColumnMapping, ExtractionResult, ExtractionStrategy


def _safe_print(msg: str) -> None:
    """Print without triggering Rich console recursion in Jupyter.

    Writes directly to sys.__stdout__ to bypass Rich's file proxy
    which can cause RecursionError in Jupyter notebooks.
    """
    try:
        stdout = sys.__stdout__
        if stdout is not None:
            stdout.write(msg + "\n")
            stdout.flush()
    except Exception:
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
        if hasattr(original_stdout, "flush"):
            original_stdout.flush()
        sys.stdout = sys.__stdout__
        yield
    finally:
        sys.__stdout__.flush()
        sys.stdout = original_stdout
        if hasattr(sys.stdout, "flush"):
            sys.stdout.flush()


class ConfigLoader:
    """Load YAML configuration files from config/ directory."""

    def __init__(self, config_dir: str | Path | None = None):
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "config"
        self.config_dir = Path(config_dir)

    def load(self, filename: str) -> dict:
        """Load a YAML config file."""
        path = self.config_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open(encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_prompt(self, prompt_key: str) -> str:
        """Get a prompt template from bank_prompts.yaml."""
        config = self.load("bank_prompts.yaml")
        prompts = config.get("prompts", {})
        if prompt_key not in prompts:
            raise KeyError(f"Prompt '{prompt_key}' not found")
        return prompts[prompt_key]["template"]

    def get_column_patterns(self) -> dict:
        """Get column pattern configuration."""
        config = self.load("bank_column_patterns.yaml")
        return config.get("patterns", {})


class ColumnMatcher:
    """Match detected headers to semantic column types."""

    def __init__(self, patterns: dict[str, dict] | None = None):
        if patterns is None:
            loader = ConfigLoader()
            patterns = loader.get_column_patterns()
        self.patterns = patterns

    def match(self, headers: list[str]) -> ColumnMapping:
        """Match headers to semantic column types."""
        mapping = ColumnMapping()
        headers_lower = [h.lower() for h in headers]

        for col_type, config in self.patterns.items():
            keywords = config.get("keywords", [])
            matched = self._find_match(headers, headers_lower, keywords)
            if matched:
                setattr(mapping, col_type, matched)

        # NOTE: Removed fallback that set debit=amount
        # Strategy selection now properly handles Amount-only statements
        # using AMOUNT_DESCRIPTION strategy instead of pretending Amount is Debit

        return mapping

    def _find_match(
        self, headers: list[str], headers_lower: list[str], keywords: list[str]
    ) -> str | None:
        """Find matching header using keywords.

        Only matches headers that look like actual column headers (short text),
        not transaction content (long descriptions).
        """
        # Exact match first
        for keyword in keywords:
            for i, header_lower in enumerate(headers_lower):
                if keyword == header_lower:
                    return headers[i]

        # Substring match - but only for short headers (likely actual column names)
        # Skip headers longer than 20 chars as they're likely transaction content
        for keyword in keywords:
            if len(keyword) > 2:
                for i, header_lower in enumerate(headers_lower):
                    # Only match short headers (actual column names)
                    if len(header_lower) <= 20 and keyword in header_lower:
                        return headers[i]

        return None


class ResponseParser:
    """Parse LLM responses into structured data."""

    @staticmethod
    def parse_headers(response: str) -> list[str]:
        """Parse Turn 0 header detection response.

        Handles multiple response formats from different VLMs:
        - One header per line (InternVL3 style): "Date\\nDescription\\nDebit"
        - Comma-separated single line (Qwen3-VL style): "Date, Description, Debit"
        - Pipe-separated (some models): "Date | Description | Debit"
        """
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        headers: list[str] = []

        for line in lines:
            # Remove numbering and bullets
            cleaned = line.lstrip("0123456789.-•* ").strip()
            # Remove markdown formatting
            cleaned = cleaned.replace("**", "").replace("__", "")
            # Skip labels like "Headers:" or empty/too-long lines
            if cleaned.endswith(":"):
                continue

            # If a single line contains multiple comma-separated or
            # pipe-separated items, split them into individual headers.
            # Heuristic: line has 2+ commas/pipes AND is long (>20 chars).
            if len(cleaned) > 20 and (
                cleaned.count(",") >= 2 or cleaned.count("|") >= 2
            ):
                sep = "|" if cleaned.count("|") >= 2 else ","
                for part in cleaned.split(sep):
                    part = part.strip()
                    if part and len(part) > 1:
                        headers.append(part)
                continue

            if len(cleaned) > 40:
                continue
            if cleaned and len(cleaned) > 2:
                headers.append(cleaned)

        return headers

    @staticmethod
    def parse_balance_description(
        response: str,
        date_col: str,
        desc_col: str,
        debit_col: str,
        credit_col: str,
        balance_col: str,
    ) -> list[dict[str, str]]:
        """Parse balance-description response into transaction rows.

        Handles multi-line descriptions where continuation lines appear as:
            - Transaction: Card Purchase K MART
              - Perth WA
        These are joined with a space to form: "Card Purchase K MART Perth WA"
        """
        rows = []
        current_date = None
        current_transaction: dict[str, str] = {}
        last_field_was_description = False  # Track if we should append continuations

        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # DATE DETECTION
            date_found = None

            # Pattern 1: "1. **Date:** 03/05/2025" or "**Date:** 03/05/2025"
            date_field_match = re.match(
                r"^\d*\.?\s*\*?\*?Date:?\*?\*?\s*(.+)$", line, re.IGNORECASE
            )
            if date_field_match:
                date_found = date_field_match.group(1).strip().strip("*").strip()

            # Pattern 2: Bold date "**03/05/2025**"
            if not date_found:
                bold_date_match = re.match(r"^\*\*(\d{1,2}/\d{1,2}/\d{4})\*\*$", line)
                if bold_date_match:
                    date_found = bold_date_match.group(1)

            # Pattern 3: Numbered bold date "1. **Thu 04 Sep 2025**"
            if not date_found:
                date_match = re.match(
                    r"^\d+\.\s*\*?\*?([A-Za-z]{3}\s+\d{1,2}\s+[A-Za-z]{3}\s+\d{4})\*?\*?",
                    line,
                )
                if date_match:
                    date_found = date_match.group(1).strip()

            # Pattern 4: "1. **04 Sep 2025**"
            if not date_found:
                date_match = re.match(
                    r"^\d+\.\s*\*?\*?(\d{1,2}\s+[A-Za-z]{3}\s+\d{4})\*?\*?", line
                )
                if date_match:
                    date_found = date_match.group(1).strip()

            # Pattern 5: "1. **03/05/2025**"
            if not date_found:
                date_match = re.match(
                    r"^\d+\.\s*\*?\*?(\d{1,2}/\d{1,2}/\d{4})\*?\*?", line
                )
                if date_match:
                    date_found = date_match.group(1).strip()

            # Pattern 6: "1. **[20 May]**" (bracketed date without year)
            if not date_found:
                date_match = re.match(
                    r"^\d+\.\s*\*?\*?\[(\d{1,2}\s+[A-Za-z]{3,9})\]\*?\*?", line
                )
                if date_match:
                    date_found = date_match.group(1).strip()

            # Pattern 7: "1. **20 May**" (date without year)
            if not date_found:
                date_match = re.match(
                    r"^\d+\.\s*\*?\*?(\d{1,2}\s+[A-Za-z]{3,9})\*?\*?$", line
                )
                if date_match:
                    date_found = date_match.group(1).strip()

            # Pattern 8: "1. **06 Aug 24**" (date with 2-digit year)
            if not date_found:
                date_match = re.match(
                    r"^\d+\.\s*\*?\*?(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2})\*?\*?$", line
                )
                if date_match:
                    date_found = date_match.group(1).strip()

            if date_found:
                # Save previous transaction
                if current_transaction and current_date:
                    current_transaction[date_col] = current_date
                    rows.append(current_transaction)
                    current_transaction = {}
                current_date = date_found
                last_field_was_description = False
                continue

            # FIELD DETECTION
            field_name = None
            field_value = None

            # Pattern 1: "**Description:** value"
            bold_field_match = re.match(
                r"^\s*\*\*([^*:]+)(?:\s*Amount)?:?\*\*\s*(.+)$", line, re.IGNORECASE
            )
            if bold_field_match:
                field_name = bold_field_match.group(1).strip().lower()
                field_value = bold_field_match.group(2).strip()

            # Pattern 2: "* Description: value"
            if not field_name:
                asterisk_match = re.match(r"^\s*\*\s*([^:]+):\s*(.+)$", line)
                if asterisk_match:
                    field_name = asterisk_match.group(1).strip().lower()
                    field_value = asterisk_match.group(2).strip()

            # Pattern 3: "- Description: value"
            if not field_name:
                dash_match = re.match(r"^\s*-\s*([^:]+):\s*(.+)$", line)
                if dash_match:
                    field_name = dash_match.group(1).strip().lower()
                    field_value = dash_match.group(2).strip()

            if field_name and field_value:
                # Normalize field name
                field_name = field_name.replace(" amount", "").strip()

                # Map to columns
                if field_name in [
                    "description",
                    "transaction",
                    "details",
                    "particulars",
                    desc_col.lower(),
                ]:
                    # New transaction under same date
                    if (
                        desc_col in current_transaction
                        and current_transaction[desc_col]
                    ):
                        if current_date:
                            current_transaction[date_col] = current_date
                        rows.append(current_transaction)
                        current_transaction = {}
                    current_transaction[desc_col] = field_value
                    last_field_was_description = True

                elif field_name in [
                    "debit",
                    "withdrawal",
                    "withdrawwal",
                    "dr",
                    debit_col.lower(),
                ]:
                    current_transaction[debit_col] = field_value
                    last_field_was_description = False

                elif field_name in ["credit", "deposit", "cr", credit_col.lower()]:
                    current_transaction[credit_col] = field_value
                    last_field_was_description = False

                elif field_name == "balance":
                    current_transaction[balance_col] = field_value
                    last_field_was_description = False

                elif field_name == "amount":
                    if debit_col not in current_transaction:
                        current_transaction[debit_col] = field_value
                    last_field_was_description = False

            else:
                # CONTINUATION LINE DETECTION (no field name, just "- value")
                # Handles InternVL3.5 format like:
                #   - Transaction: Card Purchase K MART
                #     - Perth WA
                continuation_match = re.match(r"^\s*-\s+(.+)$", line)
                if continuation_match and last_field_was_description:
                    continuation_text = continuation_match.group(1).strip()
                    # Append to existing description with a space
                    if desc_col in current_transaction:
                        current_transaction[desc_col] += " " + continuation_text

        # Don't forget last transaction
        if current_transaction and current_date:
            current_transaction[date_col] = current_date
            rows.append(current_transaction)

        return rows

    @staticmethod
    def parse_amount_description(
        response: str,
        date_col: str,
        desc_col: str,
        amount_col: str,
        balance_col: str | None = None,
    ) -> list[dict[str, str]]:
        """Parse amount-description response into transaction rows.

        Handles statements with signed Amount column (negative = withdrawal).
        Optionally parses Balance column if provided.
        Supports multi-line descriptions with continuation lines.
        """
        rows = []
        current_date = None
        current_transaction: dict[str, str] = {}
        last_field_was_description = False

        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # DATE DETECTION (same patterns as balance-description)
            date_found = None

            # Pattern 1: "1. **Date:** 03/05/2025"
            date_field_match = re.match(
                r"^\d*\.?\s*\*?\*?Date:?\*?\*?\s*(.+)$", line, re.IGNORECASE
            )
            if date_field_match:
                date_found = date_field_match.group(1).strip().strip("*").strip()

            # Pattern 2: Numbered bold date "1. **03/05/2025**" or "1. **03 Jun 2023**"
            if not date_found:
                date_match = re.match(
                    r"^\d+\.\s*\*?\*?(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\*?\*?", line
                )
                if date_match:
                    date_found = date_match.group(1).strip()

            # Pattern 3: "1. **04 Sep 2025**" or "1. **03 Jun 2023**" (4-digit year)
            if not date_found:
                date_match = re.match(
                    r"^\d+\.\s*\*?\*?(\d{1,2}\s+[A-Za-z]{3}\s+\d{4})\*?\*?", line
                )
                if date_match:
                    date_found = date_match.group(1).strip()

            # Pattern 4: "1. **[20 May]**" (bracketed date without year)
            if not date_found:
                date_match = re.match(
                    r"^\d+\.\s*\*?\*?\[(\d{1,2}\s+[A-Za-z]{3,9})\]\*?\*?", line
                )
                if date_match:
                    date_found = date_match.group(1).strip()

            # Pattern 5: "1. **20 May**" (date without year)
            if not date_found:
                date_match = re.match(
                    r"^\d+\.\s*\*?\*?(\d{1,2}\s+[A-Za-z]{3,9})\*?\*?$", line
                )
                if date_match:
                    date_found = date_match.group(1).strip()

            # Pattern 6: "1. **06 Aug 24**" (date with 2-digit year)
            if not date_found:
                date_match = re.match(
                    r"^\d+\.\s*\*?\*?(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2})\*?\*?$", line
                )
                if date_match:
                    date_found = date_match.group(1).strip()

            if date_found:
                # Save previous transaction
                if current_transaction and current_date:
                    current_transaction[date_col] = current_date
                    rows.append(current_transaction)
                    current_transaction = {}
                current_date = date_found
                last_field_was_description = False
                continue

            # FIELD DETECTION
            field_name = None
            field_value = None

            # Pattern: "- Description: value" or "* Description: value"
            field_match = re.match(r"^\s*[-*]\s*([^:]+):\s*(.+)$", line)
            if field_match:
                field_name = field_match.group(1).strip().lower()
                field_value = field_match.group(2).strip()

            if field_name and field_value:
                # Description/Transaction details
                if field_name in [
                    "description",
                    "details",
                    "transaction details",
                    desc_col.lower(),
                ]:
                    # New transaction under same date
                    if (
                        desc_col in current_transaction
                        and current_transaction[desc_col]
                    ):
                        if current_date:
                            current_transaction[date_col] = current_date
                        rows.append(current_transaction)
                        current_transaction = {}
                    current_transaction[desc_col] = field_value
                    last_field_was_description = True

                # Amount (signed value)
                elif field_name in ["amount", amount_col.lower()]:
                    current_transaction[amount_col] = field_value
                    last_field_was_description = False

                # Balance (optional)
                elif balance_col and field_name in ["balance", balance_col.lower()]:
                    current_transaction[balance_col] = field_value
                    last_field_was_description = False

                # Skip other labeled fields (Card, Value Date, etc.) - not description continuation
                else:
                    last_field_was_description = False

            else:
                # UNLABELED LINE DETECTION (no colon, just "- value")
                # Handles LLM format where description has no label:
                #   1. **03 Jun 2023**
                #      - GROCERY MARKET SUBURB NSW AUS  ← First unlabeled = description
                #      - Card xx5678                     ← Could be continuation
                unlabeled_match = re.match(r"^\s*-\s+(.+)$", line)
                if unlabeled_match:
                    unlabeled_text = unlabeled_match.group(1).strip()

                    # If no description yet, this IS the description
                    if (
                        desc_col not in current_transaction
                        or not current_transaction.get(desc_col)
                    ):
                        current_transaction[desc_col] = unlabeled_text
                        last_field_was_description = True
                    # If we already have description and last field was description, append
                    elif last_field_was_description:
                        current_transaction[desc_col] += " " + unlabeled_text

        # Don't forget last transaction
        if current_transaction and current_date:
            current_transaction[date_col] = current_date
            rows.append(current_transaction)

        return rows


class UnifiedBankExtractor:
    """Unified bank statement extractor with automatic strategy selection.

    Also serves as the pipeline entry point (replaces the former BankStatementAdapter).

    Args:
        generate_fn: Callable(image, prompt, max_tokens) -> str.
            Typically ``processor.generate`` from a DocumentProcessor.
        config_dir: Path to config directory (optional)
        use_balance_correction: Enable balance-based mathematical correction
        verbose: Enable verbose output

    Example:
        extractor = UnifiedBankExtractor(generate_fn=processor.generate)
        schema, metadata = extractor.extract_bank_statement(image_path)
    """

    def __init__(
        self,
        generate_fn: Any,
        config_dir: str | Path | None = None,
        use_balance_correction: bool = False,
        verbose: bool = True,
    ):
        self.generate_fn = generate_fn
        self.use_balance_correction = use_balance_correction
        self.verbose = verbose

        self.config_loader = ConfigLoader(config_dir)
        self.column_matcher = ColumnMatcher()
        self.parser = ResponseParser()
        self.filter = TransactionFilter()

        # Load prompts
        self._prompts = {
            "turn0": self.config_loader.get_prompt("turn0_header_detection"),
            "turn1_balance": self.config_loader.get_prompt("turn1_balance_extraction"),
            "turn1_amount": self.config_loader.get_prompt("turn1_amount_extraction"),
            "turn1_debit_credit": self.config_loader.get_prompt(
                "turn1_debit_credit_extraction"
            ),
            "schema_fallback": self.config_loader.get_prompt(
                "schema_fallback_extraction"
            ),
        }

    def _log(self, msg: str) -> None:
        """Log a message to stdout only when verbose is enabled."""
        stdout = sys.__stdout__
        if self.verbose and stdout is not None:
            stdout.write(msg + "\n")
            stdout.flush()

    def extract(
        self,
        image: Any,
        force_strategy: ExtractionStrategy | None = None,
    ) -> ExtractionResult:
        """Extract bank statement data using optimal strategy.

        Args:
            image: PIL Image or path to image
            force_strategy: Optional manual strategy override

        Returns:
            ExtractionResult with extracted data
        """
        import torch
        from PIL import Image as PILImage

        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = PILImage.open(image).convert("RGB")

        # Turn 0: Header detection
        self._log("[UBE] Turn 0: Detecting headers...")
        turn0_response = self.generate_fn(image, self._prompts["turn0"], max_tokens=500)
        headers = self.parser.parse_headers(turn0_response)
        self._log(f"  Detected {len(headers)} headers: {headers}")

        # Map headers to columns
        mapping = self.column_matcher.match(headers)
        self._log(f"  Balance column: {mapping.balance or 'NOT FOUND'}")

        # Select strategy based on detected columns
        # Key insight: Some statements have Amount+Balance (signed values),
        # others have Debit/Credit/Balance (separate columns)
        has_debit_or_credit = mapping.debit or mapping.credit
        has_amount = mapping.amount is not None

        # DEBUG: Show column detection
        self._log(f"  Amount column: {mapping.amount or 'NOT FOUND'}")
        self._log(f"  Debit column: {mapping.debit or 'NOT FOUND'}")
        self._log(f"  Credit column: {mapping.credit or 'NOT FOUND'}")
        self._log(
            f"  has_balance={mapping.has_balance}, has_amount={has_amount}, has_debit_or_credit={has_debit_or_credit}"
        )

        if force_strategy:
            strategy = force_strategy
            reason = "Manual override"
        elif mapping.has_balance and has_debit_or_credit and not has_amount:
            # Standard format: Debit/Credit columns with Balance
            strategy = ExtractionStrategy.BALANCE_DESCRIPTION
            reason = "Balance + Debit/Credit columns detected"
        elif mapping.has_balance and has_amount and not has_debit_or_credit:
            # CBA-style format: Single Amount column (signed) with Balance
            strategy = ExtractionStrategy.AMOUNT_DESCRIPTION
            reason = "Balance + Amount column detected (signed values)"
        elif mapping.has_balance and has_debit_or_credit:
            # Has both Amount and Debit/Credit - prefer Debit/Credit
            strategy = ExtractionStrategy.BALANCE_DESCRIPTION
            reason = "Balance + Debit/Credit columns detected"
        elif has_amount:
            # Amount column without balance
            strategy = ExtractionStrategy.AMOUNT_DESCRIPTION
            reason = "Amount column detected (no balance)"
        elif has_debit_or_credit:
            # Debit/Credit without balance
            strategy = ExtractionStrategy.DEBIT_CREDIT_DESCRIPTION
            reason = "Debit/Credit columns detected (no balance)"
        else:
            strategy = ExtractionStrategy.TABLE_EXTRACTION
            reason = "Schema fallback (column detection failed)"

        self._log(f"[UBE] Strategy: {strategy.name} ({reason})")

        # Execute strategy
        if strategy == ExtractionStrategy.BALANCE_DESCRIPTION:
            result = self._extract_balance_description(
                image, headers, mapping, turn0_response
            )
        elif strategy == ExtractionStrategy.AMOUNT_DESCRIPTION:
            result = self._extract_amount_description(
                image, headers, mapping, turn0_response
            )
        elif strategy == ExtractionStrategy.DEBIT_CREDIT_DESCRIPTION:
            result = self._extract_debit_credit_description(
                image, headers, mapping, turn0_response
            )
        else:
            # Schema-based fallback when column detection fails
            result = self._extract_schema_fallback(image, headers, mapping)

        # Free GPU memory
        torch.cuda.empty_cache()

        return result

    def _extract_balance_description(
        self,
        image: Any,
        headers: list[str],
        mapping: ColumnMapping,
        turn0_response: str,
    ) -> ExtractionResult:
        """Execute 2-turn balance-description extraction with balance correction."""
        import torch

        self._log("[UBE] >>> ENTERING _extract_balance_description")

        # Build Turn 1 prompt with actual column names
        prompt_template = self._prompts["turn1_balance"]
        prompt = prompt_template.format(
            balance_col=mapping.balance,
            desc_col=mapping.description or "Description",
            debit_col=mapping.debit or "Debit",
            credit_col=mapping.credit or "Credit",
        )

        self._log(f"[UBE] Turn 1 Prompt:\n{prompt}")
        self._log("[UBE] Turn 1: Calling model for extraction...")
        try:
            response = self.generate_fn(image, prompt, max_tokens=4096)
            self._log(f"[UBE]   Raw response length: {len(response)} chars")
            self._log(f"[UBE]   Response preview:\n{response[:500]}...")
        except Exception as e:
            self._log(f"[UBE] ERROR in _generate: {type(e).__name__}: {e}")
            raise

        # Column name shortcuts
        date_col = mapping.date or "Date"
        desc_col = mapping.description or "Description"
        debit_col = mapping.debit or "Debit"
        credit_col = mapping.credit or "Credit"
        balance_col = mapping.balance or "Balance"

        # Parse response
        self._log("[UBE] Parsing response...")
        all_rows = self.parser.parse_balance_description(
            response,
            date_col=date_col,
            desc_col=desc_col,
            debit_col=debit_col,
            credit_col=credit_col,
            balance_col=balance_col,
        )
        self._log(f"[UBE]   Parsed {len(all_rows)} total transactions")
        for i, row in enumerate(all_rows[:3]):
            self._log(f"[UBE]     Row {i}: {row}")
        if len(all_rows) > 3:
            self._log(f"[UBE]     ... and {len(all_rows) - 3} more rows")

        # Optionally apply balance correction (sort to chronological order first)
        correction_stats = None
        self._log(f"[UBE]   use_balance_correction={self.use_balance_correction}")
        if self.use_balance_correction:
            # Check if transactions are in chronological order
            is_chrono, order_reason = BalanceCorrector.is_chronological_order(
                all_rows, date_col
            )
            self._log(f"  Date order: {order_reason}")

            # Sort to chronological order if not already
            if is_chrono:
                sorted_rows = all_rows
            else:
                sorted_rows = BalanceCorrector.sort_by_date(all_rows, date_col)
                self._log("  Sorted to chronological order for balance correction")

            # Apply balance correction on sorted (chronological) rows
            corrector = BalanceCorrector()
            corrected_rows, correction_stats = corrector.correct_transactions(
                sorted_rows,
                balance_col=balance_col,
                debit_col=debit_col,
                credit_col=credit_col,
                desc_col=desc_col,
            )
            self._log(f"  Balance correction: {correction_stats}")
            self._log("[UBE]   Corrected rows:")
            for i, row in enumerate(corrected_rows):
                debit_val = row.get(debit_col, "")
                credit_val = row.get(credit_col, "")
                desc = row.get(desc_col, "")[:30]
                self._log(
                    f"[UBE]     {i}: D={debit_val or 'N/A':10} C={credit_val or 'N/A':10} {desc}"
                )
        else:
            corrected_rows = all_rows

        # Filter for debits (from corrected rows)
        debit_rows = self.filter.filter_debits(
            corrected_rows,
            debit_col=debit_col,
            desc_col=desc_col,
        )
        self._log(f"[UBE]   After correction: {len(corrected_rows)} rows")
        self._log(f"[UBE]   After debit filter: {len(debit_rows)} debit transactions")
        self._log("[UBE]   Debit rows:")
        for i, row in enumerate(debit_rows):
            debit_val = row.get(debit_col, "")
            desc = row.get(desc_col, "")[:30]
            self._log(f"[UBE]     {i}: D={debit_val:10} {desc}")

        # Extract schema fields - use consistent filtering to ensure all arrays have same length
        # Only include rows that have the minimum required fields (date AND description AND debit)
        dates = []
        descriptions = []
        amounts = []
        balances = []
        for r in debit_rows:
            date_val = r.get(date_col, "")
            desc_val = r.get(desc_col, "")
            debit_val = r.get(debit_col, "")
            balance_val = r.get(balance_col, "") if balance_col else ""

            # Only include if we have the minimum required fields
            if date_val and desc_val and debit_val:
                dates.append(date_val)
                descriptions.append(desc_val)
                amounts.append(debit_val)
                # Always append balance (even if empty) to maintain array alignment
                balances.append(balance_val if balance_val else "NOT_FOUND")

        # Calculate date range from ALL parsed transactions (including opening/closing balance)
        # Use all_rows, not corrected_rows, to include full statement period
        all_dates = [r.get(date_col, "") for r in all_rows if r.get(date_col)]
        date_range = self._compute_date_range(all_dates) if all_dates else "NOT_FOUND"

        self._log(
            f"[UBE]   Final arrays: dates={len(dates)}, desc={len(descriptions)}, amounts={len(amounts)}, balances={len(balances)}"
        )
        self._log(f"[UBE]   Date range: {date_range}")
        self._log("[UBE] <<< EXITING _extract_balance_description")

        # Free memory
        torch.cuda.empty_cache()

        return ExtractionResult(
            statement_date_range=date_range,
            transaction_dates=dates,
            line_item_descriptions=descriptions,
            transaction_amounts_paid=amounts,
            account_balances=balances,
            strategy_used="balance_description_2turn",
            turns_executed=2,
            headers_detected=headers,
            column_mapping=mapping,
            raw_responses={"turn0": turn0_response, "turn1": response},
            correction_stats=correction_stats,
        )

    def _extract_amount_description(
        self,
        image: Any,
        headers: list[str],
        mapping: ColumnMapping,
        turn0_response: str,
    ) -> ExtractionResult:
        """Execute 2-turn amount-description extraction for Amount-only statements.

        Used when statement has a single Amount column with signed values
        (negative = withdrawal, positive = deposit) instead of separate Debit/Credit.
        Optionally includes Balance column for validation.
        """
        import torch

        # Column name shortcuts
        date_col = mapping.date or "Date"
        desc_col = mapping.description or "Description"
        amount_col = mapping.amount or "Amount"
        balance_col = mapping.balance  # May be None

        # Build Turn 1 prompt with optional balance
        prompt_template = self._prompts["turn1_amount"]
        if balance_col:
            balance_line = f"- {balance_col}"
            balance_format = f"- {balance_col}: [balance amount]"
        else:
            balance_line = ""
            balance_format = ""

        prompt = prompt_template.format(
            amount_col=amount_col,
            desc_col=desc_col,
            balance_line=balance_line,
            balance_format=balance_format,
        )

        self._log("[UBE] Turn 1: Extracting transactions (amount-description)...")
        response = self.generate_fn(image, prompt, max_tokens=4096)
        # Note: Raw response is printed by BankStatementAdapter after bypass context

        # Parse response
        all_rows = self.parser.parse_amount_description(
            response,
            date_col=date_col,
            desc_col=desc_col,
            amount_col=amount_col,
            balance_col=balance_col,
        )
        self._log(f"[UBE]   Parsed {len(all_rows)} transactions")
        # DEBUG: Show first few parsed rows
        for i, row in enumerate(all_rows[:3]):
            self._log(f"[UBE]     Row {i}: {row}")
        if len(all_rows) > 3:
            self._log(f"[UBE]     ... and {len(all_rows) - 3} more rows")

        # Filter for negative amounts (withdrawals/debits)
        # For bank statements with signed Amount column:
        # Negative = withdrawals/debits (money out) - what we want
        # Positive = deposits/credits (money in) - excluded
        debit_rows = self.filter.filter_negative_amounts(
            all_rows,
            amount_col=amount_col,
            desc_col=desc_col,
        )
        self._log(
            f"[UBE]   Filtered to {len(debit_rows)} debit transactions (negative amounts)"
        )

        # Extract schema fields - use consistent filtering to ensure all arrays have same length
        # Only include rows that have the minimum required fields (date AND description AND amount)
        dates = []
        descriptions = []
        amounts = []
        balances = []
        for r in debit_rows:
            date_val = r.get(date_col, "")
            desc_val = r.get(desc_col, "")
            amount_val = r.get(amount_col, "")
            balance_val = r.get(balance_col, "") if balance_col else ""

            # Only include if we have the minimum required fields
            if date_val and desc_val and amount_val:
                dates.append(date_val)
                descriptions.append(desc_val)
                amounts.append(self._format_debit_amount(amount_val))
                # Always append balance (even if empty) to maintain array alignment
                balances.append(balance_val if balance_val else "NOT_FOUND")

        # Date range from all transactions
        all_dates = [r.get(date_col, "") for r in all_rows if r.get(date_col)]
        date_range = self._compute_date_range(all_dates) if all_dates else "NOT_FOUND"

        # DEBUG: Verify array lengths are consistent
        self._log(
            f"[UBE]   Array lengths: dates={len(dates)}, desc={len(descriptions)}, amounts={len(amounts)}, balances={len(balances)}"
        )

        torch.cuda.empty_cache()

        return ExtractionResult(
            statement_date_range=date_range,
            transaction_dates=dates,
            line_item_descriptions=descriptions,
            transaction_amounts_paid=amounts,
            account_balances=balances,
            strategy_used="amount_description_2turn",
            turns_executed=2,
            headers_detected=headers,
            column_mapping=mapping,
            raw_responses={"turn0": turn0_response, "turn1": response},
        )

    def _extract_debit_credit_description(
        self,
        image: Any,
        headers: list[str],
        mapping: ColumnMapping,
        turn0_response: str,
    ) -> ExtractionResult:
        """Execute 2-turn debit-credit extraction for statements without balance."""
        import torch

        self._log("[UBE] >>> ENTERING _extract_debit_credit_description")

        # Build Turn 1 prompt with actual column names
        prompt_template = self._prompts["turn1_debit_credit"]
        prompt = prompt_template.format(
            debit_col=mapping.debit or "Debit",
            credit_col=mapping.credit or "Credit",
            desc_col=mapping.description or "Transaction",
        )

        self._log(f"[UBE] Turn 1 Prompt:\n{prompt}")
        self._log("[UBE] Turn 1: Calling model for extraction (debit-credit)...")
        response = self.generate_fn(image, prompt, max_tokens=4096)
        self._log(f"[UBE]   Raw response length: {len(response)} chars")
        self._log(f"[UBE]   Response preview:\n{response[:500]}...")

        # Column name shortcuts
        date_col = mapping.date or "Date"
        desc_col = mapping.description or "Transaction"
        debit_col = mapping.debit or "Debit"
        credit_col = mapping.credit or "Credit"

        # Parse response - reuse balance-description parser (same format)
        self._log("[UBE] Parsing response...")
        all_rows = self.parser.parse_balance_description(
            response,
            date_col=date_col,
            desc_col=desc_col,
            debit_col=debit_col,
            credit_col=credit_col,
            balance_col="Balance",  # Placeholder, not used
        )
        self._log(f"[UBE]   Parsed {len(all_rows)} transactions")
        for i, row in enumerate(all_rows[:3]):
            self._log(f"[UBE]     Row {i}: {row}")
        if len(all_rows) > 3:
            self._log(f"[UBE]     ... and {len(all_rows) - 3} more rows")

        # Filter for debits
        debit_rows = self.filter.filter_debits(
            all_rows,
            debit_col=debit_col,
            desc_col=desc_col,
        )
        self._log(f"[UBE]   Filtered to {len(debit_rows)} debit transactions")

        # Extract schema fields - use consistent filtering to ensure all arrays have same length
        # Only include rows that have the minimum required fields (date AND description AND debit)
        dates = []
        descriptions = []
        amounts = []
        balances: list[str] = []  # No balance column in debit/credit strategy
        for r in debit_rows:
            date_val = r.get(date_col, "")
            desc_val = r.get(desc_col, "")
            debit_val = r.get(debit_col, "")

            # Only include if we have the minimum required fields
            if date_val and desc_val and debit_val:
                dates.append(date_val)
                descriptions.append(desc_val)
                amounts.append(debit_val)
                # No balance in this strategy, but maintain alignment
                balances.append("NOT_FOUND")

        # Date range from all transactions
        all_dates = [r.get(date_col, "") for r in all_rows if r.get(date_col)]
        date_range = self._compute_date_range(all_dates) if all_dates else "NOT_FOUND"

        self._log(
            f"[UBE]   Final arrays: dates={len(dates)}, desc={len(descriptions)}, amounts={len(amounts)}, balances={len(balances)}"
        )
        self._log(f"[UBE]   Date range: {date_range}")
        self._log("[UBE] <<< EXITING _extract_debit_credit_description")

        torch.cuda.empty_cache()

        return ExtractionResult(
            statement_date_range=date_range,
            transaction_dates=dates,
            line_item_descriptions=descriptions,
            transaction_amounts_paid=amounts,
            account_balances=balances,
            strategy_used="debit_credit_description_2turn",
            turns_executed=2,
            headers_detected=headers,
            column_mapping=mapping,
            raw_responses={"turn0": turn0_response, "turn1": response},
        )

    def _extract_schema_fallback(
        self,
        image: Any,
        headers: list[str],
        mapping: ColumnMapping,
    ) -> ExtractionResult:
        """Schema-based extraction when column detection fails.

        Used when header detection returns garbage (days of week, ATM locations, etc.)
        instead of actual column headers. Asks model to extract bank statement fields
        directly without relying on column structure.
        """
        import torch

        self._log("[UBE] Schema fallback: Extracting with direct schema prompt...")
        prompt = self._prompts["schema_fallback"]
        response = self.generate_fn(image, prompt, max_tokens=4096)

        self._log(f"[UBE]   Raw response length: {len(response)} chars")
        self._log(f"[UBE]   Raw response preview: {response[:500]}...")

        # Parse the schema-format response
        extracted = self._parse_schema_response(response)
        self._log(f"[UBE]   Parsed fields: {list(extracted.keys())}")

        # Extract fields from parsed response
        statement_date_range = extracted.get("STATEMENT_DATE_RANGE", "NOT_FOUND")
        transaction_dates_str = extracted.get("TRANSACTION_DATES", "NOT_FOUND")
        descriptions_str = extracted.get("LINE_ITEM_DESCRIPTIONS", "NOT_FOUND")
        amounts_str = extracted.get("TRANSACTION_AMOUNTS_PAID", "NOT_FOUND")

        # Parse pipe-separated lists
        def parse_list(value: str) -> list[str]:
            if not value or value == "NOT_FOUND":
                return []
            return [item.strip() for item in value.split("|") if item.strip()]

        dates = parse_list(transaction_dates_str)
        descriptions = parse_list(descriptions_str)
        amounts = parse_list(amounts_str)

        self._log(
            f"[UBE]   Parsed: {len(dates)} dates, {len(descriptions)} descriptions, {len(amounts)} amounts"
        )

        # Ensure arrays are same length (truncate to shortest)
        min_len = (
            min(len(dates), len(descriptions), len(amounts))
            if dates and descriptions and amounts
            else 0
        )
        if min_len > 0:
            dates = dates[:min_len]
            descriptions = descriptions[:min_len]
            amounts = amounts[:min_len]
        else:
            # If any is empty, set all to empty
            dates = []
            descriptions = []
            amounts = []

        # No balance info in schema fallback
        balances = ["NOT_FOUND"] * len(dates) if dates else []

        # Compute date range from transaction dates (ensures chronological: earliest - latest)
        # This handles reverse-chronological statements correctly
        if dates:
            computed_date_range = self._compute_date_range(dates)
            self._log(
                f"[UBE]   Date range: {statement_date_range} → {computed_date_range} (computed from transactions)"
            )
            statement_date_range = computed_date_range

        torch.cuda.empty_cache()

        return ExtractionResult(
            statement_date_range=statement_date_range,
            transaction_dates=dates,
            line_item_descriptions=descriptions,
            transaction_amounts_paid=amounts,
            account_balances=balances,
            strategy_used="schema_fallback",
            turns_executed=1,
            headers_detected=headers,
            column_mapping=mapping,
            raw_responses={"schema_fallback": response},
        )

    def _parse_schema_response(self, response: str) -> dict[str, str]:
        """Parse schema-format response into field dictionary."""
        result = {}
        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Match "FIELD_NAME: value" or "**FIELD_NAME**: value" pattern
            # Models sometimes wrap field names in markdown bold
            if ":" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    # Remove markdown bold markers (**) from field name
                    field = parts[0].strip().replace("*", "").upper()
                    value = parts[1].strip()

                    # Only capture known schema fields
                    if field in [
                        "DOCUMENT_TYPE",
                        "STATEMENT_DATE_RANGE",
                        "TRANSACTION_DATES",
                        "LINE_ITEM_DESCRIPTIONS",
                        "TRANSACTION_AMOUNTS_PAID",
                    ]:
                        result[field] = value

        return result

    @staticmethod
    def _format_debit_amount(amount_str: str) -> str:
        """Format debit amount, preserving negative sign for Amount strategy.

        For Amount-based statements (signed values), ground truth expects
        negative amounts (e.g., "-$78.90") to remain negative.
        This preserves the sign while normalizing format.
        """
        if not amount_str:
            return ""
        amount = amount_str.strip()

        # Check if negative
        is_negative = False
        if amount.startswith("-"):
            is_negative = True
            amount = amount[1:]
        elif amount.startswith("(") and amount.endswith(")"):
            # Handle parentheses notation for negative: ($78.90) -> -$78.90
            is_negative = True
            amount = amount[1:-1]

        # Ensure $ prefix for consistency
        if amount and not amount.startswith("$"):
            amount = "$" + amount

        # Restore negative sign if original was negative
        if is_negative:
            amount = "-" + amount

        return amount

    @staticmethod
    def _compute_date_range(dates: list[str]) -> str:
        """Compute date range string, always oldest - newest.

        Args:
            dates: List of date strings from transactions

        Returns:
            Date range string in format "oldest - newest"
        """
        if len(dates) < 2:
            return dates[0] if dates else "NOT_FOUND"

        import re
        from datetime import datetime

        first_str = dates[0].strip()
        last_str = dates[-1].strip()

        # Strip day name prefix if present (e.g., "Mon 11 Aug 2025" -> "11 Aug 2025")
        day_prefix = re.compile(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+", re.IGNORECASE)
        first_str_clean = day_prefix.sub("", first_str)
        last_str_clean = day_prefix.sub("", last_str)

        # Common date formats to try (after stripping day prefix)
        date_formats = [
            "%d/%m/%Y",  # 03/05/2025
            "%d %b %Y",  # 04 Sep 2025
            "%d %B %Y",  # 04 September 2025
            "%Y-%m-%d",  # 2025-09-04
            "%m/%d/%Y",  # 05/03/2025 (US format)
        ]

        first_date = None
        last_date = None

        for fmt in date_formats:
            if first_date is None:
                try:
                    first_date = datetime.strptime(first_str_clean, fmt)
                except ValueError:
                    pass
            if last_date is None:
                try:
                    last_date = datetime.strptime(last_str_clean, fmt)
                except ValueError:
                    pass

        # If parsing failed, return as-is (first - last) with cleaned strings
        if first_date is None or last_date is None:
            return f"{first_str_clean} - {last_str_clean}"

        # Return in chronological order (oldest - newest) using cleaned strings
        if first_date <= last_date:
            return f"{first_str_clean} - {last_str_clean}"
        else:
            return f"{last_str_clean} - {first_str_clean}"

    # ------------------------------------------------------------------
    # Pipeline entry point (replaces BankStatementAdapter)
    # ------------------------------------------------------------------

    def extract_bank_statement(
        self,
        image_path: str | Path,
        force_strategy: ExtractionStrategy | None = None,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """Extract bank statement using multi-turn strategy.

        Pipeline-compatible entry point that loads the image, runs extraction
        with Rich stdout bypass, and returns (schema_fields, metadata).

        Drop-in replacement for BankStatementAdapter.extract_bank_statement().

        Args:
            image_path: Path to bank statement image.
            force_strategy: Optional strategy override (default: auto-select).

        Returns:
            Tuple of (schema_fields, metadata).
        """
        from PIL import Image

        if self.verbose:
            _safe_print(
                f"\n[UBE] >>> START extract_bank_statement({Path(image_path).name})"
            )

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Execute extraction with stdout bypass to prevent Rich console recursion
        with _bypass_rich_stdout():
            result: ExtractionResult = self.extract(
                image=image,
                force_strategy=force_strategy,
            )

        # Print raw Turn 1 response AFTER bypass context (visible in notebook)
        if self.verbose and result.raw_responses.get("turn1"):
            _safe_print(
                f"[UBE] Raw Turn 1 response:\n{result.raw_responses['turn1']}\n"
                "[UBE] === End raw response ==="
            )

        # Convert to schema dict + metadata
        schema_fields = result.to_schema_dict()
        metadata = result.to_metadata_dict()

        if self.verbose:
            _safe_print(f"[UBE]   Strategy: {result.strategy_used}")
            _safe_print(f"[UBE]   Turns: {result.turns_executed}")
            _safe_print(
                f"[UBE]   Transactions extracted: {len(result.transaction_dates)}"
            )
            _safe_print("[UBE] <<< END extract_bank_statement")

        return schema_fields, metadata
