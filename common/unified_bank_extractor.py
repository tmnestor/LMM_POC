"""Unified Bank Statement Extractor.

Automatically selects the optimal extraction strategy based on document characteristics.
Follows the 2-turn balance-description approach when Balance column is detected.

Usage:
    from common.unified_bank_extractor import UnifiedBankExtractor

    extractor = UnifiedBankExtractor(model, tokenizer, model_type="internvl3")
    result = extractor.extract(image_path)
    schema = result.to_schema_dict()
"""

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import yaml


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


@dataclass
class ExtractionResult:
    """Standardized extraction result."""

    document_type: str = "BANK_STATEMENT"
    statement_date_range: str = "NOT_FOUND"
    transaction_dates: list[str] = field(default_factory=list)
    line_item_descriptions: list[str] = field(default_factory=list)
    transaction_amounts_paid: list[str] = field(default_factory=list)
    account_balances: list[str] = field(default_factory=list)  # For math enhancement

    # Metadata
    strategy_used: str = ""
    turns_executed: int = 0
    headers_detected: list[str] = field(default_factory=list)
    column_mapping: ColumnMapping | None = None
    raw_responses: dict[str, str] = field(default_factory=dict)
    correction_stats: Any = None  # CorrectionStats, uses Any to avoid forward ref

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
        """Find matching header using keywords."""
        # Exact match first
        for keyword in keywords:
            for i, header_lower in enumerate(headers_lower):
                if keyword == header_lower:
                    return headers[i]

        # Substring match
        for keyword in keywords:
            if len(keyword) > 2:
                for i, header_lower in enumerate(headers_lower):
                    if keyword in header_lower:
                        return headers[i]

        return None


class ResponseParser:
    """Parse LLM responses into structured data."""

    @staticmethod
    def parse_headers(response: str) -> list[str]:
        """Parse Turn 0 header detection response."""
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        headers = []

        for line in lines:
            # Remove numbering and bullets
            cleaned = line.lstrip("0123456789.-•* ").strip()
            # Remove markdown formatting
            cleaned = cleaned.replace("**", "").replace("__", "")
            # Skip labels like "Headers:" or empty/too-long lines
            if cleaned.endswith(":"):
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

            # Pattern 3: "1. **04 Sep 2025**" or "1. **03 Jun 2023**"
            if not date_found:
                date_match = re.match(
                    r"^\d+\.\s*\*?\*?(\d{1,2}\s+[A-Za-z]{3}\s+\d{4})\*?\*?", line
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
                    if desc_col not in current_transaction or not current_transaction.get(desc_col):
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


class TransactionFilter:
    """Filter and process extracted transactions."""

    @staticmethod
    def parse_amount(value: str) -> float:
        """Extract numeric value from currency string."""
        if not value or not value.strip():
            return 0.0
        cleaned = (
            value.replace("$", "")
            .replace(",", "")
            .replace("CR", "")
            .replace("DR", "")
            .strip()
        )
        try:
            return float(cleaned)
        except ValueError:
            return 0.0

    @staticmethod
    def is_non_transaction(row: dict[str, str], desc_col: str) -> bool:
        """Check if row is a non-transaction entry (opening/closing balance)."""
        desc = row.get(desc_col, "").strip().upper()
        skip_patterns = [
            "OPENING BALANCE",
            "CLOSING BALANCE",
            "BROUGHT FORWARD",
            "CARRIED FORWARD",
        ]
        return any(pattern in desc for pattern in skip_patterns)

    @classmethod
    def filter_debits(
        cls,
        rows: list[dict[str, str]],
        debit_col: str,
        desc_col: str | None = None,
    ) -> list[dict[str, str]]:
        """Filter to only debit transactions with amount > 0."""
        debit_rows = []
        for row in rows:
            debit_value = row.get(debit_col, "").strip()

            if not debit_value:
                continue
            if debit_value.upper() == "NOT_FOUND":
                continue

            amount = cls.parse_amount(debit_value)
            if amount <= 0:
                continue

            if desc_col and cls.is_non_transaction(row, desc_col):
                continue

            debit_rows.append(row)

        return debit_rows

    @classmethod
    def filter_negative_amounts(
        cls,
        rows: list[dict[str, str]],
        amount_col: str,
        desc_col: str | None = None,
    ) -> list[dict[str, str]]:
        """Filter to transactions with negative amounts (withdrawals)."""
        debit_rows = []
        for row in rows:
            amount_str = row.get(amount_col, "").strip()
            if not amount_str:
                continue

            # Skip non-transaction entries
            if desc_col and cls.is_non_transaction(row, desc_col):
                continue

            # Check for negative indicators
            is_negative = amount_str.startswith("-") or (
                amount_str.startswith("(") and amount_str.endswith(")")
            )

            if is_negative:
                debit_rows.append(row)

        return debit_rows


@dataclass
class CorrectionStats:
    """Statistics from balance correction."""

    total_transactions: int = 0
    debits_found: int = 0
    credits_found: int = 0
    corrections_made: int = 0
    type_corrections: int = 0  # debit<->credit swaps
    unparseable_balances: int = 0

    def __str__(self) -> str:
        return (
            f"Transactions: {self.total_transactions}, "
            f"Debits: {self.debits_found}, Credits: {self.credits_found}, "
            f"Corrections: {self.corrections_made} (type swaps: {self.type_corrections})"
        )


class BalanceCorrector:
    """Correct debit/credit classification using balance arithmetic.

    Uses the mathematical relationship between consecutive balances to verify
    and correct the LLM's classification of transactions as debits or credits.

    Logic:
        balance_change = current_balance - previous_balance
        if balance_change < 0: transaction is a DEBIT of abs(balance_change)
        if balance_change > 0: transaction is a CREDIT of balance_change

    IMPORTANT: Only works for chronologically ordered statements (oldest first).
    Use is_chronological_order() to check before applying corrections.
    """

    def __init__(self, tolerance: float = 0.01):
        """Initialize corrector.

        Args:
            tolerance: Allowed difference for amount matching (default 0.01)
        """
        self.tolerance = tolerance

    @staticmethod
    def parse_balance(value: str) -> float | None:
        """Parse balance string to float, return None if unparseable."""
        if not value or not value.strip():
            return None
        cleaned = (
            value.replace("$", "")
            .replace(",", "")
            .replace("CR", "")
            .replace("DR", "")
            .strip()
        )
        # Handle parentheses for negative (e.g., "(100.00)")
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = "-" + cleaned[1:-1]
        try:
            return float(cleaned)
        except ValueError:
            return None

    def correct_transactions(
        self,
        rows: list[dict[str, str]],
        balance_col: str,
        debit_col: str,
        credit_col: str,
        desc_col: str | None = None,
    ) -> tuple[list[dict[str, str]], CorrectionStats]:
        """Correct transaction classifications using balance deltas.

        Args:
            rows: Parsed transaction rows with balance, debit, credit columns
            balance_col: Name of balance column
            debit_col: Name of debit column
            credit_col: Name of credit column
            desc_col: Name of description column (for skipping non-transactions)

        Returns:
            Tuple of (corrected_rows, correction_stats)
        """
        stats = CorrectionStats(total_transactions=len(rows))
        corrected_rows = []

        prev_balance: float | None = None

        for row in rows:
            # Skip non-transaction entries
            if desc_col and TransactionFilter.is_non_transaction(row, desc_col):
                continue

            # Parse current balance
            balance_str = row.get(balance_col, "")
            current_balance = self.parse_balance(balance_str)

            if current_balance is None:
                stats.unparseable_balances += 1
                # Keep original classification if balance unparseable
                # DON'T reset prev_balance - preserve chain for next parseable row
                corrected_rows.append(row.copy())
                continue

            # Need previous balance to calculate delta
            if prev_balance is None:
                prev_balance = current_balance
                corrected_rows.append(row.copy())
                # Count based on LLM classification
                if self._has_amount(row, debit_col):
                    stats.debits_found += 1
                elif self._has_amount(row, credit_col):
                    stats.credits_found += 1
                continue

            # Calculate balance change
            balance_delta = current_balance - prev_balance
            corrected_row = row.copy()

            # Get LLM's classification
            llm_debit = TransactionFilter.parse_amount(row.get(debit_col, ""))
            llm_credit = TransactionFilter.parse_amount(row.get(credit_col, ""))

            # Determine correct classification from balance delta
            # IMPORTANT: Only SWAP values between columns - never recalculate amounts
            # This matches the llama notebook's validate_and_correct_alignment behavior
            if abs(balance_delta) < self.tolerance:
                # No significant change - keep original
                pass
            elif balance_delta < 0:
                # Balance decreased = should be DEBIT
                stats.debits_found += 1

                # If LLM misclassified (put amount in credit instead of debit), swap it
                if llm_credit > 0 and llm_debit == 0:
                    # Move credit value to debit column
                    corrected_row[debit_col] = row.get(credit_col, "")
                    corrected_row[credit_col] = ""
                    stats.corrections_made += 1
                    stats.type_corrections += 1
                else:
                    # LLM classified correctly as debit - keep amount, just clear credit
                    corrected_row[credit_col] = ""

            else:
                # Balance increased = should be CREDIT
                stats.credits_found += 1

                # If LLM misclassified (put amount in debit instead of credit), swap it
                if llm_debit > 0 and llm_credit == 0:
                    # Move debit value to credit column
                    corrected_row[credit_col] = row.get(debit_col, "")
                    corrected_row[debit_col] = ""
                    stats.corrections_made += 1
                    stats.type_corrections += 1
                else:
                    # LLM classified correctly as credit - keep amount, just clear debit
                    corrected_row[debit_col] = ""

            corrected_rows.append(corrected_row)
            prev_balance = current_balance

        return corrected_rows, stats

    @staticmethod
    def _has_amount(row: dict[str, str], col: str) -> bool:
        """Check if column has a valid amount."""
        value = row.get(col, "").strip()
        if not value or value.upper() == "NOT_FOUND":
            return False
        return TransactionFilter.parse_amount(value) > 0

    @staticmethod
    def is_chronological_order(
        rows: list[dict[str, str]], date_col: str
    ) -> tuple[bool, str]:
        """Determine if transactions are in chronological order (oldest first).

        Args:
            rows: Transaction rows with date column
            date_col: Name of the date column

        Returns:
            Tuple of (is_chronological, reason)
            - is_chronological: True if oldest transaction is first
            - reason: Description of the detection result
        """
        from datetime import datetime

        if len(rows) < 2:
            return False, "Not enough rows to determine order"

        # Get first and last dates
        first_date_str = rows[0].get(date_col, "").strip()
        last_date_str = rows[-1].get(date_col, "").strip()

        if not first_date_str or not last_date_str:
            return False, "Missing date values"

        # Try parsing dates with common formats
        date_formats = [
            "%d/%m/%Y",  # 03/05/2025
            "%d %b %Y",  # 04 Sep 2025
            "%d %B %Y",  # 04 September 2025
            "%a %d %b %Y",  # Thu 04 Sep 2025
            "%Y-%m-%d",  # 2025-09-04
            "%m/%d/%Y",  # 05/03/2025 (US format)
        ]

        first_date = None
        last_date = None

        for fmt in date_formats:
            if first_date is None:
                try:
                    first_date = datetime.strptime(first_date_str, fmt)
                except ValueError:
                    pass
            if last_date is None:
                try:
                    last_date = datetime.strptime(last_date_str, fmt)
                except ValueError:
                    pass

        if first_date is None or last_date is None:
            return (
                False,
                f"Could not parse dates: '{first_date_str}', '{last_date_str}'",
            )

        if first_date < last_date:
            return True, f"Chronological: {first_date_str} → {last_date_str}"
        elif first_date > last_date:
            return False, f"Reverse chronological: {first_date_str} → {last_date_str}"
        else:
            return False, "Same date for first and last transaction"

    @staticmethod
    def sort_by_date(
        rows: list[dict[str, str]], date_col: str
    ) -> list[dict[str, str]]:
        """Sort rows chronologically by date (oldest first).

        Args:
            rows: Transaction rows with date column
            date_col: Name of the date column

        Returns:
            Rows sorted by date (oldest first). Unparseable dates go to end.
        """
        from datetime import datetime

        date_formats = [
            "%d/%m/%Y",  # 03/05/2025
            "%d %b %Y",  # 04 Sep 2025
            "%d %B %Y",  # 04 September 2025
            "%a %d %b %Y",  # Thu 04 Sep 2025
            "%Y-%m-%d",  # 2025-09-04
            "%m/%d/%Y",  # 05/03/2025 (US format)
        ]

        def parse_date(date_str: str) -> datetime | None:
            """Parse date string to datetime object."""
            for fmt in date_formats:
                try:
                    return datetime.strptime(date_str.strip(), fmt)
                except ValueError:
                    continue
            return None

        def sort_key(row: dict[str, str]) -> tuple[int, datetime | None]:
            """Return sort key: (parseable flag, datetime)."""
            date_str = row.get(date_col, "")
            parsed = parse_date(date_str)
            if parsed:
                return (0, parsed)  # Parseable dates first, sorted by date
            return (1, None)  # Unparseable dates at end

        return sorted(rows, key=sort_key)


class UnifiedBankExtractor:
    """Unified bank statement extractor with automatic strategy selection.

    Args:
        model: Loaded model (Llama or InternVL3)
        tokenizer: Model tokenizer
        processor: Model processor (required for Llama)
        model_type: "llama" or "internvl3"
        config_dir: Path to config directory (optional)

    Example:
        extractor = UnifiedBankExtractor(model, tokenizer, model_type="internvl3")
        result = extractor.extract(image_path)
        print(result.to_schema_dict())
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        processor: Any = None,
        model_type: str = "internvl3",
        config_dir: str | Path | None = None,
        model_dtype: Any = None,
        image_processing_config: dict[str, Any] | None = None,
        use_balance_correction: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.model_type = model_type.lower()
        self.model_dtype = model_dtype
        self.use_balance_correction = use_balance_correction

        # Image processing config (from model_config.yaml)
        self.image_processing = image_processing_config or {}
        self.max_tiles = self.image_processing.get("max_tiles", 14)
        self.input_size = self.image_processing.get("input_size", 448)

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
        }

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
        print("[UBE] Turn 0: Detecting headers...")
        turn0_response = self._generate(image, self._prompts["turn0"], max_tokens=500)
        headers = self.parser.parse_headers(turn0_response)
        print(f"  Detected {len(headers)} headers: {headers}")

        # Map headers to columns
        mapping = self.column_matcher.match(headers)
        print(f"  Balance column: {mapping.balance or 'NOT FOUND'}")

        # Select strategy based on detected columns
        # Key insight: Some statements have Amount+Balance (signed values),
        # others have Debit/Credit/Balance (separate columns)
        has_debit_or_credit = mapping.debit or mapping.credit
        has_amount = mapping.amount is not None

        # DEBUG: Show column detection
        print(f"  Amount column: {mapping.amount or 'NOT FOUND'}")
        print(f"  Debit column: {mapping.debit or 'NOT FOUND'}")
        print(f"  Credit column: {mapping.credit or 'NOT FOUND'}")
        print(f"  has_balance={mapping.has_balance}, has_amount={has_amount}, has_debit_or_credit={has_debit_or_credit}")

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
            reason = "Fallback to table extraction"

        print(f"[UBE] Strategy: {strategy.name} ({reason})")

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
            # Table extraction not implemented yet
            result = ExtractionResult(
                strategy_used="table_extraction_not_implemented",
                headers_detected=headers,
                column_mapping=mapping,
            )

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

        # Build Turn 1 prompt with actual column names
        prompt_template = self._prompts["turn1_balance"]
        prompt = prompt_template.format(
            balance_col=mapping.balance,
            desc_col=mapping.description or "Description",
            debit_col=mapping.debit or "Debit",
            credit_col=mapping.credit or "Credit",
        )

        print("[UBE] Turn 1: Extracting transactions...")
        import sys as _sys
        _sys.stdout.flush()
        response = self._generate(image, prompt, max_tokens=4096)
        print(f"[UBE] Raw Turn 1 response:\n{response}\n[UBE] === End raw response ===")
        _sys.stdout.flush()

        # Column name shortcuts
        date_col = mapping.date or "Date"
        desc_col = mapping.description or "Description"
        debit_col = mapping.debit or "Debit"
        credit_col = mapping.credit or "Credit"
        balance_col = mapping.balance or "Balance"

        # Parse response
        all_rows = self.parser.parse_balance_description(
            response,
            date_col=date_col,
            desc_col=desc_col,
            debit_col=debit_col,
            credit_col=credit_col,
            balance_col=balance_col,
        )
        print(f"[UBE]   Parsed {len(all_rows)} transactions")

        # Optionally apply balance correction (sort to chronological order first)
        correction_stats = None
        if self.use_balance_correction:
            # Check if transactions are in chronological order
            is_chrono, order_reason = BalanceCorrector.is_chronological_order(
                all_rows, date_col
            )
            print(f"  Date order: {order_reason}")

            # Sort to chronological order if not already
            if is_chrono:
                sorted_rows = all_rows
            else:
                sorted_rows = BalanceCorrector.sort_by_date(all_rows, date_col)
                print("  Sorted to chronological order for balance correction")

            # Apply balance correction on sorted (chronological) rows
            corrector = BalanceCorrector()
            corrected_rows, correction_stats = corrector.correct_transactions(
                sorted_rows,
                balance_col=balance_col,
                debit_col=debit_col,
                credit_col=credit_col,
                desc_col=desc_col,
            )
            print(f"  Balance correction: {correction_stats}")
        else:
            corrected_rows = all_rows

        # Filter for debits (from corrected rows)
        debit_rows = self.filter.filter_debits(
            corrected_rows,
            debit_col=debit_col,
            desc_col=desc_col,
        )
        print(f"[UBE]   Filtered to {len(debit_rows)} debit transactions")

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

        # DEBUG: Verify array lengths are consistent
        print(f"[UBE]   Array lengths: dates={len(dates)}, desc={len(descriptions)}, amounts={len(amounts)}, balances={len(balances)}")

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

        print("[UBE] Turn 1: Extracting transactions (amount-description)...")
        response = self._generate(image, prompt, max_tokens=4096)
        print(f"[UBE] Raw Turn 1 response:\n{response}\n[UBE] === End raw response ===")

        # Parse response
        all_rows = self.parser.parse_amount_description(
            response,
            date_col=date_col,
            desc_col=desc_col,
            amount_col=amount_col,
            balance_col=balance_col,
        )
        print(f"[UBE]   Parsed {len(all_rows)} transactions")

        # Filter for withdrawals (negative amounts)
        debit_rows = self.filter.filter_negative_amounts(
            all_rows,
            amount_col=amount_col,
            desc_col=desc_col,
        )
        print(f"[UBE]   Filtered to {len(debit_rows)} withdrawal transactions")

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
        print(f"[UBE]   Array lengths: dates={len(dates)}, desc={len(descriptions)}, amounts={len(amounts)}, balances={len(balances)}")

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

        # Build Turn 1 prompt with actual column names
        prompt_template = self._prompts["turn1_debit_credit"]
        prompt = prompt_template.format(
            debit_col=mapping.debit or "Debit",
            credit_col=mapping.credit or "Credit",
            desc_col=mapping.description or "Transaction",
        )

        print("[UBE] Turn 1: Extracting transactions (debit-credit)...")
        response = self._generate(image, prompt, max_tokens=4096)
        print(f"[UBE] Raw Turn 1 response:\n{response}\n[UBE] === End raw response ===")

        # Column name shortcuts
        date_col = mapping.date or "Date"
        desc_col = mapping.description or "Transaction"
        debit_col = mapping.debit or "Debit"
        credit_col = mapping.credit or "Credit"

        # Parse response - reuse balance-description parser (same format)
        all_rows = self.parser.parse_balance_description(
            response,
            date_col=date_col,
            desc_col=desc_col,
            debit_col=debit_col,
            credit_col=credit_col,
            balance_col="Balance",  # Placeholder, not used
        )
        print(f"[UBE]   Parsed {len(all_rows)} transactions")

        # Filter for debits
        debit_rows = self.filter.filter_debits(
            all_rows,
            debit_col=debit_col,
            desc_col=desc_col,
        )
        print(f"[UBE]   Filtered to {len(debit_rows)} debit transactions")

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

        # DEBUG: Verify array lengths are consistent
        print(f"[UBE]   Array lengths: dates={len(dates)}, desc={len(descriptions)}, amounts={len(amounts)}, balances={len(balances)}")

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

    def _generate(self, image: Any, prompt: str, max_tokens: int = 4096) -> str:
        """Generate model response."""
        if self.model_type == "llama":
            return self._generate_llama(image, prompt, max_tokens)
        return self._generate_internvl3(image, prompt, max_tokens)

    def _generate_llama(self, image: Any, prompt: str, max_tokens: int) -> str:
        """Generate response using Llama model."""
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_input = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            images=[image], text=text_input, return_tensors="pt"
        ).to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

        generate_ids = output[:, inputs["input_ids"].shape[1] : -1]
        response = self.processor.decode(
            generate_ids[0], clean_up_tokenization_spaces=False
        )

        del inputs, output, generate_ids
        torch.cuda.empty_cache()

        return response

    def _generate_internvl3(self, image: Any, prompt: str, max_tokens: int) -> str:
        """Generate response using InternVL3 model."""
        import torch

        # Preprocess image
        pixel_values = self._preprocess_image_internvl3(image)
        pixel_values = pixel_values.to(
            dtype=self.model_dtype or torch.bfloat16, device="cuda:0"
        )

        response = self.model.chat(
            tokenizer=self.tokenizer,
            pixel_values=pixel_values,
            question=prompt,
            generation_config={"max_new_tokens": max_tokens, "do_sample": False},
        )

        del pixel_values
        torch.cuda.empty_cache()

        return response

    def _preprocess_image_internvl3(self, image: Any) -> Any:
        """Preprocess image for InternVL3 using config from model_config.yaml."""

        import torch
        import torchvision.transforms as T
        from PIL import Image as PILImage
        from torchvision.transforms.functional import InterpolationMode

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        max_tiles = self.max_tiles
        input_size = self.input_size

        def build_transform(input_size):
            return T.Compose(
                [
                    T.Lambda(
                        lambda img: img.convert("RGB") if img.mode != "RGB" else img
                    ),
                    T.Resize(
                        (input_size, input_size),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    T.ToTensor(),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
            )

        def find_closest_aspect_ratio(
            aspect_ratio, target_ratios, width, height, image_size
        ):
            best_ratio_diff = float("inf")
            best_ratio = (1, 1)
            area = width * height
            for ratio in target_ratios:
                target_aspect_ratio = ratio[0] / ratio[1]
                ratio_diff = abs(aspect_ratio - target_aspect_ratio)
                if ratio_diff < best_ratio_diff:
                    best_ratio_diff = ratio_diff
                    best_ratio = ratio
                elif ratio_diff == best_ratio_diff:
                    if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                        best_ratio = ratio
            return best_ratio

        def dynamic_preprocess(image, min_num=1, max_num=max_tiles, image_size=448):
            orig_width, orig_height = image.size
            aspect_ratio = orig_width / orig_height

            target_ratios = set(
                (i, j)
                for n in range(min_num, max_num + 1)
                for i in range(1, n + 1)
                for j in range(1, n + 1)
                if i * j <= max_num and i * j >= min_num
            )
            target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

            target_aspect_ratio = find_closest_aspect_ratio(
                aspect_ratio, target_ratios, orig_width, orig_height, image_size
            )

            target_width = image_size * target_aspect_ratio[0]
            target_height = image_size * target_aspect_ratio[1]
            blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

            resized_img = image.resize((target_width, target_height))
            processed_images = []
            for i in range(blocks):
                box = (
                    (i % (target_width // image_size)) * image_size,
                    (i // (target_width // image_size)) * image_size,
                    ((i % (target_width // image_size)) + 1) * image_size,
                    ((i // (target_width // image_size)) + 1) * image_size,
                )
                split_img = resized_img.crop(box)
                processed_images.append(split_img)

            # Add thumbnail
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)

            return processed_images

        # Convert to PIL if needed
        if isinstance(image, str):
            image = PILImage.open(image).convert("RGB")

        transform = build_transform(input_size)
        images = dynamic_preprocess(image, image_size=input_size, max_num=max_tiles)
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)

        return pixel_values
