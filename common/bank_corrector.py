"""Balance-arithmetic correction and transaction filtering for bank statements.

Pure computation — no GPU, no model, no PIL, no torch dependencies.
Independently testable with synthetic transaction data.

Extracted from unified_bank_extractor.py to enable unit testing
of correction logic without VLM infrastructure.
"""

from dataclasses import dataclass
from datetime import datetime


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
        """Filter to only debit transactions with amount > 0.

        Note: Some statements show debits as negative values (e.g., -$5.00).
        We use abs() to handle both positive and negative representations.
        """
        debit_rows = []
        for row in rows:
            debit_value = row.get(debit_col, "").strip()

            if not debit_value:
                continue
            if debit_value.upper() == "NOT_FOUND":
                continue

            # Use abs() to handle negative amounts (some statements show debits as negative)
            amount = abs(cls.parse_amount(debit_value))
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

    @classmethod
    def filter_positive_amounts(
        cls,
        rows: list[dict[str, str]],
        amount_col: str,
        desc_col: str | None = None,
    ) -> list[dict[str, str]]:
        """Filter to transactions with positive amounts (charges/purchases).

        For credit card statements, positive amounts represent charges
        that the customer paid for goods/services. Negative amounts
        represent payments/credits and should be marked as NOT_FOUND.
        """
        charge_rows = []
        for row in rows:
            amount_str = row.get(amount_col, "").strip()
            if not amount_str:
                continue

            # Skip non-transaction entries (Opening Balance, Closing Balance, etc.)
            if desc_col and cls.is_non_transaction(row, desc_col):
                continue

            # Check for negative indicators - skip these
            is_negative = amount_str.startswith("-") or (
                amount_str.startswith("(") and amount_str.endswith(")")
            )

            if not is_negative:
                charge_rows.append(row)

        return charge_rows


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
                expected_amount = abs(balance_delta)

                # Check if debit already has the correct value
                if llm_debit > 0 and abs(llm_debit - expected_amount) < self.tolerance:
                    # Debit is already correct - keep it, set credit to NOT_FOUND
                    corrected_row[credit_col] = "NOT_FOUND"
                # Check if credit has the correct value (misclassified)
                elif (
                    llm_credit > 0
                    and abs(llm_credit - expected_amount) < self.tolerance
                ):
                    # Move credit value to debit column
                    corrected_row[debit_col] = row.get(credit_col, "")
                    corrected_row[credit_col] = "NOT_FOUND"
                    stats.corrections_made += 1
                    stats.type_corrections += 1
                else:
                    # Neither matches - derive amount from balance delta (most reliable)
                    corrected_row[debit_col] = f"${expected_amount:.2f}"
                    corrected_row[credit_col] = "NOT_FOUND"
                    stats.corrections_made += 1

            else:
                # Balance increased = should be CREDIT
                stats.credits_found += 1
                expected_amount = abs(balance_delta)

                # Check if credit already has the correct value
                if (
                    llm_credit > 0
                    and abs(llm_credit - expected_amount) < self.tolerance
                ):
                    # Credit is already correct - keep it, set debit to NOT_FOUND
                    corrected_row[debit_col] = "NOT_FOUND"
                # Check if debit has the correct value (misclassified)
                elif (
                    llm_debit > 0 and abs(llm_debit - expected_amount) < self.tolerance
                ):
                    # Move debit value to credit column
                    corrected_row[credit_col] = row.get(debit_col, "")
                    corrected_row[debit_col] = "NOT_FOUND"
                    stats.corrections_made += 1
                    stats.type_corrections += 1
                else:
                    # Neither matches - derive amount from balance delta (most reliable)
                    corrected_row[credit_col] = f"${expected_amount:.2f}"
                    corrected_row[debit_col] = "NOT_FOUND"
                    stats.corrections_made += 1

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
            return True, f"Chronological: {first_date_str} -> {last_date_str}"
        elif first_date > last_date:
            return (
                False,
                f"Reverse chronological: {first_date_str} -> {last_date_str}",
            )
        else:
            return False, "Same date for first and last transaction"

    @staticmethod
    def sort_by_date(rows: list[dict[str, str]], date_col: str) -> list[dict[str, str]]:
        """Sort rows chronologically by date (oldest first).

        Args:
            rows: Transaction rows with date column
            date_col: Name of the date column

        Returns:
            Rows sorted by date (oldest first). Unparseable dates go to end.
        """
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
