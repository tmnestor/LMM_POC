"""
Bank Statement Mathematical Post-Processing Calculator

This module provides mathematical analysis of bank statement data to derive
transaction types (debit/credit) and amounts from reliable extracted data
(dates and running balances), bypassing problematic debit/credit columns.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rich import print as rprint

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    np = None
    PANDAS_AVAILABLE = False


@dataclass
class Transaction:
    """Represents a single bank transaction with calculated fields."""

    date: datetime
    date_str: str
    description: str
    balance: float
    amount: float
    transaction_type: str  # 'DEBIT' or 'CREDIT'
    index: int  # Original position in sequence


@dataclass
class BankStatementAnalysis:
    """Results of bank statement mathematical analysis."""

    transactions: List[Transaction]
    total_debits: float
    total_credits: float
    transaction_count: int
    calculated_amounts_paid: List[str]
    calculated_amounts_received: List[str]
    success: bool
    errors: List[str]


class BankStatementCalculator:
    """Mathematical calculator for bank statement transaction analysis."""

    def __init__(self, verbose: bool = False, use_pandas: bool = True):
        """Initialize the calculator.

        Args:
            verbose: Whether to display detailed processing information
            use_pandas: Whether to use pandas-based calculation (recommended)
                       Falls back to legacy method if pandas not available
        """
        self.verbose = verbose
        self.use_pandas = use_pandas and PANDAS_AVAILABLE

    def analyze_bank_statement(
        self, extracted_data: Dict[str, Any]
    ) -> BankStatementAnalysis:
        """
        Analyze bank statement data using mathematical balance differences.

        Routes to pandas-based implementation by default, falls back to legacy method.

        Args:
            extracted_data: Dictionary containing extracted fields including
                           TRANSACTION_DATES and ACCOUNT_BALANCE

        Returns:
            BankStatementAnalysis with calculated transaction types and amounts
        """
        # Route to pandas implementation if enabled
        if self.use_pandas:
            return self.analyze_bank_statement_pandas(extracted_data)
        else:
            return self.analyze_bank_statement_legacy(extracted_data)

    def analyze_bank_statement_legacy(
        self, extracted_data: Dict[str, Any]
    ) -> BankStatementAnalysis:
        """
        Analyze bank statement data using original manual array manipulation.

        This is the legacy implementation kept for compatibility and fallback.

        Args:
            extracted_data: Dictionary containing extracted fields including
                           TRANSACTION_DATES and ACCOUNT_BALANCE

        Returns:
            BankStatementAnalysis with calculated transaction types and amounts
        """
        try:
            # Extract required fields
            dates_str = extracted_data.get("TRANSACTION_DATES", "")
            balances_str = extracted_data.get("ACCOUNT_BALANCE", "")
            descriptions_str = extracted_data.get("LINE_ITEM_DESCRIPTIONS", "")

            # Extract additional fields for hybrid approach
            amounts_paid_str = extracted_data.get("TRANSACTION_AMOUNTS_PAID", "")
            amounts_received_str = extracted_data.get(
                "TRANSACTION_AMOUNTS_RECEIVED", ""
            )

            if not dates_str or not balances_str:
                return BankStatementAnalysis(
                    transactions=[],
                    total_debits=0.0,
                    total_credits=0.0,
                    transaction_count=0,
                    calculated_amounts_paid=[],
                    calculated_amounts_received=[],
                    success=False,
                    errors=[
                        "Missing required fields: TRANSACTION_DATES or ACCOUNT_BALANCE"
                    ],
                )

            # Parse data into lists
            dates = self._parse_dates(dates_str)
            balances = self._parse_balances(balances_str)
            descriptions = self._parse_descriptions(descriptions_str)

            # Parse extracted amounts for hybrid approach
            extracted_paid = self._parse_extracted_amounts(amounts_paid_str)
            extracted_received = self._parse_extracted_amounts(amounts_received_str)

            # Validate and correct array alignment if needed
            if extracted_paid or extracted_received:
                extracted_paid, extracted_received = self._validate_array_alignment(
                    dates,
                    extracted_paid,
                    extracted_received,
                    dates_str,
                    descriptions_str,
                )

            if len(dates) != len(balances):
                if self.verbose:
                    rprint(
                        f"[yellow]⚠️ Count mismatch: {len(dates)} dates vs {len(balances)} balances - using minimum count[/yellow]"
                    )

                # Use the minimum count and truncate the longer list
                min_count = min(len(dates), len(balances))
                if min_count == 0:
                    return BankStatementAnalysis(
                        transactions=[],
                        total_debits=0.0,
                        total_credits=0.0,
                        transaction_count=0,
                        calculated_amounts_paid=[],
                        calculated_amounts_received=[],
                        success=False,
                        errors=[
                            f"No valid transactions: {len(dates)} dates vs {len(balances)} balances"
                        ],
                    )

                # Truncate to minimum count
                dates = dates[:min_count]
                balances = balances[:min_count]
                if descriptions and len(descriptions) > min_count:
                    descriptions = descriptions[:min_count]

            if self.verbose:
                rprint(
                    f"[cyan]📊 Analyzing {len(dates)} transactions mathematically[/cyan]"
                )

            # Create transactions and sort chronologically
            transactions = self._create_transactions(dates, balances, descriptions)
            sorted_transactions = self._sort_chronologically(transactions)

            # Calculate transaction types and amounts using hybrid approach
            analyzed_transactions = self._calculate_transaction_types(
                sorted_transactions,
                extracted_paid,
                extracted_received,
                dates_str,
                descriptions_str,
            )

            # Generate summary
            analysis = self._generate_analysis(analyzed_transactions)

            if self.verbose:
                self._display_analysis_summary(analysis)

            return analysis

        except Exception as e:
            return BankStatementAnalysis(
                transactions=[],
                total_debits=0.0,
                total_credits=0.0,
                transaction_count=0,
                calculated_amounts_paid=[],
                calculated_amounts_received=[],
                success=False,
                errors=[f"Analysis failed: {str(e)}"],
            )

    def analyze_bank_statement_pandas(
        self, extracted_data: Dict[str, Any]
    ) -> BankStatementAnalysis:
        """
        Analyze bank statement data using pandas for reliable calculations.

        This pandas-based approach eliminates array index mapping issues and provides
        cleaner mathematical operations with better data integrity.

        Args:
            extracted_data: Dictionary containing extracted fields including
                           TRANSACTION_DATES and ACCOUNT_BALANCE

        Returns:
            BankStatementAnalysis with calculated transaction types and amounts
        """
        if not PANDAS_AVAILABLE:
            if self.verbose:
                rprint(
                    "[yellow]⚠️ Pandas not available, falling back to legacy method[/yellow]"
                )
            return self.analyze_bank_statement(extracted_data)

        try:
            # Extract required fields
            dates_str = extracted_data.get("TRANSACTION_DATES", "")
            balances_str = extracted_data.get("ACCOUNT_BALANCE", "")
            descriptions_str = extracted_data.get("LINE_ITEM_DESCRIPTIONS", "")
            amounts_paid_str = extracted_data.get("TRANSACTION_AMOUNTS_PAID", "")
            amounts_received_str = extracted_data.get(
                "TRANSACTION_AMOUNTS_RECEIVED", ""
            )

            if not dates_str or not balances_str:
                return BankStatementAnalysis(
                    transactions=[],
                    total_debits=0.0,
                    total_credits=0.0,
                    transaction_count=0,
                    calculated_amounts_paid=[],
                    calculated_amounts_received=[],
                    success=False,
                    errors=[
                        "Missing required fields: TRANSACTION_DATES or ACCOUNT_BALANCE"
                    ],
                )

            # Parse data using existing methods
            dates_parsed = self._parse_dates(dates_str)
            balances_parsed = self._parse_balances(balances_str)
            descriptions_parsed = self._parse_descriptions(descriptions_str)
            amounts_paid_parsed = self._parse_extracted_amounts(amounts_paid_str)
            amounts_received_parsed = self._parse_extracted_amounts(
                amounts_received_str
            )

            if not dates_parsed or not balances_parsed:
                return BankStatementAnalysis(
                    transactions=[],
                    total_debits=0.0,
                    total_credits=0.0,
                    transaction_count=0,
                    calculated_amounts_paid=[],
                    calculated_amounts_received=[],
                    success=False,
                    errors=["Failed to parse dates or balances"],
                )

            # Create DataFrame with parsed data
            max_length = max(len(dates_parsed), len(balances_parsed))

            # Ensure all arrays are same length for DataFrame creation
            dates_list = [(d[0], d[1], i) for i, d in enumerate(dates_parsed)]
            balances_list = balances_parsed[:max_length]
            descriptions_list = (
                descriptions_parsed[:max_length]
                if descriptions_parsed
                else ["Transaction"] * max_length
            )

            # Pad shorter lists
            while len(dates_list) < max_length:
                dates_list.append((None, "Unknown", len(dates_list)))
            while len(balances_list) < max_length:
                balances_list.append(0.0)
            while len(descriptions_list) < max_length:
                descriptions_list.append("Transaction")
            while len(amounts_paid_parsed) < max_length:
                amounts_paid_parsed.append(0.0)
            while len(amounts_received_parsed) < max_length:
                amounts_received_parsed.append(0.0)

            # Create DataFrame
            # NOTE: For AMOUNT_DESCRIPTION strategy, amounts are negative (withdrawals)
            # Use abs() to accept both positive and negative VLM amounts
            # Track if original amounts were signed (negative) for output formatting
            has_signed_amounts = any(x < 0 for x in amounts_paid_parsed if x != 0)

            transactions_df = pd.DataFrame(
                {
                    "date": [d[0] for d in dates_list],
                    "date_str": [d[1] for d in dates_list],
                    "original_index": [d[2] for d in dates_list],
                    "balance": balances_list[:max_length],
                    "description": descriptions_list[:max_length],
                    "extracted_paid": [
                        abs(x) if x != 0 else np.nan
                        for x in amounts_paid_parsed[:max_length]
                    ],
                    "extracted_received": [
                        abs(x) if x != 0 else np.nan
                        for x in amounts_received_parsed[:max_length]
                    ],
                }
            )
            # Store flag for later use in formatting
            transactions_df.attrs["has_signed_amounts"] = has_signed_amounts

            # Remove rows with invalid dates
            transactions_df = transactions_df.dropna(subset=["date"])

            if len(transactions_df) == 0:
                return BankStatementAnalysis(
                    transactions=[],
                    total_debits=0.0,
                    total_credits=0.0,
                    transaction_count=0,
                    calculated_amounts_paid=[],
                    calculated_amounts_received=[],
                    success=False,
                    errors=["No valid transactions after parsing"],
                )

            if self.verbose:
                rprint(
                    f"[cyan]📊 Analyzing {len(transactions_df)} transactions with pandas[/cyan]"
                )

            # Sort chronologically for mathematical calculations
            df_calc = transactions_df.sort_values("date").copy()
            # Preserve the signed amounts flag
            has_signed_amounts = transactions_df.attrs.get("has_signed_amounts", False)

            # Calculate balance differences (vectorized operation)
            df_calc["balance_change"] = df_calc["balance"].diff()

            # Calculate mathematical amounts from balance differences (ground truth)
            df_calc["calc_paid"] = (
                df_calc["balance_change"].where(df_calc["balance_change"] < 0).abs()
            )
            df_calc["calc_received"] = df_calc["balance_change"].where(
                df_calc["balance_change"] > 0
            )

            # Mathematical ground truth for transaction classification
            df_calc["math_transaction_type"] = df_calc["balance_change"].apply(
                lambda x: "DEBIT" if x < 0 else "CREDIT" if x > 0 else "UNKNOWN"
            )
            df_calc["math_amount"] = df_calc["balance_change"].abs()

            # Extract VLM amount from whichever column it appears in
            df_calc["vlm_amount"] = df_calc["extracted_paid"].fillna(
                df_calc["extracted_received"]
            )
            df_calc["vlm_source_column"] = np.where(
                df_calc["extracted_paid"].notna(),
                "PAID",
                np.where(df_calc["extracted_received"].notna(), "RECEIVED", "NONE"),
            )

            # ENHANCED APPROACH: Apply validation and description-based correction
            # Use balance differences as ground truth but validate VLM extractions

            # Apply validation to each row
            def apply_validation(row):
                # Get VLM extracted values
                vlm_paid = (
                    row["extracted_paid"] if pd.notna(row["extracted_paid"]) else 0
                )
                vlm_received = (
                    row["extracted_received"]
                    if pd.notna(row["extracted_received"])
                    else 0
                )
                balance_change = (
                    row["balance_change"] if pd.notna(row["balance_change"]) else 0
                )
                description = row["description"]

                # Apply validation and correction
                corrected_paid, corrected_received, reason = (
                    self._validate_and_correct_vlm_extraction(
                        vlm_paid, vlm_received, description, balance_change
                    )
                )

                # For first transaction or when balance change is NaN, use corrected VLM values
                if pd.isna(row["balance_change"]):
                    return pd.Series(
                        [
                            corrected_paid if corrected_paid > 0 else np.nan,
                            corrected_received if corrected_received > 0 else np.nan,
                            reason,
                        ]
                    )

                # Otherwise use calculated values from balance change
                calc_paid = abs(balance_change) if balance_change < 0 else 0
                calc_received = balance_change if balance_change > 0 else 0

                # Use corrected VLM values if available, otherwise use calculated values
                final_paid = (
                    corrected_paid
                    if corrected_paid > 0
                    else (calc_paid if calc_paid > 0 else np.nan)
                )
                final_received = (
                    corrected_received
                    if corrected_received > 0
                    else (calc_received if calc_received > 0 else np.nan)
                )

                return pd.Series([final_paid, final_received, reason])

            # Apply validation to all rows
            df_calc[["final_paid", "final_received", "validation_reason"]] = (
                df_calc.apply(apply_validation, axis=1)
            )

            # Debug output - print the COMPLETE corrected DataFrame including descriptions
            if self.verbose:
                rprint(
                    "\n[bold cyan]🔍 MATHEMATICAL CORRECTION DEBUG - COMPLETE CORRECTED DATAFRAME[/bold cyan]"
                )
                rprint("All columns showing mathematical correction results:")
                debug_df = df_calc[
                    [
                        "original_index",
                        "date_str",
                        "description",
                        "balance",
                        "balance_change",
                        "extracted_paid",
                        "extracted_received",
                        "final_paid",
                        "final_received",
                    ]
                ].copy()
                debug_df = debug_df.sort_values(
                    "original_index"
                )  # Show in original order
                for _i, row in debug_df.iterrows():
                    desc_short = (
                        row["description"][:40] + "..."
                        if len(row["description"]) > 40
                        else row["description"]
                    )
                    final_paid_str = (
                        f"${row['final_paid']:.2f}"
                        if pd.notna(row["final_paid"])
                        else "NOT_FOUND"
                    )
                    final_recv_str = (
                        f"${row['final_received']:.2f}"
                        if pd.notna(row["final_received"])
                        else "NOT_FOUND"
                    )
                    rprint(
                        f"[cyan]Pos {int(row['original_index'])}: {row['date_str']} | {desc_short} | "
                        f"Balance: ${row['balance']:.2f} | Change: {row['balance_change'] if pd.notna(row['balance_change']) else 'NaN'} | "
                        f"VLM_PAID: {row['extracted_paid'] if pd.notna(row['extracted_paid']) else 'NaN'} | "
                        f"VLM_RECV: {row['extracted_received'] if pd.notna(row['extracted_received']) else 'NaN'} | "
                        f"FINAL_PAID: {final_paid_str} | FINAL_RECV: {final_recv_str}[/cyan]"
                    )

            # Determine transaction types
            df_calc["transaction_type"] = df_calc.apply(
                lambda row: "DEBIT"
                if pd.notna(row["final_paid"]) and row["final_paid"] > 0
                else "CREDIT"
                if pd.notna(row["final_received"]) and row["final_received"] > 0
                else "UNKNOWN",
                axis=1,
            )

            # Keep NaN values - they will be handled correctly in output generation as "NOT_FOUND"

            # Add mathematical validation columns for VLM accuracy
            df_calc["vlm_classification_correct"] = np.where(
                df_calc["vlm_amount"].isna(),
                np.nan,  # No VLM data to validate
                # VLM classification matches mathematical ground truth
                (
                    (df_calc["vlm_source_column"] == "PAID")
                    & (df_calc["math_transaction_type"] == "DEBIT")
                )
                | (
                    (df_calc["vlm_source_column"] == "RECEIVED")
                    & (df_calc["math_transaction_type"] == "CREDIT")
                ),
            )

            # Amount accuracy validation (percentage difference from mathematical amount)
            df_calc["vlm_amount_accuracy"] = np.where(
                df_calc["vlm_amount"].isna() | (df_calc["math_amount"] == 0),
                np.nan,
                1.0
                - (
                    abs(df_calc["vlm_amount"] - df_calc["math_amount"])
                    / df_calc["math_amount"]
                ),
            )

            # Enhanced confidence scoring based on data source and validation
            # Create boolean masks explicitly to avoid pandas boolean operation issues
            def create_confidence_score(row):
                has_vlm = pd.notna(row["vlm_amount"])
                if not has_vlm:
                    return 0.80  # Mathematical calculation only

                classification_correct = row["vlm_classification_correct"]
                if pd.notna(classification_correct) and classification_correct:
                    return 0.95  # VLM amount + correct classification
                else:
                    return (
                        0.85  # VLM amount but wrong/missing classification (corrected)
                    )

            df_calc["confidence"] = df_calc.apply(create_confidence_score, axis=1)

            # Validate balance continuity
            balance_validation_passed = True
            try:
                expected_final_balance = (
                    df_calc["balance"].iloc[0] + df_calc["balance_change"].sum()
                )
                actual_final_balance = df_calc["balance"].iloc[-1]
                balance_validation_passed = (
                    abs(expected_final_balance - actual_final_balance) < 0.01
                )
            except Exception:
                balance_validation_passed = False

            # Convert back to original order for output
            df_final = df_calc.sort_values("original_index")

            # Generate output arrays in original extraction order
            # For AMOUNT_DESCRIPTION (signed amounts), format debits as negative
            calculated_amounts_paid = []
            calculated_amounts_received = []

            for _, row in df_final.iterrows():
                if row["transaction_type"] == "DEBIT":
                    # Format with negative sign if original amounts were signed (AMOUNT_DESCRIPTION)
                    if has_signed_amounts:
                        calculated_amounts_paid.append(f"-${row['final_paid']:.2f}")
                    else:
                        calculated_amounts_paid.append(f"${row['final_paid']:.2f}")
                    calculated_amounts_received.append("NOT_FOUND")
                elif row["transaction_type"] == "CREDIT":
                    calculated_amounts_paid.append("NOT_FOUND")
                    calculated_amounts_received.append(f"${row['final_received']:.2f}")
                else:
                    calculated_amounts_paid.append("NOT_FOUND")
                    calculated_amounts_received.append("NOT_FOUND")

            # Calculate summary statistics
            total_debits = df_calc["final_paid"].sum()
            total_credits = df_calc["final_received"].sum()

            # Create Transaction objects for compatibility
            transactions = []
            for _, row in df_calc.iterrows():
                transactions.append(
                    Transaction(
                        date=row["date"],
                        date_str=row["date_str"],
                        description=row["description"],
                        balance=row["balance"],
                        amount=row["final_paid"]
                        if row["transaction_type"] == "DEBIT"
                        else row["final_received"],
                        transaction_type=row["transaction_type"],
                        index=int(row["original_index"]),
                    )
                )

            analysis = BankStatementAnalysis(
                transactions=transactions,
                total_debits=total_debits,
                total_credits=total_credits,
                transaction_count=len(df_calc),
                calculated_amounts_paid=calculated_amounts_paid,
                calculated_amounts_received=calculated_amounts_received,
                success=True,
                errors=[]
                if balance_validation_passed
                else [
                    "Balance validation warning: calculated vs actual balance mismatch"
                ],
            )

            if self.verbose:
                self._display_pandas_analysis_summary(
                    analysis, df_calc, balance_validation_passed
                )

            return analysis

        except Exception as e:
            if self.verbose:
                rprint(f"[red]❌ Pandas analysis failed: {e}[/red]")
                rprint("[yellow]Falling back to legacy method[/yellow]")
            return self.analyze_bank_statement(extracted_data)

    def _display_pandas_analysis_summary(
        self,
        analysis: BankStatementAnalysis,
        transactions_df: "pd.DataFrame",
        balance_validation_passed: bool,
    ):
        """Display pandas-based analysis summary with enhanced debugging."""
        if not self.verbose or not PANDAS_AVAILABLE:
            return

        rprint("\n[bold blue]📊 Pandas-Based Bank Statement Analysis[/bold blue]")
        rprint(f"[cyan]📋 Transactions: {analysis.transaction_count}[/cyan]")
        rprint(f"[red]📤 Total Debits: ${analysis.total_debits:.2f}[/red]")
        rprint(f"[green]📥 Total Credits: ${analysis.total_credits:.2f}[/green]")
        rprint(
            f"[blue]💰 Net Change: ${analysis.total_credits - analysis.total_debits:.2f}[/blue]"
        )

        # Balance validation status
        validation_color = "green" if balance_validation_passed else "yellow"
        validation_status = "✅ PASSED" if balance_validation_passed else "⚠️  WARNING"
        rprint(
            f"[{validation_color}]🔍 Balance Validation: {validation_status}[/{validation_color}]"
        )

        # Show VLM correction and validation metrics
        vlm_count = transactions_df["vlm_amount"].notna().sum()
        vlm_correct_count = (
            transactions_df["vlm_classification_correct"].fillna(False).sum()
        )
        vlm_corrected_count = vlm_count - vlm_correct_count if vlm_count > 0 else 0
        mathematical_count = analysis.transaction_count - vlm_count

        rprint("[dim]📊 Data Sources and Corrections:[/dim]")
        rprint(
            f"[dim]   • {vlm_correct_count} VLM amounts (correctly classified)[/dim]"
        )
        if vlm_corrected_count > 0:
            rprint(
                f"[dim]   • {vlm_corrected_count} VLM amounts (classification corrected)[/dim]"
            )
        rprint(
            f"[dim]   • {mathematical_count} mathematical calculations (VLM gaps filled)[/dim]"
        )

        if vlm_count > 0:
            accuracy_scores = transactions_df["vlm_amount_accuracy"].dropna()
            if len(accuracy_scores) > 0:
                avg_accuracy = accuracy_scores.mean()
                rprint(f"[dim]💡 VLM Amount Accuracy: {avg_accuracy:.1%} average[/dim]")

        # Show ALL transactions with enhanced source information
        if len(transactions_df) > 0:
            rprint("\n[dim]📅 All transactions with validation:[/dim]")
            for _, row in transactions_df.iterrows():
                color = "red" if row["transaction_type"] == "DEBIT" else "green"
                amount = (
                    row["final_paid"]
                    if row["transaction_type"] == "DEBIT"
                    else row["final_received"]
                )
                confidence = f"({row['confidence']:.1f})"

                # Determine source and validation status
                if pd.notna(row["vlm_amount"]):
                    if row["vlm_classification_correct"]:
                        source = "VLM✓"  # VLM correct
                    else:
                        source = "VLM→CORRECTED"  # VLM corrected
                    if pd.notna(row["vlm_amount_accuracy"]):
                        accuracy = f" {row['vlm_amount_accuracy']:.0%}"
                    else:
                        accuracy = ""
                else:
                    source = "CALCULATED"
                    accuracy = ""

                rprint(
                    f"[{color}]  {row['date_str']}: {row['transaction_type']} ${amount:.2f} [{source}{accuracy}] {confidence}[/{color}]"
                )

    def _parse_dates(self, dates_str: str) -> List[Tuple[datetime, str]]:
        """Parse date strings into datetime objects with original strings."""
        dates = []

        # Handle both pipe-separated and space-separated formats
        if "|" in dates_str:
            # Pipe-separated format (preferred)
            date_parts = [d.strip() for d in dates_str.split("|") if d.strip()]
        else:
            # Space-separated format (fallback for Llama extraction)
            # Look for date patterns like "Thu 04 Sep 2025" - more comprehensive
            date_parts = re.findall(
                r"[A-Za-z]{3}\s+\d{1,2}\s+[A-Za-z]{3}\s+\d{4}", dates_str
            )
            if not date_parts:
                # Fallback: try to find DD/MM/YYYY patterns
                date_parts = re.findall(r"\d{1,2}\/\d{1,2}\/\d{4}", dates_str)
            if not date_parts:
                # More aggressive fallback: find any date-like patterns
                date_parts = re.findall(r"\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}", dates_str)

        for date_str in date_parts:
            try:
                # Handle different date formats
                # "Thu 04 Sep 2025" or "04/09/2025" or "04-Sep-25" etc.
                date_obj = self._parse_single_date(date_str)
                if date_obj:
                    dates.append((date_obj, date_str))
            except Exception as e:
                if self.verbose:
                    rprint(f"[yellow]⚠️ Could not parse date: {date_str} - {e}[/yellow]")

        return dates

    def _parse_single_date(self, date_str: str) -> Optional[datetime]:
        """Parse a single date string into datetime object."""
        date_str = date_str.strip()

        # Common date formats in bank statements
        formats = [
            "%a %d %b %Y",  # "Thu 04 Sep 2025"
            "%d/%m/%Y",  # "04/09/2025"
            "%d-%b-%y",  # "04-Sep-25"
            "%d %b %Y",  # "04 Sep 2025"
            "%Y-%m-%d",  # "2025-09-04"
            "%d/%m/%y",  # "04/09/25"
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        # Try to extract date components with regex
        # Look for patterns like "04 Sep 2025" in strings like "Thu 04 Sep 2025"
        date_pattern = r"(\d{1,2})\s+(\w{3})\s+(\d{4})"
        match = re.search(date_pattern, date_str)
        if match:
            try:
                day, month, year = match.groups()
                date_str_clean = f"{day} {month} {year}"
                return datetime.strptime(date_str_clean, "%d %b %Y")
            except ValueError:
                pass

        return None

    def _parse_balances(self, balances_str: str) -> List[float]:
        """Parse balance strings into float values."""
        balances = []

        # Handle both pipe-separated and space-separated formats
        if "|" in balances_str:
            # Pipe-separated format (preferred)
            balance_parts = [b.strip() for b in balances_str.split("|") if b.strip()]
        else:
            # Space-separated format (fallback for Llama extraction)
            # Enhanced regex to handle CR suffixes and various formats
            balance_parts = re.findall(r"\$[\d,]+\.?\d*(?:\s+CR)?", balances_str)
            if not balance_parts:
                # More aggressive fallback: find any currency amounts
                balance_parts = re.findall(r"\$[\d,]+\.?\d*", balances_str)
            if not balance_parts:
                # Final fallback: split on spaces and filter for currency-like patterns
                parts = balances_str.split()
                balance_parts = [
                    p for p in parts if "$" in p or re.match(r"\d+\.?\d*", p)
                ]

        for balance_str in balance_parts:
            try:
                # Remove currency symbols, commas, and "CR" indicators
                clean_balance = re.sub(r"[^\d.-]", "", balance_str)
                if clean_balance:
                    balances.append(float(clean_balance))
            except (ValueError, TypeError) as e:
                if self.verbose:
                    rprint(
                        f"[yellow]⚠️ Could not parse balance: {balance_str} - {e}[/yellow]"
                    )

        return balances

    def _parse_descriptions(self, descriptions_str: str) -> List[str]:
        """Parse transaction descriptions."""
        if not descriptions_str or descriptions_str == "NOT_FOUND":
            return []
        return [d.strip() for d in descriptions_str.split("|") if d.strip()]

    def _parse_extracted_amounts(self, amounts_str: str) -> List[float]:
        """Parse extracted transaction amounts from PAID/RECEIVED fields."""
        amounts = []
        if not amounts_str or amounts_str == "NOT_FOUND":
            return amounts

        # Handle both pipe-separated and space-separated formats
        if "|" in amounts_str:
            # Pipe-separated format (preferred) - PRESERVE ALL POSITIONS INCLUDING NOT_FOUND
            amount_parts = [
                a.strip() for a in amounts_str.split("|")
            ]  # Don't filter - keep positions!
        else:
            # Space-separated format (fallback for Llama extraction)
            amount_parts = re.findall(r"\$[\d,]+\.?\d*(?:\s+CR)?", amounts_str)
            if not amount_parts:
                # More aggressive fallback: find any currency amounts
                amount_parts = re.findall(r"\$[\d,]+\.?\d*", amounts_str)

        for amount_str in amount_parts:
            try:
                if amount_str.upper().strip() != "NOT_FOUND":
                    # Remove currency symbols, commas, and "CR" indicators
                    clean_amount = re.sub(r"[^\d.-]", "", amount_str)
                    if clean_amount:
                        amounts.append(float(clean_amount))
                    else:
                        amounts.append(0.0)
                else:
                    amounts.append(0.0)
            except (ValueError, TypeError):
                amounts.append(0.0)

        return amounts

    def _create_transactions(
        self,
        dates: List[Tuple[datetime, str]],
        balances: List[float],
        descriptions: List[str],
    ) -> List[Transaction]:
        """Create Transaction objects from parsed data."""
        transactions = []

        # Ensure descriptions list is same length as dates/balances
        while len(descriptions) < len(dates):
            descriptions.append("Transaction")

        for i, ((date_obj, date_str), balance) in enumerate(
            zip(dates, balances, strict=False)
        ):
            description = descriptions[i] if i < len(descriptions) else "Transaction"

            transactions.append(
                Transaction(
                    date=date_obj,
                    date_str=date_str,
                    description=description,
                    balance=balance,
                    amount=0.0,  # Will be calculated
                    transaction_type="",  # Will be calculated
                    index=i,
                )
            )

        return transactions

    def _sort_chronologically(
        self, transactions: List[Transaction]
    ) -> List[Transaction]:
        """Sort transactions chronologically by date."""
        return sorted(transactions, key=lambda t: t.date)

    def _calculate_transaction_types(
        self,
        transactions: List[Transaction],
        extracted_paid: List[float] = None,
        extracted_received: List[float] = None,
        dates_str: str = "",
        descriptions_str: str = "",
    ) -> List[Transaction]:
        """Calculate transaction types and amounts using hybrid approach.

        Uses extracted data for first transaction and mathematical differences for subsequent ones.
        """
        if len(transactions) < 1:
            return transactions

        analyzed = []

        for i in range(len(transactions)):
            transaction = transactions[i]

            if i == 0:
                # First transaction - use extracted data if available, otherwise mark as unknown
                used_extracted_data = False

                # Try to use extracted PAID data first
                if extracted_paid and len(extracted_paid) > 0 and extracted_paid[0] > 0:
                    transaction.transaction_type = "DEBIT"
                    transaction.amount = extracted_paid[0]
                    used_extracted_data = True

                # If no PAID data, try RECEIVED data
                elif (
                    extracted_received
                    and len(extracted_received) > 0
                    and extracted_received[0] > 0
                ):
                    transaction.transaction_type = "CREDIT"
                    transaction.amount = extracted_received[0]
                    used_extracted_data = True

                # Fallback to unknown if no extracted data available
                else:
                    transaction.transaction_type = "UNKNOWN"
                    transaction.amount = 0.0

                # Display detailed earliest transaction information when verbose
                if self.verbose:
                    # Find which position in the original arrays corresponds to this earliest transaction
                    earliest_array_index = self._find_earliest_transaction_array_index(
                        transaction, dates_str, descriptions_str
                    )
                    self._display_earliest_transaction_details(
                        transaction,
                        extracted_paid,
                        extracted_received,
                        used_extracted_data,
                        earliest_array_index,
                    )
            else:
                # Calculate change from previous transaction
                prev_balance = transactions[i - 1].balance
                current_balance = transaction.balance
                balance_change = current_balance - prev_balance

                if balance_change > 0:
                    # Balance increased = money came in = CREDIT
                    transaction.transaction_type = "CREDIT"
                    transaction.amount = abs(balance_change)
                elif balance_change < 0:
                    # Balance decreased = money went out = DEBIT
                    transaction.transaction_type = "DEBIT"
                    transaction.amount = abs(balance_change)
                else:
                    # No change in balance
                    transaction.transaction_type = "UNKNOWN"
                    transaction.amount = 0.0

            analyzed.append(transaction)

        return analyzed

    def _generate_analysis(
        self, transactions: List[Transaction]
    ) -> BankStatementAnalysis:
        """Generate final analysis results."""
        total_debits = sum(
            t.amount for t in transactions if t.transaction_type == "DEBIT"
        )
        total_credits = sum(
            t.amount for t in transactions if t.transaction_type == "CREDIT"
        )

        # Generate calculated amounts in original order
        amounts_paid = []
        amounts_received = []

        # Sort back to original order for output
        original_order = sorted(transactions, key=lambda t: t.index)

        for transaction in original_order:
            if transaction.transaction_type == "DEBIT":
                amounts_paid.append(f"${transaction.amount:.2f}")
                amounts_received.append("NOT_FOUND")
            elif transaction.transaction_type == "CREDIT":
                amounts_paid.append("NOT_FOUND")
                amounts_received.append(f"${transaction.amount:.2f}")
            else:
                amounts_paid.append("NOT_FOUND")
                amounts_received.append("NOT_FOUND")

        return BankStatementAnalysis(
            transactions=transactions,
            total_debits=total_debits,
            total_credits=total_credits,
            transaction_count=len(transactions),
            calculated_amounts_paid=amounts_paid,
            calculated_amounts_received=amounts_received,
            success=True,
            errors=[],
        )

    def _display_analysis_summary(self, analysis: BankStatementAnalysis):
        """Display analysis summary if verbose mode is enabled."""
        if not self.verbose:
            return

        rprint("\n[bold blue]📊 Bank Statement Mathematical Analysis[/bold blue]")
        rprint(f"[cyan]📋 Transactions: {analysis.transaction_count}[/cyan]")
        rprint(f"[red]📤 Total Debits: ${analysis.total_debits:.2f}[/red]")
        rprint(f"[green]📥 Total Credits: ${analysis.total_credits:.2f}[/green]")
        rprint(
            f"[blue]💰 Net Change: ${analysis.total_credits - analysis.total_debits:.2f}[/blue]"
        )

        if analysis.transactions:
            rprint("\n[dim]📅 Sample transactions:[/dim]")
            for txn in analysis.transactions[:3]:
                color = "red" if txn.transaction_type == "DEBIT" else "green"
                rprint(
                    f"[{color}]  {txn.date_str}: {txn.transaction_type} ${txn.amount:.2f}[/{color}]"
                )
            if len(analysis.transactions) > 3:
                rprint(f"[dim]  ... and {len(analysis.transactions) - 3} more[/dim]")

    def _display_earliest_transaction_details(
        self,
        transaction: Transaction,
        extracted_paid: List[float],
        extracted_received: List[float],
        used_extracted_data: bool,
        array_index: int,
    ):
        """Display detailed information about the earliest transaction when verbose mode is enabled."""
        rprint("\n[bold yellow]🔍 EARLIEST TRANSACTION ANALYSIS[/bold yellow]")
        rprint(f"[cyan]📅 Date: {transaction.date_str}[/cyan]")
        rprint(f"[cyan]💰 Balance: ${transaction.balance:.2f}[/cyan]")
        rprint(
            f"[cyan]📝 Description: {transaction.description[:50]}{'...' if len(transaction.description) > 50 else ''}[/cyan]"
        )
        rprint(
            f"[cyan]📍 Array Position: {array_index} (position in original extraction data)[/cyan]"
        )

        # Show extracted data analysis
        rprint("\n[bold blue]📊 Extracted Data Analysis:[/bold blue]")

        # Display PAID data for the correct transaction index
        if extracted_paid and len(extracted_paid) > array_index:
            paid_amount = extracted_paid[array_index]
            if paid_amount > 0:
                rprint(
                    f"[red]  💸 TRANSACTION_AMOUNTS_PAID[{array_index}]: ${paid_amount:.2f}[/red]"
                )
            else:
                rprint(
                    f"[dim]  💸 TRANSACTION_AMOUNTS_PAID[{array_index}]: NOT_FOUND (zero/invalid)[/dim]"
                )
        else:
            rprint(
                f"[dim]  💸 TRANSACTION_AMOUNTS_PAID[{array_index}]: No data available[/dim]"
            )

        # Display RECEIVED data for the correct transaction index
        if extracted_received and len(extracted_received) > array_index:
            received_amount = extracted_received[array_index]
            if received_amount > 0:
                rprint(
                    f"[green]  💰 TRANSACTION_AMOUNTS_RECEIVED[{array_index}]: ${received_amount:.2f}[/green]"
                )
            else:
                rprint(
                    f"[dim]  💰 TRANSACTION_AMOUNTS_RECEIVED[{array_index}]: NOT_FOUND (zero/invalid)[/dim]"
                )
        else:
            rprint(
                f"[dim]  💰 TRANSACTION_AMOUNTS_RECEIVED[{array_index}]: No data available[/dim]"
            )

        # Show final determination
        rprint("\n[bold green]✅ FINAL DETERMINATION:[/bold green]")
        if used_extracted_data:
            color = "red" if transaction.transaction_type == "DEBIT" else "green"
            rprint(f"[{color}]  🎯 Type: {transaction.transaction_type}[/{color}]")
            rprint(f"[{color}]  💵 Amount: ${transaction.amount:.2f}[/{color}]")
            rprint("[cyan]  📋 Source: Extracted data (hybrid approach)[/cyan]")
        else:
            rprint(f"[yellow]  ⚠️ Type: {transaction.transaction_type}[/yellow]")
            rprint(f"[yellow]  💵 Amount: ${transaction.amount:.2f}[/yellow]")
            rprint(
                "[yellow]  📋 Source: No extracted data available - marked as UNKNOWN[/yellow]"
            )

        rprint("[dim]" + "─" * 60 + "[/dim]")

    def _find_earliest_transaction_array_index(
        self, earliest_transaction: Transaction, dates_str: str, descriptions_str: str
    ) -> int:
        """Find which position in the original extraction arrays corresponds to the earliest transaction."""

        # Parse the original dates and descriptions to find the matching index
        dates = self._parse_dates(dates_str)
        descriptions = self._parse_descriptions(descriptions_str)

        # Look for the earliest transaction by date and description
        for i, (date_obj, _date_str) in enumerate(dates):
            # Match by date
            if date_obj.date() == earliest_transaction.date.date():
                # If we have descriptions, also match by description for accuracy
                if descriptions and i < len(descriptions):
                    if (
                        descriptions[i].strip()
                        == earliest_transaction.description.strip()
                    ):
                        return i
                else:
                    # No descriptions available, match by date only
                    return i

        # Fallback: if we can't find a match, return the transaction's original index
        # This should rarely happen but provides a safe fallback
        return earliest_transaction.index

    def _validate_array_alignment(
        self,
        dates: List[Tuple[datetime, str]],
        extracted_paid: List[float],
        extracted_received: List[float],
        dates_str: str,
        descriptions_str: str,
    ) -> Tuple[List[float], List[float]]:
        """Validate and correct array alignment issues in extracted amounts."""

        if not dates:
            return extracted_paid, extracted_received

        num_transactions = len(dates)

        # Check if arrays are properly aligned (same length as dates)
        paid_aligned = (
            len(extracted_paid) == num_transactions if extracted_paid else True
        )
        received_aligned = (
            len(extracted_received) == num_transactions if extracted_received else True
        )

        if paid_aligned and received_aligned:
            # Arrays are already properly aligned
            return extracted_paid, extracted_received

        if self.verbose:
            rprint(
                "[yellow]⚠️ Detecting array misalignment - applying correction...[/yellow]"
            )

        # Apply correction: resize arrays to match transaction count
        corrected_paid = []
        corrected_received = []

        for i in range(num_transactions):
            # For paid amounts
            if extracted_paid and i < len(extracted_paid):
                corrected_paid.append(extracted_paid[i])
            else:
                corrected_paid.append(0.0)  # NOT_FOUND equivalent

            # For received amounts
            if extracted_received and i < len(extracted_received):
                corrected_received.append(extracted_received[i])
            else:
                corrected_received.append(0.0)  # NOT_FOUND equivalent

        if self.verbose:
            rprint(
                f"[green]✅ Array alignment corrected: {num_transactions} transactions[/green]"
            )

        return corrected_paid, corrected_received

    def _infer_transaction_type_from_description(
        self, description: str
    ) -> Optional[str]:
        """
        Infer transaction type (DEBIT/CREDIT) from description keywords.

        Returns:
            'DEBIT', 'CREDIT', or None if uncertain
        """
        description_lower = description.lower()

        # Keywords strongly indicating DEBIT (money out)
        DEBIT_KEYWORDS = [
            "withdrawal",
            "atm",
            "cash out",
            "payment",
            "purchase",
            "eftpos",
            "direct debit",
            "fee",
            "charge",
            "debit",
            "transfer to",
            "repayment",
            "mortgage",
            "home loan",
            "bill",
            "subscription",
            "insurance",
        ]

        # Keywords strongly indicating CREDIT (money in)
        CREDIT_KEYWORDS = [
            "salary",
            "wage",
            "payroll",
            "deposit",
            "credit",
            "transfer from",
            "received",
            "refund",
            "return",
            "dividend",
            "interest",
            "payment from",
            "income",
            "centrelink",
            "jobseeker",
            "pension",
        ]

        debit_score = sum(
            1 for keyword in DEBIT_KEYWORDS if keyword in description_lower
        )
        credit_score = sum(
            1 for keyword in CREDIT_KEYWORDS if keyword in description_lower
        )

        if debit_score > credit_score:
            return "DEBIT"
        elif credit_score > debit_score:
            return "CREDIT"

        return None

    def _validate_and_correct_vlm_extraction(
        self,
        vlm_paid: float,
        vlm_received: float,
        description: str,
        balance_change: float,
    ) -> Tuple[float, float, str]:
        """
        Validate VLM extracted amounts and correct if necessary.

        Returns:
            Tuple of (corrected_paid, corrected_received, correction_reason)
        """
        # Infer transaction type from description
        inferred_type = self._infer_transaction_type_from_description(description)

        # No numerical validation - trust the VLM extraction and balance calculations

        # Check if VLM classification matches description inference
        if inferred_type:
            if inferred_type == "DEBIT" and vlm_received > 0 and vlm_paid == 0:
                # Misclassified as credit, should be debit
                if self.verbose:
                    rprint(
                        "[yellow]⚠️ Description suggests DEBIT but VLM extracted as CREDIT - correcting[/yellow]"
                    )
                return vlm_received, 0, "Reclassified based on description"
            elif inferred_type == "CREDIT" and vlm_paid > 0 and vlm_received == 0:
                # Misclassified as debit, should be credit
                if self.verbose:
                    rprint(
                        "[yellow]⚠️ Description suggests CREDIT but VLM extracted as DEBIT - correcting[/yellow]"
                    )
                return 0, vlm_paid, "Reclassified based on description"

        return vlm_paid, vlm_received, "No correction needed"


def enhance_bank_statement_extraction(
    extracted_data: Dict[str, Any], verbose: bool = False, use_pandas: bool = True
) -> Dict[str, Any]:
    """
    Enhance bank statement extraction with mathematical post-processing.

    Args:
        extracted_data: Dictionary containing extracted fields
        verbose: Whether to display detailed processing information
        use_pandas: Whether to use pandas-based calculation (recommended)

    Returns:
        Enhanced extracted_data with calculated TRANSACTION_AMOUNTS_PAID
        and TRANSACTION_AMOUNTS_RECEIVED fields
    """
    calculator = BankStatementCalculator(verbose=verbose, use_pandas=use_pandas)
    analysis = calculator.analyze_bank_statement(extracted_data)

    if analysis.success:
        # Update extracted data with calculated fields
        extracted_data["TRANSACTION_AMOUNTS_PAID"] = " | ".join(
            analysis.calculated_amounts_paid
        )
        extracted_data["TRANSACTION_AMOUNTS_RECEIVED"] = " | ".join(
            analysis.calculated_amounts_received
        )

        # Add metadata about the calculation
        extracted_data["_mathematical_analysis"] = {
            "total_debits": analysis.total_debits,
            "total_credits": analysis.total_credits,
            "transaction_count": analysis.transaction_count,
            "calculation_success": True,
        }

        if verbose:
            rprint("[green]✅ Mathematical enhancement completed successfully[/green]")
    else:
        if verbose:
            rprint("[red]❌ Mathematical enhancement failed[/red]")
            for error in analysis.errors:
                rprint(f"[red]  • {error}[/red]")

        # Add error metadata
        extracted_data["_mathematical_analysis"] = {
            "calculation_success": False,
            "errors": analysis.errors,
        }

    return extracted_data
