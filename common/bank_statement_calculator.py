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

from rich import print as rprint


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

    def __init__(self, verbose: bool = False):
        """Initialize the calculator.

        Args:
            verbose: Whether to display detailed processing information
        """
        self.verbose = verbose

    def analyze_bank_statement(
        self,
        extracted_data: Dict[str, Any]
    ) -> BankStatementAnalysis:
        """
        Analyze bank statement data using mathematical balance differences.

        Args:
            extracted_data: Dictionary containing extracted fields including
                           TRANSACTION_DATES and ACCOUNT_BALANCE

        Returns:
            BankStatementAnalysis with calculated transaction types and amounts
        """
        try:
            # Extract required fields
            dates_str = extracted_data.get('TRANSACTION_DATES', '')
            balances_str = extracted_data.get('ACCOUNT_BALANCE', '')
            descriptions_str = extracted_data.get('LINE_ITEM_DESCRIPTIONS', '')

            if not dates_str or not balances_str:
                return BankStatementAnalysis(
                    transactions=[],
                    total_debits=0.0,
                    total_credits=0.0,
                    transaction_count=0,
                    calculated_amounts_paid=[],
                    calculated_amounts_received=[],
                    success=False,
                    errors=["Missing required fields: TRANSACTION_DATES or ACCOUNT_BALANCE"]
                )

            # Parse data into lists
            dates = self._parse_dates(dates_str)
            balances = self._parse_balances(balances_str)
            descriptions = self._parse_descriptions(descriptions_str)

            if len(dates) != len(balances):
                if self.verbose:
                    rprint(f"[yellow]⚠️ Count mismatch: {len(dates)} dates vs {len(balances)} balances - using minimum count[/yellow]")

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
                        errors=[f"No valid transactions: {len(dates)} dates vs {len(balances)} balances"]
                    )

                # Truncate to minimum count
                dates = dates[:min_count]
                balances = balances[:min_count]
                if descriptions and len(descriptions) > min_count:
                    descriptions = descriptions[:min_count]

            if self.verbose:
                rprint(f"[cyan]📊 Analyzing {len(dates)} transactions mathematically[/cyan]")

            # Create transactions and sort chronologically
            transactions = self._create_transactions(dates, balances, descriptions)
            sorted_transactions = self._sort_chronologically(transactions)

            # Calculate transaction types and amounts
            analyzed_transactions = self._calculate_transaction_types(sorted_transactions)

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
                errors=[f"Analysis failed: {str(e)}"]
            )

    def _parse_dates(self, dates_str: str) -> List[Tuple[datetime, str]]:
        """Parse date strings into datetime objects with original strings."""
        dates = []

        # Handle both pipe-separated and space-separated formats
        if '|' in dates_str:
            # Pipe-separated format (preferred)
            date_parts = [d.strip() for d in dates_str.split('|') if d.strip()]
        else:
            # Space-separated format (fallback for Llama extraction)
            # Look for date patterns like "Thu 04 Sep 2025" - more comprehensive
            date_parts = re.findall(r'[A-Za-z]{3}\s+\d{1,2}\s+[A-Za-z]{3}\s+\d{4}', dates_str)
            if not date_parts:
                # Fallback: try to find DD/MM/YYYY patterns
                date_parts = re.findall(r'\d{1,2}\/\d{1,2}\/\d{4}', dates_str)
            if not date_parts:
                # More aggressive fallback: find any date-like patterns
                date_parts = re.findall(r'\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}', dates_str)

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
            "%a %d %b %Y",      # "Thu 04 Sep 2025"
            "%d/%m/%Y",         # "04/09/2025"
            "%d-%b-%y",         # "04-Sep-25"
            "%d %b %Y",         # "04 Sep 2025"
            "%Y-%m-%d",         # "2025-09-04"
            "%d/%m/%y",         # "04/09/25"
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        # Try to extract date components with regex
        # Look for patterns like "04 Sep 2025" in strings like "Thu 04 Sep 2025"
        date_pattern = r'(\d{1,2})\s+(\w{3})\s+(\d{4})'
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
        if '|' in balances_str:
            # Pipe-separated format (preferred)
            balance_parts = [b.strip() for b in balances_str.split('|') if b.strip()]
        else:
            # Space-separated format (fallback for Llama extraction)
            # Enhanced regex to handle CR suffixes and various formats
            balance_parts = re.findall(r'\$[\d,]+\.?\d*(?:\s+CR)?', balances_str)
            if not balance_parts:
                # More aggressive fallback: find any currency amounts
                balance_parts = re.findall(r'\$[\d,]+\.?\d*', balances_str)
            if not balance_parts:
                # Final fallback: split on spaces and filter for currency-like patterns
                parts = balances_str.split()
                balance_parts = [p for p in parts if '$' in p or re.match(r'\d+\.?\d*', p)]

        for balance_str in balance_parts:
            try:
                # Remove currency symbols, commas, and "CR" indicators
                clean_balance = re.sub(r'[^\d.-]', '', balance_str)
                if clean_balance:
                    balances.append(float(clean_balance))
            except (ValueError, TypeError) as e:
                if self.verbose:
                    rprint(f"[yellow]⚠️ Could not parse balance: {balance_str} - {e}[/yellow]")

        return balances

    def _parse_descriptions(self, descriptions_str: str) -> List[str]:
        """Parse transaction descriptions."""
        if not descriptions_str or descriptions_str == "NOT_FOUND":
            return []
        return [d.strip() for d in descriptions_str.split('|') if d.strip()]

    def _create_transactions(
        self,
        dates: List[Tuple[datetime, str]],
        balances: List[float],
        descriptions: List[str]
    ) -> List[Transaction]:
        """Create Transaction objects from parsed data."""
        transactions = []

        # Ensure descriptions list is same length as dates/balances
        while len(descriptions) < len(dates):
            descriptions.append("Transaction")

        for i, ((date_obj, date_str), balance) in enumerate(zip(dates, balances, strict=False)):
            description = descriptions[i] if i < len(descriptions) else "Transaction"

            transactions.append(Transaction(
                date=date_obj,
                date_str=date_str,
                description=description,
                balance=balance,
                amount=0.0,  # Will be calculated
                transaction_type="",  # Will be calculated
                index=i
            ))

        return transactions

    def _sort_chronologically(self, transactions: List[Transaction]) -> List[Transaction]:
        """Sort transactions chronologically by date."""
        return sorted(transactions, key=lambda t: t.date)

    def _calculate_transaction_types(self, transactions: List[Transaction]) -> List[Transaction]:
        """Calculate transaction types and amounts using balance differences."""
        if len(transactions) < 1:
            return transactions

        analyzed = []

        for i in range(len(transactions)):
            transaction = transactions[i]

            if i == 0:
                # First transaction - we can't determine type without opening balance
                # Mark as unknown unless we have additional context
                transaction.transaction_type = "UNKNOWN"
                transaction.amount = 0.0
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

    def _generate_analysis(self, transactions: List[Transaction]) -> BankStatementAnalysis:
        """Generate final analysis results."""
        total_debits = sum(t.amount for t in transactions if t.transaction_type == "DEBIT")
        total_credits = sum(t.amount for t in transactions if t.transaction_type == "CREDIT")

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
            errors=[]
        )

    def _display_analysis_summary(self, analysis: BankStatementAnalysis):
        """Display analysis summary if verbose mode is enabled."""
        if not self.verbose:
            return

        rprint("\n[bold blue]📊 Bank Statement Mathematical Analysis[/bold blue]")
        rprint(f"[cyan]📋 Transactions: {analysis.transaction_count}[/cyan]")
        rprint(f"[red]📤 Total Debits: ${analysis.total_debits:.2f}[/red]")
        rprint(f"[green]📥 Total Credits: ${analysis.total_credits:.2f}[/green]")
        rprint(f"[blue]💰 Net Change: ${analysis.total_credits - analysis.total_debits:.2f}[/blue]")

        if analysis.transactions:
            rprint("\n[dim]📅 Sample transactions:[/dim]")
            for txn in analysis.transactions[:3]:
                color = "red" if txn.transaction_type == "DEBIT" else "green"
                rprint(f"[{color}]  {txn.date_str}: {txn.transaction_type} ${txn.amount:.2f}[/{color}]")
            if len(analysis.transactions) > 3:
                rprint(f"[dim]  ... and {len(analysis.transactions) - 3} more[/dim]")


def enhance_bank_statement_extraction(extracted_data: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """
    Enhance bank statement extraction with mathematical post-processing.

    Args:
        extracted_data: Dictionary containing extracted fields
        verbose: Whether to display detailed processing information

    Returns:
        Enhanced extracted_data with calculated TRANSACTION_AMOUNTS_PAID
        and TRANSACTION_AMOUNTS_RECEIVED fields
    """
    calculator = BankStatementCalculator(verbose=verbose)
    analysis = calculator.analyze_bank_statement(extracted_data)

    if analysis.success:
        # Update extracted data with calculated fields
        extracted_data['TRANSACTION_AMOUNTS_PAID'] = ' | '.join(analysis.calculated_amounts_paid)
        extracted_data['TRANSACTION_AMOUNTS_RECEIVED'] = ' | '.join(analysis.calculated_amounts_received)

        # Add metadata about the calculation
        extracted_data['_mathematical_analysis'] = {
            'total_debits': analysis.total_debits,
            'total_credits': analysis.total_credits,
            'transaction_count': analysis.transaction_count,
            'calculation_success': True
        }

        if verbose:
            rprint("[green]✅ Mathematical enhancement completed successfully[/green]")
    else:
        if verbose:
            rprint("[red]❌ Mathematical enhancement failed[/red]")
            for error in analysis.errors:
                rprint(f"[red]  • {error}[/red]")

        # Add error metadata
        extracted_data['_mathematical_analysis'] = {
            'calculation_success': False,
            'errors': analysis.errors
        }

    return extracted_data