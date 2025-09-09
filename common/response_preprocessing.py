"""Response preprocessing utilities for bank statement extraction.

This module contains functions for cleaning, parsing, and mapping
model responses from bank statement extraction to universal field formats.
"""

import re

from rich import print as rprint


def clean_markdown_response(response: str) -> str:
    """Clean markdown formatting from model response before parsing.

    Removes asterisks from field names while preserving the field structure.

    Args:
        response: Raw response string from model

    Returns:
        Cleaned response string with markdown formatting removed
    """
    # Remove asterisks from field names (e.g., **FIELD_NAME:** -> FIELD_NAME:)
    cleaned = re.sub(r"\*+([A-Z_]+):\*+", r"\1:", response)

    # Also handle case where there might be spaces
    cleaned = re.sub(r"\*+\s*([A-Z_]+)\s*:\s*\*+", r"\1:", cleaned)

    return cleaned


def extract_transaction_data_from_table(response: str) -> dict:
    """Extract transaction data from markdown table in response.

    Parses markdown table rows to extract transaction dates, descriptions,
    and debit amounts for universal field mapping.

    Args:
        response: Response string containing markdown table

    Returns:
        Dictionary with transaction_dates, transaction_descriptions,
        and transaction_amounts_paid fields
    """
    transaction_dates = []
    transaction_descriptions = []
    transaction_amounts_paid = []

    lines = response.split("\n")

    # Parse markdown table rows
    for line in lines:
        if (
            "|" in line
            and not line.strip().startswith("|---")
            and "Date" not in line
            and "Description" not in line
        ):
            # Parse table row
            cells = [
                cell.strip() for cell in line.split("|")[1:-1]
            ]  # Remove first/last empty cells
            if len(cells) >= 4:  # Date, Description, Debit, Credit, Balance
                date = cells[0].strip()
                description = cells[1].strip()
                debit = cells[2].strip()

                if date and date != "NOT_FOUND":
                    transaction_dates.append(date)

                if description and description != "NOT_FOUND":
                    transaction_descriptions.append(description)

                # Only include debit amounts (money OUT), use NOT_FOUND for credits
                if debit and debit != "NOT_FOUND":
                    transaction_amounts_paid.append(debit)
                else:
                    transaction_amounts_paid.append("NOT_FOUND")

    return {
        "transaction_dates": " | ".join(transaction_dates)
        if transaction_dates
        else "NOT_FOUND",
        "transaction_descriptions": " | ".join(transaction_descriptions)
        if transaction_descriptions
        else "NOT_FOUND",
        "transaction_amounts_paid": " | ".join(transaction_amounts_paid)
        if transaction_amounts_paid
        else "NOT_FOUND",
    }


def extract_statement_date_range(response: str) -> str:
    """Extract STATEMENT_DATE_RANGE from date fields in response.

    Extracts EARLIEST_TRANSACTION_DATE and LATEST_TRANSACTION_DATE from response
    and formats them as a date range string.

    Args:
        response: Response string containing date fields

    Returns:
        Date range string in format 'DD/MM/YYYY to DD/MM/YYYY' or 'NOT_FOUND'
    """
    # First clean markdown asterisks from the response
    cleaned_response = clean_markdown_response(response)
    lines = cleaned_response.split("\n")

    earliest_date = None
    latest_date = None

    # Debug: Print some lines to see what we're working with
    rprint(f"[dim]DEBUG: Looking for date fields in {len(lines)} lines[/dim]")

    for line in lines:
        line = line.strip()

        # More flexible matching - handle variations
        if "EARLIEST_TRANSACTION_DATE" in line and ":" in line:
            parts = line.split(":", 1)
            if len(parts) == 2:
                date_value = parts[1].strip()
                if date_value and date_value != "NOT_FOUND":
                    earliest_date = date_value
                    rprint(f"[dim]DEBUG: Found earliest date: {earliest_date}[/dim]")

        if "LATEST_TRANSACTION_DATE" in line and ":" in line:
            parts = line.split(":", 1)
            if len(parts) == 2:
                date_value = parts[1].strip()
                if date_value and date_value != "NOT_FOUND":
                    latest_date = date_value
                    rprint(f"[dim]DEBUG: Found latest date: {latest_date}[/dim]")

    if earliest_date and latest_date:
        result = f"{earliest_date} to {latest_date}"
        rprint(f"[dim]DEBUG: Constructed date range: {result}[/dim]")
        return result
    else:
        rprint(
            f"[dim]DEBUG: Date range construction failed - earliest: {earliest_date}, latest: {latest_date}[/dim]"
        )
        return "NOT_FOUND"


def map_bank_fields_to_universal(bank_response: str) -> str:
    """Map bank-specific field names to universal field names for evaluation.

    Converts bank statement extraction fields to the universal field format
    used for evaluation across different document types. This preserves the
    excellent extraction quality while enabling standardized evaluation.

    Args:
        bank_response: Raw response from bank statement extraction

    Returns:
        Response with mapped universal field names and additional required fields
    """
    # Bank-specific to universal field mapping
    field_mapping = {
        "BANK_NAME": "SUPPLIER_NAME",
        "BANK_ACCOUNT_HOLDER": "PAYER_NAME",
    }

    # Apply field name mappings
    mapped_response = bank_response
    for bank_field, universal_field in field_mapping.items():
        pattern = rf"\b{bank_field}:"
        mapped_response = re.sub(pattern, f"{universal_field}:", mapped_response)

    # Extract transaction data from the table
    transaction_data = extract_transaction_data_from_table(bank_response)

    # Extract statement date range from the new date fields
    statement_date_range = extract_statement_date_range(bank_response)

    # Add required universal fields with extracted transaction data
    universal_additions = [
        "DOCUMENT_TYPE: STATEMENT",
        "INVOICE_DATE: NOT_FOUND",
        "BUSINESS_ABN: NOT_FOUND",
        "BUSINESS_ADDRESS: NOT_FOUND",
        "PAYER_ADDRESS: NOT_FOUND",
        f"LINE_ITEM_DESCRIPTIONS: {transaction_data['transaction_descriptions']}",
        "LINE_ITEM_QUANTITIES: NOT_FOUND",
        "LINE_ITEM_PRICES: NOT_FOUND",
        "LINE_ITEM_TOTAL_PRICES: NOT_FOUND",
        "GST_AMOUNT: NOT_FOUND",
        "IS_GST_INCLUDED: NOT_FOUND",
        "TOTAL_AMOUNT: NOT_FOUND",
        f"STATEMENT_DATE_RANGE: {statement_date_range}",
        f"TRANSACTION_DATES: {transaction_data['transaction_dates']}",
        f"TRANSACTION_AMOUNTS_PAID: {transaction_data['transaction_amounts_paid']}",
    ]

    # Insert universal fields at the beginning
    mapped_response = "\n".join(universal_additions) + "\n\n" + mapped_response

    return mapped_response
