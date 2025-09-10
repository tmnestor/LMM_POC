"""Response preprocessing utilities for document extraction.

This module contains functions for cleaning, parsing, and mapping
model responses from document extraction (invoices, receipts, bank statements)
to universal field formats for evaluation.
"""

import re

from rich import print as rprint


def clean_markdown_response(response: str) -> str:
    """Clean markdown formatting from model response before parsing.

    Removes asterisks from field names while preserving the field structure.
    Enhanced to handle various markdown patterns that cause evaluation failures.

    Args:
        response: Raw response string from model

    Returns:
        Cleaned response string with markdown formatting removed
    """
    if not response:
        return response
    
    cleaned = response
    
    # Pattern 1: **FIELD_NAME:** -> FIELD_NAME: (exact match)
    cleaned = re.sub(r"\*\*([A-Z_]+):\*\*", r"\1:", cleaned)
    
    # Pattern 2: **FIELD_NAME:** value -> FIELD_NAME: value (with value on same line)
    cleaned = re.sub(r"\*\*([A-Z_]+):\*\*\s*", r"\1: ", cleaned)
    
    # Pattern 3: Handle generic asterisk wrapping around field names
    cleaned = re.sub(r"\*+([A-Z_]+):\*+", r"\1:", cleaned)
    
    # Pattern 4: Handle spaces around field names and asterisks
    cleaned = re.sub(r"\*+\s*([A-Z_]+)\s*:\s*\*+", r"\1:", cleaned)
    
    # Pattern 5: Clean up any remaining double asterisks around words
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    
    # Pattern 6: Handle cases where asterisks appear at start of line with field names
    cleaned = re.sub(r"^\*+\s*([A-Z_]+):", r"\1:", cleaned, flags=re.MULTILINE)
    
    # Pattern 7: Clean up any remaining single or multiple asterisks around field patterns
    cleaned = re.sub(r"\*+\s*([A-Z_][A-Z0-9_]*)\s*:\s*\*+", r"\1:", cleaned)
    
    # Pattern 8: Handle bold markdown at the beginning of field names only
    cleaned = re.sub(r"^(\s*)\*\*([A-Z_]+):\*\*(.*)$", r"\1\2:\3", cleaned, flags=re.MULTILINE)
    
    # Final cleanup: Remove any stray asterisks that might be left around colons
    cleaned = re.sub(r"\*+:\*+", ":", cleaned)
    cleaned = re.sub(r"\*+:", ":", cleaned)
    cleaned = re.sub(r":\*+", ":", cleaned)
    
    return cleaned


def extract_transaction_data_from_table(response: str) -> dict:
    """Extract transaction data from markdown table in response.

    Parses markdown table rows to extract transaction dates, descriptions,
    and debit amounts for universal field mapping. Enhanced for complex documents.

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
    
    # Debug: Count potential table lines
    table_lines = [line for line in lines if "|" in line]
    print(f"DEBUG: Found {len(table_lines)} lines with '|' in response")

    # Parse markdown table rows with enhanced detection
    for _line_num, line in enumerate(lines):
        # More flexible table row detection
        if "|" in line and line.strip():
            # Skip header separators and header rows
            if (line.strip().startswith("|---") or 
                "-" in line and "|" in line and len(line.replace("-", "").replace("|", "").strip()) < 5):
                continue
                
            # Skip obvious header rows (but be flexible about case and spacing)
            line_lower = line.lower()
            if ("date" in line_lower and "description" in line_lower) or \
               ("date" in line_lower and "debit" in line_lower) or \
               ("description" in line_lower and "credit" in line_lower):
                continue

            # Parse table row - handle different cell counts gracefully
            cells = [cell.strip() for cell in line.split("|")]
            # Remove empty cells from start/end
            while cells and not cells[0]:
                cells.pop(0)
            while cells and not cells[-1]:
                cells.pop()
                
            if len(cells) >= 3:  # Minimum: Date, Description, Amount
                date = cells[0].strip() if len(cells) > 0 else ""
                description = cells[1].strip() if len(cells) > 1 else ""
                
                # For amount, try debit column first, then credit if debit is empty
                amount = ""
                if len(cells) >= 4:  # Has separate debit/credit columns
                    debit = cells[2].strip() if len(cells) > 2 else ""
                    credit = cells[3].strip() if len(cells) > 3 else ""
                    
                    # Prefer debit, fall back to credit, then NOT_FOUND
                    if debit and debit != "NOT_FOUND" and debit != "":
                        amount = debit
                    elif credit and credit != "NOT_FOUND" and credit != "":
                        amount = credit
                    else:
                        amount = "NOT_FOUND"
                else:
                    # Single amount column
                    amount = cells[2].strip() if len(cells) > 2 else "NOT_FOUND"

                # Add data if we have meaningful values
                if date and date != "NOT_FOUND" and date != "":
                    transaction_dates.append(date)
                    
                    # Always add description and amount for each date
                    if description and description != "NOT_FOUND":
                        transaction_descriptions.append(description)
                    else:
                        transaction_descriptions.append("NOT_FOUND")
                        
                    transaction_amounts_paid.append(amount)
                    
                    # Debug output for complex documents
                    if len(transaction_dates) <= 5 or len(transaction_dates) % 10 == 0:
                        print(f"DEBUG: Row {len(transaction_dates)}: {date} | {description[:30]}... | {amount}")

    print(f"DEBUG: Extracted {len(transaction_dates)} transactions from table")
    
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


def map_invoice_fields_to_universal(invoice_response: str) -> str:
    """Map invoice/bill field names to universal field names for evaluation.

    Most invoice fields already use universal names, but this ensures
    consistency and adds any missing fields needed for evaluation.

    Args:
        invoice_response: Raw response from invoice extraction

    Returns:
        Response with universal field names
    """
    # Clean markdown formatting first
    cleaned_response = clean_markdown_response(invoice_response)

    # Invoice fields are mostly already universal, just ensure DOCUMENT_TYPE is set correctly
    if "DOCUMENT_TYPE:" not in cleaned_response:
        cleaned_response = "DOCUMENT_TYPE: INVOICE\n" + cleaned_response

    # Add fields that aren't applicable to invoices
    lines = cleaned_response.split("\n")
    fields_present = {line.split(":")[0].strip() for line in lines if ":" in line}

    # Fields specific to bank statements that need to be added as NOT_FOUND
    bank_statement_fields = {
        "STATEMENT_DATE_RANGE": "NOT_FOUND",
        "TRANSACTION_DATES": "NOT_FOUND",
        "TRANSACTION_AMOUNTS_PAID": "NOT_FOUND",
    }

    additions = []
    for field, default_value in bank_statement_fields.items():
        if field not in fields_present:
            additions.append(f"{field}: {default_value}")

    if additions:
        cleaned_response = cleaned_response + "\n" + "\n".join(additions)

    return cleaned_response


def map_receipt_fields_to_universal(receipt_response: str) -> str:
    """Map receipt field names to universal field names for evaluation.

    Receipt fields use the same structure as invoices but may have
    fewer populated fields.

    Args:
        receipt_response: Raw response from receipt extraction

    Returns:
        Response with universal field names
    """
    # Clean markdown formatting first
    cleaned_response = clean_markdown_response(receipt_response)

    # Ensure DOCUMENT_TYPE is set correctly
    if "DOCUMENT_TYPE:" not in cleaned_response:
        cleaned_response = "DOCUMENT_TYPE: RECEIPT\n" + cleaned_response

    # Add fields that aren't applicable to receipts
    lines = cleaned_response.split("\n")
    fields_present = {line.split(":")[0].strip() for line in lines if ":" in line}

    # Fields specific to bank statements that need to be added as NOT_FOUND
    bank_statement_fields = {
        "STATEMENT_DATE_RANGE": "NOT_FOUND",
        "TRANSACTION_DATES": "NOT_FOUND",
        "TRANSACTION_AMOUNTS_PAID": "NOT_FOUND",
    }

    additions = []
    for field, default_value in bank_statement_fields.items():
        if field not in fields_present:
            additions.append(f"{field}: {default_value}")

    if additions:
        cleaned_response = cleaned_response + "\n" + "\n".join(additions)

    return cleaned_response


def map_fields_to_universal(response: str, document_type: str) -> str:
    """Map document-specific field names to universal field names.

    Routes to the appropriate mapping function based on document type.

    Args:
        response: Raw response from extraction
        document_type: Type of document (INVOICE, RECEIPT, BANK_STATEMENT, or STATEMENT)

    Returns:
        Response with universal field names for evaluation
    """
    # Normalize document type
    doc_type_upper = document_type.upper()

    if doc_type_upper in ["BANK_STATEMENT", "STATEMENT"]:
        return map_bank_fields_to_universal(response)
    elif doc_type_upper == "INVOICE":
        return map_invoice_fields_to_universal(response)
    elif doc_type_upper == "RECEIPT":
        return map_receipt_fields_to_universal(response)
    else:
        # Default to invoice mapping if unknown
        rprint(
            f"[yellow]Warning: Unknown document type '{document_type}', using invoice mapping[/yellow]"
        )
        return map_invoice_fields_to_universal(response)
