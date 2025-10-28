"""Markdown table parsing and transformation utilities for bank statement extraction."""

import re
from typing import List, Tuple


def parse_markdown_table(markdown_text: str) -> List[List[str]]:
    """
    Parse markdown table into list of rows.

    Args:
        markdown_text: Markdown text containing table

    Returns:
        List of rows, where each row is a list of cell values

    Example:
        >>> table = "| Date | Amount |\\n| --- | --- |\\n| 01/01 | $100 |"
        >>> rows = parse_markdown_table(table)
        >>> rows
        [['01/01', '$100']]
    """
    lines = markdown_text.split('\n')
    table_rows = []
    found_separator = False

    for line in lines:
        line = line.strip()
        if not line or '|' not in line:
            continue

        # Skip separator line (| --- | --- |)
        if re.match(r'^\|\s*-+\s*\|', line):
            found_separator = True
            continue

        # Only process rows after separator
        if found_separator:
            # Parse data row: split by | and clean
            cells = [cell.strip() for cell in line.split('|')]
            # Remove empty cells from leading/trailing |
            cells = [c for c in cells if c]

            if cells:  # Only add non-empty rows
                table_rows.append(cells)

    return table_rows


def extract_columns(
    table_rows: List[List[str]],
    *column_indices: int
) -> Tuple[List[str], ...]:
    """
    Extract specific columns from parsed table rows.

    Args:
        table_rows: List of rows from parse_markdown_table()
        *column_indices: Column indices to extract (0-based)

    Returns:
        Tuple of lists, one per requested column

    Example:
        >>> rows = [['01/01', 'Store', '$100'], ['01/02', 'Gas', '$50']]
        >>> dates, amounts = extract_columns(rows, 0, 2)
        >>> dates
        ['01/01', '01/02']
        >>> amounts
        ['$100', '$50']
    """
    if not column_indices:
        raise ValueError("At least one column index must be specified")

    # Initialize result lists
    results = [[] for _ in column_indices]

    for row in table_rows:
        for i, col_idx in enumerate(column_indices):
            if col_idx < len(row):
                results[i].append(row[col_idx])
            else:
                results[i].append("NOT_FOUND")

    return tuple(results)


def format_structured_extraction(
    dates: List[str],
    descriptions: List[str],
    withdrawals: List[str]
) -> str:
    """
    Format extracted columns into structured extraction output (STEP 2-5 format).

    Args:
        dates: List of transaction dates
        descriptions: List of transaction descriptions
        withdrawals: List of withdrawal amounts

    Returns:
        Formatted structured output string

    Example:
        >>> dates = ['01/01', '01/02']
        >>> descriptions = ['Store', 'Gas']
        >>> withdrawals = ['$100', '$50']
        >>> output = format_structured_extraction(dates, descriptions, withdrawals)
        >>> 'STATEMENT_DATE_RANGE:' in output
        True
    """
    # STEP 2: Extract date range
    if dates:
        first_date = dates[0]
        last_date = dates[-1]
        date_range = f"STATEMENT_DATE_RANGE: [ {first_date} - {last_date} ]"
    else:
        date_range = "STATEMENT_DATE_RANGE: [ NOT_FOUND ]"

    # STEP 3: Extract all dates
    dates_str = " | ".join(dates)
    transaction_dates = f"TRANSACTION_DATES: [ {dates_str} ]"

    # STEP 4: Extract all descriptions
    descriptions_str = " | ".join(descriptions)
    line_item_descriptions = f"LINE_ITEM_DESCRIPTIONS: [ {descriptions_str} ]"

    # STEP 5: Extract all withdrawals
    withdrawals_str = " | ".join(withdrawals)
    transaction_amounts = f"TRANSACTION_AMOUNTS_PAID: [ {withdrawals_str} ]"

    # Combine into structured output
    structured_output = f"""DOCUMENT_TYPE: BANK_STATEMENT

{date_range}

{transaction_dates}

{line_item_descriptions}

{transaction_amounts}
"""
    return structured_output


def filter_not_found_rows(
    dates: List[str],
    descriptions: List[str],
    amounts: List[str]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Filter out rows where amounts contain 'NOT_FOUND'.

    Args:
        dates: List of transaction dates
        descriptions: List of transaction descriptions
        amounts: List of amounts (withdrawals)

    Returns:
        Tuple of (filtered_dates, filtered_descriptions, filtered_amounts)

    Example:
        >>> dates = ['01/01', '01/02', '01/03']
        >>> descriptions = ['A', 'B', 'C']
        >>> amounts = ['$100', 'NOT_FOUND', '$200']
        >>> f_dates, f_desc, f_amt = filter_not_found_rows(dates, descriptions, amounts)
        >>> f_dates
        ['01/01', '01/03']
        >>> f_amt
        ['$100', '$200']
    """
    filtered_dates = []
    filtered_descriptions = []
    filtered_amounts = []

    for i in range(len(dates)):
        if i < len(amounts) and amounts[i] != "NOT_FOUND":
            filtered_dates.append(dates[i])
            filtered_descriptions.append(descriptions[i])
            filtered_amounts.append(amounts[i])

    return filtered_dates, filtered_descriptions, filtered_amounts


def parse_and_extract_bank_statement(
    markdown_text: str,
    date_col_idx: int = 0,
    description_col_idx: int = 1,
    withdrawal_col_idx: int = 2,
    filter_not_found: bool = False
) -> Tuple[str, str]:
    """
    Complete pipeline: Parse markdown table and generate structured extraction.

    This is a convenience function that combines all steps:
    1. Parse markdown table
    2. Extract specific columns
    3. Format structured output
    4. Optionally filter NOT_FOUND rows

    Args:
        markdown_text: Markdown table text
        date_col_idx: Column index for dates (default: 0)
        description_col_idx: Column index for descriptions (default: 1)
        withdrawal_col_idx: Column index for withdrawals (default: 2)
        filter_not_found: If True, filter out rows with NOT_FOUND in withdrawals

    Returns:
        Tuple of (full_structured_output, filtered_structured_output)
        If filter_not_found=False, both outputs will be the same

    Example:
        >>> table = "| Date | Desc | Amount |\\n| --- | --- | --- |\\n| 01/01 | A | $100 |"
        >>> full, filtered = parse_and_extract_bank_statement(table)
        >>> 'TRANSACTION_DATES:' in full
        True
    """
    # Parse markdown table
    table_rows = parse_markdown_table(markdown_text)

    if not table_rows:
        return "NO_TRANSACTIONS_FOUND", "NO_TRANSACTIONS_FOUND"

    # Extract columns
    dates, descriptions, withdrawals = extract_columns(
        table_rows,
        date_col_idx,
        description_col_idx,
        withdrawal_col_idx
    )

    # Generate full structured output
    full_output = format_structured_extraction(dates, descriptions, withdrawals)

    # Generate filtered output if requested
    if filter_not_found:
        filtered_dates, filtered_descriptions, filtered_amounts = filter_not_found_rows(
            dates, descriptions, withdrawals
        )
        filtered_output = format_structured_extraction(
            filtered_dates, filtered_descriptions, filtered_amounts
        )
    else:
        filtered_output = full_output

    return full_output, filtered_output
