"""Header mapping utilities for bank statement column identification."""

from difflib import SequenceMatcher


def fuzzy_match(text1: str, text2: str) -> float:
    """
    Calculate similarity ratio between two strings (0.0 to 1.0).

    Args:
        text1: First string
        text2: Second string

    Returns:
        Similarity ratio (1.0 = exact match, 0.0 = no similarity)
    """
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def find_best_match(target_keywords: list[str], available_headers: list[str], threshold: float = 0.4) -> str | None:
    """
    Find the best matching header from available headers.

    Args:
        target_keywords: List of keywords to search for (e.g., ["Date", "Date of Transaction"])
        available_headers: List of actual column headers from the image
        threshold: Minimum similarity score to consider a match

    Returns:
        Best matching header or None if no match found
    """
    best_match = None
    best_score = threshold

    for header in available_headers:
        for keyword in target_keywords:
            score = fuzzy_match(keyword, header)
            if score > best_score:
                best_score = score
                best_match = header

    return best_match


def map_headers_to_fields(headers_pipe_separated: str) -> dict[str, str | None]:
    """
    Map detected column headers to semantic field types.

    Args:
        headers_pipe_separated: Pipe-separated header string from classifier
                                e.g., "Date | Transaction Details | Debit | Credit | Balance"

    Returns:
        Dictionary mapping semantic fields to actual column names:
        {
            'DATE': 'Date',
            'DESCRIPTION': 'Transaction Details',
            'DEBIT': 'Debit',
            'CREDIT': 'Credit',
            'BALANCE': 'Balance'
        }
    """
    if not headers_pipe_separated or headers_pipe_separated in ["NO_HEADERS", "N/A", "UNKNOWN"]:
        return {
            'DATE': None,
            'DESCRIPTION': None,
            'DEBIT': None,
            'CREDIT': None,
            'BALANCE': None
        }

    # Parse headers
    headers = [h.strip() for h in headers_pipe_separated.split('|')]

    # Define keyword patterns for each field type
    field_patterns = {
        'DATE': ['Date', 'Date of Transaction', 'Transaction Date', 'Dt', 'Day'],
        'DESCRIPTION': [
            'Description', 'Details', 'Transaction Details', 'Particulars',
            'Transaction', 'Narrative', 'Transaction Description', 'Remarks'
        ],
        'DEBIT': [
            'Debit', 'Debit Amount', 'Debit ($)', 'Withdrawal', 'Withdrawals',
            'Debit (AUD)', 'Money Out', 'Spent', 'Payments'
        ],
        'CREDIT': [
            'Credit', 'Credit Amount', 'Credit ($)', 'Deposit', 'Deposits',
            'Credit (AUD)', 'Money In', 'Received', 'Receipts'
        ],
        'BALANCE': [
            'Balance', 'Running Balance', 'Closing Balance', 'Available Balance',
            'Current Balance', 'Bal', 'Balance ($)', 'Balance (AUD)'
        ]
    }

    # Find best match for each field
    mapping = {}
    for field, keywords in field_patterns.items():
        mapping[field] = find_best_match(keywords, headers, threshold=0.4)

    return mapping


def validate_mapping(mapping: dict[str, str | None], required_fields: list[str] | None = None) -> tuple[bool, list[str]]:
    """
    Validate that required fields have been mapped.

    Args:
        mapping: Result from map_headers_to_fields()
        required_fields: List of required field names (e.g., ['DATE', 'DESCRIPTION', 'DEBIT'])
                        If None, defaults to ['DATE', 'DESCRIPTION']

    Returns:
        Tuple of (is_valid, missing_fields)
    """
    if required_fields is None:
        required_fields = ['DATE', 'DESCRIPTION']

    missing_fields = [field for field in required_fields if not mapping.get(field)]

    return len(missing_fields) == 0, missing_fields


def generate_extraction_instruction(mapping: dict[str, str | None], headers_pipe_separated: str) -> str:
    """
    Generate dynamic extraction instruction using mapped column names.

    Args:
        mapping: Result from map_headers_to_fields()
        headers_pipe_separated: Original pipe-separated headers

    Returns:
        Formatted instruction string with actual column names
    """
    date_col = mapping.get('DATE', 'Date')
    desc_col = mapping.get('DESCRIPTION', 'Description')
    debit_col = mapping.get('DEBIT', 'Debit')

    instruction = f"""Look at the transaction table in this bank statement.

The table has these columns: {headers_pipe_separated}

Extract ONLY these columns for each transaction row:
- {date_col} (the transaction date)
- {desc_col} (the transaction description/details)
- {debit_col} (the debit/withdrawal amount - leave blank if not present or if value is in Credit column)

IMPORTANT:
- Read each row from top to bottom
- Skip the header row
- Extract EVERY transaction row you see
- For the {debit_col} column, only extract if there is a value (ignore if blank or zero)
- If a transaction is a credit/deposit (not a debit), leave the {debit_col} field blank"""

    return instruction
