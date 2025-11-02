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


def map_headers_by_position(headers_pipe_separated: str) -> dict[str, str | None]:
    """
    Map headers by position based on column count.

    Bank statement formats:
    - 3 columns: Date | Description | Amount
    - 4 columns: Date | Description | [varies] | [varies] (use fuzzy matching for positions 3-4)
    - 5 columns: Date | Description | Debit | Credit | Balance

    Args:
        headers_pipe_separated: Pipe-separated header string

    Returns:
        Dictionary mapping semantic fields to actual column names
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
    column_count = len(headers)

    # Initialize mapping
    mapping = {
        'DATE': None,
        'DESCRIPTION': None,
        'DEBIT': None,
        'CREDIT': None,
        'BALANCE': None
    }

    # Positions 0-1 are always Date and Description
    if column_count >= 2:
        mapping['DATE'] = headers[0]
        mapping['DESCRIPTION'] = headers[1]

    if column_count == 3:
        # 3-column format: Date | Description | Amount
        # Map position 2 to DEBIT (assume single amount column represents debits/withdrawals)
        mapping['DEBIT'] = headers[2]

    elif column_count == 4:
        # 4-column format: Date | Description | [varies] | [varies]
        # Use fuzzy matching for positions 3 and 4 since they vary
        # Return None to trigger fuzzy matching in map_headers_smart
        return None  # Signal to use fuzzy matching

    elif column_count >= 5:
        # 5-column format: Date | Description | Debit | Credit | Balance
        mapping['DEBIT'] = headers[2]
        mapping['CREDIT'] = headers[3]
        mapping['BALANCE'] = headers[4]

    return mapping


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


def map_headers_smart(headers_pipe_separated: str, use_positional: bool = True) -> dict[str, str | None]:
    """
    Smart mapping with positional-first approach and fuzzy matching fallback.

    Strategy:
    1. Try positional mapping first (default for standard 5-column bank statements)
    2. Validate that positional mappings make semantic sense using fuzzy matching
    3. Fall back to full fuzzy matching if positional mapping produces poor matches

    Args:
        headers_pipe_separated: Pipe-separated header string
        use_positional: If True, try positional mapping first (default: True)

    Returns:
        Dictionary mapping semantic fields to actual column names

    Example:
        >>> headers = "Transaction Date | Particulars | Withdrawals | Deposits | Running Balance"
        >>> mapping = map_headers_smart(headers)
        >>> # Uses positional mapping (fast, simple)

        >>> weird_headers = "Balance | Date | Description | Amount | Type"
        >>> mapping = map_headers_smart(weird_headers)
        >>> # Falls back to fuzzy matching (handles non-standard order)
    """
    if use_positional:
        # Try positional mapping first
        positional_mapping = map_headers_by_position(headers_pipe_separated)

        # Check if positional mapping returned None (4-column case requiring fuzzy matching)
        if positional_mapping is None:
            print("⚠️  4-column format detected, using fuzzy matching for variable columns...")
            return map_headers_to_fields(headers_pipe_separated)

        # Basic validation: check required fields exist
        is_valid, missing = validate_mapping(positional_mapping, required_fields=['DATE', 'DESCRIPTION', 'DEBIT'])

        if not is_valid:
            print("⚠️  Positional mapping missing required fields, falling back to fuzzy matching...")
            return map_headers_to_fields(headers_pipe_separated)

        # Semantic validation: check that mapped names actually match expected field types
        # Define keywords for each field
        field_keywords = {
            'DATE': ['date', 'dt', 'day', 'transaction date'],
            'DESCRIPTION': ['description', 'details', 'particulars', 'transaction', 'narrative'],
            'DEBIT': ['debit', 'withdrawal', 'withdrawals', 'money out', 'spent', 'payment'],
        }

        # Check if positionally-mapped names make semantic sense
        semantic_scores = {}
        for field in ['DATE', 'DESCRIPTION', 'DEBIT']:
            mapped_name = positional_mapping.get(field)
            if mapped_name:
                keywords = field_keywords[field]
                best_score = max([fuzzy_match(keyword, mapped_name) for keyword in keywords])
                semantic_scores[field] = best_score

        # Check if ANY required field has poor semantic match (indicates wrong order)
        min_threshold = 0.4  # Each field must have at least this match score
        avg_threshold = 0.5  # Average across all fields must exceed this

        min_score = min(semantic_scores.values()) if semantic_scores else 0
        avg_score = sum(semantic_scores.values()) / len(semantic_scores) if semantic_scores else 0

        if min_score < min_threshold or avg_score < avg_threshold:
            print(f"⚠️  Positional mapping has poor semantic match (min: {min_score:.2f}, avg: {avg_score:.2f}), falling back to fuzzy matching...")
            return map_headers_to_fields(headers_pipe_separated)

        # Positional mapping succeeded with good semantic matches
        return positional_mapping

    # Use fuzzy matching as fallback
    return map_headers_to_fields(headers_pipe_separated)


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
- If a transaction is a credit/deposit (not a debit), leave the {debit_col} field blank

ANTI-HALLUCINATION RULES:
- YOU MUST NOT GUESS values you are unsure of
- Rows may have missing values
- Rows NEVER HAVE REPEATED AMOUNTS, SO YOU MUST NOT REPEAT VALUES THAT YOU ARE UNSURE OF
- If a value is unclear or missing, leave that field empty"""

    return instruction


def generate_flat_table_prompt(mapping: dict[str, str | None], headers_pipe_separated: str, yaml_config: dict) -> str:
    """
    Generate dynamic flat table extraction prompt using mapped column names.

    This function takes the flat_table_extraction.yaml template and replaces placeholders
    with actual column names detected from the bank statement.

    Args:
        mapping: Result from map_headers_to_fields()
        headers_pipe_separated: Original pipe-separated headers (e.g., "Date | Description | Debit | Credit | Balance")
        yaml_config: Loaded YAML config from flat_table_extraction.yaml

    Returns:
        Complete prompt with placeholders replaced by actual column names

    Example:
        >>> mapping = {
        ...     'DATE': 'Transaction Date',
        ...     'DESCRIPTION': 'Particulars',
        ...     'DEBIT': 'Withdrawals',
        ...     'CREDIT': 'Deposits',
        ...     'BALANCE': 'Running Balance'
        ... }
        >>> headers = "Transaction Date | Particulars | Withdrawals | Deposits | Running Balance"
        >>> with open('prompts/flat_table_extraction.yaml') as f:
        ...     config = yaml.safe_load(f)
        >>> prompt = generate_flat_table_prompt(mapping, headers, config)
    """
    # Get column names from mapping, with fallback defaults
    date_col = mapping.get('DATE', 'Date')
    desc_col = mapping.get('DESCRIPTION', 'Description')
    debit_col = mapping.get('DEBIT', 'Withdrawal')
    credit_col = mapping.get('CREDIT', 'Deposit')
    balance_col = mapping.get('BALANCE', 'Balance')

    # Get instruction from YAML
    instruction_template = yaml_config.get('instruction', '')

    # Replace placeholders in instruction with actual column names
    instruction = instruction_template.format(
        headers_pipe_separated=headers_pipe_separated,
        date_column=date_col,
        description_column=desc_col,
        debit_column=debit_col,
        credit_column=credit_col,
        balance_column=balance_col
    )

    return instruction


def parse_5_column_headers(turn0_response: str) -> dict[str, str]:
    """Parse Turn 0 response from multi-turn extraction to extract 5-column bank statement headers.

    Designed for standard 5-column bank statement structure:
    - Position 1 (index 0) = Date column
    - Position 2 (index 1) = Description/Transaction column
    - Position 3 (index 2) = Debit/Withdrawal column
    - Position 4 (index 3) = Credit/Deposit column
    - Position 5 (index 4) = Balance column

    Args:
        turn0_response: String response from Turn 0 header identification

    Returns:
        Dictionary mapping semantic field names to actual column names:
        {'date': str, 'description': str, 'debit': str, 'credit': str, 'balance': str}

    Note:
        Falls back to intelligent defaults if parsing fails or returns != 5 columns.
        Uses keyword matching to assign extracted headers to appropriate positions.
    """
    # Default column mapping (fallback for standard bank statements)
    DEFAULT_COLUMNS = {
        'date': 'Date',
        'description': 'Description',
        'debit': 'Debit',
        'credit': 'Credit',
        'balance': 'Balance'
    }

    # Extract the header line from the response
    lines = turn0_response.strip().split('\n')
    header_line = None

    for line in lines:
        # Look for line with commas (likely the headers)
        if ',' in line and 'HEADERS' not in line.upper():
            header_line = line
            break

    if header_line is None:
        print("⚠️  Failed to parse headers from Turn 0 response. Using defaults.")
        print(f"Response was:\n{turn0_response}\n")
        return DEFAULT_COLUMNS

    # Parse comma-separated headers and strip whitespace
    headers = [h.strip() for h in header_line.split(',')]

    # Remove markdown bullet points from first header
    if headers and headers[0].startswith(('- ', '* ')):
        headers[0] = headers[0][2:]

    # If we got exactly 5 headers, use position-based mapping
    if len(headers) == 5:
        column_map = {
            'date': headers[0],
            'description': headers[1],
            'debit': headers[2],
            'credit': headers[3],
            'balance': headers[4]
        }
        print("✅ Successfully parsed 5 column headers")
        return column_map

    # If not exactly 5, try to infer positions intelligently
    print(f"⚠️  Expected 5 column headers, but found {len(headers)}.")
    print(f"Headers found: {headers}")
    print("Applying intelligent defaults with keyword matching...")

    # Start with defaults
    column_map = DEFAULT_COLUMNS.copy()

    # Try to match extracted headers to positions based on keywords
    for header in headers:
        header_lower = header.lower()

        # Date column
        if any(kw in header_lower for kw in ['date', 'day']):
            column_map['date'] = header

        # Description column
        elif any(kw in header_lower for kw in ['description', 'transaction', 'details', 'particulars']):
            column_map['description'] = header

        # Debit column
        elif any(kw in header_lower for kw in ['debit', 'withdrawal', 'dr']):
            column_map['debit'] = header

        # Credit column
        elif any(kw in header_lower for kw in ['credit', 'deposit', 'cr']):
            column_map['credit'] = header

        # Balance column
        elif any(kw in header_lower for kw in ['balance', 'bal']):
            column_map['balance'] = header

    print("✅ Applied column mapping with defaults:")
    for key, value in column_map.items():
        print(f"   {key}: {value}")

    return column_map
