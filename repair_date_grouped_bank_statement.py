"""
Repair date-grouped bank statement extractions.

Handles malformed table rows where VLM combined multiple transactions from
date-grouped bank statements (where multiple transactions share a single date row).
"""

import re
from pathlib import Path
from typing import Dict, List

from rich.console import Console

console = Console()


def split_description_by_transaction_types(description: str) -> List[str]:
    """
    Split combined descriptions by detecting transaction type boundaries.

    Generic approach: Look for common transaction type keywords that
    typically start new transactions. Handles compound phrases like
    "Salary Payment" as single transactions.
    """
    # Common transaction type starters (generic patterns)
    # Order matters: longer patterns first to avoid false splits
    transaction_starters = [
        r'\bHome Loan Payment\b',  # Compound phrase
        r'\bSalary Payment\b',     # Compound phrase
        r'\bDirect Debit\b',
        r'\bCash Withdrawal\b',
        r'\bATM Withdrawal\b',
        r'\bEFTPOS\b',
        r'\bTransfer (?:To|From)\b',
        r'\bSalary\b',
        r'\bHome Loan\b',
    ]

    # Create combined pattern
    combined_pattern = '|'.join(f'({starter})' for starter in transaction_starters)

    # Find all matches with positions
    matches = list(re.finditer(combined_pattern, description))

    if len(matches) <= 1:
        # Single transaction or no matches
        return [description]

    # Filter out overlapping matches (keep longer/earlier matches)
    filtered_matches = []
    for match in matches:
        # Check if this match overlaps with any already filtered match
        overlaps = False
        for existing in filtered_matches:
            if not (match.end() <= existing.start() or match.start() >= existing.end()):
                overlaps = True
                break
        if not overlaps:
            filtered_matches.append(match)

    if len(filtered_matches) <= 1:
        return [description]

    # Split at each match position
    segments = []
    for i, match in enumerate(filtered_matches):
        start = match.start()
        end = filtered_matches[i + 1].start() if i < len(filtered_matches) - 1 else len(description)
        segment = description[start:end].strip()
        if segment:
            segments.append(segment)

    return segments


def extract_amounts_with_positions(text: str) -> List[tuple]:
    """
    Extract currency amounts with their positions in text.

    Returns list of (amount_string, start_position, end_position) tuples.
    """
    amount_pattern = r'[$£€¥]\s*[\d,]+\.[\d]{2}(?:\s*CR)?'
    matches = []
    for match in re.finditer(amount_pattern, text):
        matches.append((match.group().strip(), match.start(), match.end()))
    return matches


def repair_combined_row_with_balances(date: str, description: str, debit_col: str, balances: List[str]) -> List[Dict[str, str]]:
    """
    Repair a row with combined transactions using extracted balances list.

    Priority: Correctly assign balances (most important) and dates to each transaction,
    followed by debit amounts.

    Args:
        date: Transaction date
        description: Combined description
        debit_col: Debit amount column
        balances: List of all balance amounts (with CR suffix)

    Returns:
        List of repaired transaction dictionaries
    """
    # Split description into individual transactions
    desc_segments = split_description_by_transaction_types(description)

    # Extract debit amounts
    debit_amounts_data = extract_amounts_with_positions(debit_col)
    debit_amounts = [amt[0] for amt in debit_amounts_data if 'CR' not in amt[0]]

    # Determine number of transactions from max of desc segments or balance count
    num_transactions = max(len(desc_segments), len(balances))

    transactions = []
    for i in range(num_transactions):
        trans_desc = desc_segments[i] if i < len(desc_segments) else description
        trans_debit = debit_amounts[i] if i < len(debit_amounts) else ""
        trans_balance = balances[i] if i < len(balances) else ""

        transactions.append({
            'date': date,
            'description': trans_desc,
            'debit': trans_debit,
            'credit': "",
            'balance': trans_balance
        })

    return transactions


def repair_combined_row(date: str, description: str, col2: str, col3: str, col4: str = "") -> List[Dict[str, str]]:
    """
    Repair a row with combined transactions.

    Priority: Extract and correctly assign balances (most important) and dates,
    followed by debit amounts.

    Args:
        date: Transaction date
        description: Combined description
        col2: Amount column (could be debit, credit, or balance)
        col3: Amount column (could be credit or balance)
        col4: Balance column (if present)

    Returns:
        List of repaired transaction dictionaries
    """
    # Split description into individual transactions
    desc_segments = split_description_by_transaction_types(description)

    # Extract all amounts from columns
    col2_amounts = extract_amounts_with_positions(col2)
    col3_amounts = extract_amounts_with_positions(col3)
    col4_amounts = extract_amounts_with_positions(col4) if col4 else []

    # Determine column structure and extract balances (highest priority)
    balance_amounts = []
    debit_amounts = []
    credit_amounts = []

    # Balances always end with "CR"
    if col4:
        balance_amounts = [amt[0] for amt in col4_amounts if 'CR' in amt[0]]
        # If col4 has balances, col2 is debit, col3 is credit
        debit_amounts = [amt[0] for amt in col2_amounts]
        credit_amounts = [amt[0] for amt in col3_amounts if 'CR' not in amt[0]]
    elif col3:
        # Check if col3 contains balances (CR suffix)
        col3_balances = [amt[0] for amt in col3_amounts if 'CR' in amt[0]]
        if col3_balances:
            # col3 has balances, so col2 is debits
            balance_amounts = col3_balances
            debit_amounts = [amt[0] for amt in col2_amounts]
            credit_amounts = []
        else:
            # col3 doesn't have balances, might be debit + credit
            debit_amounts = [amt[0] for amt in col2_amounts]
            credit_amounts = [amt[0] for amt in col3_amounts]
            balance_amounts = []
    else:
        # 3 columns: date | desc | amount (likely debit)
        debit_amounts = [amt[0] for amt in col2_amounts]
        credit_amounts = []
        balance_amounts = []

    # Determine number of transactions (use max of desc segments or balance count)
    num_transactions = max(len(desc_segments), len(balance_amounts))

    transactions = []
    for i in range(num_transactions):
        trans_desc = desc_segments[i] if i < len(desc_segments) else description
        trans_debit = debit_amounts[i] if i < len(debit_amounts) else ""
        trans_credit = credit_amounts[i] if i < len(credit_amounts) else ""
        trans_balance = balance_amounts[i] if i < len(balance_amounts) else ""

        transactions.append({
            'date': date,
            'description': trans_desc,
            'debit': trans_debit,
            'credit': trans_credit,
            'balance': trans_balance
        })

    return transactions


def repair_extraction(input_file: str, output_file: str = None):
    """
    Repair date-grouped bank statement extraction with combined transactions.

    Args:
        input_file: Path to extracted text file (markdown table format)
        output_file: Path for repaired output (optional)
    """
    input_path = Path(input_file)

    with input_path.open() as f:
        lines = f.readlines()

    # Find table structure
    header_idx = None
    separator_idx = None
    data_start = None

    for i, line in enumerate(lines):
        if '| Date |' in line and '| Transaction |' in line:
            header_idx = i
        elif header_idx is not None and '| ---' in line:
            separator_idx = i
            data_start = i + 1
            break

    if header_idx is None or data_start is None:
        console.print("[red]✗ Could not find table structure[/red]")
        return

    header = lines[header_idx].rstrip('\n')
    separator = lines[separator_idx].rstrip('\n')

    # Process data rows
    repaired_rows = []
    skipped_lines = 0
    i = data_start

    while i < len(lines):
        line = lines[i].rstrip('\n')

        # Skip empty lines
        if not line.strip():
            skipped_lines += 1
            i += 1
            continue

        # Check for continuation lines (lines with balance amounts but no date)
        if line.strip().startswith('$') and not line.startswith('|'):
            skipped_lines += 1
            i += 1
            continue

        # Skip non-table lines
        if not line.strip().startswith('|'):
            skipped_lines += 1
            i += 1
            continue

        # Parse columns
        parts = [p.strip() for p in line.split('|')[1:-1]]

        if len(parts) < 2:
            i += 1
            continue

        date = parts[0]
        description = parts[1]
        col2 = parts[2] if len(parts) > 2 else ""
        col3 = parts[3] if len(parts) > 3 else ""
        col4 = parts[4] if len(parts) > 4 else ""

        # Look ahead for continuation lines with additional balances
        continuation_balances = []
        j = i + 1
        while j < len(lines):
            next_line = lines[j].rstrip('\n').strip()
            # Check if next line is a continuation (starts with $ or is a partial table row)
            if next_line.startswith('$') and 'CR' in next_line:
                # Extract ALL balances from continuation line (may have multiple)
                balance_matches = re.findall(r'[$£€¥]\s*[\d,]+\.[\d]{2}\s*CR', next_line)
                continuation_balances.extend(balance_matches)
                j += 1
                skipped_lines += 1
            elif next_line.startswith('|') and not lines[j].split('|')[1].strip():
                # Partial table row with no date (continuation of previous row)
                cont_parts = [p.strip() for p in lines[j].split('|')[1:-1]]
                for part in cont_parts:
                    if part and 'CR' in part:
                        balance_matches = re.findall(r'[$£€¥]\s*[\d,]+\.[\d]{2}\s*CR', part)
                        continuation_balances.extend(balance_matches)
                j += 1
                skipped_lines += 1
            else:
                break

        # Check if this is a combined row
        desc_segments = split_description_by_transaction_types(description)

        if len(desc_segments) > 1 or len(parts) < 5 or continuation_balances:
            # Combined or malformed row - repair it
            # Collect all balance amounts from all sources
            all_balances = []
            if col3 and 'CR' in col3:
                balance_matches = re.findall(r'[$£€¥]\s*[\d,]+\.[\d]{2}\s*CR', col3)
                all_balances.extend(balance_matches)
            if col4 and 'CR' in col4:
                balance_matches = re.findall(r'[$£€¥]\s*[\d,]+\.[\d]{2}\s*CR', col4)
                all_balances.extend(balance_matches)
            all_balances.extend(continuation_balances)

            # Get repaired transactions with all balances passed separately
            repaired_transactions = repair_combined_row_with_balances(
                date, description, col2, all_balances
            )

            for trans in repaired_transactions:
                row = f"| {trans['date']} | {trans['description']} | {trans['debit']} | {trans['credit']} | {trans['balance']} |"
                repaired_rows.append(row)
        else:
            # Normal row - keep as is
            repaired_rows.append(line)

        i = j if continuation_balances else i + 1

    # Write output
    if output_file is None:
        output_path = input_path.parent / f"{input_path.stem}_repaired{input_path.suffix}"
    else:
        output_path = Path(output_file)

    with output_path.open('w') as f:
        f.write(header + '\n')
        f.write(separator + '\n')
        for row in repaired_rows:
            f.write(row + '\n')

    console.print(f"[green]✓ Repaired extraction saved to: {output_path}[/green]")
    console.print(f"[cyan]Original data rows: {len(lines) - data_start - skipped_lines}[/cyan]")
    console.print(f"[cyan]Repaired rows: {len(repaired_rows)}[/cyan]")
    console.print(f"[yellow]Net change: {len(repaired_rows) - (len(lines) - data_start - skipped_lines):+d}[/yellow]")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        console.print("[yellow]Usage: python repair_date_grouped_bank_statement.py <input_file> [output_file][/yellow]")
        console.print("[yellow]Example: python repair_date_grouped_bank_statement.py image_009_extracted.txt[/yellow]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    repair_extraction(input_file, output_file)