"""
Post-Processing Module

Functions for cleaning, validating, and parsing extracted bank statement data
"""

import csv
import json
import re
from pathlib import Path


def clean_llama_output(text):
    """
    Clean up Llama's output
    
    Args:
        text: Raw text from Llama Vision
        
    Returns:
        str: Cleaned text
    """
    # Remove any model artifacts
    text = re.sub(r'<\|.*?\|>', '', text)
    text = re.sub(r'\[INST\].*?\[/INST\]', '', text, flags=re.DOTALL)
    
    # Fix common OCR errors in amounts
    # Ensure proper decimal places
    def fix_amount(match):
        amount = match.group(0)
        # Remove extra spaces
        amount = re.sub(r'\s+', '', amount)
        return amount
    
    text = re.sub(r'\$\s*[\d,]+\.?\d*', fix_amount, text)
    
    # Fix BSB format
    text = re.sub(r'(\d{3})\s*-?\s*(\d{3})', r'\1-\2', text)
    
    # Standardize dates to DD/MM/YYYY
    text = re.sub(r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})', 
                  lambda m: f"{m.group(1).zfill(2)}/{m.group(2).zfill(2)}/{m.group(3)}", 
                  text)
    
    return text.strip()


def parse_transactions_table(text):
    """
    Parse transaction table from Llama output
    
    Args:
        text: Text containing transaction table
        
    Returns:
        list: List of transaction dictionaries
    """
    transactions = []
    
    # Find lines that look like transactions
    lines = text.split('\n')
    
    for line in lines:
        # Look for pattern: Date | Description | Debit | Credit | Balance
        if '|' in line and re.search(r'\d{2}/\d{2}/\d{4}', line):
            parts = [p.strip() for p in line.split('|')]
            
            if len(parts) >= 5:
                txn = {
                    'date': parts[0],
                    'description': parts[1],
                    'debit': parts[2] if parts[2] != '-' else '',
                    'credit': parts[3] if parts[3] != '-' else '',
                    'balance': parts[4]
                }
                transactions.append(txn)
    
    return transactions


def clean_amount(amount_str):
    """
    Clean and convert amount string to float
    
    Args:
        amount_str: String representation of amount (e.g., "$1,234.56")
        
    Returns:
        float: Cleaned amount
    """
    if isinstance(amount_str, (int, float)):
        return float(amount_str)
    if isinstance(amount_str, str):
        # Remove $, commas, whitespace
        cleaned = re.sub(r'[$,\s]', '', amount_str)
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return 0.0
    return 0.0


def validate_bsb(bsb_str):
    """
    Validate and format BSB number
    
    Args:
        bsb_str: BSB string
        
    Returns:
        str: Formatted BSB (XXX-XXX) or None if invalid
    """
    if not bsb_str:
        return None
    
    # Remove all non-digits
    digits = re.sub(r'\D', '', bsb_str)
    
    if len(digits) == 6:
        return f"{digits[:3]}-{digits[3:]}"
    
    return None


def validate_date(date_str):
    """
    Validate date format
    
    Args:
        date_str: Date string
        
    Returns:
        bool: True if valid DD/MM/YYYY format
    """
    pattern = r'^\d{2}/\d{2}/\d{4}$'
    return bool(re.match(pattern, date_str))


def extract_header_info(text):
    """
    Extract header information from text
    
    Args:
        text: Extracted text containing header info
        
    Returns:
        dict: Dictionary with header fields
    """
    header = {}
    
    # Extract bank name
    bank_match = re.search(r'Bank:\s*(.+)', text, re.IGNORECASE)
    if bank_match:
        header['bank'] = bank_match.group(1).strip()
    
    # Extract account holder
    holder_match = re.search(r'Account\s+Holder:\s*(.+)', text, re.IGNORECASE)
    if holder_match:
        header['account_holder'] = holder_match.group(1).strip()
    
    # Extract BSB
    bsb_match = re.search(r'BSB:\s*(\d{3}-?\d{3})', text, re.IGNORECASE)
    if bsb_match:
        header['bsb'] = validate_bsb(bsb_match.group(1))
    
    # Extract account number
    acc_match = re.search(r'Account\s+Number:\s*(\d+)', text, re.IGNORECASE)
    if acc_match:
        header['account_number'] = acc_match.group(1).strip()
    
    # Extract statement period
    period_match = re.search(r'Period:\s*(\d{2}/\d{2}/\d{4})\s+to\s+(\d{2}/\d{2}/\d{4})', text, re.IGNORECASE)
    if period_match:
        header['period_from'] = period_match.group(1)
        header['period_to'] = period_match.group(2)
    
    return header


def extract_summary_info(text):
    """
    Extract summary information from text
    
    Args:
        text: Extracted text containing summary info
        
    Returns:
        dict: Dictionary with summary fields
    """
    summary = {}
    
    # Extract opening balance
    opening_match = re.search(r'Opening\s+Balance[:\s]+\$?([\d,]+\.?\d*)', text, re.IGNORECASE)
    if opening_match:
        summary['opening_balance'] = clean_amount(opening_match.group(1))
    
    # Extract closing balance
    closing_match = re.search(r'Closing\s+Balance[:\s]+\$?([\d,]+\.?\d*)', text, re.IGNORECASE)
    if closing_match:
        summary['closing_balance'] = clean_amount(closing_match.group(1))
    
    # Extract total credits
    credits_match = re.search(r'Total\s+Credits[:\s]+\$?([\d,]+\.?\d*)', text, re.IGNORECASE)
    if credits_match:
        summary['total_credits'] = clean_amount(credits_match.group(1))
    
    # Extract total debits
    debits_match = re.search(r'Total\s+Debits[:\s]+\$?([\d,]+\.?\d*)', text, re.IGNORECASE)
    if debits_match:
        summary['total_debits'] = clean_amount(debits_match.group(1))
    
    return summary


def validate_transactions(transactions, opening_balance=None):
    """
    Validate transaction data and check balance calculations
    
    Args:
        transactions: List of transaction dictionaries
        opening_balance: Opening balance for validation
        
    Returns:
        tuple: (validated_transactions, list of errors)
    """
    errors = []
    
    if not transactions:
        return transactions, ["No transactions found"]
    
    # Validate each transaction
    for i, txn in enumerate(transactions):
        # Check date format
        if 'date' in txn and not validate_date(txn['date']):
            errors.append(f"Transaction {i+1}: Invalid date format - {txn.get('date', 'N/A')}")
        
        # Check that either debit or credit is present
        has_debit = txn.get('debit') and txn['debit'] != '-'
        has_credit = txn.get('credit') and txn['credit'] != '-'
        
        if not has_debit and not has_credit:
            errors.append(f"Transaction {i+1}: No debit or credit amount")
        
        # Clean amounts
        if has_debit:
            txn['debit'] = clean_amount(txn['debit'])
        else:
            txn['debit'] = 0.0
            
        if has_credit:
            txn['credit'] = clean_amount(txn['credit'])
        else:
            txn['credit'] = 0.0
        
        if 'balance' in txn:
            txn['balance'] = clean_amount(txn['balance'])
    
    # Validate running balance if opening balance provided
    if opening_balance is not None:
        calculated_balance = opening_balance
        
        for i, txn in enumerate(transactions):
            calculated_balance += txn.get('credit', 0.0)
            calculated_balance -= txn.get('debit', 0.0)
            
            stated_balance = txn.get('balance', 0.0)
            
            if stated_balance > 0 and abs(calculated_balance - stated_balance) > 0.02:
                errors.append(f"Transaction {i+1}: Balance mismatch - calculated: ${calculated_balance:.2f}, stated: ${stated_balance:.2f}")
                txn['balance_mismatch'] = True
    
    return transactions, errors


def parse_to_structured_format(text):
    """
    Parse complete extraction into structured format
    
    Args:
        text: Complete extracted text
        
    Returns:
        dict: Structured data with header, summary, and transactions
    """
    # Clean the text first
    text = clean_llama_output(text)
    
    # Split into sections
    sections = {
        'header': '',
        'summary': '',
        'transactions': ''
    }
    
    # Try to identify sections
    if 'HEADER INFORMATION' in text:
        header_start = text.find('HEADER INFORMATION')
        summary_start = text.find('ACCOUNT SUMMARY', header_start)
        trans_start = text.find('TRANSACTIONS', summary_start if summary_start > 0 else header_start)
        
        if header_start >= 0:
            sections['header'] = text[header_start:summary_start if summary_start > 0 else trans_start]
        if summary_start >= 0:
            sections['summary'] = text[summary_start:trans_start if trans_start > 0 else len(text)]
        if trans_start >= 0:
            sections['transactions'] = text[trans_start:]
    else:
        # Try to parse without clear sections
        sections['header'] = text
        sections['summary'] = text
        sections['transactions'] = text
    
    # Extract structured data
    result = {}
    
    # Extract header info
    result.update(extract_header_info(sections['header']))
    
    # Extract summary info
    result.update(extract_summary_info(sections['summary']))
    
    # Parse transactions
    transactions = parse_transactions_table(sections['transactions'])
    
    # Validate transactions
    opening_balance = result.get('opening_balance')
    transactions, errors = validate_transactions(transactions, opening_balance)
    
    result['transactions'] = transactions
    result['transaction_count'] = len(transactions)
    
    if errors:
        result['validation_errors'] = errors
    
    return result


def export_to_json(data, output_path):
    """
    Export structured data to JSON file

    Args:
        data: Structured data dictionary
        output_path: Path to save JSON file
    """
    with Path(output_path).open('w') as f:
        json.dump(data, f, indent=2)


def export_to_csv(transactions, output_path):
    """
    Export transactions to CSV file

    Args:
        transactions: List of transaction dictionaries
        output_path: Path to save CSV file
    """
    if not transactions:
        return

    fieldnames = ['date', 'description', 'debit', 'credit', 'balance']

    with Path(output_path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(transactions)


def generate_summary_report(data):
    """
    Generate a text summary report
    
    Args:
        data: Structured data dictionary
        
    Returns:
        str: Formatted summary report
    """
    report = []
    report.append("=" * 60)
    report.append("BANK STATEMENT EXTRACTION SUMMARY")
    report.append("=" * 60)
    report.append("")
    
    # Header info
    if 'bank' in data:
        report.append(f"Bank: {data['bank']}")
    if 'account_holder' in data:
        report.append(f"Account Holder: {data['account_holder']}")
    if 'bsb' in data:
        report.append(f"BSB: {data['bsb']}")
    if 'account_number' in data:
        report.append(f"Account Number: {data['account_number']}")
    if 'period_from' in data and 'period_to' in data:
        report.append(f"Statement Period: {data['period_from']} to {data['period_to']}")
    
    report.append("")
    report.append("-" * 60)
    
    # Summary info
    if 'opening_balance' in data:
        report.append(f"Opening Balance: ${data['opening_balance']:.2f}")
    if 'closing_balance' in data:
        report.append(f"Closing Balance: ${data['closing_balance']:.2f}")
    if 'total_credits' in data:
        report.append(f"Total Credits: ${data['total_credits']:.2f}")
    if 'total_debits' in data:
        report.append(f"Total Debits: ${data['total_debits']:.2f}")
    
    report.append("")
    report.append("-" * 60)
    
    # Transaction count
    if 'transaction_count' in data:
        report.append(f"Total Transactions: {data['transaction_count']}")
    
    # Validation errors
    if 'validation_errors' in data and data['validation_errors']:
        report.append("")
        report.append("VALIDATION WARNINGS:")
        for error in data['validation_errors']:
            report.append(f"  - {error}")
    
    report.append("")
    report.append("=" * 60)
    
    return "\n".join(report)
