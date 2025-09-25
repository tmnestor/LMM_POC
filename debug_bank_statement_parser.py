#!/usr/bin/env python3
"""
Debug script to test bank statement parsing issue.

This script reproduces the parsing issue where bank statement fields
from the model response are not being correctly extracted.
"""

from common.extraction_parser import parse_extraction_response

# Bank statement fields that should be expected
BANK_STATEMENT_FIELDS = [
    "DOCUMENT_TYPE",
    "STATEMENT_DATE_RANGE",
    "LINE_ITEM_DESCRIPTIONS",
    "TRANSACTION_DATES",
    "TRANSACTION_AMOUNTS_PAID",
    "TRANSACTION_AMOUNTS_RECEIVED",
    "ACCOUNT_BALANCE"
]

# Sample model response from the error output (trimmed for testing)
SAMPLE_RESPONSE = """DOCUMENT_TYPE: STATEMENT

STATEMENT_DATE_RANGE: 03/05/2025 - 10/05/2025

LINE_ITEM_DESCRIPTIONS: ONLINE PURCHASE AMAZON AU | EFTPOS PURCHASE COLES EXP | EFTPOS PURCHASE COLES EXP | DIRECT
CREDIT SALARY | ATM WITHDRAWAL ANZ ATM | EFTPOS PURCHASE COLES EXP | INTEREST PAYMENT | ATM WITHDRAWAL ANZ ATM

TRANSACTION_DATES: 03/05/2025 | 04/05/2025 | 05/05/2025 | 06/05/2025 | 07/05/2025 | 08/05/2025 | 09/05/2025 |
10/05/2025

TRANSACTION_AMOUNTS_PAID: $288.03 | $22.50 | $114.66 | $187.59 | $112.50 | $146.72 | $5.16 | $50.00

TRANSACTION_AMOUNTS_RECEIVED: NOT_FOUND | $3497.47 | NOT_FOUND | NOT_FOUND | NOT_FOUND | NOT_FOUND | NOT_FOUND |
NOT_FOUND

ACCOUNT_BALANCE: $13387.44 | $13344.94 | $13230.27 | $16727.74 | $16540.15 | $16427.65 | $16432.81 | $16286.08"""

def test_bank_statement_parsing():
    """Test bank statement parsing with debug output."""

    print("🧪 Testing Bank Statement Parser")
    print("=" * 60)
    print(f"Expected fields: {BANK_STATEMENT_FIELDS}")
    print(f"Response length: {len(SAMPLE_RESPONSE)} chars")
    print()

    # Test parsing
    result = parse_extraction_response(
        SAMPLE_RESPONSE,
        expected_fields=BANK_STATEMENT_FIELDS
    )

    print("\n📊 Parsing Results:")
    print("=" * 60)

    found_fields = []
    missing_fields = []

    for field in BANK_STATEMENT_FIELDS:
        value = result.get(field, "NOT_FOUND")
        if value != "NOT_FOUND":
            found_fields.append(field)
            print(f"✅ {field}: {value[:50]}{'...' if len(value) > 50 else ''}")
        else:
            missing_fields.append(field)
            print(f"❌ {field}: NOT_FOUND")

    print(f"\n📈 Summary:")
    print(f"Found: {len(found_fields)}/{len(BANK_STATEMENT_FIELDS)} fields ({len(found_fields)/len(BANK_STATEMENT_FIELDS)*100:.1f}%)")
    print(f"Missing: {missing_fields}")

    return result, found_fields, missing_fields

if __name__ == "__main__":
    test_bank_statement_parsing()