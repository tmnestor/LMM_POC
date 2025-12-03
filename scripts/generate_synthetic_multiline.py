#!/usr/bin/env python3
"""
Generate synthetic_multiline.png with:
1. Multiline transaction descriptions (2 lines per transaction)
2. Higher starting balance to ensure all balances remain positive
3. Mathematically consistent balances
"""

import random
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Transaction templates with second line for location/reference
DEBIT_TEMPLATES = [
    ("Card Purchase KMART", "{location}"),
    ("ATM Withdrawal WESTPAC ATM", "{location}"),
    ("Insurance Premium AGL", "Policy {ref}"),
    ("Auto Payment UTILITIES SIMPLY ENERGY", "Ref: {ref}"),
    ("Mortgage Repayment MORT", "{ref} {bank}"),
    ("EFTPOS Purchase COLES SUPERMARKET", "{location}"),
    ("Online Purchase gumtree.com.au", "AUS"),
    ("Contactless Payment DAVID JONES", "{location}"),
    ("EFTPOS Cash Out OFFICEWORKS", "{location}"),
    ("Direct Debit ACME CORP PTY LTD", "Ref: {ref}"),
    ("Contactless Payment KMART", "{location}"),
    ("EFTPOS Purchase WOOLWORTHS", "{location}"),
    ("Transfer To DAVID CHEN", "NetBank {ref}"),
    ("Direct Debit TELSTRA PTY LTD", "Ref: {ref}"),
    ("EFTPOS Cash Out JB HIFI", "{location}"),
    ("EFTPOS Cash Out KMART", "{location}"),
    ("Credit Card Payment CC", "Ref: {ref}"),
    ("Contactless Payment BUNNINGS", "{location}"),
    ("EFTPOS Cash Out DAVID JONES", "{location}"),
    ("Subscription Spotify", "Monthly {ref}"),
]

CREDIT_TEMPLATES = [
    ("Salary Payment AGL", "Ref: {ref}"),
    ("Fortnightly Pay AGL", "PAYROLL {ref}"),
    ("Interest Payment", "Ref: {ref}"),
    ("Direct Credit SALARY", "Ref: {ref}"),
    ("Refund BIG W", "Ref: {ref}"),
]

LOCATIONS = ["SYDNEY NSW", "MELBOURNE VIC", "BRISBANE QLD", "PERTH WA", "ADELAIDE SA",
             "HOBART TAS", "DARWIN NT", "CANBERRA ACT", "GOLD COAST QLD"]
BANKS = ["SUNCORP", "WESTPAC", "BOQ", "ANZ", "NAB"]


def generate_ref():
    return f"{random.randint(10000, 99999)}P{random.randint(10000000, 99999999)}"


def generate_transactions(
    num_transactions: int = 35,
    start_date: datetime = None,
    opening_balance: Decimal = None,
) -> list[dict]:
    """Generate mathematically consistent transactions with multiline descriptions."""

    if start_date is None:
        start_date = datetime(2025, 8, 8)

    if opening_balance is None:
        opening_balance = Decimal("6884.90")

    transactions = []
    current_balance = opening_balance
    current_date = start_date

    random.seed(42)  # For reproducibility

    for _ in range(num_transactions):
        # Control debit/credit ratio for a good mix (~60% debits, 40% credits)
        # Adjust based on balance to keep it positive
        if current_balance < Decimal("3000"):
            is_debit = random.random() < 0.3  # 30% debits when low
        elif current_balance < Decimal("6000"):
            is_debit = random.random() < 0.5  # 50% debits when moderate
        else:
            is_debit = random.random() < 0.65  # 65% debits when high

        if is_debit:
            template = random.choice(DEBIT_TEMPLATES)
            # Limit debit to not exceed balance minus buffer
            max_debit = float(current_balance - Decimal("100"))
            max_debit = max(50, min(max_debit, 2000))
            amount = Decimal(str(random.uniform(20, max_debit))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            debit = amount
            credit = Decimal("0")
            current_balance = current_balance - amount
        else:
            template = random.choice(CREDIT_TEMPLATES)
            amount = Decimal(str(random.uniform(500, 5000))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            debit = Decimal("0")
            credit = amount
            current_balance = current_balance + amount

        # Generate description with second line
        line1 = template[0]
        line2 = template[1].format(
            location=random.choice(LOCATIONS),
            bank=random.choice(BANKS),
            ref=generate_ref(),
        )

        # Advance date by 0-3 days
        current_date = current_date + timedelta(days=random.randint(0, 3))

        transactions.append({
            "date": current_date.strftime("%d/%m/%Y"),
            "description_line1": line1,
            "description_line2": line2,
            "description_full": f"{line1} {line2}",
            "debit": f"${debit:.2f}" if debit > 0 else "",
            "credit": f"${credit:.2f}" if credit > 0 else "",
            "balance": f"${current_balance:.2f}",
        })

    return transactions, opening_balance


def create_multiline_bank_statement(
    transactions: list[dict],
    output_path: Path,
    account_holder: str = "CHRISTOPHER PAUL WHITE",
    bank_name: str = "COMMONWEALTH BANK",
):
    """Create a bank statement image with multiline transaction descriptions."""

    # Image dimensions - taller rows for 2-line descriptions
    width = 800
    row_height = 45  # Taller for 2 lines
    header_height = 100
    height = header_height + (len(transactions) + 2) * row_height + 30

    # Create image
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Load fonts
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 12)
        font_small = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 10)
        font_bold = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14)
        font_header = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font
        font_bold = font
        font_header = font

    # Draw header
    y = 15
    draw.text((30, y), bank_name, fill="navy", font=font_header)
    y += 22
    draw.text((30, y), "TRANSACTION STATEMENT", fill="black", font=font_bold)
    y += 20
    draw.text((30, y), f"Account Holder: {account_holder}", fill="gray", font=font_small)
    y += 16

    # Date range
    date_range = f"{transactions[0]['date']} - {transactions[-1]['date']}"
    draw.text((30, y), f"Statement Period: {date_range}", fill="gray", font=font_small)

    y = header_height

    # Draw table header
    draw.rectangle([(20, y), (width - 20, y + 25)], fill="#e0e0e0")
    columns = [
        ("Date", 30),
        ("Transaction", 120),
        ("Withdrawal", 460),
        ("Deposit", 560),
        ("Balance", 670),
    ]
    for col_name, x in columns:
        draw.text((x, y + 5), col_name, fill="black", font=font_bold)

    y += 25

    # Draw transactions
    for i, txn in enumerate(transactions):
        # Alternate row background
        if i % 2 == 0:
            draw.rectangle([(20, y), (width - 20, y + row_height)], fill="#f5f5f5")

        # Date
        draw.text((30, y + 8), txn["date"], fill="black", font=font)

        # Description - 2 lines
        draw.text((120, y + 4), txn["description_line1"], fill="black", font=font)
        draw.text((120, y + 20), txn["description_line2"], fill="gray", font=font_small)

        # Withdrawal (red)
        if txn["debit"]:
            debit_width = draw.textlength(txn["debit"], font=font)
            draw.text((540 - debit_width, y + 12), txn["debit"], fill="red", font=font)

        # Deposit (green)
        if txn["credit"]:
            credit_width = draw.textlength(txn["credit"], font=font)
            draw.text((630 - credit_width, y + 12), txn["credit"], fill="green", font=font)

        # Balance
        balance_width = draw.textlength(txn["balance"], font=font)
        # Color based on positive/negative
        balance_val = float(txn["balance"].replace("$", "").replace(",", ""))
        balance_color = "gray" if balance_val >= 0 else "red"
        draw.text((760 - balance_width, y + 12), txn["balance"], fill=balance_color, font=font)

        y += row_height

    # Save image
    img.save(output_path, "PNG")
    print(f"Created: {output_path}")
    return transactions


def generate_ground_truth(transactions: list[dict], image_file: str) -> dict:
    """Generate ground truth row for the bank statement."""

    # Extract debits only (in document order)
    debit_txns = [t for t in transactions if t["debit"]]

    # Date range (oldest - newest, chronological)
    date_range = f"{transactions[0]['date']} - {transactions[-1]['date']}"

    # Extract fields from debit transactions
    debit_amounts = " | ".join(t["debit"] for t in debit_txns)
    debit_descriptions = " | ".join(t["description_full"] for t in debit_txns)
    debit_dates = " | ".join(t["date"] for t in debit_txns)

    return {
        "image_file": image_file,
        "DOCUMENT_TYPE": "BANK_STATEMENT",
        "STATEMENT_DATE_RANGE": date_range,
        "LINE_ITEM_DESCRIPTIONS": debit_descriptions,
        "TRANSACTION_AMOUNTS_PAID": debit_amounts,
        "TRANSACTION_DATES": debit_dates,
        "PAYER_NAME": "CHRISTOPHER PAUL WHITE",
        "SUPPLIER_NAME": "Commonwealth Bank",
    }


def main():
    output_dir = Path("/Users/tod/Desktop/LMM_POC/evaluation_data/bank")

    print("Generating synthetic_multiline.png with 15 transactions...")

    # Generate 15 transactions with mix of debits/credits, high starting balance to stay positive
    transactions, opening = generate_transactions(
        num_transactions=15,
        start_date=datetime(2025, 8, 8),
        opening_balance=Decimal("10000.00"),  # Good starting balance for 15 txns
    )

    # Verify all balances are positive
    for txn in transactions:
        balance = float(txn["balance"].replace("$", "").replace(",", ""))
        if balance < 0:
            print(f"WARNING: Negative balance found: {txn['balance']}")

    output_path = output_dir / "synthetic_multiline.png"
    create_multiline_bank_statement(transactions, output_path)

    # Generate ground truth
    gt = generate_ground_truth(transactions, "synthetic_multiline.png")

    print("\n" + "=" * 60)
    print("GROUND TRUTH (copy to CSV):")
    print("=" * 60)

    # Print as CSV-compatible format
    print(f"\nSTATEMENT_DATE_RANGE: {gt['STATEMENT_DATE_RANGE']}")
    print(f"\nTRANSACTION_DATES ({len(gt['TRANSACTION_DATES'].split(' | '))} items):")
    print(gt["TRANSACTION_DATES"])
    print(f"\nLINE_ITEM_DESCRIPTIONS ({len(gt['LINE_ITEM_DESCRIPTIONS'].split(' | '))} items):")
    print(gt["LINE_ITEM_DESCRIPTIONS"])
    print(f"\nTRANSACTION_AMOUNTS_PAID ({len(gt['TRANSACTION_AMOUNTS_PAID'].split(' | '))} items):")
    print(gt["TRANSACTION_AMOUNTS_PAID"])

    # Show balance summary
    print("\n" + "=" * 60)
    print("BALANCE SUMMARY:")
    print("=" * 60)
    print(f"Opening balance: ${opening}")
    print(f"First balance: {transactions[0]['balance']}")
    print(f"Last balance: {transactions[-1]['balance']}")

    # Check min balance
    balances = [float(t["balance"].replace("$", "").replace(",", "")) for t in transactions]
    print(f"Minimum balance: ${min(balances):.2f}")
    print(f"Maximum balance: ${max(balances):.2f}")

    return gt


if __name__ == "__main__":
    gt = main()
