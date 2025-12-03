#!/usr/bin/env python3
"""
Generate mathematically correct synthetic bank statement images.

Creates bank statements where:
1. Transactions are in strict chronological or reverse chronological order
2. Each balance = previous balance - debit + credit (mathematically consistent)
3. Realistic Australian transaction descriptions
"""

import random
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from rich import print as rprint

# Australian transaction description templates
DEBIT_DESCRIPTIONS = [
    "EFTPOS Purchase COLES SUPERMARKET {location}",
    "EFTPOS Purchase WOOLWORTHS {location}",
    "ATM Withdrawal {bank} ATM {location}",
    "Direct Debit {company} PTY LTD",
    "BPAY Payment BILLER{code} CRN {ref}",
    "Transfer To {name} NetBank {ref}",
    "Card Purchase {merchant} {location}",
    "Contactless Payment {merchant} {location}",
    "Auto Payment UTILITIES {utility} {ref}",
    "Subscription {service} {ref}",
    "Online Purchase {website} AUS",
    "EFTPOS Cash Out {merchant} {location}",
    "Mortgage Repayment MORT {ref} {bank}",
    "Credit Card Payment CC {ref}",
    "Insurance Premium {company}",
]

CREDIT_DESCRIPTIONS = [
    "Salary Payment {company} {ref}",
    "Direct Credit SALARY {ref}",
    "Transfer From {name} {ref}",
    "Fortnightly Pay {company} PAYROLL {ref}",
    "Dividend Payment {company} PTY LTD {ref}",
    "Tax Refund ATO {ref}",
    "Interest Payment {ref}",
    "Refund {merchant} {ref}",
]

LOCATIONS = ["SYDNEY NSW", "MELBOURNE VIC", "BRISBANE QLD", "PERTH WA", "ADELAIDE SA",
             "HOBART TAS", "DARWIN NT", "CANBERRA ACT", "GOLD COAST QLD", "CAIRNS QLD"]
BANKS = ["NAB", "ANZ", "CBA", "WESTPAC", "BOQ", "SUNCORP"]
COMPANIES = ["ACME CORP", "TELSTRA", "OPTUS", "AGL", "ORIGIN ENERGY", "MEDIBANK"]
MERCHANTS = ["BUNNINGS", "JB HIFI", "OFFICEWORKS", "KMART", "BIG W", "MYER", "DAVID JONES"]
UTILITIES = ["AGL", "ORIGIN", "RED ENERGY", "ENERGY AUSTRALIA", "SIMPLY ENERGY"]
SERVICES = ["Netflix", "Spotify", "Stan", "Binge", "Disney+", "Amazon Prime"]
NAMES = ["JOHN SMITH", "SARAH JONES", "MICHAEL BROWN", "EMMA WILSON", "DAVID CHEN"]


def generate_ref():
    return f"{random.randint(10000, 99999)}P{random.randint(10000000, 99999999)}"


def generate_description(is_debit: bool) -> str:
    if is_debit:
        template = random.choice(DEBIT_DESCRIPTIONS)
    else:
        template = random.choice(CREDIT_DESCRIPTIONS)

    return template.format(
        location=random.choice(LOCATIONS),
        bank=random.choice(BANKS),
        company=random.choice(COMPANIES),
        merchant=random.choice(MERCHANTS),
        utility=random.choice(UTILITIES),
        service=random.choice(SERVICES),
        name=random.choice(NAMES),
        code=random.randint(10000, 99999),
        ref=generate_ref(),
        website=random.choice(["amazon.com.au", "ebay.com.au", "gumtree.com.au"]),
    )


def generate_transactions(
    num_transactions: int = 30,
    start_date: datetime = None,
    opening_balance: Decimal = None,
    reverse_chronological: bool = False,
) -> list[dict]:
    """Generate mathematically consistent transactions."""

    if start_date is None:
        start_date = datetime(2025, 8, 1)

    if opening_balance is None:
        opening_balance = Decimal(str(random.randint(5000, 50000)))

    transactions = []
    current_balance = opening_balance
    current_date = start_date

    for _ in range(num_transactions):
        # Randomly decide if debit or credit (70% debits, 30% credits)
        is_debit = random.random() < 0.7

        # Generate amount
        if is_debit:
            # Debits: $10 to $2000
            amount = Decimal(str(random.uniform(10, 2000))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            debit = amount
            credit = Decimal("0")
            current_balance = current_balance - amount
        else:
            # Credits: $100 to $5000
            amount = Decimal(str(random.uniform(100, 5000))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            debit = Decimal("0")
            credit = amount
            current_balance = current_balance + amount

        # Generate description
        description = generate_description(is_debit)

        # Advance date by 0-3 days
        current_date = current_date + timedelta(days=random.randint(0, 3))

        transactions.append({
            "date": current_date.strftime("%d/%m/%Y"),
            "description": description[:50],  # Truncate long descriptions
            "debit": f"${debit:.2f}" if debit > 0 else "",
            "credit": f"${credit:.2f}" if credit > 0 else "",
            "balance": f"${current_balance:.2f}",
        })

    if reverse_chronological:
        transactions = list(reversed(transactions))

    return transactions, opening_balance


def create_bank_statement_image(
    transactions: list[dict],
    output_path: Path,
    account_holder: str = "JOHN SMITH",
    bank_name: str = "Commonwealth Bank",
    reverse_chronological: bool = False,
):
    """Create a bank statement image from transactions."""

    # Image dimensions
    width = 1200
    row_height = 30
    header_height = 150
    height = header_height + (len(transactions) + 2) * row_height + 50

    # Create image
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Try to use a monospace font, fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14)
        font_bold = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 16)
        font_header = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 20)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_bold = font
        font_header = font

    # Draw header
    y = 20
    draw.text((50, y), bank_name.upper(), fill="navy", font=font_header)
    y += 35
    draw.text((50, y), "TRANSACTION STATEMENT", fill="black", font=font_bold)
    y += 30
    draw.text((50, y), f"Account Holder: {account_holder}", fill="black", font=font)
    y += 25

    # Date range
    if transactions:
        if reverse_chronological:
            date_range = f"{transactions[-1]['date']} - {transactions[0]['date']}"
        else:
            date_range = f"{transactions[0]['date']} - {transactions[-1]['date']}"
        draw.text((50, y), f"Statement Period: {date_range}", fill="black", font=font)

    y = header_height

    # Draw table header
    draw.rectangle([(40, y), (width - 40, y + row_height)], fill="lightgray")
    columns = [("Date", 50), ("Description", 170), ("Withdrawal", 650), ("Deposit", 800), ("Balance", 950)]
    for col_name, x in columns:
        draw.text((x, y + 5), col_name, fill="black", font=font_bold)

    y += row_height

    # Draw separator line
    draw.line([(40, y), (width - 40, y)], fill="gray", width=1)

    # Draw transactions
    for i, txn in enumerate(transactions):
        y += row_height

        # Alternate row background
        if i % 2 == 0:
            draw.rectangle([(40, y - row_height + 5), (width - 40, y + 5)], fill="#f8f8f8")

        draw.text((50, y - row_height + 8), txn["date"], fill="black", font=font)
        draw.text((170, y - row_height + 8), txn["description"], fill="black", font=font)

        # Right-align amounts
        if txn["debit"]:
            debit_width = draw.textlength(txn["debit"], font=font)
            draw.text((750 - debit_width, y - row_height + 8), txn["debit"], fill="red", font=font)

        if txn["credit"]:
            credit_width = draw.textlength(txn["credit"], font=font)
            draw.text((900 - credit_width, y - row_height + 8), txn["credit"], fill="green", font=font)

        balance_width = draw.textlength(txn["balance"], font=font)
        draw.text((1100 - balance_width, y - row_height + 8), txn["balance"], fill="black", font=font)

    # Save image
    img.save(output_path, "PNG")
    return output_path


def generate_ground_truth_row(
    image_file: str,
    transactions: list[dict],
    account_holder: str,
    bank_name: str,
    reverse_chronological: bool = False,
) -> dict:
    """Generate ground truth CSV row for the bank statement."""

    # Extract debits only
    debit_txns = [t for t in transactions if t["debit"]]

    # Sort by date for ground truth (always chronological)
    if reverse_chronological:
        date_ordered = list(reversed(transactions))
    else:
        date_ordered = transactions

    # Date range (always oldest - newest)
    date_range = f"{date_ordered[0]['date']} - {date_ordered[-1]['date']}"

    # Extract debit amounts and dates (in document order for matching)
    debit_amounts = " | ".join(t["debit"] for t in debit_txns)
    debit_descriptions = " | ".join(t["description"] for t in debit_txns)
    debit_dates = " | ".join(t["date"] for t in debit_txns)

    return {
        "image_file": image_file,
        "DOCUMENT_TYPE": "BANK_STATEMENT",
        "BUSINESS_ABN": "NOT_FOUND",
        "BUSINESS_ADDRESS": "NOT_FOUND",
        "GST_AMOUNT": "NOT_FOUND",
        "INVOICE_DATE": "NOT_FOUND",
        "IS_GST_INCLUDED": "NOT_FOUND",
        "LINE_ITEM_DESCRIPTIONS": debit_descriptions,
        "LINE_ITEM_QUANTITIES": "NOT_FOUND",
        "LINE_ITEM_PRICES": "NOT_FOUND",
        "LINE_ITEM_TOTAL_PRICES": "NOT_FOUND",
        "PAYER_ADDRESS": "NOT_FOUND",
        "PAYER_NAME": account_holder,
        "STATEMENT_DATE_RANGE": date_range,
        "SUPPLIER_NAME": bank_name,
        "TOTAL_AMOUNT": "NOT_FOUND",
        "TRANSACTION_AMOUNTS_PAID": debit_amounts,
        "TRANSACTION_DATES": debit_dates,
        "TRANSACTION_AMOUNTS_RECEIVED": "NOT_FOUND",
        "ACCOUNT_BALANCE": "NOT_FOUND",
    }


def main():
    output_dir = Path("/Users/tod/Desktop/LMM_POC/evaluation_data/bank")

    # Generate a mathematically consistent bank statement (chronological order)
    rprint("[blue]Generating chronological bank statement...[/blue]")
    transactions_chrono, opening = generate_transactions(
        num_transactions=35,
        start_date=datetime(2025, 8, 6),
        opening_balance=Decimal("6884.90"),
        reverse_chronological=False,
    )

    output_path = output_dir / "synthetic_chrono.png"
    create_bank_statement_image(
        transactions_chrono,
        output_path,
        account_holder="CHRISTOPHER PAUL WHITE",
        bank_name="Commonwealth Bank",
        reverse_chronological=False,
    )
    rprint(f"[green]Created: {output_path}[/green]")

    # Generate ground truth
    gt = generate_ground_truth_row(
        "synthetic_chrono.png",
        transactions_chrono,
        "CHRISTOPHER PAUL WHITE",
        "Commonwealth Bank",
        reverse_chronological=False,
    )

    rprint("\n[yellow]Ground Truth Row:[/yellow]")
    for key, value in gt.items():
        if value and value != "NOT_FOUND":
            rprint(f"  {key}: {value[:80]}...")

    # Generate reverse chronological version
    rprint("\n[blue]Generating reverse chronological bank statement...[/blue]")
    transactions_rev, _ = generate_transactions(
        num_transactions=35,
        start_date=datetime(2025, 8, 6),
        opening_balance=Decimal("6884.90"),
        reverse_chronological=True,
    )

    output_path_rev = output_dir / "synthetic_reverse_chrono.png"
    create_bank_statement_image(
        transactions_rev,
        output_path_rev,
        account_holder="CHRISTOPHER PAUL WHITE",
        bank_name="Commonwealth Bank",
        reverse_chronological=True,
    )
    rprint(f"[green]Created: {output_path_rev}[/green]")

    rprint("\n[green]Done! Generated mathematically consistent bank statements.[/green]")
    rprint("[yellow]Note: Add ground truth rows to evaluation_data/bank/ground_truth_bank.csv[/yellow]")


if __name__ == "__main__":
    main()
