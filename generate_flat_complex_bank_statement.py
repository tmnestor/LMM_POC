#!/usr/bin/env python3
"""
Flat Complex Bank Statement Generator

Creates bank statements with:
- FLAT table structure (like commbank_statement_basic.png)
- REVERSE chronological order (newest first)
- LARGE number of complex transactions (20-50 transactions)
- EACH transaction on its own unique date
- COMPLEX transaction descriptions with references and codes
"""

import random
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Dict, List

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ PIL not available - only data generation supported")


class FlatComplexBankStatementGenerator:
    def __init__(self):
        """Initialize with complex Australian banking transaction patterns."""

        # Complex Australian transaction types with realistic patterns
        self.complex_transaction_types = {
            "salary_payments": [
                "Salary ATO PAYROLL {ref}",
                "PAY RUN {company} {ref}",
                "Salary Payment ATO {ref}",
                "DIRECT CREDIT SALARY {ref}",
                "Fortnightly Pay ATO PAYROLL {ref}",
            ],
            "complex_transfers": [
                "Transfer To Western Port Marina NetBank From Tod",
                "Internet Transfer to xx{account} BSB {bsb}",
                "Transfer To Vicks Account NetBank {ref}",
                "OSKO Payment to {name} {ref}",
                "PayID Transfer {email} {ref}",
                "NPP Payment {phone} {ref}",
            ],
            "loan_repayments": [
                "Home Loan Payment LN REPAY {ref}",
                "Personal Loan PLOAN {ref} MHF {id}",
                "Car Loan Payment AUTO {ref}",
                "Credit Card Payment CC {ref}",
                "Mortgage Repayment MORT {ref} {bank}",
            ],
            "direct_debits": [
                "Direct Debit {ref} MHF {id}",
                "DD INSURANCE {company} {ref}",
                "Auto Payment UTILITIES {provider} {ref}",
                "BPAY Payment {biller} CRN {crn}",
                "Subscription {service} {ref}",
                "Gym Membership {gym} DD {ref}",
            ],
            "atm_transactions": [
                "Wdl ATM WBC WESTPAC {location}",
                "ATM Withdrawal ANZ {suburb} NSW",
                "Cash Withdrawal CBA {location}",
                "EFTPOS Cash Out {merchant} {location}",
                "International ATM {country} USD",
            ],
            "eftpos_purchases": [
                "EFTPOS Purchase {merchant} {location}",
                "Card Purchase {store} {suburb} {state}",
                "Contactless Payment {venue} {location}",
                "Online Purchase {website} AUS",
                "Subscription {service} Monthly {ref}",
            ],
            "government_payments": [
                "Tax Refund ATO {ref}",
                "Centrelink Payment {benefit} {ref}",
                "Medicare Benefit {ref}",
                "GST Refund ATO {ref}",
                "Family Tax Benefit {ref}",
            ],
            "business_transactions": [
                "Invoice Payment {business} {ref}",
                "Professional Services {provider} {ref}",
                "Contractor Payment ABN {abn} {ref}",
                "Business Expense {category} {ref}",
                "Equipment Purchase {supplier} {ref}",
            ],
            "investment_transactions": [
                "Share Purchase {broker} {ref}",
                "Dividend Payment {company} {ref}",
                "Super Contribution {fund} {ref}",
                "Investment Transfer {platform} {ref}",
                "Interest Payment TERM DEP {ref}",
            ],
            "fees_and_charges": [
                "Account Fee Monthly Service",
                "Overdraft Interest Charge",
                "International Transaction Fee",
                "ATM Fee Third Party",
                "Paper Statement Fee",
                "Dishonour Fee NSF",
            ],
        }

        # Australian locations for realistic transactions
        self.australian_locations = [
            "SYDNEY NSW",
            "MELBOURNE VIC",
            "BRISBANE QLD",
            "PERTH WA",
            "ADELAIDE SA",
            "GOLD COAST QLD",
            "NEWCASTLE NSW",
            "CANBERRA ACT",
            "SUNSHINE COAST QLD",
            "WOLLONGONG NSW",
            "HOBART TAS",
            "GEELONG VIC",
            "TOWNSVILLE QLD",
            "CAIRNS QLD",
            "DARWIN NT",
            "BALLARAT VIC",
            "BENDIGO VIC",
            "LAUNCESTON TAS",
            "MACKAY QLD",
            "ROCKHAMPTON QLD",
        ]

        # Complex business names and merchants
        self.complex_merchants = [
            "WOOLWORTHS METRO",
            "COLES EXPRESS",
            "IGA XPRESS",
            "BUNNINGS WAREHOUSE",
            "HARVEY NORMAN FLAGSHIP",
            "JB HI-FI HOME",
            "OFFICEWORKS BUSINESS",
            "CHEMIST WAREHOUSE DISCOUNT",
            "PRICELINE PHARMACY",
            "TERRY WHITE",
            "MCDONALD'S FAMILY RESTAURANT",
            "SUBWAY FRESH FIT",
            "DOMINO'S PIZZA",
            "RED ROOSTER",
            "HUNGRY JACK'S",
            "PIZZA HUT DELIVERY",
            "WESTFIELD SHOPPING CENTRE",
            "CENTRO SHOPPING",
            "EASTLAND SHOPPING",
            "PACIFIC FAIR",
            "GARDEN CITY",
            "MARION SHOPPING CENTRE",
        ]

        # Reference number generators for complex transactions
        self.reference_generators = {
            "payroll": lambda: f"{random.randint(10000, 99999)}P{random.randint(10000000, 99999999)}",
            "loan": lambda: f"LN{random.randint(100000000, 999999999)}",
            "account": lambda: f"{random.randint(1000, 9999)}",
            "bsb": lambda: f"{random.randint(10, 99)}{random.randint(1, 8)}-{random.randint(100, 999)}",
            "mhf": lambda: f"MHF {random.randint(10000, 99999)}",
            "crn": lambda: f"{random.randint(1000000000, 9999999999)}",
            "abn": lambda: f"{random.randint(10, 99)} {random.randint(100, 999)} {random.randint(100, 999)} {random.randint(100, 999)}",
            "ato": lambda: f"ATO{random.randint(100000000, 999999999)}",
        }

    def generate_complex_transaction_description(
        self, transaction_category: str
    ) -> str:
        """Generate a complex, realistic transaction description."""
        patterns = self.complex_transaction_types[transaction_category]
        pattern = random.choice(patterns)

        # Fill in placeholders with realistic data
        return pattern.format(
            ref=self.reference_generators["payroll"](),
            account=self.reference_generators["account"](),
            bsb=self.reference_generators["bsb"](),
            id=random.randint(10000, 99999),
            location=random.choice(self.australian_locations),
            merchant=random.choice(self.complex_merchants),
            company=f"{random.choice(['ACME', 'GLOBAL', 'PREMIUM', 'ELITE', 'SUPERIOR'])} CORP PTY LTD",
            name=random.choice(
                ["JOHN SMITH", "SARAH JONES", "MIKE CHEN", "LISA BROWN"]
            ),
            email=f"user{random.randint(100, 999)}@{random.choice(['gmail.com', 'outlook.com', 'yahoo.com'])}",
            phone=f"04{random.randint(10000000, 99999999)}",
            provider=random.choice(
                ["Origin Energy", "AGL", "Energy Australia", "Red Energy"]
            ),
            service=random.choice(["Netflix", "Spotify", "Amazon Prime", "Disney+"]),
            gym=random.choice(["Anytime Fitness", "Snap Fitness", "Goodlife Health"]),
            biller=f"BILLER{random.randint(100000, 999999)}",
            crn=self.reference_generators["crn"](),
            abn=self.reference_generators["abn"](),
            broker=random.choice(["CommSec", "Westpac Share Trading", "NAB Trade"]),
            fund=random.choice(["Australian Super", "REST Super", "HESTA"]),
            platform=random.choice(["Vanguard", "BetaShares", "iShares"]),
            business=f"{random.choice(self.complex_merchants[:10])} PTY LTD",
            category=random.choice(
                ["Office Supplies", "IT Equipment", "Professional Dev"]
            ),
            supplier=f"{random.choice(['TechCorp', 'OfficeMax', 'ProSupply'])} Australia",
            benefit=random.choice(["JobSeeker", "Family Tax Benefit", "Carer Payment"]),
            bank=random.choice(["CBA", "ANZ", "NAB", "WBC"]),
            suburb=random.choice(["Parramatta", "Chatswood", "Richmond", "Frankston"]),
            state=random.choice(["NSW", "VIC", "QLD", "WA", "SA"]),
            store=random.choice(self.complex_merchants),
            venue=random.choice(["Restaurant", "Cafe", "Bar", "Cinema"]),
            website=random.choice(
                ["amazon.com.au", "ebay.com.au", "kogan.com", "catch.com.au"]
            ),
            country=random.choice(["USA", "UK", "NZ", "SG"]),
        )

    def generate_unique_dates(
        self, num_transactions: int, days_span: int = 30
    ) -> List[datetime]:
        """Generate unique dates for each transaction in reverse chronological order."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_span)

        # Generate random dates within the span
        dates = []
        for _ in range(num_transactions):
            random_days = random.randint(0, days_span)
            transaction_date = start_date + timedelta(days=random_days)
            # Add random hours to make them more unique
            transaction_date = transaction_date.replace(
                hour=random.randint(0, 23),
                minute=random.randint(0, 59),
                second=random.randint(0, 59),
            )
            dates.append(transaction_date)

        # Sort in reverse chronological order (newest first)
        dates.sort(reverse=True)

        # Ensure all dates are unique by adding small time differences
        for i in range(1, len(dates)):
            if dates[i] >= dates[i - 1]:
                dates[i] = dates[i - 1] - timedelta(minutes=random.randint(1, 30))

        return dates

    def generate_complex_amounts(self, transaction_category: str) -> Decimal:
        """Generate realistic amounts based on transaction type."""
        amount_ranges = {
            "salary_payments": (3500, 8500),  # Salary payments
            "complex_transfers": (100, 5000),  # Personal transfers
            "loan_repayments": (500, 3000),  # Loan payments
            "direct_debits": (25, 400),  # Bills and subscriptions
            "atm_transactions": (40, 400),  # ATM withdrawals
            "eftpos_purchases": (15, 300),  # Daily purchases
            "government_payments": (200, 4000),  # Tax refunds, benefits
            "business_transactions": (150, 2500),  # Professional services
            "investment_transactions": (500, 10000),  # Investments
            "fees_and_charges": (5, 35),  # Bank fees
        }

        min_amt, max_amt = amount_ranges.get(transaction_category, (50, 500))
        amount = Decimal(str(random.uniform(min_amt, max_amt))).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        return amount

    def determine_transaction_type(self, transaction_category: str) -> str:
        """Determine if transaction is debit or credit based on category."""
        credit_categories = {
            "salary_payments",
            "government_payments",
            "investment_transactions",
        }
        return "credit" if transaction_category in credit_categories else "debit"

    def generate_flat_complex_statement(
        self, account_holder: str = "CHRISTOPHER PAUL WHITE", num_transactions: int = 35
    ) -> Dict:
        """Generate a flat table format bank statement with many complex transactions."""

        # Generate unique dates for each transaction
        transaction_dates = self.generate_unique_dates(num_transactions)

        # Account details
        account_details = {
            "bank_name": "Commonwealth Bank",
            "account_holder": account_holder,
            "bsb": "062-274",
            "account_number": "457789123",
            "statement_period": f"{transaction_dates[-1].strftime('%d/%m/%Y')} to {transaction_dates[0].strftime('%d/%m/%Y')}",
        }

        # Generate opening balance
        opening_balance = Decimal(str(random.uniform(2000, 8000))).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        # Generate transactions
        transactions = []
        current_balance = opening_balance

        # Process in chronological order for balance calculation (oldest first)
        chronological_dates = sorted(transaction_dates)

        for date in chronological_dates:
            # Select random transaction category with weights
            categories = list(self.complex_transaction_types.keys())
            weights = [
                0.15,
                0.12,
                0.10,
                0.15,
                0.12,
                0.15,
                0.08,
                0.08,
                0.03,
                0.02,
            ]  # Sum = 1.0

            category = random.choices(categories, weights=weights)[0]
            description = self.generate_complex_transaction_description(category)
            amount = self.generate_complex_amounts(category)
            transaction_type = self.determine_transaction_type(category)

            # Calculate new balance
            if transaction_type == "debit":
                current_balance -= amount
                withdrawal = amount
                deposit = None
            else:
                current_balance += amount
                withdrawal = None
                deposit = amount

            # Create transaction record
            transaction = {
                "date": date,
                "description": description,
                "withdrawal": withdrawal,
                "deposit": deposit,
                "balance": current_balance.quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                ),
                "type": transaction_type,
                "category": category,
            }

            transactions.append(transaction)

        # Sort transactions in reverse chronological order for display (newest first)
        transactions.sort(key=lambda x: x["date"], reverse=True)

        return {
            **account_details,
            "opening_balance": opening_balance,
            "closing_balance": current_balance.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            ),
            "transactions": transactions,
            "transaction_count": len(transactions),
        }

    def generate_flat_png(self, statement_data: Dict, output_path: str) -> None:
        """Generate PNG with FLAT table structure (like commbank_statement_basic.png)."""

        if not PIL_AVAILABLE:
            print("⚠️ Skipping PNG generation - PIL not available")
            return

        # Calculate image height based on transaction count
        base_height = 600
        transaction_height = len(statement_data["transactions"]) * 18
        total_height = base_height + transaction_height

        # Image dimensions
        width, height = 900, total_height
        background_color = "white"
        text_color = "black"
        header_color = "#003F7F"  # Commonwealth Bank blue

        # Create image and drawing context
        img = Image.new("RGB", (width, height), background_color)
        draw = ImageDraw.Draw(img)

        # Load fonts
        try:
            title_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
            header_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
            normal_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 11)
            small_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 10)
        except (OSError, IOError):
            title_font = ImageFont.load_default()
            header_font = ImageFont.load_default()
            normal_font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        y_pos = 30

        # Bank header
        draw.text(
            (50, y_pos), statement_data["bank_name"], fill=header_color, font=title_font
        )
        y_pos += 35

        # Account information
        account_info = [
            f"Account Holder: {statement_data['account_holder']}",
            f"BSB: {statement_data['bsb']}",
            f"Account Number: {statement_data['account_number']}",
            f"Statement Period: {statement_data['statement_period']}",
        ]

        for info in account_info:
            draw.text((50, y_pos), info, fill=text_color, font=normal_font)
            y_pos += 16

        y_pos += 20

        # FLAT TABLE STRUCTURE - Single header row
        # Table headers
        draw.text((50, y_pos), "Date", fill=text_color, font=header_font)
        draw.text((130, y_pos), "Description", fill=text_color, font=header_font)
        draw.text((550, y_pos), "Withdrawal", fill=text_color, font=header_font)
        draw.text((650, y_pos), "Deposit", fill=text_color, font=header_font)
        draw.text((750, y_pos), "Balance", fill=text_color, font=header_font)
        y_pos += 18

        # Header underline
        draw.line([(50, y_pos), (850, y_pos)], fill=text_color, width=1)
        y_pos += 10

        # Transaction rows - FLAT format (one row per transaction)
        for transaction in statement_data["transactions"]:
            date_str = transaction["date"].strftime("%d/%m/%Y")
            description = transaction["description"]

            # Truncate long descriptions
            if len(description) > 50:
                description = description[:47] + "..."

            # Draw transaction row
            draw.text((50, y_pos), date_str, fill=text_color, font=small_font)
            draw.text((130, y_pos), description, fill=text_color, font=small_font)

            # Withdrawal column
            if transaction["withdrawal"]:
                draw.text(
                    (550, y_pos),
                    f"${transaction['withdrawal']:.2f}",
                    fill=text_color,
                    font=small_font,
                )

            # Deposit column
            if transaction["deposit"]:
                draw.text(
                    (650, y_pos),
                    f"${transaction['deposit']:.2f}",
                    fill=text_color,
                    font=small_font,
                )

            # Balance column
            draw.text(
                (750, y_pos),
                f"${transaction['balance']:.2f}",
                fill=text_color,
                font=small_font,
            )

            y_pos += 16

        # Footer line
        y_pos += 10
        draw.line([(50, y_pos), (850, y_pos)], fill=text_color, width=1)
        y_pos += 20

        # Summary information
        draw.text(
            (50, y_pos),
            f"Total Transactions: {statement_data['transaction_count']}",
            fill=text_color,
            font=normal_font,
        )
        y_pos += 16
        draw.text(
            (50, y_pos),
            f"Closing Balance: ${statement_data['closing_balance']:.2f}",
            fill=header_color,
            font=normal_font,
        )

        # Save the image
        img.save(output_path, "PNG", quality=95, dpi=(300, 300))
        print(f"📄 Generated flat complex PNG: {Path(output_path).name}")
        print(f"   📊 {statement_data['transaction_count']} complex transactions")
        print("   📅 Reverse chronological order (newest first)")
        print("   🔧 Flat table structure")


def generate_flat_complex_bank_statement(
    output_filename: str = "commbank_flat_complex.png", num_transactions: int = 35
) -> Dict:
    """Generate a single flat complex bank statement."""

    generator = FlatComplexBankStatementGenerator()

    print("🏦 Generating flat complex bank statement...")
    print(f"   📊 Transactions: {num_transactions}")
    print("   📋 Format: Flat table (like commbank_statement_basic.png)")
    print("   📅 Order: Reverse chronological (newest first)")
    print("   🔧 Complexity: High (complex descriptions with references)")

    # Generate statement data
    statement_data = generator.generate_flat_complex_statement(
        num_transactions=num_transactions
    )

    # Generate PNG
    generator.generate_flat_png(statement_data, output_filename)

    # Print summary
    print(f"\n✅ Generated: {output_filename}")
    print(f"   🏦 Account: {statement_data['account_holder']}")
    print(f"   📊 Transactions: {statement_data['transaction_count']}")
    print(f"   💰 Opening: ${statement_data['opening_balance']:.2f}")
    print(f"   💰 Closing: ${statement_data['closing_balance']:.2f}")
    print(f"   📅 Period: {statement_data['statement_period']}")

    # Show sample transactions
    print("\n📋 Sample Complex Transactions (showing first 5):")
    for i, transaction in enumerate(statement_data["transactions"][:5], 1):
        date_str = transaction["date"].strftime("%d/%m/%Y")
        desc = (
            transaction["description"][:60] + "..."
            if len(transaction["description"]) > 60
            else transaction["description"]
        )
        amount = transaction["withdrawal"] or transaction["deposit"]
        type_str = "Dr" if transaction["withdrawal"] else "Cr"
        print(f"   {i}. {date_str} | {desc} | ${amount:.2f} {type_str}")

    return statement_data


def main():
    """Generate flat complex bank statement for testing."""

    # Generate statement in evaluation_data directory
    output_path = "evaluation_data/commbank_flat_complex.png"

    # Create evaluation_data directory if it doesn't exist
    Path("evaluation_data").mkdir(exist_ok=True)

    # Generate with many complex transactions
    statement_data = generate_flat_complex_bank_statement(
        output_filename=output_path,
        num_transactions=40,  # Large number of complex transactions
    )

    print("\n🎯 FLAT COMPLEX STATEMENT FEATURES:")
    print("   ✅ Flat table structure (simple layout)")
    print("   ✅ Reverse chronological order (newest first)")
    print(f"   ✅ {statement_data['transaction_count']} complex transactions")
    print("   ✅ Each transaction on unique date")
    print("   ✅ Complex descriptions with references")
    print("   ✅ Ready for V100 testing")


if __name__ == "__main__":
    main()
