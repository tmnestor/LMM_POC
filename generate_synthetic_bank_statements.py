#!/usr/bin/env python3
"""
Synthetic Australian Bank Statement Generator

Generates realistic Australian bank statements that match Big 4 bank formats:
- Reverse chronological order (newest transactions first)
- Proper debit/credit column formatting (positive amounts only)
- Mathematically correct running balances
- Realistic Australian transaction types and amounts
- BSB and account number formats
"""

import csv
import json
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
    print("⚠️ PIL not available - only JSON/HTML generation supported")


class AustralianBankStatementGenerator:
    def __init__(self):
        """Initialize with realistic Australian banking data."""
        
        # Field schema mapping for ground truth CSV generation
        # IMPORTANT: Bank-specific fields for bank statement extraction evaluation
        self.field_schema_mapping = {
            "BANK_NAME": "bank_name",
            "BANK_BSB_NUMBER": "bsb_number", 
            "BANK_ACCOUNT_NUMBER": "account_number",
            "BANK_ACCOUNT_HOLDER": "account_holder",
            "ACCOUNT_OPENING_BALANCE": "opening_balance",
            "ACCOUNT_CLOSING_BALANCE": "closing_balance",
            "TRANSACTION_DATES": "transaction_dates_chronological",  # Chronological order (oldest first)
            "TRANSACTION_AMOUNTS_PAID": "debit_amounts_only",        # Debit amounts only
            "TRANSACTION_AMOUNTS_RECEIVED": "credit_amounts_only",   # Credit amounts only
            "TRANSACTION_BALANCES": "running_balances",             # Running balance after each transaction
            "TOTAL_CREDITS": "total_credits",
            "TOTAL_DEBITS": "total_debits",
            "CREDIT_CARD_DUE_DATE": None  # Not applicable to bank statements
        }
        
        # Australian Big 4 Banks
        self.banks = {
            "CommBank": {
                "name": "Commonwealth Bank of Australia",
                "bsb_prefix": "06",  # CommBank BSB codes start with 06
                "account_types": ["Smart Access", "Complete Access", "Streamline"],
                "logo_color": "#FFD100"  # CommBank yellow
            },
            "ANZ": {
                "name": "Australia and New Zealand Banking Group",
                "bsb_prefix": "01",  # ANZ BSB codes start with 01
                "account_types": ["ANZ Access", "ANZ Progress", "ANZ Online Saver"],
                "logo_color": "#003F5C"  # ANZ blue
            },
            "NAB": {
                "name": "National Australia Bank",
                "bsb_prefix": "08",  # NAB BSB codes start with 08
                "account_types": ["NAB Classic", "NAB Choice", "NAB Reward Saver"],
                "logo_color": "#E50019"  # NAB red
            },
            "Westpac": {
                "name": "Westpac Banking Corporation", 
                "bsb_prefix": "03",  # Westpac BSB codes start with 03
                "account_types": ["Choice", "Everyday", "eSaver"],
                "logo_color": "#DA020E"  # Westpac red
            }
        }
        
        # Australian account holder names
        self.account_holders = [
            "MAURICE TOD NESTOR",
            "SARAH JANE SMITH",
            "MICHAEL ANTHONY CHEN", 
            "EMMA LOUISE WILLIAMS",
            "DAVID ROBERT BROWN",
            "LISA MARIE WILSON",
            "JAMES PATRICK TAYLOR",
            "SOPHIE ANNE MARTIN",
            "ROBERT JOHN DAVIS",
            "ANNA ELIZABETH THOMPSON",
            "MARK STEVEN ANDERSON",
            "JENNIFER MARY CLARK",
            "ANDREW WILLIAM JONES",
            "REBECCA JANE MILLER",
            "CHRISTOPHER PAUL WHITE"
        ]
        
        # Australian transaction types and patterns
        self.transaction_types = {
            "salary": {
                "patterns": [
                    "Salary ATO PAYROLL {ref}",
                    "Salary - ATO PAYROLL {ref}", 
                    "PAY ATO PAYROLL {ref}",
                    "Salary Payment ATO {ref}"
                ],
                "amounts": (2000, 8000),  # Bi-weekly/monthly salaries
                "frequency": "monthly",
                "type": "credit"
            },
            "bank_transfer_out": {
                "patterns": [
                    "Transfer To Western Port Marina NetBank",
                    "Transfer to xx{account} NetBank", 
                    "Transfer To Vicks Account NetBank From Tod",
                    "Transfer to {account} NetBank",
                    "Internet Transfer to {account}"
                ],
                "amounts": (50, 5000),
                "frequency": "weekly",
                "type": "debit"
            },
            "loan_repayment": {
                "patterns": [
                    "Loan Repayment LN REPAY {ref}",
                    "Home Loan Payment LN REPAY {ref}",
                    "Personal Loan LN REPAY {ref}"
                ],
                "amounts": (25, 2500),
                "frequency": "monthly", 
                "type": "debit"
            },
            "atm_withdrawal": {
                "patterns": [
                    "Wdl ATM WBC WESTPAC GLEN WAVE",
                    "ATM Withdrawal {location}",
                    "EFTPOS Withdrawal {merchant}",
                    "Cash Withdrawal ATM {location}"
                ],
                "amounts": (20, 500),
                "frequency": "weekly",
                "type": "debit"
            },
            "direct_debit": {
                "patterns": [
                    "Direct Debit {ref} MHF {id}",
                    "DD {merchant} {ref}",
                    "Direct Debit {company}",
                    "Auto Payment {provider}"
                ],
                "amounts": (10, 300),
                "frequency": "monthly",
                "type": "debit"
            },
            "refund": {
                "patterns": [
                    "Refund Purchase Medicare Benefit",
                    "Refund {merchant}",
                    "Credit Adjustment {ref}",
                    "Return/Refund {store}"
                ],
                "amounts": (5, 500),
                "frequency": "occasional",
                "type": "credit"
            },
            "interest_fee": {
                "patterns": [
                    "Debit Interest",
                    "Account Fee",
                    "Overdraft Usage Fee",
                    "Monthly Service Fee"
                ],
                "amounts": (0.39, 25.00),
                "frequency": "monthly",
                "type": "debit"
            }
        }
        
        # Australian locations and businesses
        self.locations = [
            "SYDNEY NSW", "MELBOURNE VIC", "BRISBANE QLD", "PERTH WA",
            "ADELAIDE SA", "GOLD COAST QLD", "NEWCASTLE NSW", "CANBERRA ACT",
            "SUNSHINE COAST QLD", "WOLLONGONG NSW", "HOBART TAS", "GEELONG VIC",
            "TOWNSVILLE QLD", "CAIRNS QLD", "DARWIN NT", "BALLARAT VIC"
        ]
        
        self.merchants = [
            "WOOLWORTHS", "COLES", "IGA", "BUNNINGS", "KMART", "TARGET",
            "MYER", "HARVEY NORMAN", "JB HI-FI", "OFFICEWORKS", "CHEMIST WAREHOUSE",
            "MCDONALD'S", "SUBWAY", "DOMINO'S", "PIZZA HUT", "KFC"
        ]
        
        self.reference_generators = {
            "payroll": lambda: f"{random.randint(10000, 99999)}P{random.randint(10000000, 99999999)}",
            "loan": lambda: f"{random.randint(100000000, 999999999)}",
            "account": lambda: f"{random.randint(1000, 9999)}",
            "direct_debit": lambda: f"{random.randint(100000, 999999)} MHF {random.randint(10000, 99999)}"
        }

    def generate_bsb(self, bank_code: str) -> str:
        """Generate a valid Australian BSB number."""
        # BSB format: XXY-ZZZ where XX=bank, Y=state, ZZZ=branch
        prefix = self.banks[bank_code]["bsb_prefix"]
        state = random.randint(0, 8)  # Australian states/territories
        branch = random.randint(100, 999)
        return f"{prefix}{state}-{branch}"

    def generate_account_number(self) -> str:
        """Generate a realistic Australian account number (8-9 digits)."""
        return str(random.randint(10000000, 999999999))

    def generate_transaction(self, transaction_type: str, date: datetime) -> Dict:
        """Generate a single transaction of the specified type."""
        config = self.transaction_types[transaction_type]
        
        # Select random pattern and generate description
        pattern = random.choice(config["patterns"])
        description = pattern.format(
            ref=self.reference_generators.get("payroll", lambda: str(random.randint(10000, 99999)))(),
            account=self.reference_generators["account"](),
            location=random.choice(self.locations),
            merchant=random.choice(self.merchants),
            company=f"{random.choice(self.merchants)} PTY LTD",
            provider=f"{random.choice(['Telstra', 'Optus', 'Vodafone', 'Origin', 'AGL'])}",
            store=random.choice(self.merchants),
            id=str(random.randint(10000, 99999))
        )
        
        # Generate realistic amount
        min_amt, max_amt = config["amounts"]
        amount = Decimal(str(random.uniform(min_amt, max_amt))).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        
        return {
            "date": date,
            "description": description,
            "amount": amount,
            "type": config["type"],  # "debit" or "credit"
            "category": transaction_type
        }

    def get_chronological_transactions(self, transactions: List[Dict]) -> List[Dict]:
        """Return transactions in chronological order (oldest first) for CSV extraction."""
        return sorted(transactions, key=lambda x: x["date"])

    def get_debit_amounts_only(self, transactions: List[Dict]) -> List[str]:
        """Extract debit amounts with NOT_FOUND placeholders for credits to maintain positional alignment."""
        chronological_transactions = self.get_chronological_transactions(transactions)
        debit_amounts = []
        
        for transaction in chronological_transactions:
            if transaction["type"] == "debit":
                debit_amounts.append(f"{transaction['amount']:.2f}")
            else:
                # For credits, use NOT_FOUND placeholder to maintain alignment with transaction dates
                debit_amounts.append("NOT_FOUND")
            
        return debit_amounts

    def get_credit_amounts_only(self, transactions: List[Dict]) -> List[str]:
        """Extract credit amounts with NOT_FOUND placeholders for debits to maintain positional alignment."""
        chronological_transactions = self.get_chronological_transactions(transactions)
        credit_amounts = []
        
        for transaction in chronological_transactions:
            if transaction["type"] == "credit":
                credit_amounts.append(f"{transaction['amount']:.2f}")
            else:
                # For debits, use NOT_FOUND placeholder to maintain alignment with transaction dates
                credit_amounts.append("NOT_FOUND")
            
        return credit_amounts

    def get_running_balances(self, transactions: List[Dict]) -> List[str]:
        """Extract running balances in chronological order with CR/DR suffixes."""
        chronological_transactions = self.get_chronological_transactions(transactions)
        balances = []
        
        for transaction in chronological_transactions:
            balance = transaction["balance"]
            # Format with CR/DR suffix (no negative signs)
            balance_amount = abs(balance)
            if balance < 0:
                balance_str = f"{balance_amount:.2f} DR"
            else:
                balance_str = f"{balance_amount:.2f} CR"
            balances.append(balance_str)
        
        return balances

    def calculate_running_balance(self, transactions: List[Dict], opening_balance: Decimal) -> List[Dict]:
        """Calculate running balance for transactions in chronological order."""
        balance = opening_balance
        
        # Sort transactions chronologically (oldest first) for balance calculation
        sorted_transactions = sorted(transactions, key=lambda x: x["date"])
        
        for transaction in sorted_transactions:
            if transaction["type"] == "debit":
                balance -= transaction["amount"]
            else:  # credit
                balance += transaction["amount"]
            
            transaction["balance"] = balance.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        
        return sorted_transactions

    def generate_statement_data(self, bank: str = "CommBank", 
                               statement_date: datetime = None,
                               num_transactions: int = None) -> Dict:
        """Generate complete bank statement data."""
        
        if statement_date is None:
            statement_date = datetime.now()
        
        if num_transactions is None:
            num_transactions = random.randint(10, 30)
        
        # Generate statement period (typically 1 month)
        end_date = statement_date
        start_date = end_date - timedelta(days=30)
        
        # Basic statement details
        bank_info = self.banks[bank]
        bsb = self.generate_bsb(bank)
        account_number = self.generate_account_number()
        account_holder = random.choice(self.account_holders)
        account_type = random.choice(bank_info["account_types"])
        
        # Generate opening balance (realistic range for Australian accounts)
        opening_balance = Decimal(str(random.uniform(500, 10000))).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        
        # Generate transactions across the statement period
        transactions = []
        
        # Ensure we have some salary/major credit transactions
        salary_count = random.randint(1, 2)  # 1-2 salary payments per month
        for _ in range(salary_count):
            salary_date = start_date + timedelta(days=random.randint(5, 25))
            transactions.append(self.generate_transaction("salary", salary_date))
        
        # Generate other transaction types
        remaining_transactions = num_transactions - salary_count
        transaction_types = list(self.transaction_types.keys())
        transaction_types.remove("salary")  # Already added
        
        for _ in range(remaining_transactions):
            # Random date within statement period
            days_offset = random.randint(0, 29)
            trans_date = start_date + timedelta(days=days_offset)
            
            # Weighted selection of transaction types (match number of available types)
            weights = [0.25, 0.20, 0.20, 0.15, 0.10, 0.10]  # 6 types remaining after salary
            trans_type = random.choices(transaction_types, weights=weights)[0]
            
            transactions.append(self.generate_transaction(trans_type, trans_date))
        
        # Calculate running balances
        transactions_with_balance = self.calculate_running_balance(transactions, opening_balance)
        
        # Calculate totals
        total_credits = sum(t["amount"] for t in transactions_with_balance if t["type"] == "credit")
        total_debits = sum(t["amount"] for t in transactions_with_balance if t["type"] == "debit") 
        closing_balance = transactions_with_balance[-1]["balance"] if transactions_with_balance else opening_balance
        
        # Sort transactions in reverse chronological order for display (newest first)
        display_transactions = sorted(transactions_with_balance, key=lambda x: x["date"], reverse=True)
        
        return {
            # Bank and account details
            "document_type": "BANK STATEMENT",
            "bank_name": bank_info["name"],
            "bank_code": bank,
            "account_type": account_type,
            "bsb_number": bsb,
            "account_number": account_number,
            "account_holder": account_holder,
            
            # Statement period
            "statement_period": f"{start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}",
            "start_date": start_date.strftime("%d/%m/%Y"),
            "end_date": end_date.strftime("%d/%m/%Y"),
            
            # Financial summary (Australian format with CR/DR suffixes)
            "opening_balance": f"${abs(opening_balance):.2f} {'DR' if opening_balance < 0 else 'CR'}",
            "closing_balance": f"${abs(closing_balance):.2f} {'DR' if closing_balance < 0 else 'CR'}",
            "total_credits": f"${total_credits:.2f}",
            "total_debits": f"${total_debits:.2f}",
            
            # Transaction data (in display order - reverse chronological)
            "transactions": display_transactions,
            
            # Derived fields for ground truth
            "transaction_count": len(transactions_with_balance),
            "generation_timestamp": datetime.now().isoformat()
        }

    def generate_statement_png(self, statement_data: Dict, output_path: str) -> None:
        """Generate a PNG image of the bank statement matching Australian format."""
        
        if not PIL_AVAILABLE:
            print("⚠️ Skipping PNG generation - PIL not available")
            return
        
        # Image dimensions - taller for statements
        width, height = 800, 1200
        background_color = "white"
        text_color = "black"
        header_color = "#2E4B8B"  # Professional blue
        
        # Create image and drawing context
        img = Image.new('RGB', (width, height), background_color)
        draw = ImageDraw.Draw(img)
        
        # Load fonts
        try:
            title_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 18)
            header_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
            normal_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
            small_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 10)
        except (OSError, IOError):
            title_font = ImageFont.load_default()
            header_font = ImageFont.load_default() 
            normal_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        y_pos = 30
        
        # Bank header
        bank_name = statement_data["bank_name"]
        draw.text((50, y_pos), bank_name, fill=header_color, font=title_font)
        y_pos += 40
        
        # Account information section
        account_info = [
            f"Account: {statement_data['account_type']}",
            f"BSB: {statement_data['bsb_number']}",
            f"Account Number: {statement_data['account_number']}",
            f"Name: {statement_data['account_holder']}",
            f"Statement Period: {statement_data['statement_period']}"
        ]
        
        for info in account_info:
            draw.text((50, y_pos), info, fill=text_color, font=normal_font)
            y_pos += 18
        
        y_pos += 20
        
        # Balance summary (already formatted with CR/DR suffixes)
        opening_str = statement_data['opening_balance'] 
        closing_str = statement_data['closing_balance']
        
        draw.text((50, y_pos), f"Opening Balance: {opening_str}", 
                 fill=text_color, font=normal_font)
        y_pos += 16
        draw.text((50, y_pos), f"Closing Balance: {closing_str}", 
                 fill=text_color, font=normal_font)
        y_pos += 30
        
        # Transaction table header
        draw.text((50, y_pos), "Date", fill=text_color, font=header_font)
        draw.text((150, y_pos), "Transaction", fill=text_color, font=header_font)
        draw.text((500, y_pos), "Debit", fill=text_color, font=header_font)
        draw.text((600, y_pos), "Credit", fill=text_color, font=header_font)
        draw.text((700, y_pos), "Balance", fill=text_color, font=header_font)
        y_pos += 20
        
        # Header underline
        draw.line([(50, y_pos), (750, y_pos)], fill=text_color, width=1)
        y_pos += 15
        
        # Group transactions by date for display (reverse chronological)
        current_date = None
        
        for transaction in statement_data["transactions"]:
            trans_date = transaction["date"]
            date_str = trans_date.strftime("%a %d %b %Y")
            
            # Date header if new date
            if current_date != trans_date.date():
                if current_date is not None:
                    y_pos += 10  # Extra spacing between date groups
                
                draw.text((50, y_pos), date_str, fill=header_color, font=header_font)
                y_pos += 25
                
                # Date separator line
                draw.line([(50, y_pos), (750, y_pos)], fill="#CCCCCC", width=1)
                y_pos += 10
                
                current_date = trans_date.date()
            
            # Transaction row
            description = transaction["description"]
            if len(description) > 45:
                description = description[:42] + "..."
            
            # Date (empty for grouped format)
            draw.text((150, y_pos), description, fill=text_color, font=small_font)
            
            # Amount in appropriate column (positive amounts only)
            amount_str = f"${transaction['amount']:.2f}"
            if transaction["type"] == "debit":
                draw.text((500, y_pos), amount_str, fill=text_color, font=small_font)
            else:  # credit
                draw.text((600, y_pos), amount_str, fill=text_color, font=small_font)
            
            # Balance (always positive amount with CR/DR suffix)
            balance_amount = abs(transaction['balance'])
            if transaction['balance'] < 0:
                balance_str = f"${balance_amount:.2f} DR"
            else:
                balance_str = f"${balance_amount:.2f} CR"
            draw.text((700, y_pos), balance_str, fill=text_color, font=small_font)
            
            y_pos += 16
            
            # Check if we're running out of space
            if y_pos > height - 100:
                draw.text((50, y_pos + 10), "... (truncated for display)", 
                         fill="#888888", font=small_font)
                break
        
        y_pos += 30
        
        # Account summary footer (like Smart Access note in your example)
        footer_text = f"{statement_data['account_type']}"
        draw.text((50, y_pos), footer_text, fill=header_color, font=normal_font)
        y_pos += 20
        
        # Account features note
        features_text = "Enjoy the convenience and security of withdrawing what you need, when you need it."
        draw.text((50, y_pos), features_text, fill=text_color, font=small_font)
        
        # Save the image
        img.save(output_path, 'PNG', quality=95, dpi=(300, 300))
        print(f"📄 Generated PNG: {Path(output_path).name}")

    def map_to_extraction_fields(self, statement_data: Dict, image_filename: str) -> Dict[str, str]:
        """Map statement data to extraction field format for ground truth CSV."""
        
        result = {"image_file": image_filename}
        transactions = statement_data["transactions"]
        
        # Map each field from schema to statement data
        for csv_field, data_key in self.field_schema_mapping.items():
            if data_key is None:
                # Field not applicable to bank statements
                result[csv_field] = "NOT_FOUND"
            elif data_key in statement_data and statement_data[data_key] is not None:
                result[csv_field] = str(statement_data[data_key])
            else:
                # Handle derived fields with new helper methods
                if csv_field == "TRANSACTION_DATES" and data_key == "transaction_dates_chronological":
                    # Transaction dates in chronological order (oldest first)
                    chronological_transactions = self.get_chronological_transactions(transactions)
                    dates = [t["date"].strftime("%d/%m/%Y") for t in chronological_transactions]
                    result[csv_field] = " | ".join(dates) if dates else "NOT_FOUND"
                    
                elif csv_field == "TRANSACTION_AMOUNTS_PAID" and data_key == "debit_amounts_only":
                    # Debit amounts only (positive values, chronological order) with NOT_FOUND placeholders
                    debit_amounts = self.get_debit_amounts_only(transactions)
                    result[csv_field] = " | ".join(debit_amounts) if debit_amounts else "NOT_FOUND"
                    
                elif csv_field == "TRANSACTION_AMOUNTS_RECEIVED" and data_key == "credit_amounts_only":
                    # Credit amounts only (positive values, chronological order) with NOT_FOUND placeholders
                    credit_amounts = self.get_credit_amounts_only(transactions)
                    result[csv_field] = " | ".join(credit_amounts) if credit_amounts else "NOT_FOUND"
                    
                elif csv_field == "TRANSACTION_BALANCES" and data_key == "running_balances":
                    # Running balances with CR/DR suffixes (chronological order)
                    balances = self.get_running_balances(transactions)
                    result[csv_field] = " | ".join(balances) if balances else "NOT_FOUND"
                    
                else:
                    result[csv_field] = "NOT_FOUND"
        
        return result

    def generate_ground_truth_csv(self, statements_data: List[Dict], csv_path: str, 
                                 append_mode: bool = False) -> None:
        """Generate or append to ground truth CSV file for model evaluation."""
        
        # Define CSV headers (all extraction fields)
        fieldnames = ["image_file"] + list(self.field_schema_mapping.keys())
        
        csv_file = Path(csv_path)
        
        # Determine write mode and whether to write header
        write_header = True
        mode = 'w'
        
        if append_mode and csv_file.exists():
            mode = 'a'
            write_header = False
            print(f"📝 Appending to existing CSV: {csv_file}")
        else:
            print(f"📝 Creating new CSV: {csv_file}")
        
        with csv_file.open(mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if write_header:
                writer.writeheader()
            
            for statement_data in statements_data:
                image_filename = statement_data.get('image_filename', 
                    f"statement_{statement_data.get('account_number', 'unknown')}.png")
                row_data = self.map_to_extraction_fields(statement_data, image_filename)
                writer.writerow(row_data)
        
        action = "Appended to" if mode == 'a' else "Generated"
        print(f"✅ {action} ground truth CSV: {csv_file}")
        print(f"📊 Fields mapped: {len(fieldnames)} columns, {len(statements_data)} rows")

    def generate_batch(self, count: int = 10, bank: str = "CommBank", 
                      output_dir: str = "synthetic_bank_statements",
                      with_ground_truth: bool = True, start_number: int = 1,
                      append_csv: bool = False) -> List[Dict]:
        """Generate a batch of synthetic bank statements.
        
        Args:
            count: Number of statements to generate
            bank: Bank to generate statements for (CommBank, ANZ, NAB, Westpac)
            output_dir: Directory to save files
            with_ground_truth: Whether to generate ground truth CSV
            start_number: Starting statement number
            append_csv: Whether to append to existing CSV
        """
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        statements = []
        
        for i in range(count):
            # Vary statement dates
            days_back = random.randint(0, 365)
            statement_date = datetime.now() - timedelta(days=days_back)
            
            statement_data = self.generate_statement_data(
                bank=bank,
                statement_date=statement_date
            )
            
            # Add image filename for ground truth tracking
            statement_num = start_number + i
            image_filename = f"synthetic_statement_{statement_num:03d}.png"
            statement_data['image_filename'] = image_filename
            
            statements.append(statement_data)
            
            # Save as PNG image
            png_file = output_path / image_filename
            self.generate_statement_png(statement_data, str(png_file))
            
            print(f"Generated statement {statement_num}: {statement_data['account_holder'][:20]}... - "
                  f"{statement_data['closing_balance']} ({statement_data['transaction_count']} transactions)")
        
        # Generate ground truth CSV if requested
        if with_ground_truth:
            csv_path = output_path / "ground_truth.csv"
            self.generate_ground_truth_csv(statements, str(csv_path), append_mode=append_csv)
        
        # Generate summary
        summary = {
            "total_generated": count,
            "bank": bank,
            "generation_date": datetime.now().isoformat(),
            "reverse_chronological": True,
            "ground_truth_csv": with_ground_truth,
            "avg_transactions": sum(s['transaction_count'] for s in statements) / count
        }
        
        summary_file = output_path / "generation_summary.json"
        with summary_file.open('w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Generated {count} Australian bank statements in {output_path}")
        print(f"   Bank: {self.banks[bank]['name']}")
        print(f"   Avg transactions per statement: {summary['avg_transactions']:.1f}")
        print("   Reverse chronological order: ✅")
        if with_ground_truth:
            print("   Ground truth CSV: ✅ Generated")
        
        return statements


def generate_all_banks(count_per_bank: int = 5, output_base_dir: str = "australian_synthetic_statements") -> Dict[str, List[Dict]]:
    """Generate statements for all Big 4 Australian banks."""
    
    generator = AustralianBankStatementGenerator()
    all_statements = {}
    
    banks = list(generator.banks.keys())
    
    for bank in banks:
        print(f"\n🏦 Generating {count_per_bank} statements for {generator.banks[bank]['name']}...")
        
        # Create bank-specific output directory
        bank_dir = f"{output_base_dir}_{bank.lower()}"
        
        statements = generator.generate_batch(
            count=count_per_bank,
            bank=bank,
            output_dir=bank_dir,
            with_ground_truth=True,
            start_number=1
        )
        
        all_statements[bank] = statements
    
    # Generate combined ground truth CSV
    print("\n📋 Generating combined ground truth CSV...")
    combined_statements = []
    
    for bank, statements in all_statements.items():
        for statement in statements:
            # Update image filename to include bank prefix
            original_filename = statement['image_filename']
            statement['image_filename'] = f"{bank.lower()}_{original_filename}"
            combined_statements.append(statement)
    
    # Save combined CSV
    from pathlib import Path
    combined_dir = Path(output_base_dir + "_combined")
    combined_dir.mkdir(exist_ok=True)
    
    generator.generate_ground_truth_csv(
        combined_statements, 
        str(combined_dir / "ground_truth_all_banks.csv")
    )
    
    print("\n✅ Generated statements for all Big 4 banks:")
    for bank, statements in all_statements.items():
        bank_name = generator.banks[bank]['name']
        print(f"   {bank_name}: {len(statements)} statements")
    
    print(f"   Combined ground truth CSV: {len(combined_statements)} total statements")
    
    return all_statements


def generate_single_bank(bank: str = "CommBank", count: int = 10, output_dir: str = None) -> List[Dict]:
    """Generate statements for a single bank - convenience function."""
    
    generator = AustralianBankStatementGenerator()
    
    if output_dir is None:
        output_dir = f"synthetic_statements_{bank.lower()}"
    
    if bank not in generator.banks:
        available_banks = list(generator.banks.keys())
        raise ValueError(f"Bank '{bank}' not supported. Available: {available_banks}")
    
    return generator.generate_batch(
        count=count,
        bank=bank,
        output_dir=output_dir,
        with_ground_truth=True
    )


def generate_one_per_bank(output_dir: str = "big4_bank_statements") -> Dict[str, Dict]:
    """Generate exactly one statement for each Big 4 bank in a single folder."""
    
    generator = AustralianBankStatementGenerator()
    
    # Create output directory
    from pathlib import Path
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    bank_statements = {}
    all_statements_for_csv = []
    
    # Generate one statement per bank
    for i, (bank_code, bank_info) in enumerate(generator.banks.items(), 1):
        print(f"🏦 Generating statement {i}/4: {bank_info['name']}")
        
        # Generate statement data
        statement_data = generator.generate_statement_data(bank=bank_code)
        
        # Create filename with bank prefix
        image_filename = f"{bank_code.lower()}_statement_001.png"
        statement_data['image_filename'] = image_filename
        
        # Save PNG in the single folder
        png_file = output_path / image_filename
        generator.generate_statement_png(statement_data, str(png_file))
        
        bank_statements[bank_code] = statement_data
        all_statements_for_csv.append(statement_data)
        
        print(f"   ✅ {image_filename} - {statement_data['account_holder'][:25]}...")
    
    # Generate single ground truth CSV for all 4 statements
    csv_path = output_path / "ground_truth.csv"
    generator.generate_ground_truth_csv(all_statements_for_csv, str(csv_path))
    
    # Generate summary
    summary = {
        "total_statements": 4,
        "banks_included": list(generator.banks.keys()),
        "files_generated": [f"{bank.lower()}_statement_001.png" for bank in generator.banks.keys()],
        "ground_truth_csv": "ground_truth.csv",
        "generation_date": datetime.now().isoformat()
    }
    
    summary_file = output_path / "generation_summary.json"
    with summary_file.open('w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Generated Big 4 bank statements in {output_path}")
    print("   📁 Single folder contains:")
    print("   🏦 commbank_statement_001.png - Commonwealth Bank")
    print("   🏦 anz_statement_001.png - ANZ Bank") 
    print("   🏦 nab_statement_001.png - NAB Bank")
    print("   🏦 westpac_statement_001.png - Westpac Bank")
    print("   📄 ground_truth.csv - Evaluation data")
    print("   📊 generation_summary.json - Metadata")
    
    return bank_statements


def main():
    """Generate synthetic Australian bank statements with ground truth CSV."""
    
    # Option 1: Generate one statement per bank in single folder (recommended)
    print("🏦 Generating one statement per Big 4 bank...")
    bank_statements = generate_one_per_bank("big4_bank_statements")
    
    # Option 2: Generate multiple statements for a single bank
    print("\n🏦 Generating multiple CommBank statements...")
    generator = AustralianBankStatementGenerator()
    statements = generator.generate_batch(
        count=3,
        bank="CommBank", 
        output_dir="commbank_statements_sample",
        with_ground_truth=True
    )
    
    # Option 3: Generate for all Big 4 banks (separate folders)
    print("\n🏦 Generating statements for all Big 4 banks (separate folders)...")
    all_bank_statements = generate_all_banks(count_per_bank=2)
    
    print("\\n📊 Sample Statement Data:")
    print(f"   CommBank statements: {len(statements)}")
    # Parse closing balance from CR/DR format
    def parse_balance(balance_str):
        """Parse balance string like '$1234.56 CR' or '$1234.56 DR' to float."""
        parts = balance_str.split()
        amount = float(parts[0][1:])  # Remove $ sign
        return -amount if parts[1] == 'DR' else amount
    
    avg_balance = sum(parse_balance(stmt['closing_balance']) for stmt in statements) / len(statements)
    print(f"   Average closing balance: ${avg_balance:.2f}")
    print("   Reverse chronological order: ✅")
    print("   Australian format compliant: ✅")
    
    print("\\n📊 All Banks Summary:")
    total_statements = sum(len(stmts) for stmts in all_bank_statements.values())
    print(f"   Total statements generated: {total_statements}")
    print("   Banks covered: CommBank, ANZ, NAB, Westpac")
    print("   Combined ground truth CSV: ✅")


if __name__ == "__main__":
    main()