#!/usr/bin/env python3
"""
Generate one bank statement for each Australian Big 4 bank in a single folder.

This creates a sample set with one statement from each bank:
- CommBank (Commonwealth Bank of Australia)
- ANZ (Australia and New Zealand Banking Group) 
- NAB (National Australia Bank)
- Westpac (Westpac Banking Corporation)

Usage:
    python generate_big4_sample.py
"""

from generate_synthetic_bank_statements import generate_one_per_bank

if __name__ == "__main__":
    print("🏦 Generating Big 4 Australian Bank Statements")
    print("=" * 50)
    
    # Generate one statement per bank
    statements = generate_one_per_bank("big4_sample_statements")
    
    print("\n📊 Generation Summary:")
    for bank_code, statement_data in statements.items():
        bank_name = statement_data['bank_name']
        account_holder = statement_data['account_holder']
        closing_balance = statement_data['closing_balance']
        transaction_count = statement_data['transaction_count']
        
        print(f"   {bank_code}: {bank_name}")
        print(f"      Account: {account_holder}")
        print(f"      Balance: {closing_balance}")
        print(f"      Transactions: {transaction_count}")
        print()
    
    print("✅ Complete! Check the 'big4_sample_statements' folder for:")
    print("   • PNG images of all 4 bank statements")
    print("   • ground_truth.csv for model evaluation")
    print("   • generation_summary.json with metadata")