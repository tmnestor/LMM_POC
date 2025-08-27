#!/usr/bin/env python3
"""
Ground Truth CSV Migration Script: v3 → v4 Schema

Migrates existing ground truth CSV from v3 schema (34 fields) to v4 schema (49 fields)
by adding the 15 new fields introduced in the comprehensive field expansion.

This script handles:
- Field mapping from v3 to v4
- Default value assignment for new fields
- Data validation and integrity checks
- Backup creation of original file
"""

import shutil
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd


class GroundTruthMigrator:
    """Handles migration from v3 to v4 ground truth CSV schema."""
    
    def __init__(self, input_csv: str, output_csv: str = None, backup: bool = True):
        """
        Initialize migrator.
        
        Args:
            input_csv (str): Path to existing v3 ground truth CSV
            output_csv (str): Path for output v4 CSV (default: add _v4 suffix)
            backup (bool): Whether to backup original file
        """
        self.input_path = Path(input_csv)
        self.output_path = Path(output_csv) if output_csv else self.input_path.with_stem(f"{self.input_path.stem}_v4")
        self.backup = backup
        
        # Expected v3 schema fields (34 total)
        self.v3_fields = [
            "image_file", "DOCUMENT_TYPE", "INVOICE_NUMBER", "INVOICE_DATE", "DUE_DATE",
            "SUPPLIER_NAME", "BUSINESS_ABN", "BUSINESS_ADDRESS", "BUSINESS_PHONE", "SUPPLIER_WEBSITE",
            "PAYER_NAME", "PAYER_ABN", "PAYER_ADDRESS", "PAYER_PHONE", "PAYER_EMAIL",
            "LINE_ITEM_DESCRIPTIONS", "LINE_ITEM_QUANTITIES", "LINE_ITEM_PRICES", 
            "SUBTOTAL_AMOUNT", "GST_AMOUNT", "TOTAL_AMOUNT",
            "RECEIPT_NUMBER", "TRANSACTION_DATE", "PAYMENT_METHOD", "STORE_LOCATION",
            "BANK_NAME", "BANK_BSB_NUMBER", "BANK_ACCOUNT_NUMBER", "BANK_ACCOUNT_HOLDER", 
            "STATEMENT_DATE_RANGE", "ACCOUNT_OPENING_BALANCE", "ACCOUNT_CLOSING_BALANCE", 
            "TOTAL_CREDITS", "TOTAL_DEBITS"
        ]
        
        # New fields in v4 schema (15 additional fields)
        self.v4_new_fields = {
            # New common fields
            "SUPPLIER_EMAIL": "NOT_FOUND",
            
            # New invoice-specific fields  
            "LINE_ITEM_TOTAL_PRICES": "NOT_FOUND",
            "LINE_ITEM_GST_AMOUNTS": "NOT_FOUND", 
            "LINE_ITEM_DISCOUNT_AMOUNTS": "NOT_FOUND",
            "TOTAL_DISCOUNT_AMOUNT": "NOT_FOUND",
            "IS_GST_INCLUDED": "NOT_FOUND",
            "TOTAL_AMOUNT_PAID": "NOT_FOUND",
            "BALANCE_OF_PAYMENT": "NOT_FOUND", 
            "TOTAL_AMOUNT_PAYABLE": "NOT_FOUND",
            
            # New bank statement fields
            "CREDIT_CARD_DUE_DATE": "NOT_FOUND",
            "TRANSACTION_DATES": "NOT_FOUND",
            "TRANSACTION_AMOUNTS_PAID": "NOT_FOUND", 
            "TRANSACTION_AMOUNTS_RECEIVED": "NOT_FOUND",
            "TRANSACTION_BALANCES": "NOT_FOUND",
            
            # New calculated field for bank statements
            "NET_TRANSACTION_AMOUNT": "NOT_FOUND"
        }
        
        # Expected v4 field order (49 total)
        self.v4_field_order = self._build_v4_field_order()
    
    def _build_v4_field_order(self) -> List[str]:
        """Build the expected v4 field order for consistent CSV output."""
        return [
            # Metadata
            "image_file",
            "DOCUMENT_TYPE", 
            
            # Invoice fields
            "INVOICE_NUMBER", "INVOICE_DATE", "DUE_DATE",
            
            # Supplier/Business information
            "SUPPLIER_NAME", "BUSINESS_ABN", "BUSINESS_ADDRESS", "BUSINESS_PHONE", 
            "SUPPLIER_WEBSITE", "SUPPLIER_EMAIL",
            
            # Payer information
            "PAYER_NAME", "PAYER_ABN", "PAYER_ADDRESS", "PAYER_PHONE", "PAYER_EMAIL",
            
            # Line item details
            "LINE_ITEM_DESCRIPTIONS", "LINE_ITEM_QUANTITIES", "LINE_ITEM_PRICES",
            "LINE_ITEM_TOTAL_PRICES", "LINE_ITEM_GST_AMOUNTS", "LINE_ITEM_DISCOUNT_AMOUNTS",
            
            # Financial totals
            "SUBTOTAL_AMOUNT", "TOTAL_DISCOUNT_AMOUNT", "GST_AMOUNT", "IS_GST_INCLUDED",
            "TOTAL_AMOUNT", "TOTAL_AMOUNT_PAID", "BALANCE_OF_PAYMENT", "TOTAL_AMOUNT_PAYABLE",
            
            # Receipt fields
            "RECEIPT_NUMBER", "TRANSACTION_DATE", "PAYMENT_METHOD", "STORE_LOCATION",
            
            # Bank statement fields
            "BANK_NAME", "BANK_BSB_NUMBER", "BANK_ACCOUNT_NUMBER", "BANK_ACCOUNT_HOLDER",
            "STATEMENT_DATE_RANGE", "CREDIT_CARD_DUE_DATE",
            "ACCOUNT_OPENING_BALANCE", "ACCOUNT_CLOSING_BALANCE", 
            "TOTAL_CREDITS", "TOTAL_DEBITS", "NET_TRANSACTION_AMOUNT",
            
            # Transaction details
            "TRANSACTION_DATES", "TRANSACTION_AMOUNTS_PAID", "TRANSACTION_AMOUNTS_RECEIVED", 
            "TRANSACTION_BALANCES"
        ]
    
    def validate_input(self) -> bool:
        """Validate the input CSV has expected v3 structure."""
        if not self.input_path.exists():
            print(f"❌ Error: Input file not found: {self.input_path}")
            return False
        
        try:
            ground_truth_df = pd.read_csv(self.input_path)
            current_columns = list(ground_truth_df.columns)
            
            print("📊 Input CSV validation:")
            print(f"   File: {self.input_path}")
            print(f"   Rows: {len(ground_truth_df)}")
            print(f"   Current columns: {len(current_columns)}")
            print(f"   Expected v3 columns: {len(self.v3_fields)}")
            
            # Check if all v3 fields are present
            missing_v3_fields = set(self.v3_fields) - set(current_columns)
            extra_fields = set(current_columns) - set(self.v3_fields)
            
            if missing_v3_fields:
                print(f"⚠️  Missing expected v3 fields: {missing_v3_fields}")
            
            if extra_fields:
                print(f"ℹ️  Extra fields (will be preserved): {extra_fields}")
            
            # Check if this might already be v4
            v4_fields_present = set(self.v4_new_fields.keys()) & set(current_columns)
            if v4_fields_present:
                print(f"⚠️  Some v4 fields already present: {v4_fields_present}")
                print("   This may already be a v4 CSV or partially migrated")
            
            return True
            
        except Exception as e:
            print(f"❌ Error reading input CSV: {e}")
            return False
    
    def create_backup(self):
        """Create backup of original CSV."""
        if self.backup:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.input_path.with_stem(f"{self.input_path.stem}_backup_{timestamp}")
            shutil.copy2(self.input_path, backup_path)
            print(f"💾 Backup created: {backup_path}")
    
    def migrate(self) -> bool:
        """Perform the migration from v3 to v4 schema."""
        print("🔄 Starting ground truth CSV migration: v3 → v4")
        print("=" * 60)
        
        # Validation
        if not self.validate_input():
            return False
        
        # Create backup
        self.create_backup()
        
        try:
            # Load v3 data
            print("📖 Loading v3 ground truth data...")
            ground_truth_df = pd.read_csv(self.input_path)
            
            # Add new v4 fields with default values
            print("🆕 Adding new v4 fields...")
            for field, default_value in self.v4_new_fields.items():
                if field not in ground_truth_df.columns:
                    ground_truth_df[field] = default_value
                    print(f"   ✅ Added {field}: {default_value}")
                else:
                    print(f"   ⚠️  {field} already exists, preserving values")
            
            # Reorder columns to match v4 schema
            print("🔄 Reordering columns to v4 schema...")
            
            # Only include fields that exist in the dataframe
            available_fields = [field for field in self.v4_field_order if field in ground_truth_df.columns]
            extra_fields = [field for field in ground_truth_df.columns if field not in self.v4_field_order]
            
            # Reorder with v4 schema first, then any extra fields
            final_column_order = available_fields + extra_fields
            df_v4 = ground_truth_df[final_column_order]
            
            # Apply field-specific transformations
            df_v4 = self._apply_field_transformations(df_v4)
            
            # Save v4 CSV
            print("💾 Saving v4 ground truth CSV...")
            df_v4.to_csv(self.output_path, index=False)
            
            print("✅ Migration completed successfully!")
            print("=" * 60)
            print("📊 Migration summary:")
            print(f"   Input: {self.input_path} ({len(ground_truth_df.columns)} columns)")
            print(f"   Output: {self.output_path} ({len(df_v4.columns)} columns)")
            print(f"   Rows processed: {len(df_v4)}")
            print(f"   New fields added: {len(self.v4_new_fields)}")
            
            # Show sample of new fields
            print("\\n🔍 Sample of added fields:")
            for field in list(self.v4_new_fields.keys())[:5]:
                if field in df_v4.columns:
                    print(f"   {field}: {df_v4[field].iloc[0] if len(df_v4) > 0 else 'N/A'}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during migration: {e}")
            return False
    
    def _apply_field_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply field-specific transformations for v4 compatibility."""
        
        # Example transformations based on document type
        for idx, row in df.iterrows():
            doc_type = row.get('DOCUMENT_TYPE', '').upper()
            
            # For invoice documents, set some realistic defaults
            if 'INVOICE' in doc_type:
                # Set GST inclusion based on presence of GST_AMOUNT
                gst_amount = row.get('GST_AMOUNT', 'NOT_FOUND')
                if gst_amount != 'NOT_FOUND' and gst_amount != '':
                    df.loc[idx, 'IS_GST_INCLUDED'] = 'false'  # GST separate if amount exists
                
                # Calculate line item totals if possible
                quantities = row.get('LINE_ITEM_QUANTITIES', 'NOT_FOUND')
                prices = row.get('LINE_ITEM_PRICES', 'NOT_FOUND') 
                
                if quantities != 'NOT_FOUND' and prices != 'NOT_FOUND':
                    try:
                        qty_list = [q.strip() for q in quantities.split('|')]
                        price_list = [p.strip() for p in prices.split('|')]
                        
                        if len(qty_list) == len(price_list):
                            totals = []
                            for q, p in zip(qty_list, price_list, strict=False):
                                # Extract numeric values
                                qty_num = float(''.join(filter(str.isdigit, q)) or '1')
                                price_num = float(''.join(filter(lambda x: x.isdigit() or x == '.', p.replace('$', '').replace(',', '').strip())))
                                total = qty_num * price_num
                                totals.append(f"${total:.2f}")
                            
                            df.loc[idx, 'LINE_ITEM_TOTAL_PRICES'] = ' | '.join(totals)
                    except (ValueError, AttributeError):
                        pass  # Keep default NOT_FOUND
        
        return df


def main():
    """Main function to run the migration."""
    input_file = "/Users/tod/Desktop/LMM_POC/evaluation_data/ground_truth.csv"
    output_file = "/Users/tod/Desktop/LMM_POC/evaluation_data/ground_truth_v4.csv"
    
    migrator = GroundTruthMigrator(input_file, output_file, backup=True)
    
    success = migrator.migrate()
    
    if success:
        print("\\n🎉 Ground truth migration to v4 schema completed successfully!")
        print("\\nNext steps:")
        print("1. Review the migrated CSV for data integrity")
        print("2. Update any ground truth generation scripts to use v4 schema")
        print("3. Test extraction evaluation with the new v4 ground truth")
    else:
        print("\\n❌ Migration failed. Please check the errors above and retry.")


if __name__ == "__main__":
    main()