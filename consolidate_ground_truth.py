#!/usr/bin/env python3
"""
Consolidate ground truth CSV files with consistent column headers.
Merges evaluation_data and ato_synthetic_invoices CSV files.
"""

import csv
from pathlib import Path

def consolidate_ground_truth():
    """Merge ground truth CSV files with consistent columns."""
    
    # Define the unified column order (superset of all columns)
    unified_columns = [
        # File reference
        "image_file",
        
        # Document metadata
        "DOCUMENT_TYPE",
        "INVOICE_NUMBER",
        "INVOICE_DATE", 
        "DUE_DATE",
        
        # Supplier/Business information
        "SUPPLIER_NAME",
        "BUSINESS_ABN",
        "BUSINESS_ADDRESS",
        "BUSINESS_PHONE",
        "SUPPLIER_WEBSITE",
        
        # Payer/Customer information
        "PAYER_NAME",
        "PAYER_ABN",
        "PAYER_ADDRESS",
        "PAYER_PHONE",
        "PAYER_EMAIL",
        
        # Line items
        "LINE_ITEM_DESCRIPTIONS",
        "LINE_ITEM_QUANTITIES",
        "LINE_ITEM_PRICES",
        
        # Financial amounts
        "SUBTOTAL_AMOUNT",
        "GST_AMOUNT",
        "TOTAL_AMOUNT",
        
        # Bank statement specific fields
        "BANK_NAME",
        "BANK_BSB_NUMBER",
        "BANK_ACCOUNT_NUMBER",
        "BANK_ACCOUNT_HOLDER",
        "STATEMENT_DATE_RANGE",
        "ACCOUNT_OPENING_BALANCE",
        "ACCOUNT_CLOSING_BALANCE",
        "TOTAL_CREDITS",
        "TOTAL_DEBITS"
    ]
    
    # Read evaluation_data CSV
    eval_csv = Path("/Users/tod/Desktop/LMM_POC/evaluation_data/evaluation_ground_truth.csv")
    eval_rows = []
    
    with open(eval_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Standardize field names and fill missing columns
            standardized_row = {}
            for col in unified_columns:
                if col in row:
                    standardized_row[col] = row[col]
                else:
                    # Handle missing columns
                    if col == "INVOICE_NUMBER":
                        standardized_row[col] = "NOT_FOUND"  # Old data doesn't have invoice numbers
                    elif col == "PAYER_ABN":
                        standardized_row[col] = "NOT_FOUND"  # Old data doesn't have payer ABN
                    elif col == "LINE_ITEM_DESCRIPTIONS":
                        # Check if it exists under different name
                        if "LINE_ITEM_DESCRIPTIONS" in row:
                            standardized_row[col] = row["LINE_ITEM_DESCRIPTIONS"]
                        elif "DESCRIPTIONS" in row:
                            standardized_row[col] = row["DESCRIPTIONS"]
                        else:
                            standardized_row[col] = "NOT_FOUND"
                    elif col == "TOTAL_CREDITS":
                        standardized_row[col] = "NOT_FOUND"  # Old format doesn't have this
                    elif col == "TOTAL_DEBITS":
                        standardized_row[col] = "NOT_FOUND"  # Old format doesn't have this
                    else:
                        standardized_row[col] = "NOT_FOUND"
            
            eval_rows.append(standardized_row)
    
    print(f"✅ Read {len(eval_rows)} rows from evaluation_data")
    
    # Read ato_synthetic_invoices CSV
    ato_csv = Path("/Users/tod/Desktop/LMM_POC/ato_synthetic_invoices/ground_truth.csv")
    ato_rows = []
    
    with open(ato_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # All columns should already exist in the ATO CSV
            standardized_row = {}
            for col in unified_columns:
                if col in row:
                    standardized_row[col] = row[col]
                else:
                    standardized_row[col] = "NOT_FOUND"
            
            ato_rows.append(standardized_row)
    
    print(f"✅ Read {len(ato_rows)} rows from ato_synthetic_invoices")
    
    # Combine all rows
    all_rows = eval_rows + ato_rows
    
    # Write consolidated CSV
    output_path = Path("/Users/tod/Desktop/LMM_POC/consolidated_evaluation_data/ground_truth.csv")
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=unified_columns)
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"\n✅ Consolidated ground truth CSV created: {output_path}")
    print(f"📊 Total rows: {len(all_rows)} (invoices 001-020 + 021-040)")
    print(f"📋 Unified columns: {len(unified_columns)} fields")
    
    # Verify image files match CSV entries
    consolidated_dir = Path("/Users/tod/Desktop/LMM_POC/consolidated_evaluation_data")
    png_files = sorted(consolidated_dir.glob("synthetic_invoice_*.png"))
    
    print(f"\n🖼️ PNG files in consolidated directory: {len(png_files)}")
    
    # Check if all CSV entries have corresponding images
    csv_images = {row['image_file'] for row in all_rows}
    png_names = {f.name for f in png_files}
    
    if csv_images == png_names:
        print("✅ All CSV entries have matching PNG files!")
    else:
        missing_from_dir = csv_images - png_names
        extra_in_dir = png_names - csv_images
        
        if missing_from_dir:
            print(f"⚠️ Missing PNG files: {missing_from_dir}")
        if extra_in_dir:
            print(f"⚠️ Extra PNG files not in CSV: {extra_in_dir}")
    
    return all_rows


if __name__ == "__main__":
    consolidate_ground_truth()