#!/usr/bin/env python3
"""
Rename evaluation dataset from synthetic_invoice_XXX.png to image_XXX.png
to reflect the fact that the dataset contains mixed document types.

This script:
1. Renames all image files from synthetic_invoice_XXX.png to image_XXX.png  
2. Updates the ground_truth.csv file to use the new naming scheme
3. Creates backups of original files
"""

import csv
import shutil
from pathlib import Path


def main():
    eval_dir = Path("evaluation_data")
    ground_truth_file = eval_dir / "ground_truth.csv"
    
    print("🔄 RENAMING EVALUATION DATASET")
    print("=" * 50)
    
    # Step 1: Create backup of ground truth
    backup_file = eval_dir / "ground_truth_backup.csv"
    shutil.copy2(ground_truth_file, backup_file)
    print(f"✅ Backup created: {backup_file}")
    
    # Step 2: Read current ground truth and update image names
    updated_rows = []
    rename_mapping = {}
    
    with ground_truth_file.open('r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        updated_rows.append(headers)
        
        for row in reader:
            old_name = row[0]  # image_file column
            if old_name.startswith('synthetic_invoice_'):
                # Extract number: synthetic_invoice_001.png -> 001
                number = old_name.split('_')[2].split('.')[0]
                new_name = f"image_{number}.png"
                row[0] = new_name
                rename_mapping[old_name] = new_name
                print(f"   {old_name} → {new_name}")
            
            updated_rows.append(row)
    
    # Step 3: Write updated ground truth
    with ground_truth_file.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(updated_rows)
    
    print(f"✅ Updated ground_truth.csv with {len(rename_mapping)} changes")
    
    # Step 4: Rename image files
    print(f"\n🖼️ Renaming {len(rename_mapping)} image files:")
    for old_name, new_name in rename_mapping.items():
        old_path = eval_dir / old_name
        new_path = eval_dir / new_name
        
        if old_path.exists():
            shutil.move(old_path, new_path)
            print(f"   ✅ {old_name} → {new_name}")
        else:
            print(f"   ⚠️  {old_name} not found")
    
    # Step 5: Summary
    print("\n📊 SUMMARY:")
    print(f"   Files renamed: {len(rename_mapping)}")
    print("   Ground truth updated: ✅")
    print(f"   Backup created: {backup_file}")
    
    # Count document types
    doc_types = {}
    for row in updated_rows[1:]:  # Skip header
        doc_type = row[1]  # DOCUMENT_TYPE column
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
    
    print("\n📋 DOCUMENT TYPE DISTRIBUTION:")
    for doc_type, count in sorted(doc_types.items()):
        print(f"   {doc_type}: {count} files")
    
    print("\n✅ DATASET RENAMING COMPLETE!")
    print("   Now using generic image_XXX.png naming for mixed document types")


if __name__ == "__main__":
    main()