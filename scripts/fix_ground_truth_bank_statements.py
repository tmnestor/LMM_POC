#!/usr/bin/env python
"""
Fix Ground Truth for Bank Statements

This script filters bank statement ground truth data to only include DEBIT transactions.

RULE (BANK_STATEMENT only):
- TRANSACTION_DATES, LINE_ITEM_DESCRIPTIONS, and TRANSACTION_AMOUNTS_PAID are aligned fields
- Each index position corresponds to the same transaction
- When TRANSACTION_AMOUNTS_PAID[i] == "NOT_FOUND", that's a CREDIT (money IN), not a debit
- For tax purposes, we only care about DEBITS (what taxpayer PAID)
- This script filters out all credit entries from these three aligned fields

This rule ONLY applies to DOCUMENT_TYPE == "BANK_STATEMENT"
Invoices and receipts are NOT modified.

Usage:
    python fix_ground_truth_bank_statements.py [--input PATH] [--output PATH] [--dry-run]
"""

import argparse
from pathlib import Path

import pandas as pd


def filter_aligned_fields(row: pd.Series, verbose: bool = True) -> pd.Series:
    """Filter aligned fields to remove entries where TRANSACTION_AMOUNTS_PAID == 'NOT_FOUND'.

    For BANK_STATEMENT documents only, this removes credit transactions from the
    three aligned fields: TRANSACTION_DATES, LINE_ITEM_DESCRIPTIONS, TRANSACTION_AMOUNTS_PAID.

    Args:
        row: DataFrame row to process
        verbose: Print progress information

    Returns:
        Modified row with filtered fields (only for bank statements)
    """
    # Only process bank statements
    if row["DOCUMENT_TYPE"] != "BANK_STATEMENT":
        return row

    image_file = row.get("image_file", "unknown")

    # Get the three aligned fields
    dates_str = str(row.get("TRANSACTION_DATES", ""))
    descriptions_str = str(row.get("LINE_ITEM_DESCRIPTIONS", ""))
    amounts_str = str(row.get("TRANSACTION_AMOUNTS_PAID", ""))

    # Skip if amounts field is empty or NOT_FOUND entirely
    if not amounts_str or amounts_str == "NOT_FOUND" or amounts_str == "nan":
        if verbose:
            print(f"  ℹ️  {image_file}: No TRANSACTION_AMOUNTS_PAID data, skipping")
        return row

    # Split by delimiter
    delimiter = " | "
    dates = dates_str.split(delimiter) if dates_str and dates_str != "NOT_FOUND" else []
    descriptions = (
        descriptions_str.split(delimiter)
        if descriptions_str and descriptions_str != "NOT_FOUND"
        else []
    )
    amounts = amounts_str.split(delimiter) if amounts_str else []

    # Check alignment
    if not (len(dates) == len(descriptions) == len(amounts)):
        print(
            f"  ⚠️  {image_file}: Length mismatch - dates={len(dates)}, desc={len(descriptions)}, amounts={len(amounts)}"
        )
        if len(amounts) == 0:
            return row

    # Filter: keep only entries where amount is NOT "NOT_FOUND"
    filtered_dates = []
    filtered_descriptions = []
    filtered_amounts = []

    removed_count = 0
    for i, amount in enumerate(amounts):
        if amount.strip() != "NOT_FOUND":
            # Keep this entry (it's a DEBIT)
            if i < len(dates):
                filtered_dates.append(dates[i])
            if i < len(descriptions):
                filtered_descriptions.append(descriptions[i])
            filtered_amounts.append(amount)
        else:
            # Skip this entry (it's a CREDIT)
            removed_count += 1

    # Update row with filtered values
    if removed_count > 0:
        if verbose:
            print(
                f"  ✅ {image_file}: Removed {removed_count} credit entries, kept {len(filtered_amounts)} debits"
            )
        row["TRANSACTION_DATES"] = (
            delimiter.join(filtered_dates) if filtered_dates else "NOT_FOUND"
        )
        row["LINE_ITEM_DESCRIPTIONS"] = (
            delimiter.join(filtered_descriptions)
            if filtered_descriptions
            else "NOT_FOUND"
        )
        row["TRANSACTION_AMOUNTS_PAID"] = (
            delimiter.join(filtered_amounts) if filtered_amounts else "NOT_FOUND"
        )
    else:
        if verbose:
            print(
                f"  ℹ️  {image_file}: No credits to remove (all {len(amounts)} are debits)"
            )

    return row


def fix_ground_truth(
    input_path: Path, output_path: Path, dry_run: bool = False
) -> None:
    """Fix ground truth CSV by filtering bank statement credits.

    Args:
        input_path: Path to input ground_truth.csv
        output_path: Path to save fixed CSV
        dry_run: If True, don't save changes
    """
    print("=" * 80)
    print("BANK STATEMENT GROUND TRUTH FILTER")
    print("=" * 80)

    # Load ground truth
    ground_truth_df = pd.read_csv(input_path)
    print(f"\nLoaded {len(ground_truth_df)} rows from {input_path}")

    # Count bank statements
    bank_statements = ground_truth_df[
        ground_truth_df["DOCUMENT_TYPE"] == "BANK_STATEMENT"
    ]
    print(f"Found {len(bank_statements)} BANK_STATEMENT rows to process")
    print(
        f"Found {len(ground_truth_df) - len(bank_statements)} non-bank-statement rows (unchanged)"
    )

    print("\nProcessing bank statements...")
    print("-" * 80)

    # Apply filter to each row
    fixed_df = ground_truth_df.apply(
        lambda row: filter_aligned_fields(row, verbose=True), axis=1
    )

    # Verification - show sample before/after
    print("\n" + "=" * 80)
    print("VERIFICATION SAMPLES")
    print("=" * 80)

    for image_file in ["cba_date_grouped.png", "cba_date_grouped_cont.png"]:
        if image_file in ground_truth_df["image_file"].to_numpy():
            row_before = ground_truth_df[
                ground_truth_df["image_file"] == image_file
            ].iloc[0]
            row_after = fixed_df[fixed_df["image_file"] == image_file].iloc[0]

            before_count = len(row_before["TRANSACTION_AMOUNTS_PAID"].split(" | "))
            after_count = len(row_after["TRANSACTION_AMOUNTS_PAID"].split(" | "))

            print(f"\n{image_file}:")
            print(f"  BEFORE: {before_count} entries")
            print(
                f"  AFTER:  {after_count} entries (removed {before_count - after_count} credits)"
            )

    if dry_run:
        print("\n🔍 DRY RUN - No changes saved")
    else:
        # Save backup
        backup_path = input_path.with_suffix(".csv.bak")
        ground_truth_df.to_csv(backup_path, index=False)
        print(f"\n✅ Backup saved to: {backup_path}")

        # Save fixed file
        fixed_df.to_csv(output_path, index=False)
        print(f"✅ Fixed ground truth saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fix bank statement ground truth by filtering out credit transactions"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("evaluation_data/ground_truth.csv"),
        help="Path to input ground_truth.csv",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Path to save fixed CSV (default: overwrite input)",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Show what would be changed without saving",
    )

    args = parser.parse_args()

    output_path = args.output if args.output else args.input

    fix_ground_truth(args.input, output_path, args.dry_run)


if __name__ == "__main__":
    main()
