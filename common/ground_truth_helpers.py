"""
Ground Truth Validation Helpers - V100 Optimized

Helper functions for validating extraction results against ground truth data.
Used across multiple notebooks and scripts for consistent evaluation.
"""

from pathlib import Path

import pandas as pd
from rich import print as rprint


def get_expected_transaction_count(image_path: str, ground_truth_csv: str) -> int:
    """
    Get the expected number of transactions from ground truth CSV.

    Args:
        image_path: Path to the image file
        ground_truth_csv: Path to the ground truth CSV file

    Returns:
        Expected transaction count (0 if not found or error)
    """
    try:
        if not Path(ground_truth_csv).exists():
            rprint(
                f"[yellow]⚠️ Ground truth file not found: {ground_truth_csv}[/yellow]"
            )
            return 0

        # Read ground truth CSV
        ground_truth_df = pd.read_csv(ground_truth_csv)

        # Extract just the filename from the path
        image_filename = Path(image_path).name

        # Find the row for this image
        image_row = ground_truth_df[ground_truth_df["image_file"] == image_filename]

        if image_row.empty:
            rprint(f"[yellow]⚠️ No ground truth found for {image_filename}[/yellow]")
            return 0

        # Count transactions - use TRANSACTION_DATES column
        transaction_dates = image_row.iloc[0]["TRANSACTION_DATES"]

        if pd.isna(transaction_dates) or transaction_dates == "NOT_FOUND":
            # Try LINE_ITEM_DESCRIPTIONS as fallback for receipts/invoices
            line_items = image_row.iloc[0]["LINE_ITEM_DESCRIPTIONS"]
            if pd.isna(line_items) or line_items == "NOT_FOUND":
                return 0
            return len(line_items.split(" | "))

        # Count the number of dates (transactions)
        return len(transaction_dates.split(" | "))

    except Exception as e:
        rprint(f"[red]❌ Error reading ground truth: {e}[/red]")
        return 0
