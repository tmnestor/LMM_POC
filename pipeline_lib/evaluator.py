"""
Lightweight evaluator module for llama_batch_pipeline.

Extracts only load_ground_truth and calculate_field_accuracy from common/evaluation_metrics.py.
Simplified implementation without heavy config dependencies - self-contained.
"""

import re
from pathlib import Path
from typing import Dict

import pandas as pd


def load_ground_truth(
    csv_path: str, show_sample: bool = False, verbose: bool = True
) -> Dict[str, Dict]:
    """
    Load ground truth data from CSV file.

    Args:
        csv_path: Path to the ground truth CSV file
        show_sample: Whether to display a sample of the data
        verbose: Whether to print loading messages

    Returns:
        dict: Dictionary mapping image filenames to ground truth data

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV has invalid structure
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Ground truth file not found: {csv_path}")

    try:
        ground_truth_df = pd.read_csv(csv_path)
        if verbose:
            print(
                f"📊 Ground truth CSV loaded with {len(ground_truth_df)} rows and {len(ground_truth_df.columns)} columns"
            )

    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}") from e

    # Find image identifier column
    image_col = None
    possible_names = ["image_file", "filename", "image_name", "file"]
    for col in possible_names:
        if col in ground_truth_df.columns:
            image_col = col
            break

    if image_col is None:
        raise ValueError(
            f"No image identifier column found. Expected one of: {possible_names}"
        )

    if verbose:
        print(f"✅ Using '{image_col}' as image identifier column")

    if show_sample and len(ground_truth_df) > 0 and verbose:
        print("📄 Sample ground truth data:")
        print(ground_truth_df.head(2).to_string(index=False))

    # Convert to dictionary mapping with stem normalization
    # Normalize keys to stems (remove extensions) for flexible matching
    # This allows "image_001", "image_001.png", and "image_001.jpeg" to all match
    ground_truth_map = {}
    for _, row in ground_truth_df.iterrows():
        image_name = row[image_col]
        if pd.isna(image_name):
            continue
        # Normalize to stem (filename without extension)
        stem_key = Path(str(image_name)).stem
        ground_truth_map[stem_key] = row.to_dict()

    if verbose:
        print(f"✅ Ground truth mapping created for {len(ground_truth_map)} images")
    return ground_truth_map


def calculate_field_accuracy(
    extracted_value: str, ground_truth_value: str, field_name: str, debug: bool = False
) -> float:
    """
    Calculate accuracy for a single field comparison.

    Simplified version for pipeline use - handles common field types without
    heavy config dependencies.

    Args:
        extracted_value: Value extracted by the model
        ground_truth_value: Expected correct value
        field_name: Name of the field being compared
        debug: Whether to print debug information

    Returns:
        float: Accuracy score (0.0 to 1.0)
    """
    # Convert to strings and clean
    extracted = str(extracted_value).strip() if extracted_value else "NOT_FOUND"
    ground_truth = (
        str(ground_truth_value).strip() if ground_truth_value else "NOT_FOUND"
    )

    if debug:
        print(f"    🔍 DEBUG FIELD {field_name}: '{extracted}' vs '{ground_truth}'")

    # Handle missing values
    extracted_is_missing = extracted.upper() == "NOT_FOUND"
    ground_truth_is_missing = ground_truth.upper() == "NOT_FOUND"

    if extracted_is_missing and ground_truth_is_missing:
        return 1.0

    if extracted_is_missing != ground_truth_is_missing:
        return 0.0

    # Normalize for comparison
    extracted_lower = extracted.lower()
    ground_truth_lower = ground_truth.lower()

    # Normalize pipes to spaces for text fields
    if not any(
        pattern in field_name
        for pattern in ["AMOUNT", "PRICE", "QUANTITY", "ABN", "BSB", "NUMBER"]
    ):
        extracted_lower = extracted_lower.replace("|", " ")
        ground_truth_lower = ground_truth_lower.replace("|", " ")
        extracted_lower = " ".join(extracted_lower.split())
        ground_truth_lower = " ".join(ground_truth_lower.split())

    # Create normalized versions with formatting removed
    extracted_normalized = extracted_lower
    ground_truth_normalized = ground_truth_lower
    for char in [",", "$", "%", "(", ")", " "]:
        extracted_normalized = extracted_normalized.replace(char, "")
        ground_truth_normalized = ground_truth_normalized.replace(char, "")

    # Exact match after normalization
    if extracted_normalized == ground_truth_normalized:
        if debug:
            print("    ✅ Exact match - score: 1.0")
        return 1.0

    # DOCUMENT_TYPE special handling
    if field_name == "DOCUMENT_TYPE":
        type_mapping = {
            "invoice": "invoice",
            "tax invoice": "invoice",
            "receipt": "receipt",
            "bank statement": "bank_statement",
            "statement": "bank_statement",
        }
        extracted_canonical = type_mapping.get(extracted_lower, extracted_lower)
        ground_truth_canonical = type_mapping.get(ground_truth_lower, ground_truth_lower)

        return 1.0 if extracted_canonical == ground_truth_canonical else 0.0

    # Monetary fields
    if any(pattern in field_name for pattern in ["AMOUNT", "PRICE", "TOTAL", "GST"]):
        try:
            extracted_num = float(re.sub(r"[^\d.-]", "", extracted))
            ground_truth_num = float(re.sub(r"[^\d.-]", "", ground_truth))
            tolerance = abs(ground_truth_num * 0.01) if ground_truth_num != 0 else 0.01
            return 1.0 if abs(extracted_num - ground_truth_num) <= tolerance else 0.0
        except (ValueError, AttributeError):
            return 0.0

    # Numeric IDs (ABN, BSB, etc)
    if any(pattern in field_name for pattern in ["ABN", "BSB", "NUMBER", "ID"]):
        extracted_digits = re.sub(r"\D", "", extracted)
        ground_truth_digits = re.sub(r"\D", "", ground_truth)
        return 1.0 if extracted_digits == ground_truth_digits else 0.0

    # Date fields
    if "DATE" in field_name:
        extracted_numbers = re.findall(r"\d+", extracted)
        ground_truth_numbers = re.findall(r"\d+", ground_truth)

        if set(extracted_numbers) == set(ground_truth_numbers):
            return 1.0

        common = set(extracted_numbers) & set(ground_truth_numbers)
        if common and len(common) >= 2:
            return 0.8

        return 0.0

    # List fields
    if any(
        pattern in field_name
        for pattern in [
            "LINE_ITEM",
            "TRANSACTION",
            "DESCRIPTIONS",
            "QUANTITIES",
            "PRICES",
        ]
    ):
        extracted_items = [
            item.strip() for item in re.split(r"[,;|\n]", extracted) if item.strip()
        ]
        ground_truth_items = [
            item.strip()
            for item in re.split(r"[,;|\n]", ground_truth)
            if item.strip()
        ]

        if not extracted_items and not ground_truth_items:
            return 1.0

        if not extracted_items or not ground_truth_items:
            return 0.0

        # Normalize items
        extracted_items_norm = [item.lower().replace(" ", "") for item in extracted_items]
        ground_truth_items_norm = [
            item.lower().replace(" ", "") for item in ground_truth_items
        ]

        # Calculate overlap
        matches = sum(
            1
            for item in extracted_items_norm
            if item in ground_truth_items_norm
        )

        # Score based on precision and recall
        precision = matches / len(extracted_items) if extracted_items else 0.0
        recall = matches / len(ground_truth_items) if ground_truth_items else 0.0

        if precision == 1.0 and recall == 1.0:
            return 1.0
        elif precision >= 0.8 and recall >= 0.8:
            return 0.9
        elif precision >= 0.6 and recall >= 0.6:
            return 0.7
        elif matches > 0:
            return 0.5
        else:
            return 0.0

    # Boolean fields
    if "GST_INCLUDED" in field_name or "IS_" in field_name:
        bool_values = {
            "yes": True,
            "no": False,
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        }
        extracted_bool = bool_values.get(extracted_lower, None)
        ground_truth_bool = bool_values.get(ground_truth_lower, None)

        if extracted_bool is not None and ground_truth_bool is not None:
            return 1.0 if extracted_bool == ground_truth_bool else 0.0

    # Default: case-insensitive substring matching
    if extracted_lower in ground_truth_lower or ground_truth_lower in extracted_lower:
        return 0.8

    return 0.0
