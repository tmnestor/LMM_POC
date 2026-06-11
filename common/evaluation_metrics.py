"""
Evaluation and accuracy assessment utilities for model outputs - DOCUMENT AWARE REDUCTION.

This module handles ground truth comparison, accuracy calculations, and evaluation
reporting. It provides comprehensive metrics for assessing model performance
against known correct answers.

DOCUMENT AWARE REDUCTION COMPATIBILITY:
- Works with reduced field schemas (11 invoice/receipt, 5 bank statement)
- Dynamic evaluation based on field types from config.py (already updated)
- Ground truth loading automatically adapts to available fields
- No hardcoded field assumptions - fully flexible
"""

import json
import re
from pathlib import Path

import pandas as pd

from .field_schema import FieldSchema, get_field_schema

# Month name → number mapping used by _parse_single_date.
_MONTHS: dict[str, int] = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def _parse_single_date(date_str: str) -> tuple[int, int, int] | None:
    """Parse a single date string to a ``(day, month, year)`` tuple.

    Handles day-name prefixes (``"Thu 04 Sep 2025"``), month names,
    numeric separators (``/``, ``-``), and 2-digit year normalization.

    Returns ``None`` when the string cannot be parsed.
    """
    date_str = date_str.strip().lower()

    # Strip day name prefix (Mon, Tue, etc.)
    _day_names = [
        "mon",
        "tue",
        "wed",
        "thu",
        "fri",
        "sat",
        "sun",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]
    for day_name in _day_names:
        if date_str.startswith(day_name):
            date_str = date_str[len(day_name) :].strip()
            break

    # Extract all numbers
    nums = re.findall(r"\d+", date_str)

    # Check for month names
    month_num = None
    for month_name, month_val in _MONTHS.items():
        if month_name in date_str:
            month_num = month_val
            break

    if not nums:
        return None

    # Parse based on available information
    if len(nums) >= 3:
        # Full numeric date: DD/MM/YYYY or similar
        day, month, year = int(nums[0]), int(nums[1]), int(nums[2])
    elif len(nums) == 2 and month_num:
        # Date with month name: "28 June 2024"
        day, month, year = int(nums[0]), month_num, int(nums[1])
    elif len(nums) == 2:
        # Ambiguous - assume DD/MM or MM/YY
        day, month, year = int(nums[0]), int(nums[1]), 0
    else:
        return None

    # Normalize 2-digit years
    if year < 100:
        year = 2000 + year if year <= 50 else 1900 + year

    return (day, month, year)


def load_ground_truth(gt_path: str, show_sample: bool = False, verbose: bool = True) -> dict[str, dict]:
    """Load ground truth data from CSV or JSONL file.

    Detects file format by extension:
    - ``.jsonl``: One JSON object per line, each record carries only its
      type's fields (no cross-schema NOT_FOUNDs).
    - ``.csv``: Legacy pandas-based loader (all columns for all rows).

    Args:
        gt_path: Path to the ground truth file (.csv or .jsonl).
        show_sample: Whether to display a sample of the data.
        verbose: Whether to print loading messages.

    Returns:
        Dictionary mapping image filenames to ground truth data.

    Raises:
        FileNotFoundError: If ground truth file doesn't exist.
        ValueError: If file has invalid structure.
    """
    path = Path(gt_path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    if path.suffix == ".jsonl":
        return _load_ground_truth_jsonl(path, show_sample=show_sample, verbose=verbose)
    return _load_ground_truth_csv(str(path), show_sample=show_sample, verbose=verbose)


def _load_ground_truth_jsonl(
    path: Path, *, show_sample: bool = False, verbose: bool = True
) -> dict[str, dict]:
    """Load ground truth from JSONL (one JSON object per line).

    Each record must have a ``filename`` (or ``image_name``) key plus
    per-type schema fields.  Cross-schema NOT_FOUNDs should already be
    stripped during JSONL generation.
    """
    ground_truth_map: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            filename = record.get("filename") or record.get("image_name")
            if filename:
                ground_truth_map[str(filename)] = record

    if verbose:
        from collections import Counter

        types = Counter(r.get("DOCUMENT_TYPE", "UNKNOWN") for r in ground_truth_map.values())
        print(f"📊 Ground truth JSONL loaded: {len(ground_truth_map)} records")
        for doc_type, count in sorted(types.items()):
            sample = next(r for r in ground_truth_map.values() if r.get("DOCUMENT_TYPE") == doc_type)
            n_fields = len([k for k in sample if k not in ("filename", "image_name")])
            print(f"   {doc_type}: {count} records, {n_fields} fields")

    if show_sample and ground_truth_map and verbose:
        first = next(iter(ground_truth_map.values()))
        print(f"📄 Sample: {json.dumps(first, ensure_ascii=False)[:300]}...")

    return ground_truth_map


def _load_ground_truth_csv(
    csv_path: str, *, show_sample: bool = False, verbose: bool = True
) -> dict[str, dict]:
    """Load ground truth from CSV (legacy pandas-based loader)."""
    try:
        # CRITICAL: Use dtype=str to prevent pandas from converting "False" strings to bool False
        # This was causing type mismatch: extracted='False' (str) vs ground_truth=False (bool)
        ground_truth_df = pd.read_csv(csv_path, dtype=str)
        if verbose:
            print(
                f"📊 Ground truth CSV loaded with {len(ground_truth_df)} rows and {len(ground_truth_df.columns)} columns"
            )
            print(f"📋 Available columns: {list(ground_truth_df.columns)}")

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
        raise ValueError(f"No image identifier column found. Expected one of: {possible_names}")

    if verbose:
        print(f"✅ Using '{image_col}' as image identifier column")

    if show_sample and len(ground_truth_df) > 0 and verbose:
        print("📄 Sample ground truth data:")
        print(ground_truth_df.head(2).to_string(index=False))

    # Convert to dictionary mapping
    ground_truth_map = {}
    for _, row in ground_truth_df.iterrows():
        image_name = row[image_col]
        if pd.isna(image_name):
            continue
        ground_truth_map[str(image_name)] = row.to_dict()

    if verbose:
        print(f"✅ Ground truth mapping created for {len(ground_truth_map)} images")
    return ground_truth_map


# ============================================================================
# V4 FIELD TYPE EVALUATION HELPERS
# ============================================================================


def _parse_boolean_value(value: str) -> bool | None:
    """Parse boolean value from text string with strict matching."""
    if not value or value == "NOT_FOUND":
        return None

    # Convert to string first to handle boolean objects
    value_lower = str(value).lower().strip()

    # Boolean matching - accept common boolean representations
    true_values = ["true", "1", "yes", "y"]
    false_values = ["false", "0", "no", "n"]

    if value_lower in true_values:
        return True
    elif value_lower in false_values:
        return False
    else:
        return None


def _compare_monetary_values(extracted: str, ground_truth: str, debug: bool = False) -> float:
    """Compare monetary values with tolerance."""
    _tol = get_field_schema().get_threshold("monetary_tolerance", 0.01)
    try:
        extracted_num = float(re.sub(r"[^\d.-]", "", extracted))
        ground_truth_num = float(re.sub(r"[^\d.-]", "", ground_truth))

        # Allow configurable tolerance for rounding
        tolerance = abs(ground_truth_num * _tol) if ground_truth_num != 0 else _tol
        score = 1.0 if abs(extracted_num - ground_truth_num) <= tolerance else 0.0

        if debug:
            print(
                f"    💰 MONETARY: {extracted_num} vs {ground_truth_num} (tolerance: {tolerance}) = {score}"
            )
        return score
    except (ValueError, TypeError):
        if debug:
            print(f"    💰 MONETARY: Parse error - {extracted} vs {ground_truth}")
        return 0.0


def _transaction_item_matches(extracted_item: str, ground_truth_item: str, field_name: str) -> bool:
    """Check if individual transaction items match."""
    if "AMOUNT" in field_name:
        # Monetary comparison for transaction amounts
        return _compare_monetary_values(extracted_item, ground_truth_item, False) == 1.0
    elif "DATE" in field_name:
        # Date comparison for transaction dates
        return _compare_dates_fuzzy(extracted_item, ground_truth_item)
    elif "BALANCE" in field_name:
        # Monetary comparison for balances
        return _compare_monetary_values(extracted_item, ground_truth_item, False) == 1.0
    else:
        # Text comparison for descriptions
        return extracted_item.lower().strip() == ground_truth_item.lower().strip()


def _compare_date_field(extracted: str, ground_truth: str, field_name: str, debug: bool = False) -> float:
    """
    Compare date fields with semantic understanding.

    Handles:
    - Date ranges: "1 February 2024 - 28 June 2024" vs "01/02/2024 to 28/06/2024"
    - Single dates: "28 June 2024" vs "28/06/2024"
    - Month names vs numbers: "February" = "02"
    - Day name prefixes: "Tue 2 Apr 2024" = "02/04/2024"
    """

    def _parse_date_range(text: str) -> list[tuple[int, int, int]]:
        """Parse a date range string into list of (day, month, year) tuples."""
        separators = [" - ", " to ", " – ", " — ", "-"]
        dates: list[tuple[int, int, int]] = []

        for sep in separators:
            if sep in text.lower():
                parts = text.split(sep) if sep != "-" else re.split(r"\s*-\s*", text)
                parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 4]
                if len(parts) >= 2:
                    for part in parts[:2]:
                        parsed = _parse_single_date(part)
                        if parsed:
                            dates.append(parsed)
                    break

        if not dates:
            parsed = _parse_single_date(text)
            if parsed:
                dates.append(parsed)

        return dates

    # Parse both values
    extracted_dates = _parse_date_range(extracted)
    ground_truth_dates = _parse_date_range(ground_truth)

    if debug:
        print(f"    📅 DATE FIELD: {field_name}")
        print(f"       Extracted: '{extracted}' → {extracted_dates}")
        print(f"       Ground truth: '{ground_truth}' → {ground_truth_dates}")

    if not extracted_dates or not ground_truth_dates:
        if debug:
            print("       No dates parsed - score: 0.0")
        return 0.0

    # Compare dates (order-independent for ranges)
    extracted_set = set(extracted_dates)
    ground_truth_set = set(ground_truth_dates)

    if extracted_set == ground_truth_set:
        if debug:
            print("       Exact match - score: 1.0")
        return 1.0

    # Partial match - check overlap
    common = extracted_set & ground_truth_set
    if common:
        score = len(common) / max(len(extracted_set), len(ground_truth_set))
        if debug:
            print(f"       Partial match: {len(common)} common dates - score: {score}")
        return score

    # Check if dates are semantically close (same day/month, different format issues)
    # Compare first dates from each
    if extracted_dates and ground_truth_dates:
        e_day, e_month, e_year = extracted_dates[0]
        g_day, g_month, g_year = ground_truth_dates[0]

        # If day and month match (year might differ due to parsing issues)
        if e_day == g_day and e_month == g_month:
            if debug:
                print("       Day/month match (year differs) - score: 0.8")
            return 0.8

    if debug:
        print("       No match - score: 0.0")
    return 0.0


def _compare_dates_fuzzy(extracted_date: str, ground_truth_date: str) -> bool:
    """Fuzzy date comparison allowing for different formats."""
    if extracted_date.strip() == ground_truth_date.strip():
        return True

    # Month name to number mapping
    months = {
        "jan": "01",
        "january": "01",
        "feb": "02",
        "february": "02",
        "mar": "03",
        "march": "03",
        "apr": "04",
        "april": "04",
        "may": "05",
        "jun": "06",
        "june": "06",
        "jul": "07",
        "july": "07",
        "aug": "08",
        "august": "08",
        "sep": "09",
        "sept": "09",
        "september": "09",
        "oct": "10",
        "october": "10",
        "nov": "11",
        "november": "11",
        "dec": "12",
        "december": "12",
    }

    def normalize_date(date_str):
        """Extract day, month, year from date string, handling month names."""
        date_lower = date_str.lower()

        # Extract all numbers
        nums = re.findall(r"\d+", date_str)

        # Check for month names
        month_num = None
        for month_name, month_val in months.items():
            if month_name in date_lower:
                month_num = month_val
                break

        if not nums:
            return None

        # Try to extract day, month, year based on available information
        if len(nums) == 3:
            # Full date like DD/MM/YYYY or MM/DD/YYYY or YYYY/MM/DD
            day, month, year = nums[0], nums[1], nums[2]
        elif len(nums) == 2 and month_num:
            # Date with month name like "16-Jul-25"
            day, month, year = nums[0], month_num, nums[1]
        elif len(nums) == 2:
            # Ambiguous - assume day and year with month missing
            day, month, year = nums[0], None, nums[1]
        elif len(nums) == 1 and month_num:
            # Just day with month name
            day, month, year = nums[0], month_num, None
        else:
            return None

        # Normalize 2-digit years to 4-digit
        if year and len(year) == 2:
            year_int = int(year)
            # Assume 00-50 is 2000-2050, 51-99 is 1951-1999
            year = str(2000 + year_int) if year_int <= 50 else str(1900 + year_int)

        # Pad day and month to 2 digits
        if day:
            day = day.zfill(2)
        if month:
            month = month.zfill(2)

        return (day, month, year)

    extracted_parts = normalize_date(extracted_date)
    ground_truth_parts = normalize_date(ground_truth_date)

    if extracted_parts is None or ground_truth_parts is None:
        return False

    # Compare available components (day, month, year)
    # All non-None components must match
    for ext, gt in zip(extracted_parts, ground_truth_parts, strict=False):
        if ext is not None and gt is not None and ext != gt:
            return False

    # At least day and one other component must match
    matches = sum(
        1
        for ext, gt in zip(extracted_parts, ground_truth_parts, strict=False)
        if ext == gt and ext is not None
    )
    return matches >= 2


def calculate_correlation_aware_f1(
    extracted_data: dict,
    ground_truth_data: dict,
    document_type: str,
    debug: bool = False,
    *,
    fields: FieldSchema | None = None,
) -> dict:
    """
    Calculate F1 with cross-list correlation validation.

    This validates that related lists maintain semantic alignment across fields.
    Critical for transaction data where dates, descriptions, and amounts must
    correspond to the same transaction at the same index position.

    Args:
        extracted_data: Dict with extracted fields
        ground_truth_data: Dict with ground truth fields
        document_type: Type of document (determines which fields are related)
        debug: Whether to print debug information

    Returns:
        dict with standard_f1, alignment_score, combined_f1, field_f1_scores, alignment_details
    """
    # Define related field groups for each document type
    doc_type_lower = document_type.lower()

    related_groups: list[tuple[str, ...]] = []
    if "bank" in doc_type_lower or "statement" in doc_type_lower:
        related_groups = [("TRANSACTION_DATES", "LINE_ITEM_DESCRIPTIONS", "TRANSACTION_AMOUNTS_PAID")]
    elif "invoice" in doc_type_lower or "receipt" in doc_type_lower:
        related_groups = [
            (
                "LINE_ITEM_DESCRIPTIONS",
                "LINE_ITEM_QUANTITIES",
                "LINE_ITEM_PRICES",
                "LINE_ITEM_TOTAL_PRICES",
            )
        ]
    else:
        related_groups = []

    # Calculate standard F1 for each field (position-agnostic)
    field_f1_scores = {}
    for field in extracted_data.keys():
        if field in ground_truth_data:
            f1_metrics = calculate_field_accuracy_f1_position_agnostic(
                extracted_data.get(field, "NOT_FOUND"),
                ground_truth_data.get(field, "NOT_FOUND"),
                field,
                debug=False,
                fields=fields,
            )
            field_f1_scores[field] = f1_metrics["f1_score"]

    # Calculate alignment scores for related field groups
    alignment_scores = []

    for field_group in related_groups:
        # Parse all fields in the group into lists
        extracted_lists = {}
        ground_truth_lists = {}

        for field in field_group:
            ext_value = str(extracted_data.get(field, "NOT_FOUND"))
            gt_value = str(ground_truth_data.get(field, "NOT_FOUND"))

            # Skip if field is missing
            if ext_value == "NOT_FOUND" or gt_value == "NOT_FOUND":
                continue

            extracted_lists[field] = [i.strip() for i in ext_value.split("|") if i.strip()]
            ground_truth_lists[field] = [i.strip() for i in gt_value.split("|") if i.strip()]

        # Check alignment row-by-row (strict mode)
        if ground_truth_lists:
            min_len = min(len(lst) for lst in ground_truth_lists.values())
            aligned_rows = 0

            for i in range(min_len):
                # Check if all fields match at position i
                row_aligned = True
                for field in field_group:
                    if field in extracted_lists and field in ground_truth_lists:
                        if i < len(extracted_lists[field]):
                            # Use field-specific matching
                            if not _transaction_item_matches(
                                extracted_lists[field][i],
                                ground_truth_lists[field][i],
                                field,
                            ):
                                row_aligned = False
                                break
                        else:
                            row_aligned = False
                            break

                if row_aligned:
                    aligned_rows += 1

            # Alignment score for this group
            alignment_score = aligned_rows / min_len if min_len > 0 else 0.0
            alignment_scores.append(alignment_score)

            if debug:
                print(f"  Field Group {field_group}:")
                print(f"    Aligned rows: {aligned_rows}/{min_len}")
                print(f"    Alignment score: {alignment_score:.1%}")

    # Overall alignment score
    overall_alignment = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 1.0

    # Overall standard F1
    overall_f1 = sum(field_f1_scores.values()) / len(field_f1_scores) if field_f1_scores else 0.0

    # Combined score (weighted average)
    combined_f1 = (overall_f1 + overall_alignment) / 2

    if debug:
        print("\n📊 Correlation-Aware F1 Results:")
        print(f"  Standard F1:      {overall_f1:.1%}")
        print(f"  Alignment Score:  {overall_alignment:.1%}")
        print(f"  Combined F1:      {combined_f1:.1%}")

    return {
        "f1_score": combined_f1,
        "standard_f1": overall_f1,
        "alignment_score": overall_alignment,
        "combined_f1": combined_f1,
        "field_f1_scores": field_f1_scores,
        "alignment_details": alignment_scores,
        "precision": combined_f1,  # For compatibility
        "recall": combined_f1,  # For compatibility
        "tp": 0,  # Not applicable for correlation metric
        "fp": 0,
        "fn": 0,
    }


def calculate_field_accuracy_with_method(
    extracted_value: str,
    ground_truth_value: str,
    field_name: str,
    method: str = "order_aware_f1",
    debug: bool = False,
    extracted_data: dict | None = None,
    ground_truth_data: dict | None = None,
    document_type: str | None = None,
    *,
    fields: FieldSchema | None = None,
) -> dict:
    """
    Router function to calculate field accuracy using the specified evaluation method.

    Available methods:
        - 'order_aware_f1': Position-aware F1 (stricter - order matters) [DEFAULT]
        - 'f1': Position-agnostic F1 (lenient - only values matter)
        - 'kieval': KIEval correction cost metric (application-centric)
        - 'correlation': Correlation-Aware F1 (cross-list validation)

    Args:
        extracted_value: Value extracted by the model
        ground_truth_value: Expected correct value
        field_name: Name of the field being compared
        method: Evaluation method to use
        debug: Whether to print debug information
        extracted_data: Full extracted data dict (required for correlation method)
        ground_truth_data: Full ground truth dict (required for correlation method)
        document_type: Document type (required for correlation method)

    Returns:
        dict: Metrics dictionary (contents depend on method chosen)
    """
    if method == "correlation" or method == "correlation_aware_f1":
        # Correlation method requires full data dictionaries
        if extracted_data is None or ground_truth_data is None or document_type is None:
            # Fallback to order-aware F1 if full data not provided
            return calculate_field_accuracy_f1(
                extracted_value,
                ground_truth_value,
                field_name,
                debug,
                fields=fields,
            )
        # Return correlation metrics (calculated once for all fields)
        return calculate_correlation_aware_f1(
            extracted_data,
            ground_truth_data,
            document_type,
            debug,
            fields=fields,
        )
    elif method == "f1" or method == "position_agnostic_f1":
        return calculate_field_accuracy_f1_position_agnostic(
            extracted_value,
            ground_truth_value,
            field_name,
            debug,
            fields=fields,
        )
    elif method == "kieval":
        return calculate_field_accuracy_kieval(
            extracted_value,
            ground_truth_value,
            field_name,
            debug,
            fields=fields,
        )
    elif method == "order_aware_f1" or method == "position_aware_f1":
        return calculate_field_accuracy_f1(
            extracted_value,
            ground_truth_value,
            field_name,
            debug,
            fields=fields,
        )
    else:
        # Default to order-aware F1 (current implementation)
        return calculate_field_accuracy_f1(
            extracted_value,
            ground_truth_value,
            field_name,
            debug,
            fields=fields,
        )


def _fuzzy_text_match(text1: str, text2: str, threshold: float | None = None) -> bool:
    """
    Check if two text strings match using fuzzy word-based comparison.

    Args:
        text1: First text string
        text2: Second text string
        threshold: Minimum word overlap ratio (0.0-1.0) for match

    Returns:
        bool: True if texts match above threshold
    """
    if threshold is None:
        threshold = get_field_schema().get_threshold("fuzzy_text_match", 0.75)
    # Normalize and extract words
    words1 = set(text1.lower().strip().split())
    words2 = set(text2.lower().strip().split())

    # Handle empty cases
    if not words1 or not words2:
        return text1.lower().strip() == text2.lower().strip()

    # Calculate word overlap ratio
    intersection = words1 & words2
    union = words1 | words2

    if not union:
        return False

    similarity = len(intersection) / len(union)
    return similarity >= threshold


def calculate_field_accuracy_f1_position_agnostic(
    extracted_value: str,
    ground_truth_value: str,
    field_name: str,
    debug: bool = False,
    *,
    fields: FieldSchema | None = None,
) -> dict:
    """
    Calculate F1 score using POSITION-AGNOSTIC (set-based) matching.

    Items only need to match in value, regardless of position. This is more lenient
    than position-aware matching and is suitable when order doesn't matter.

    Example:
        Extracted:    ["apple", "banana", "cherry"]
        Ground Truth: ["banana", "apple", "cherry"]
        Result:       100% F1 (all items present, order doesn't matter)

    Args:
        extracted_value: Value extracted by the model
        ground_truth_value: Expected correct value
        field_name: Name of the field being compared
        debug: Whether to print debug information

    Returns:
        dict: F1 metrics (f1_score, precision, recall, tp, fp, fn)
    """
    if fields is None:
        fields = get_field_schema()

    # Convert to strings and clean
    extracted = str(extracted_value).strip() if extracted_value else "NOT_FOUND"
    ground_truth = str(ground_truth_value).strip() if ground_truth_value else "NOT_FOUND"

    # Handle NOT_FOUND cases
    if ground_truth.upper() == "NOT_FOUND":
        is_correct = extracted.upper() == "NOT_FOUND"
        return {
            "f1_score": 1.0 if is_correct else 0.0,
            "precision": 1.0 if is_correct else 0.0,
            "recall": 1.0,
            "tp": 0,
            "fp": 0 if is_correct else 1,
            "fn": 0,
        }

    if extracted.upper() == "NOT_FOUND":
        gt_items = [i.strip() for i in str(ground_truth).split("|") if i.strip()]
        return {
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": len(gt_items) if gt_items else 1,
        }

    # Handle single values (non-list fields) - same as position-aware
    if "|" not in str(extracted) and "|" not in str(ground_truth):
        if field_name in fields.transaction_list_fields:
            match = _transaction_item_matches(extracted, ground_truth, field_name)
            return {
                "f1_score": 1.0 if match else 0.0,
                "precision": 1.0 if match else 0.0,
                "recall": 1.0 if match else 0.0,
                "tp": 1 if match else 0,
                "fp": 0 if match else 1,
                "fn": 0 if match else 1,
            }
        else:
            # Use fuzzy text matching
            match = _fuzzy_text_match(extracted, ground_truth)
            return {
                "f1_score": 1.0 if match else 0.0,
                "precision": 1.0 if match else 0.0,
                "recall": 1.0 if match else 0.0,
                "tp": 1 if match else 0,
                "fp": 0 if match else 1,
                "fn": 0 if match else 1,
            }

    # Handle list values - POSITION-AGNOSTIC (set-based matching)
    extracted_items = [i.strip() for i in str(extracted).split("|") if i.strip()]
    ground_truth_items = [i.strip() for i in str(ground_truth).split("|") if i.strip()]

    # True Positives: Count extracted items that match any ground truth item
    tp = 0
    matched_gt_indices = set()

    for ext_item in extracted_items:
        for i, gt_item in enumerate(ground_truth_items):
            if i not in matched_gt_indices:
                if field_name in fields.transaction_list_fields:
                    match = _transaction_item_matches(ext_item, gt_item, field_name)
                else:
                    match = _fuzzy_text_match(ext_item, gt_item)

                if match:
                    tp += 1
                    matched_gt_indices.add(i)
                    break

    # False Positives: Extracted items that don't match any ground truth
    fp = len(extracted_items) - tp

    # False Negatives: Ground truth items that weren't matched
    fn = len(ground_truth_items) - tp

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    if debug:
        print(f"  📊 F1 Metrics (Position-Agnostic) for {field_name}:")
        print(f"     TP={tp}, FP={fp}, FN={fn}")
        print(f"     Precision={precision:.2%}, Recall={recall:.2%}, F1={f1_score:.2%}")

    return {
        "f1_score": f1_score,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def calculate_field_accuracy_kieval(
    extracted_value: str,
    ground_truth_value: str,
    field_name: str,
    debug: bool = False,
    *,
    fields: FieldSchema | None = None,
) -> dict:
    """
    Calculate KIEval score based on correction costs.

    KIEval focuses on "How much effort to fix extraction?" rather than just accuracy.
    It differentiates error types: substitution, addition, deletion.

    Args:
        extracted_value: Value extracted by the model
        ground_truth_value: Expected correct value
        field_name: Name of the field being compared
        debug: Whether to print debug information

    Returns:
        dict: KIEval metrics (score, substitution, addition, deletion, total_error)
    """
    # First get F1 metrics (position-agnostic) to get TP/FP/FN counts
    f1_metrics = calculate_field_accuracy_f1_position_agnostic(
        extracted_value,
        ground_truth_value,
        field_name,
        debug=False,
        fields=fields,
    )

    tp = f1_metrics["tp"]
    fp = f1_metrics["fp"]
    fn = f1_metrics["fn"]

    # Calculate correction operations
    substitution = min(fp, fn)  # Items requiring value edits
    addition = fn - substitution  # Missing items to add
    deletion = fp - substitution  # Extra items to delete

    total_error = substitution + addition + deletion

    # Count total items for normalization
    extracted_items = (
        [i.strip() for i in str(extracted_value).split("|") if i.strip()]
        if "|" in str(extracted_value)
        else ([str(extracted_value).strip()] if str(extracted_value).strip() != "NOT_FOUND" else [])
    )

    ground_truth_items = (
        [i.strip() for i in str(ground_truth_value).split("|") if i.strip()]
        if "|" in str(ground_truth_value)
        else ([str(ground_truth_value).strip()] if str(ground_truth_value).strip() != "NOT_FOUND" else [])
    )

    total_items = max(len(extracted_items), len(ground_truth_items))

    # KIEval score: 1.0 - (correction_cost / total_items)
    score = 1.0 - (total_error / total_items) if total_items > 0 else 0.0

    if debug:
        print(f"  🔧 KIEval Metrics for {field_name}:")
        print(f"     Substitution: {substitution} (items to edit)")
        print(f"     Addition: {addition} (items to add)")
        print(f"     Deletion: {deletion} (items to delete)")
        print(f"     Total Error: {total_error}")
        print(f"     Score: {score:.2%}")

    return {
        "score": score,
        "f1_score": score,  # Alias for compatibility
        "precision": f1_metrics["precision"],
        "recall": f1_metrics["recall"],
        "substitution": substitution,
        "addition": addition,
        "deletion": deletion,
        "total_error": total_error,
        "total_items": total_items,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def calculate_field_accuracy_f1(
    extracted_value: str,
    ground_truth_value: str,
    field_name: str,
    debug: bool = False,
    *,
    fields: FieldSchema | None = None,
) -> dict:
    """
    Calculate F1-based accuracy for a field with proper false positive handling.

    This is the POSITION-AWARE (order-aware) F1 implementation.
    Items must match both in value AND position. This is stricter than
    position-agnostic F1 and is used when order matters (e.g., transactions).

    Example:
        Extracted:    ["apple", "banana", "cherry"]
        Ground Truth: ["banana", "apple", "cherry"]
        Result:       33.3% F1 (only position 2 matches)

    This function uses Precision, Recall, and F1 Score to evaluate list extractions,
    properly penalizing both false positives (over-extraction) and false negatives
    (under-extraction).

    Args:
        extracted_value (str): Value extracted by the model
        ground_truth_value (str): Expected correct value
        field_name (str): Name of the field being compared
        debug (bool): Whether to print debug information

    Returns:
        dict: Dictionary with keys:
            - f1_score (float): F1 score (0.0 to 1.0)
            - precision (float): Precision (0.0 to 1.0)
            - recall (float): Recall (0.0 to 1.0)
            - tp (int): True positives count
            - fp (int): False positives count
            - fn (int): False negatives count
    """
    if fields is None:
        fields = get_field_schema()
    _tol = get_field_schema().get_threshold("monetary_tolerance", 0.01)

    # Convert to strings and clean
    extracted = str(extracted_value).strip() if extracted_value else "NOT_FOUND"
    ground_truth = str(ground_truth_value).strip() if ground_truth_value else "NOT_FOUND"

    # Handle NOT_FOUND cases
    if ground_truth.upper() == "NOT_FOUND":
        is_correct = extracted.upper() == "NOT_FOUND"
        return {
            "f1_score": 1.0 if is_correct else 0.0,
            "precision": 1.0 if is_correct else 0.0,
            "recall": 1.0,
            "tp": 0,
            "fp": 0 if is_correct else 1,
            "fn": 0,
        }

    if extracted.upper() == "NOT_FOUND":
        # Missing extraction - all ground truth items are false negatives
        gt_items = [i.strip() for i in str(ground_truth).split("|") if i.strip()]
        return {
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": len(gt_items) if gt_items else 1,
        }

    # Handle single values (non-list fields)
    if "|" not in str(extracted) and "|" not in str(ground_truth):
        # Use existing comparison logic for single values
        if field_name in fields.transaction_list_fields:
            match = _transaction_item_matches(extracted, ground_truth, field_name)
            # For transaction fields, keep binary matching
            return {
                "f1_score": 1.0 if match else 0.0,
                "precision": 1.0 if match else 0.0,
                "recall": 1.0 if match else 0.0,
                "tp": 1 if match else 0,
                "fp": 0 if match else 1,
                "fn": 0 if match else 1,
            }
        else:
            # Normalize whitespace first
            extracted_normalized = " ".join(extracted.split())
            ground_truth_normalized = " ".join(ground_truth.split())

            # For boolean fields (IS_GST_INCLUDED), use case-insensitive boolean comparison
            if field_name in fields.boolean_fields:
                if debug:
                    print(f"🔵 BOOLEAN FIELD DETECTED: {field_name}")
                    print(f"   Extracted: {extracted_normalized}")
                    print(f"   Ground truth: {ground_truth_normalized}")

                # Parse both values to boolean (handles "true"/"True"/"TRUE", "false"/"False"/"FALSE")
                extracted_bool = _parse_boolean_value(extracted_normalized)
                ground_truth_bool = _parse_boolean_value(ground_truth_normalized)

                if debug:
                    print(f"   Parsed extracted: {extracted_bool}")
                    print(f"   Parsed ground truth: {ground_truth_bool}")

                # Business logic: If ground truth is NOT_FOUND and extracted is "false", that's correct
                # Rationale: No GST field on document means IS_GST_INCLUDED = false
                if ground_truth.upper() == "NOT_FOUND" and extracted_bool is False:
                    match = True
                elif extracted_bool is not None and ground_truth_bool is not None:
                    match = extracted_bool == ground_truth_bool
                else:
                    match = False

                if debug:
                    print(f"   Match: {match}")

                return {
                    "f1_score": 1.0 if match else 0.0,
                    "precision": 1.0 if match else 0.0,
                    "recall": 1.0 if match else 0.0,
                    "tp": 1 if match else 0,
                    "fp": 0 if match else 1,
                    "fn": 0 if match else 1,
                }

            # For date fields, use semantic date comparison that handles:
            # - Date ranges: "1 February 2024 - 28 June 2024" vs "01/02/2024 to 28/06/2024"
            # - Single dates: "28 June 2024" vs "28/06/2024"
            # - Month names vs numbers
            # - Day name prefixes (Tue, Mon, etc.)
            date_field_keywords = ["DATE", "DUE_DATE", "INVOICE_DATE", "STATEMENT_DATE"]
            is_date_field = any(keyword in field_name.upper() for keyword in date_field_keywords)

            if is_date_field:
                # Use semantic date comparison that handles ranges and format variations
                score = _compare_date_field(
                    extracted_normalized, ground_truth_normalized, field_name, debug
                )
                return {
                    "f1_score": score,
                    "precision": score,
                    "recall": score,
                    "tp": 1 if score > 0 else 0,
                    "fp": 0 if score > 0 else 1,
                    "fn": 0 if score > 0 else 1,
                }

            # For single-value monetary fields (GST_AMOUNT, TOTAL_AMOUNT), use monetary comparison with F1-style penalty
            # This ensures incorrect amounts get 0.0 score (penalizing false positives)
            # NOTE: List fields like LINE_ITEM_PRICES are handled later by the list F1 logic
            monetary_single_fields = [
                "GST_AMOUNT",
                "TOTAL_AMOUNT",
                "INVOICE_TOTAL",
                "SUBTOTAL",
            ]
            is_monetary_field = field_name in monetary_single_fields

            if is_monetary_field:
                try:
                    extracted_num = float(re.sub(r"[^\d.-]", "", extracted_normalized))
                    ground_truth_num = float(re.sub(r"[^\d.-]", "", ground_truth_normalized))

                    # Allow configurable tolerance for rounding
                    tolerance = abs(ground_truth_num * _tol) if ground_truth_num != 0 else _tol
                    match = abs(extracted_num - ground_truth_num) <= tolerance

                    return {
                        "f1_score": 1.0 if match else 0.0,
                        "precision": 1.0 if match else 0.0,
                        "recall": 1.0 if match else 0.0,
                        "tp": 1 if match else 0,
                        "fp": 0 if match else 1,
                        "fn": 0 if match else 1,
                    }
                except (ValueError, TypeError):
                    # Parse error - treat as mismatch
                    return {
                        "f1_score": 0.0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "tp": 0,
                        "fp": 1,
                        "fn": 1,
                    }

            # For ID fields (ABN, invoice numbers, etc.), require exact match
            # These are critical identifiers where fuzzy matching is inappropriate
            id_field_keywords = ["ABN", "NUMBER", "ID", "REFERENCE", "BSB"]
            is_id_field = any(keyword in field_name.upper() for keyword in id_field_keywords)

            if is_id_field:
                # ID fields require exact match (no fuzzy matching)
                # ROBUST NORMALIZATION: Handle variations in both extracted and ground truth
                # Step 1: Remove field label prefixes (ABN, ABN:, BSB, BSB:, etc.)
                id_label_pattern = r"^(ABN|BSB|ACN|GST|TAX|ID|NUMBER|NO\.?|#)\s*:?\s*"
                extracted_clean = re.sub(id_label_pattern, "", extracted_normalized, flags=re.IGNORECASE)
                ground_truth_clean = re.sub(
                    id_label_pattern, "", ground_truth_normalized, flags=re.IGNORECASE
                )

                # Step 2: Remove ALL spaces, dashes, and formatting
                extracted_clean = re.sub(r"[\s\-]", "", extracted_clean)
                ground_truth_clean = re.sub(r"[\s\-]", "", ground_truth_clean)

                # Step 3: Case-insensitive comparison
                if extracted_clean.lower() == ground_truth_clean.lower():
                    f1_score = 1.0
                else:
                    f1_score = 0.0
            else:
                # For text fields (addresses, names), use fuzzy matching with Levenshtein distance
                try:
                    from Levenshtein import distance as levenshtein_distance

                    # Calculate normalized similarity (ANLS-style)
                    edit_dist = levenshtein_distance(
                        extracted_normalized.lower(), ground_truth_normalized.lower()
                    )
                    max_len = max(len(extracted_normalized), len(ground_truth_normalized))

                    if max_len == 0:
                        similarity = 1.0
                    else:
                        similarity = 1.0 - (edit_dist / max_len)

                    # Apply 0.5 threshold like ANLS (standard in DocVQA)
                    # Below 50% similarity = 0.0, above = give partial credit
                    if similarity >= 0.5:
                        f1_score = similarity
                    else:
                        f1_score = 0.0

                except ImportError:
                    # Fallback to exact match if Levenshtein not installed
                    if extracted_normalized.lower() == ground_truth_normalized.lower():
                        f1_score = 1.0
                    else:
                        f1_score = 0.0

            # For text fields, precision = recall = f1 (single value)
            return {
                "f1_score": f1_score,
                "precision": f1_score,
                "recall": f1_score,
                "tp": 1 if f1_score > 0.5 else 0,
                "fp": 0 if f1_score > 0.5 else 1,
                "fn": 0 if f1_score > 0.5 else 1,
            }

    # Handle list values (transaction fields)
    extracted_items = [i.strip() for i in str(extracted).split("|") if i.strip()]
    ground_truth_items = [i.strip() for i in str(ground_truth).split("|") if i.strip()]

    # POSITION-AWARE MATCHING: Items must be in correct positions
    # This penalizes order errors (e.g., reversed lists)
    tp = 0
    fp = 0
    fn = 0

    # Compare position-by-position
    max_len = max(len(extracted_items), len(ground_truth_items))

    for i in range(max_len):
        if i < len(ground_truth_items) and i < len(extracted_items):
            # Both have an item at this position - check if they match
            if field_name in fields.transaction_list_fields:
                match = _transaction_item_matches(extracted_items[i], ground_truth_items[i], field_name)
            else:
                # Use fuzzy text matching (threshold from field_definitions.yaml)
                # This allows "EATS Sydney" to match "UBER EATS Sydney" (0.80 similarity)
                match = _fuzzy_text_match(extracted_items[i], ground_truth_items[i])

            if match:
                tp += 1
            else:
                # Substitution error: Wrong item at this position counts as 1 FN only
                # (We expected GT item but got wrong extraction - that's a false negative)
                # DO NOT also count FP - that would double-penalize a single mistake
                fn += 1
        elif i < len(ground_truth_items):
            # Ground truth has item but extraction doesn't (missing)
            fn += 1
        else:
            # Extraction has item but ground truth doesn't (extra)
            fp += 1

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    if debug:
        print(f"  📊 F1 Metrics for {field_name}:")
        print(f"     TP={tp}, FP={fp}, FN={fn}")
        print(f"     Precision={precision:.2%}, Recall={recall:.2%}, F1={f1_score:.2%}")
        print(f"     Extracted items: {len(extracted_items)}")
        print(f"     Ground truth items: {len(ground_truth_items)}")

    return {
        "f1_score": f1_score,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
