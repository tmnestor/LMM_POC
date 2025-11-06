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

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import (
    get_all_field_types,
    get_boolean_fields,
    get_calculated_fields,
    get_list_fields,
    get_monetary_fields,
    get_phone_fields,
    get_transaction_list_fields,
)


def load_ground_truth(
    csv_path: str, show_sample: bool = False, verbose: bool = True
) -> Dict[str, Dict]:
    """
    Load ground truth data from CSV file.

    Args:
        csv_path (str): Path to the ground truth CSV file
        show_sample (bool): Whether to display a sample of the data
        verbose (bool): Whether to print loading messages

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
        raise ValueError(
            f"No image identifier column found. Expected one of: {possible_names}"
        )

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


def calculate_field_accuracy(
    extracted_value: str, ground_truth_value: str, field_name: str, debug=False
) -> float:
    """
    Calculate accuracy for a single field comparison with partial credit scoring.

    This function handles different types of fields with appropriate comparison
    methods (exact match, numeric comparison, date parsing, etc.) and returns
    float scores from 0.0 to 1.0 to allow partial credit for fuzzy matches.

    Args:
        extracted_value (str): Value extracted by the model
        ground_truth_value (str): Expected correct value
        field_name (str): Name of the field being compared
        debug (bool): Whether to print debug information

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

    # Handle missing value indicator - both should use 'NOT_FOUND' now
    extracted_is_missing = extracted.upper() == "NOT_FOUND"
    ground_truth_is_missing = ground_truth.upper() == "NOT_FOUND"

    # Both are NOT_FOUND - correct
    if extracted_is_missing and ground_truth_is_missing:
        if debug:
            print("    ✅ Both NOT_FOUND - score: 1.0")
        return 1.0

    # One is NOT_FOUND but not the other - incorrect
    if extracted_is_missing != ground_truth_is_missing:
        if debug:
            print(
                f"    ❌ One NOT_FOUND, other not ('{extracted}' vs '{ground_truth}') - score: 0.0"
            )
        return 0.0

    # Normalize for comparison
    extracted_lower = extracted.lower()
    ground_truth_lower = ground_truth.lower()

    # Normalize pipes to spaces for text fields (handles multi-line addresses/names in ground truth)
    # This allows "123 Main St | Sydney NSW" to match "123 Main St Sydney NSW"
    field_types = get_all_field_types()
    if field_types.get(field_name) == "text":
        extracted_lower = extracted_lower.replace("|", " ")
        ground_truth_lower = ground_truth_lower.replace("|", " ")
        # Clean up multiple consecutive spaces from pipe replacement
        extracted_lower = " ".join(extracted_lower.split())
        ground_truth_lower = " ".join(ground_truth_lower.split())

    # For exact match checking, create normalized versions with formatting removed
    extracted_normalized = extracted_lower
    ground_truth_normalized = ground_truth_lower
    # Remove common formatting
    for char in [",", "$", "%", "(", ")", " "]:
        extracted_normalized = extracted_normalized.replace(char, "")
        ground_truth_normalized = ground_truth_normalized.replace(char, "")

    if debug:
        print(f"    🔍 Normalized: '{extracted_normalized}' vs '{ground_truth_normalized}'")

    # Exact match after normalization
    if extracted_normalized == ground_truth_normalized:
        if debug:
            print("    ✅ Exact match - score: 1.0")
        return 1.0

    # Special handling for DOCUMENT_TYPE field with canonical type mapping
    if field_name == "DOCUMENT_TYPE":
        # Define canonical document type mappings (same as detection system)
        type_mapping = {
            "invoice": "invoice",
            "tax invoice": "invoice",
            "estimate": "invoice",
            "quote": "invoice",
            "quotation": "invoice",
            "proforma invoice": "invoice",
            "receipt": "receipt",
            "purchase receipt": "receipt",
            "sales receipt": "receipt",
            "bank statement": "bank_statement",
            "account statement": "bank_statement",
            "credit card statement": "bank_statement",
            "statement": "bank_statement"
        }

        # Map both values to canonical types
        extracted_canonical = type_mapping.get(extracted_lower, extracted_lower)
        ground_truth_canonical = type_mapping.get(ground_truth_lower, ground_truth_lower)

        # Compare canonical types
        if extracted_canonical == ground_truth_canonical:
            if debug:
                print(f"    📋 DOCUMENT_TYPE: '{extracted}' ({extracted_canonical}) matches '{ground_truth}' ({ground_truth_canonical}) - score: 1.0")
            return 1.0
        else:
            if debug:
                print(f"    📋 DOCUMENT_TYPE: '{extracted}' ({extracted_canonical}) != '{ground_truth}' ({ground_truth_canonical}) - score: 0.0")
            return 0.0

    # Field-specific comparison logic using centralized field type definitions
    field_types = get_all_field_types()
    if field_types.get(field_name) == "numeric_id":
        # Numeric identifiers - exact match required
        extracted_digits = re.sub(r"\D", "", extracted)
        ground_truth_digits = re.sub(r"\D", "", ground_truth)
        score = 1.0 if extracted_digits == ground_truth_digits else 0.0
        if debug:
            print(
                f"    🔢 NUMERIC_ID: '{extracted_digits}' vs '{ground_truth_digits}' = {score}"
            )
        return score

    elif field_name in get_monetary_fields():
        # Monetary values - numeric comparison
        try:
            extracted_num = float(re.sub(r"[^\d.-]", "", extracted))
            ground_truth_num = float(re.sub(r"[^\d.-]", "", ground_truth))
            # Allow 1% tolerance for rounding
            tolerance = abs(ground_truth_num * 0.01) if ground_truth_num != 0 else 0.01
            score = 1.0 if abs(extracted_num - ground_truth_num) <= tolerance else 0.0
            if debug:
                print(
                    f"    💰 MONETARY: {extracted_num} vs {ground_truth_num} (tolerance: {tolerance}) = {score}"
                )
            return score
        except (ValueError, AttributeError):
            if debug:
                print("    💰 MONETARY: Parsing failed - score: 0.0")
            return 0.0

    elif field_name in get_phone_fields():
        # Phone number fields - digit-based with partial matching for OCR errors
        extracted_digits = re.sub(r"\D", "", extracted)
        ground_truth_digits = re.sub(r"\D", "", ground_truth)

        if extracted_digits == ground_truth_digits:
            score = 1.0
        elif len(extracted_digits) == len(ground_truth_digits):
            # Same length - check how many digits match
            matches = sum(
                1
                for e, g in zip(extracted_digits, ground_truth_digits, strict=False)
                if e == g
            )
            match_ratio = matches / len(ground_truth_digits)
            # Give partial credit for phone numbers with mostly correct digits (OCR tolerance)
            score = 0.8 if match_ratio >= 0.8 else (0.5 if match_ratio >= 0.6 else 0.0)
        else:
            score = 0.0

        if debug:
            print(
                f"    📞 PHONE: '{extracted_digits}' vs '{ground_truth_digits}' = {score}"
            )
        return score

    elif field_types.get(field_name) == "date":
        # Date fields - flexible matching
        # Extract date components
        extracted_numbers = re.findall(r"\d+", extracted)
        ground_truth_numbers = re.findall(r"\d+", ground_truth)

        # Check if same date components are present
        if set(extracted_numbers) == set(ground_truth_numbers):
            if debug:
                print(
                    f"    📅 DATE: Components match {extracted_numbers} = {ground_truth_numbers} - score: 1.0"
                )
            return 1.0

        # Partial match for dates
        common = set(extracted_numbers) & set(ground_truth_numbers)
        if common and len(common) >= 2:  # At least month and day match
            if debug:
                print(f"    📅 DATE: Partial match {common} - score: 0.8")
            return 0.8

        if debug:
            print(
                f"    📅 DATE: No match {extracted_numbers} vs {ground_truth_numbers} - score: 0.0"
            )
        return 0.0

    elif field_name in get_list_fields():
        # List fields - check overlap
        # These fields may contain multiple items
        extracted_items = [
            item.strip() for item in re.split(r"[,;|\n]", extracted) if item.strip()
        ]
        ground_truth_items = [
            item.strip() for item in re.split(r"[,;|\n]", ground_truth) if item.strip()
        ]

        if not ground_truth_items:
            score = 1.0 if not extracted_items else 0.0
            if debug:
                print(
                    f"    📋 LIST: Empty GT, extracted empty: {not extracted_items} - score: {score}"
                )
            return score

        # Calculate overlap
        matches = sum(
            1
            for item in extracted_items
            if any(
                item.lower() in gt_item.lower() or gt_item.lower() in item.lower()
                for gt_item in ground_truth_items
            )
        )

        score = (
            matches / max(len(ground_truth_items), len(extracted_items))
            if ground_truth_items
            else 0.0
        )
        if debug:
            print(
                f"    📋 LIST: {matches}/{max(len(ground_truth_items), len(extracted_items))} matches - score: {score}"
            )
        return score

    elif field_name in get_boolean_fields():
        # Boolean fields - exact match for true/false values
        extracted_bool = _parse_boolean_value(extracted)
        ground_truth_bool = _parse_boolean_value(ground_truth)

        if extracted_bool is None or ground_truth_bool is None:
            score = 0.0
        else:
            score = 1.0 if extracted_bool == ground_truth_bool else 0.0

        if debug:
            print(f"    ✅ BOOLEAN: {extracted_bool} vs {ground_truth_bool} = {score}")
        return score

    elif field_name in get_calculated_fields():
        # Calculated fields - validate calculations or compare values
        score = _evaluate_calculated_field(extracted, ground_truth, field_name, debug)
        return score

    elif field_name in get_transaction_list_fields():
        # Transaction list fields - compare structured transaction data
        score = _evaluate_transaction_list(extracted, ground_truth, field_name, debug)
        return score

    else:
        # Text fields - fuzzy matching
        # Check for substring match
        if (
            extracted_lower in ground_truth_lower
            or ground_truth_lower in extracted_lower
        ):
            if debug:
                print("    📝 TEXT: Substring match - score: 0.9")
            return 0.9

        # Check word overlap for longer text
        extracted_words = set(extracted_lower.split())
        ground_truth_words = set(ground_truth_lower.split())

        if ground_truth_words:
            overlap = len(extracted_words & ground_truth_words) / len(
                ground_truth_words
            )
            if overlap >= 0.8:
                if debug:
                    print(f"    📝 TEXT: Word overlap {overlap:.2f} - score: {overlap}")
                return overlap

        if debug:
            print("    📝 TEXT: No match - score: 0.0")
        return 0.0


def _compare_monetary_values(extracted: str, ground_truth: str) -> Tuple[bool, str]:
    """Compare monetary values with normalization."""

    def normalize_money(value):
        # Remove currency symbols and spaces, normalize decimal places
        clean = re.sub(r"[$,\s]", "", value)
        try:
            return float(clean)
        except ValueError:
            return value.lower()

    try:
        ext_val = normalize_money(extracted)
        gt_val = normalize_money(ground_truth)

        if isinstance(ext_val, float) and isinstance(gt_val, float):
            if abs(ext_val - gt_val) < 0.01:  # Penny precision
                return True, "Monetary match"
            else:
                return False, f"Amount mismatch: {ext_val} vs {gt_val}"
        else:
            # Fallback to text comparison
            return _compare_text_values(extracted, ground_truth)
    except Exception:
        return _compare_text_values(extracted, ground_truth)


def _compare_date_values(extracted: str, ground_truth: str) -> Tuple[bool, str]:
    """Compare date values with format normalization."""

    def normalize_date(date_str):
        # Try to extract date components regardless of format
        # Handle formats like DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
        date_patterns = [
            r"(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{4})",
            r"(\d{4})[\/\-\.](\d{1,2})[\/\-\.](\d{1,2})",
        ]

        for pattern in date_patterns:
            match = re.search(pattern, date_str)
            if match:
                return tuple(int(x) for x in match.groups())
        return date_str.lower()

    ext_date = normalize_date(extracted)
    gt_date = normalize_date(ground_truth)

    if ext_date == gt_date:
        return True, "Date match"
    else:
        return False, f"Date mismatch: {extracted} vs {ground_truth}"


def _compare_numeric_ids(extracted: str, ground_truth: str) -> Tuple[bool, str]:
    """Compare numeric IDs with space/formatting normalization."""

    def normalize_id(id_str):
        # Remove spaces and common formatting
        return re.sub(r"[\s\-]", "", id_str)

    ext_norm = normalize_id(extracted)
    gt_norm = normalize_id(ground_truth)

    if ext_norm.lower() == gt_norm.lower():
        return True, "ID match (normalized)"
    else:
        return False, f"ID mismatch: {extracted} vs {ground_truth}"


def _compare_list_values(extracted: str, ground_truth: str) -> Tuple[bool, str]:
    """Compare comma-separated list values."""

    def normalize_list(list_str):
        # Split by comma, strip whitespace, sort for comparison
        items = [item.strip().lower() for item in list_str.split(",")]
        return sorted([item for item in items if item])

    ext_list = normalize_list(extracted)
    gt_list = normalize_list(ground_truth)

    if ext_list == gt_list:
        return True, "List match"
    else:
        # Calculate partial match score
        intersection = set(ext_list) & set(gt_list)
        union = set(ext_list) | set(gt_list)
        if union:
            similarity = len(intersection) / len(union)
            if similarity >= 0.8:  # 80% similarity threshold
                return True, f"List partial match ({similarity:.2f})"

        return False, f"List mismatch: {extracted} vs {ground_truth}"


def _compare_text_values(extracted: str, ground_truth: str) -> Tuple[bool, str]:
    """Compare text values with fuzzy matching."""

    # Simple fuzzy matching based on common words
    def get_words(text):
        return set(re.findall(r"\w+", text.lower()))

    ext_words = get_words(extracted)
    gt_words = get_words(ground_truth)

    if not gt_words:
        return extracted.lower() == ground_truth.lower(), "Text comparison"

    # Calculate word overlap
    intersection = ext_words & gt_words
    similarity = len(intersection) / len(gt_words) if gt_words else 0

    if similarity >= 0.8:  # 80% word overlap threshold
        return True, f"Text fuzzy match ({similarity:.2f})"
    elif similarity >= 0.6:
        return True, f"Text partial match ({similarity:.2f})"
    else:
        return False, f"Text mismatch: {extracted} vs {ground_truth}"


def evaluate_extraction_results(
    extraction_results: List[Dict], ground_truth_map: Dict
) -> Dict:
    """
    Evaluate extraction results against ground truth data.

    Args:
        extraction_results (list): List of extraction result dictionaries
        ground_truth_map (dict): Ground truth data mapping

    Returns:
        dict: Comprehensive evaluation summary with accuracy metrics
    """
    if not extraction_results or not ground_truth_map:
        return {"error": "No data to evaluate"}

    print(f"🔍 Evaluating {len(extraction_results)} extraction results...")

    # Track field-level accuracies - will be populated dynamically per document type
    field_accuracies = {}

    # Detailed results for analysis
    detailed_results = []

    for _idx, result in enumerate(extraction_results):
        image_name = result.get("image_name", "")
        extracted_data = result.get("extracted_data", {})

        # Processing image silently

        # Find corresponding ground truth
        gt_data = None
        for gt_key, gt_value in ground_truth_map.items():
            if image_name in gt_key or gt_key in image_name:
                gt_data = gt_value
                # Found ground truth match
                break

        if gt_data is None:
            print(f"⚠️  No ground truth found for image: {image_name}")
            continue

        # Compare each field
        result_details = {"image_name": image_name, "fields": {}}
        image_accuracies = {}

        # Get document type to determine which fields to evaluate
        doc_type_raw = extracted_data.get("DOCUMENT_TYPE", "invoice").lower()

        # Map detected type to schema type (robust mapping like document_aware)
        type_mapping = {
            "invoice": "invoice",
            "tax invoice": "invoice",
            "estimate": "invoice",
            "quote": "invoice",
            "quotation": "invoice",
            "receipt": "receipt",
            "bank statement": "bank_statement",
            "statement": "bank_statement",
        }
        doc_type = type_mapping.get(doc_type_raw, "invoice")

        # Get document-specific fields for evaluation
        from common.config import get_document_type_fields

        fields_to_evaluate = get_document_type_fields(doc_type)

        # Compare each field
        perfect_matches = 0
        partial_matches = 0
        no_matches = 0

        for field in fields_to_evaluate:
            extracted_value = extracted_data.get(field, "NOT_FOUND")
            ground_truth_value = gt_data.get(field, "NOT_FOUND")

            # Get float accuracy score (0.0 to 1.0)
            accuracy_score = calculate_field_accuracy(
                extracted_value, ground_truth_value, field, debug=False
            )

            # Track score breakdown
            if accuracy_score == 1.0:
                perfect_matches += 1
            elif accuracy_score > 0.0:
                partial_matches += 1
            else:
                no_matches += 1

            image_accuracies[field] = accuracy_score
            is_correct = accuracy_score > 0.5  # Convert to boolean for detailed results

            # Initialize field accuracy tracking if needed (document-aware)
            if field not in field_accuracies:
                field_accuracies[field] = {"correct": 0, "total": 0, "details": []}

            field_accuracies[field]["total"] += 1
            field_accuracies[field]["correct"] += (
                accuracy_score  # Use float score for partial credit
            )

            # Store detailed result
            field_accuracies[field]["details"].append(
                {
                    "image": image_name,
                    "extracted": extracted_value,
                    "ground_truth": ground_truth_value,
                    "correct": is_correct,
                    "accuracy_score": accuracy_score,
                }
            )

            result_details["fields"][field] = {
                "extracted": extracted_value,
                "ground_truth": ground_truth_value,
                "correct": is_correct,
                "accuracy_score": accuracy_score,
            }

        # Field match summary tracked internally

        # Calculate overall accuracy for this image (like the old system)
        image_overall_accuracy = (
            sum(image_accuracies.values()) / len(image_accuracies)
            if image_accuracies
            else 0.0
        )
        result_details["overall_accuracy"] = image_overall_accuracy

        # Image accuracy calculated and stored

        detailed_results.append(result_details)

    # Processing complete

    # Calculate summary statistics (average of per-image accuracies, like the old system)
    if detailed_results:
        individual_accuracies = [
            result["overall_accuracy"] for result in detailed_results
        ]
        overall_accuracy = sum(individual_accuracies) / len(individual_accuracies)
        # Summary statistics calculated
    else:
        overall_accuracy = 0.0
        # No results to process

    field_summary = {}
    for field, data in field_accuracies.items():
        # data["correct"] is now sum of float scores, data["total"] is count of fields
        accuracy = data["correct"] / data["total"] if data["total"] > 0 else 0
        field_summary[field] = {
            "accuracy": accuracy,
            "correct": data["correct"],
            "total": data["total"],
        }

    # Field accuracy summary calculated

    # Calculate equivalent overall statistics (document-aware)
    # Count actual fields evaluated across all documents
    total_fields_evaluated = sum(data["total"] for data in field_accuracies.values())
    total_accuracy_score = sum(data["correct"] for data in field_accuracies.values())

    # Total statistics calculated

    # Calculate best and worst performing images
    if detailed_results:
        best_result = max(detailed_results, key=lambda x: x["overall_accuracy"])
        worst_result = min(detailed_results, key=lambda x: x["overall_accuracy"])
        perfect_documents = sum(
            1 for r in detailed_results if r["overall_accuracy"] >= 0.99
        )

        best_performing_image = best_result["image_name"]
        best_performance_accuracy = best_result["overall_accuracy"]
        worst_performing_image = worst_result["image_name"]
        worst_performance_accuracy = worst_result["overall_accuracy"]
    else:
        best_performing_image = "None"
        best_performance_accuracy = 0.0
        worst_performing_image = "None"
        worst_performance_accuracy = 0.0
        perfect_documents = 0

    # Generate summary report
    evaluation_summary = {
        "overall_accuracy": overall_accuracy,
        "overall_correct": total_accuracy_score,
        "overall_total": total_fields_evaluated,
        "field_accuracies": field_summary,
        "detailed_results": detailed_results,
        "images_evaluated": len(detailed_results),
        "total_images": len(detailed_results),  # Add this for reporting compatibility
        "best_performing_image": best_performing_image,
        "best_performance_accuracy": best_performance_accuracy,
        "worst_performing_image": worst_performing_image,
        "worst_performance_accuracy": worst_performance_accuracy,
        "perfect_documents": perfect_documents,
        "summary_stats": {
            "best_fields": sorted(
                field_summary.items(), key=lambda x: x[1]["accuracy"], reverse=True
            )[:5],
            "worst_fields": sorted(
                field_summary.items(), key=lambda x: x[1]["accuracy"]
            )[:5],
            "avg_field_accuracy": np.mean(
                [data["accuracy"] for data in field_summary.values()]
            ),
        },
    }

    print("✅ Evaluation complete:")
    print(f"   Overall accuracy: {overall_accuracy:.1%}")
    print(f"   Fields evaluated: {len(field_summary)}")
    print(f"   Images processed: {len(detailed_results)}")

    return evaluation_summary


def prepare_classification_data(
    detailed_results: List[Dict],
) -> Tuple[List, List, List]:
    """
    Prepare data for sklearn classification reporting.

    Args:
        detailed_results: Detailed evaluation results

    Returns:
        tuple: (y_true, y_pred, field_names)
    """
    y_true = []
    y_pred = []
    field_names = []

    for result in detailed_results:
        for field, data in result["fields"].items():
            y_true.append(1 if data["correct"] else 0)
            y_pred.append(1 if data["correct"] else 0)  # This is for consistency
            field_names.append(field)

    return y_true, y_pred, field_names


def generate_field_classification_report(evaluation_summary: Dict) -> str:
    """
    Generate a detailed classification report for field-level accuracy.

    Args:
        evaluation_summary: Results from evaluate_extraction_results

    Returns:
        str: Formatted classification report
    """
    field_accuracies = evaluation_summary.get("field_accuracies", {})

    if not field_accuracies:
        return "No field accuracy data available."

    # Create report
    report_lines = []
    report_lines.append("📊 FIELD-LEVEL ACCURACY REPORT")
    report_lines.append("=" * 50)

    # Sort fields by accuracy
    sorted_fields = sorted(
        field_accuracies.items(), key=lambda x: x[1]["accuracy"], reverse=True
    )

    report_lines.append(f"{'Field':<20} {'Accuracy':<10} {'Correct':<8} {'Total':<8}")
    report_lines.append("-" * 50)

    for field, data in sorted_fields:
        accuracy = data["accuracy"]
        correct = data["correct"]
        total = data["total"]
        report_lines.append(f"{field:<20} {accuracy:>7.1%} {correct:>7} {total:>7}")

    # Summary statistics
    report_lines.append("-" * 50)
    avg_accuracy = evaluation_summary["summary_stats"]["avg_field_accuracy"]
    report_lines.append(f"{'Average':<20} {avg_accuracy:>7.1%}")

    return "\n".join(report_lines)


def generate_overall_classification_summary(evaluation_summary: Dict) -> Dict:
    """
    Generate classification summary for sklearn metrics visualization.

    Args:
        evaluation_summary: Evaluation summary from evaluate_extraction_results

    Returns:
        dict: Classification summary with metrics and field data
    """
    try:
        # Extract data from evaluation summary
        field_accuracies = evaluation_summary.get("field_accuracies", {})
        overall_accuracy = evaluation_summary.get("overall_accuracy", 0)

        # Generate field-level metrics using our field accuracy data
        field_metrics = {}

        for field, accuracy_data in field_accuracies.items():
            # Convert our field accuracy to classification metrics
            acc = (
                accuracy_data.get("accuracy", 0)
                if isinstance(accuracy_data, dict)
                else accuracy_data
            )

            # Use accuracy as a proxy for precision, recall, f1
            field_metrics[field] = {
                "precision": float(acc),
                "recall": float(acc),
                "f1_score": float(acc),
                "support": 1,
            }

        # Create overall metrics from our evaluation data
        overall_metrics = {
            "macro_avg": {
                "precision": overall_accuracy,
                "recall": overall_accuracy,
                "f1_score": overall_accuracy,
            },
            "accuracy": overall_accuracy,
            "total_predictions": len(field_accuracies),
        }

        return {
            "overall_metrics": overall_metrics,
            "field_metrics": field_metrics,
        }

    except Exception as e:
        return {
            "overall_metrics": {"error": str(e)},
            "field_metrics": {},
        }


# ============================================================================
# V4 FIELD TYPE EVALUATION HELPERS
# ============================================================================


def _parse_boolean_value(value: str) -> bool:
    """Parse boolean value from text string with strict matching."""
    if not value or value == "NOT_FOUND":
        return None

    # Convert to string first to handle boolean objects
    value_lower = str(value).lower().strip()

    # Strict boolean matching - accept both lowercase and Python boolean string representations
    true_values = ["true", "1"]
    false_values = ["false", "0"]

    if value_lower in true_values:
        return True
    elif value_lower in false_values:
        return False
    else:
        return None


def _evaluate_calculated_field(
    extracted: str, ground_truth: str, field_name: str, debug: bool = False
) -> float:
    """Evaluate calculated fields with validation logic."""
    if not extracted or extracted == "NOT_FOUND":
        return 0.0 if ground_truth and ground_truth != "NOT_FOUND" else 1.0

    if not ground_truth or ground_truth == "NOT_FOUND":
        return 0.0

    # For LINE_ITEM_TOTAL_PRICES - could validate against quantities × prices
    # For now, treat as list comparison
    if "TOTAL_PRICES" in field_name:
        return _evaluate_calculated_totals(extracted, ground_truth, debug)
    else:
        # Default to monetary comparison for other calculated fields
        return _compare_monetary_values(extracted, ground_truth, debug)


def _evaluate_calculated_totals(
    extracted: str, ground_truth: str, debug: bool = False
) -> float:
    """Evaluate line item total calculations."""
    try:
        # Parse pipe-separated values
        extracted_items = [item.strip() for item in extracted.split("|")]
        ground_truth_items = [item.strip() for item in ground_truth.split("|")]

        if len(extracted_items) != len(ground_truth_items):
            if debug:
                print(
                    f"    🧮 CALCULATED: Length mismatch {len(extracted_items)} vs {len(ground_truth_items)}"
                )
            return 0.0

        matches = 0
        for ext_val, gt_val in zip(extracted_items, ground_truth_items, strict=False):
            if _compare_monetary_values(ext_val, gt_val, False) == 1.0:
                matches += 1

        score = matches / len(ground_truth_items) if ground_truth_items else 0.0
        if debug:
            print(
                f"    🧮 CALCULATED: {matches}/{len(ground_truth_items)} totals match = {score}"
            )
        return score

    except Exception as e:
        if debug:
            print(f"    🧮 CALCULATED: Error evaluating totals: {e}")
        return 0.0


def _compare_monetary_values(
    extracted: str, ground_truth: str, debug: bool = False
) -> float:
    """Compare monetary values with tolerance."""
    try:
        extracted_num = float(re.sub(r"[^\d.-]", "", extracted))
        ground_truth_num = float(re.sub(r"[^\d.-]", "", ground_truth))

        # Allow 1% tolerance for rounding
        tolerance = abs(ground_truth_num * 0.01) if ground_truth_num != 0 else 0.01
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


def _evaluate_transaction_list(
    extracted: str, ground_truth: str, field_name: str, debug: bool = False
) -> float:
    """Evaluate transaction list fields with structured comparison."""
    if not extracted or extracted == "NOT_FOUND":
        return 0.0 if ground_truth and ground_truth != "NOT_FOUND" else 1.0

    if not ground_truth or ground_truth == "NOT_FOUND":
        return 0.0

    try:
        # Parse pipe-separated transaction data
        extracted_items = [item.strip() for item in extracted.split("|")]
        ground_truth_items = [item.strip() for item in ground_truth.split("|")]

        # For transaction lists, order matters and length should match
        if len(extracted_items) != len(ground_truth_items):
            # Partial credit based on overlap
            # Check positional matches up to the length of the shorter list
            overlap = min(len(extracted_items), len(ground_truth_items))
            matches = 0
            for i in range(overlap):
                if _transaction_item_matches(
                    extracted_items[i], ground_truth_items[i], field_name
                ):
                    matches += 1

            # Score based on ground truth length (what we expect to find)
            # This rewards extracting correct items even if extras are present
            score = matches / len(ground_truth_items) if ground_truth_items else 0.0
            if debug:
                print(f"    📊 TRANSACTION: Length mismatch - partial score: {score} ({matches}/{len(ground_truth_items)} correct)")
            return score

        # Full comparison when lengths match
        matches = 0
        for ext_item, gt_item in zip(extracted_items, ground_truth_items, strict=False):
            if _transaction_item_matches(ext_item, gt_item, field_name):
                matches += 1

        score = matches / len(ground_truth_items) if ground_truth_items else 0.0
        if debug:
            print(
                f"    📊 TRANSACTION: {matches}/{len(ground_truth_items)} transactions match = {score}"
            )
        return score

    except Exception as e:
        if debug:
            print(f"    📊 TRANSACTION: Error evaluating transactions: {e}")
        return 0.0


def _transaction_item_matches(
    extracted_item: str, ground_truth_item: str, field_name: str
) -> bool:
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


def _compare_dates_fuzzy(extracted_date: str, ground_truth_date: str) -> bool:
    """Fuzzy date comparison allowing for different formats."""
    if extracted_date.strip() == ground_truth_date.strip():
        return True

    # Month name to number mapping
    months = {
        'jan': '01', 'january': '01',
        'feb': '02', 'february': '02',
        'mar': '03', 'march': '03',
        'apr': '04', 'april': '04',
        'may': '05',
        'jun': '06', 'june': '06',
        'jul': '07', 'july': '07',
        'aug': '08', 'august': '08',
        'sep': '09', 'sept': '09', 'september': '09',
        'oct': '10', 'october': '10',
        'nov': '11', 'november': '11',
        'dec': '12', 'december': '12',
    }

    def normalize_date(date_str):
        """Extract day, month, year from date string, handling month names."""
        date_lower = date_str.lower()

        # Extract all numbers
        nums = re.findall(r'\d+', date_str)

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
    matches = sum(1 for ext, gt in zip(extracted_parts, ground_truth_parts, strict=False) if ext == gt and ext is not None)
    return matches >= 2


def calculate_correlation_aware_f1(
    extracted_data: dict,
    ground_truth_data: dict,
    document_type: str,
    debug: bool = False,
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

    if "bank" in doc_type_lower or "statement" in doc_type_lower:
        related_groups = [
            ("TRANSACTION_DATES", "LINE_ITEM_DESCRIPTIONS", "TRANSACTION_AMOUNTS_PAID")
        ]
    elif "invoice" in doc_type_lower or "receipt" in doc_type_lower:
        related_groups = [
            ("LINE_ITEM_DESCRIPTIONS", "LINE_ITEM_QUANTITIES",
             "LINE_ITEM_PRICES", "LINE_ITEM_TOTAL_PRICES")
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
                debug=False
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

            extracted_lists[field] = [
                i.strip() for i in ext_value.split('|') if i.strip()
            ]
            ground_truth_lists[field] = [
                i.strip() for i in gt_value.split('|') if i.strip()
            ]

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
                                field
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
        "recall": combined_f1,      # For compatibility
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
    extracted_data: dict = None,
    ground_truth_data: dict = None,
    document_type: str = None,
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
                extracted_value, ground_truth_value, field_name, debug
            )
        # Return correlation metrics (calculated once for all fields)
        return calculate_correlation_aware_f1(
            extracted_data, ground_truth_data, document_type, debug
        )
    elif method == "f1" or method == "position_agnostic_f1":
        return calculate_field_accuracy_f1_position_agnostic(
            extracted_value, ground_truth_value, field_name, debug
        )
    elif method == "kieval":
        return calculate_field_accuracy_kieval(
            extracted_value, ground_truth_value, field_name, debug
        )
    elif method == "order_aware_f1" or method == "position_aware_f1":
        return calculate_field_accuracy_f1(
            extracted_value, ground_truth_value, field_name, debug
        )
    else:
        # Default to order-aware F1 (current implementation)
        return calculate_field_accuracy_f1(
            extracted_value, ground_truth_value, field_name, debug
        )


def _fuzzy_text_match(text1: str, text2: str, threshold: float = 0.75) -> bool:
    """
    Check if two text strings match using fuzzy word-based comparison.

    Args:
        text1: First text string
        text2: Second text string
        threshold: Minimum word overlap ratio (0.0-1.0) for match

    Returns:
        bool: True if texts match above threshold
    """
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
    extracted_value: str, ground_truth_value: str, field_name: str, debug: bool = False
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
    # Convert to strings and clean
    extracted = str(extracted_value).strip() if extracted_value else "NOT_FOUND"
    ground_truth = (
        str(ground_truth_value).strip() if ground_truth_value else "NOT_FOUND"
    )

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
        gt_items = [
            i.strip() for i in str(ground_truth).split("|") if i.strip()
        ]
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
        if field_name in get_transaction_list_fields():
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
            match = _fuzzy_text_match(extracted, ground_truth, threshold=0.75)
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
    ground_truth_items = [
        i.strip() for i in str(ground_truth).split("|") if i.strip()
    ]

    # True Positives: Count extracted items that match any ground truth item
    tp = 0
    matched_gt_indices = set()

    for ext_item in extracted_items:
        for i, gt_item in enumerate(ground_truth_items):
            if i not in matched_gt_indices:
                if field_name in get_transaction_list_fields():
                    match = _transaction_item_matches(ext_item, gt_item, field_name)
                else:
                    match = _fuzzy_text_match(ext_item, gt_item, threshold=0.75)

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
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    if debug:
        print(f"  📊 F1 Metrics (Position-Agnostic) for {field_name}:")
        print(f"     TP={tp}, FP={fp}, FN={fn}")
        print(
            f"     Precision={precision:.2%}, Recall={recall:.2%}, F1={f1_score:.2%}"
        )

    return {
        "f1_score": f1_score,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def calculate_field_accuracy_kieval(
    extracted_value: str, ground_truth_value: str, field_name: str, debug: bool = False
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
        extracted_value, ground_truth_value, field_name, debug=False
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
    extracted_items = [
        i.strip() for i in str(extracted_value).split("|") if i.strip()
    ] if "|" in str(extracted_value) else ([str(extracted_value).strip()] if str(extracted_value).strip() != "NOT_FOUND" else [])

    ground_truth_items = [
        i.strip() for i in str(ground_truth_value).split("|") if i.strip()
    ] if "|" in str(ground_truth_value) else ([str(ground_truth_value).strip()] if str(ground_truth_value).strip() != "NOT_FOUND" else [])

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
    extracted_value: str, ground_truth_value: str, field_name: str, debug: bool = False
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
    # Convert to strings and clean
    extracted = str(extracted_value).strip() if extracted_value else "NOT_FOUND"
    ground_truth = (
        str(ground_truth_value).strip() if ground_truth_value else "NOT_FOUND"
    )

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
        gt_items = [
            i.strip() for i in str(ground_truth).split("|") if i.strip()
        ]
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
        if field_name in get_transaction_list_fields():
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

            # For date fields, use date-aware comparison that normalizes formats
            # This handles cases like "16/07/2025" vs "16-Jul-25" (same date, different format)
            date_field_keywords = ["DATE", "DUE_DATE", "INVOICE_DATE", "STATEMENT_DATE"]
            is_date_field = any(keyword in field_name.upper() for keyword in date_field_keywords)

            if is_date_field:
                # Use fuzzy date comparison from _compare_dates_fuzzy
                match = _compare_dates_fuzzy(extracted_normalized, ground_truth_normalized)
                return {
                    "f1_score": 1.0 if match else 0.0,
                    "precision": 1.0 if match else 0.0,
                    "recall": 1.0 if match else 0.0,
                    "tp": 1 if match else 0,
                    "fp": 0 if match else 1,
                    "fn": 0 if match else 1,
                }

            # For single-value monetary fields (GST_AMOUNT, TOTAL_AMOUNT), use monetary comparison with F1-style penalty
            # This ensures incorrect amounts get 0.0 score (penalizing false positives)
            # NOTE: List fields like LINE_ITEM_PRICES are handled later by the list F1 logic
            monetary_single_fields = ["GST_AMOUNT", "TOTAL_AMOUNT", "INVOICE_TOTAL", "SUBTOTAL"]
            is_monetary_field = field_name in monetary_single_fields

            if is_monetary_field:
                try:
                    extracted_num = float(re.sub(r"[^\d.-]", "", extracted_normalized))
                    ground_truth_num = float(re.sub(r"[^\d.-]", "", ground_truth_normalized))

                    # Allow 1% tolerance for rounding (same as calculate_field_accuracy)
                    tolerance = abs(ground_truth_num * 0.01) if ground_truth_num != 0 else 0.01
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
                ground_truth_clean = re.sub(id_label_pattern, "", ground_truth_normalized, flags=re.IGNORECASE)

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
                        extracted_normalized.lower(),
                        ground_truth_normalized.lower()
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
    ground_truth_items = [
        i.strip() for i in str(ground_truth).split("|") if i.strip()
    ]

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
            if field_name in get_transaction_list_fields():
                match = _transaction_item_matches(
                    extracted_items[i], ground_truth_items[i], field_name
                )
            else:
                # Use fuzzy text matching with generous 0.75 threshold
                # This allows "EATS Sydney" to match "UBER EATS Sydney" (0.80 similarity)
                match = _fuzzy_text_match(extracted_items[i], ground_truth_items[i], threshold=0.75)

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
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    if debug:
        print(f"  📊 F1 Metrics for {field_name}:")
        print(f"     TP={tp}, FP={fp}, FN={fn}")
        print(
            f"     Precision={precision:.2%}, Recall={recall:.2%}, F1={f1_score:.2%}"
        )
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
