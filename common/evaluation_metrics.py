"""
Evaluation and accuracy assessment utilities for model outputs.

This module handles ground truth comparison, accuracy calculations, and evaluation
reporting. It provides comprehensive metrics for assessing model performance
against known correct answers.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import (
    DATE_FIELDS,
    EXTRACTION_FIELDS,
    LIST_FIELDS,
    MONETARY_FIELDS,
    NUMERIC_ID_FIELDS,
)


def load_ground_truth(csv_path: str, show_sample: bool = False) -> Dict[str, Dict]:
    """
    Load ground truth data from CSV file.
    
    Args:
        csv_path (str): Path to the ground truth CSV file
        show_sample (bool): Whether to display a sample of the data
        
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
        print(f"📊 Ground truth CSV loaded with {len(ground_truth_df)} rows and {len(ground_truth_df.columns)} columns")
        print(f"📋 Available columns: {list(ground_truth_df.columns)}")
        
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}") from e
    
    # Find image identifier column
    image_col = None
    possible_names = ['image_file', 'filename', 'image_name', 'file']
    for col in possible_names:
        if col in ground_truth_df.columns:
            image_col = col
            break
    
    if image_col is None:
        raise ValueError(f"No image identifier column found. Expected one of: {possible_names}")
    
    print(f"✅ Using '{image_col}' as image identifier column")
    
    if show_sample and len(ground_truth_df) > 0:
        print("📄 Sample ground truth data:")
        print(ground_truth_df.head(2).to_string(index=False))
    
    # Convert to dictionary mapping
    ground_truth_map = {}
    for _, row in ground_truth_df.iterrows():
        image_name = row[image_col]
        if pd.isna(image_name):
            continue
        ground_truth_map[str(image_name)] = row.to_dict()
    
    print(f"✅ Ground truth mapping created for {len(ground_truth_map)} images")
    return ground_truth_map


def calculate_field_accuracy(extracted_value: str, ground_truth_value: str, field_name: str, debug=False) -> float:
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
    ground_truth = str(ground_truth_value).strip() if ground_truth_value else "NOT_FOUND"

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
            print(f"    ❌ One NOT_FOUND, other not ('{extracted}' vs '{ground_truth}') - score: 0.0")
        return 0.0

    # Normalize for comparison
    extracted_lower = extracted.lower()
    ground_truth_lower = ground_truth.lower()

    # Remove common formatting
    for char in [",", "$", "%", "(", ")", " "]:
        extracted_lower = extracted_lower.replace(char, "")
        ground_truth_lower = ground_truth_lower.replace(char, "")

    if debug:
        print(f"    🔍 Normalized: '{extracted_lower}' vs '{ground_truth_lower}'")

    # Exact match after normalization
    if extracted_lower == ground_truth_lower:
        if debug:
            print("    ✅ Exact match - score: 1.0")
        return 1.0

    # Field-specific comparison logic using centralized field type definitions
    if field_name in NUMERIC_ID_FIELDS:
        # Numeric identifiers - exact match required
        extracted_digits = re.sub(r"\D", "", extracted)
        ground_truth_digits = re.sub(r"\D", "", ground_truth)
        score = 1.0 if extracted_digits == ground_truth_digits else 0.0
        if debug:
            print(f"    🔢 NUMERIC_ID: '{extracted_digits}' vs '{ground_truth_digits}' = {score}")
        return score

    elif field_name in MONETARY_FIELDS:
        # Monetary values - numeric comparison
        try:
            extracted_num = float(re.sub(r"[^\d.-]", "", extracted))
            ground_truth_num = float(re.sub(r"[^\d.-]", "", ground_truth))
            # Allow 1% tolerance for rounding
            tolerance = abs(ground_truth_num * 0.01) if ground_truth_num != 0 else 0.01
            score = 1.0 if abs(extracted_num - ground_truth_num) <= tolerance else 0.0
            if debug:
                print(f"    💰 MONETARY: {extracted_num} vs {ground_truth_num} (tolerance: {tolerance}) = {score}")
            return score
        except (ValueError, AttributeError):
            if debug:
                print("    💰 MONETARY: Parsing failed - score: 0.0")
            return 0.0

    elif field_name in DATE_FIELDS:
        # Date fields - flexible matching
        # Extract date components
        extracted_numbers = re.findall(r"\d+", extracted)
        ground_truth_numbers = re.findall(r"\d+", ground_truth)

        # Check if same date components are present
        if set(extracted_numbers) == set(ground_truth_numbers):
            if debug:
                print(f"    📅 DATE: Components match {extracted_numbers} = {ground_truth_numbers} - score: 1.0")
            return 1.0

        # Partial match for dates
        common = set(extracted_numbers) & set(ground_truth_numbers)
        if common and len(common) >= 2:  # At least month and day match
            if debug:
                print(f"    📅 DATE: Partial match {common} - score: 0.8")
            return 0.8

        if debug:
            print(f"    📅 DATE: No match {extracted_numbers} vs {ground_truth_numbers} - score: 0.0")
        return 0.0

    elif field_name in LIST_FIELDS:
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
                print(f"    📋 LIST: Empty GT, extracted empty: {not extracted_items} - score: {score}")
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
            print(f"    📋 LIST: {matches}/{max(len(ground_truth_items), len(extracted_items))} matches - score: {score}")
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
        clean = re.sub(r'[$,\s]', '', value)
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
            r'(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{4})',
            r'(\d{4})[\/\-\.](\d{1,2})[\/\-\.](\d{1,2})',
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
        return re.sub(r'[\s\-]', '', id_str)
    
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
        items = [item.strip().lower() for item in list_str.split(',')]
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
        return set(re.findall(r'\w+', text.lower()))
    
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


def evaluate_extraction_results(extraction_results: List[Dict], ground_truth_map: Dict) -> Dict:
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
    
    # Track field-level accuracies
    field_accuracies = {field: {"correct": 0, "total": 0, "details": []} for field in EXTRACTION_FIELDS}
    
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
        
        # Compare each field
        perfect_matches = 0
        partial_matches = 0
        no_matches = 0
        
        for field in EXTRACTION_FIELDS:
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
            
            field_accuracies[field]["total"] += 1
            field_accuracies[field]["correct"] += accuracy_score  # Use float score for partial credit
            
            # Store detailed result
            field_accuracies[field]["details"].append({
                "image": image_name,
                "extracted": extracted_value,
                "ground_truth": ground_truth_value,
                "correct": is_correct,
                "accuracy_score": accuracy_score
            })
            
            result_details["fields"][field] = {
                "extracted": extracted_value,
                "ground_truth": ground_truth_value,
                "correct": is_correct,
                "accuracy_score": accuracy_score
            }
        
        # Field match summary tracked internally
        
        # Calculate overall accuracy for this image (like the old system)
        image_overall_accuracy = sum(image_accuracies.values()) / len(image_accuracies) if image_accuracies else 0.0
        result_details["overall_accuracy"] = image_overall_accuracy
        
        # Image accuracy calculated and stored
        
        detailed_results.append(result_details)
    
    # Processing complete
    
    # Calculate summary statistics (average of per-image accuracies, like the old system)
    if detailed_results:
        individual_accuracies = [result["overall_accuracy"] for result in detailed_results]
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
            "total": data["total"]
        }
    
    # Field accuracy summary calculated
    
    # Calculate equivalent overall statistics
    total_fields_evaluated = len(detailed_results) * len(EXTRACTION_FIELDS) if detailed_results else 0
    total_accuracy_score = overall_accuracy * total_fields_evaluated if total_fields_evaluated else 0
    
    # Total statistics calculated
    
    # Generate summary report
    evaluation_summary = {
        "overall_accuracy": overall_accuracy,
        "overall_correct": total_accuracy_score,
        "overall_total": total_fields_evaluated,
        "field_accuracies": field_summary,
        "detailed_results": detailed_results,
        "images_evaluated": len(detailed_results),
        "summary_stats": {
            "best_fields": sorted(field_summary.items(), key=lambda x: x[1]["accuracy"], reverse=True)[:5],
            "worst_fields": sorted(field_summary.items(), key=lambda x: x[1]["accuracy"])[:5],
            "avg_field_accuracy": np.mean([data["accuracy"] for data in field_summary.values()]),
        }
    }
    
    print("✅ Evaluation complete:")
    print(f"   Overall accuracy: {overall_accuracy:.1%}")
    print(f"   Fields evaluated: {len(field_summary)}")
    print(f"   Images processed: {len(detailed_results)}")
    
    return evaluation_summary


def prepare_classification_data(detailed_results: List[Dict]) -> Tuple[List, List, List]:
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
    sorted_fields = sorted(field_accuracies.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    
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


def generate_overall_classification_summary(evaluation_summary: Dict) -> str:
    """
    Generate an overall summary of the evaluation results.
    
    Args:
        evaluation_summary: Results from evaluate_extraction_results
        
    Returns:
        str: Formatted summary report
    """
    overall_accuracy = evaluation_summary.get("overall_accuracy", 0)
    overall_correct = evaluation_summary.get("overall_correct", 0)
    overall_total = evaluation_summary.get("overall_total", 0)
    images_evaluated = evaluation_summary.get("images_evaluated", 0)
    
    summary_lines = []
    summary_lines.append("🎯 EXTRACTION EVALUATION SUMMARY")
    summary_lines.append("=" * 40)
    summary_lines.append(f"Overall Accuracy: {overall_accuracy:.1%}")
    summary_lines.append(f"Correct Fields: {overall_correct:,}")
    summary_lines.append(f"Total Fields: {overall_total:,}")
    summary_lines.append(f"Images Evaluated: {images_evaluated:,}")
    
    # Best and worst performing fields
    summary_stats = evaluation_summary.get("summary_stats", {})
    if "best_fields" in summary_stats:
        summary_lines.append("\n🏆 Top Performing Fields:")
        for field, data in summary_stats["best_fields"]:
            summary_lines.append(f"  {field}: {data['accuracy']:.1%}")
    
    if "worst_fields" in summary_stats:
        summary_lines.append("\n⚠️  Fields Needing Improvement:")
        for field, data in summary_stats["worst_fields"]:
            summary_lines.append(f"  {field}: {data['accuracy']:.1%}")
    
    return "\n".join(summary_lines)