"""
Pipeline stage functions for document extraction workflow.

This module provides the core pipeline stages that transform data through
the extraction → parsing → cleaning → evaluation workflow.
"""

from pathlib import Path
from typing import Dict

import pandas as pd

from .cleaner import ExtractionCleaner
from .evaluator import calculate_field_accuracy
from .parser import hybrid_parse_response


def stage_3_parsing(extraction_response: str, expected_fields: list[str]) -> Dict[str, str]:
    """
    Stage 3: Parse extraction response into structured fields.

    Uses hybrid_parse_response which handles:
    - JSON format responses
    - Plain text key:value format
    - Markdown formatted responses
    - Multi-line values
    - Truncated JSON repair

    Args:
        extraction_response: Raw text response from VLM
        expected_fields: List of field names to extract

    Returns:
        dict: Parsed fields (all fields present, NOT_FOUND if missing)
    """
    parsed_fields = hybrid_parse_response(
        response_text=extraction_response,
        expected_fields=expected_fields
    )
    return parsed_fields


def stage_4_cleaning(parsed_fields: Dict[str, str], cleaner: ExtractionCleaner) -> Dict[str, str]:
    """
    Stage 4: Clean and normalize field values.

    Uses ExtractionCleaner which provides:
    - Monetary field cleaning (remove suffixes, standardize currency, remove commas)
    - List field cleaning (markdown removal, consistent pipe-separation)
    - Address field cleaning (remove phone numbers, emails, commas)
    - ID field normalization (ABN, BSB formatting)
    - Business knowledge validation (pricing, GST calculations)

    Args:
        parsed_fields: Dictionary of parsed field values
        cleaner: ExtractionCleaner instance

    Returns:
        dict: Cleaned and validated field values
    """
    cleaned_fields = cleaner.clean_extraction_dict(parsed_fields)
    return cleaned_fields


def stage_5_evaluation(
    row: pd.Series,
    ground_truth: Dict[str, Dict],
    expected_fields: list[str]
) -> Dict | None:
    """
    Stage 5: Evaluate extraction against ground truth.

    Args:
        row: DataFrame row with 'image_path' and 'cleaned_fields'
        ground_truth: Ground truth dictionary (image_name -> field dict)
        expected_fields: List of field names to evaluate

    Returns:
        dict: Evaluation metrics or None if no ground truth
            - overall_accuracy: float (0.0 to 1.0)
            - fields_matched: int (number of perfect matches)
            - fields_extracted: int (number of non-NOT_FOUND values)
            - total_fields: int (number of fields in ground truth)
            - field_results: dict (field_name -> accuracy score)
    """
    if ground_truth is None:
        return None

    image_name = Path(row['image_path']).name

    if image_name not in ground_truth:
        return {'error': 'No ground truth available'}

    gt_data = ground_truth[image_name]
    extracted_data = row['cleaned_fields']

    # Evaluate each field individually
    fields_matched = 0
    fields_extracted = 0
    total_fields = 0
    field_results = {}

    for field in expected_fields:
        # Get ground truth value
        gt_value = gt_data.get(field, 'NOT_FOUND')

        # Skip if ground truth doesn't have this field
        if pd.isna(gt_value) or str(gt_value).strip() == '' or str(gt_value) == 'NOT_FOUND':
            continue

        total_fields += 1

        # Get extracted value
        extracted_value = extracted_data.get(field, 'NOT_FOUND')

        # Count as extracted if not NOT_FOUND
        if extracted_value != 'NOT_FOUND':
            fields_extracted += 1

        # Calculate accuracy for this field
        accuracy = calculate_field_accuracy(
            extracted_value=str(extracted_value),
            ground_truth_value=str(gt_value),
            field_name=field,
            debug=False
        )

        field_results[field] = accuracy

        # Count as matched if accuracy is 1.0 (perfect match)
        if accuracy == 1.0:
            fields_matched += 1

    # Calculate overall accuracy
    overall_accuracy = fields_matched / total_fields if total_fields > 0 else 0.0

    return {
        'overall_accuracy': overall_accuracy,
        'fields_matched': fields_matched,
        'fields_extracted': fields_extracted,
        'total_fields': total_fields,
        'field_results': field_results
    }
