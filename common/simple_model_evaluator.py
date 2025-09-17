#!/usr/bin/env python3
"""
Simple Model Evaluator - Focused on Model Performance Comparison

Replaces the complex DocumentTypeEvaluator with a simple field-by-field
comparison system designed for comparing Llama vs InternVL3 extraction accuracy.

No enterprise complexity - just: extracted_fields vs ground_truth = accuracy%
"""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class SimpleEvaluationResult:
    """Simple evaluation result for model comparison."""

    total_fields: int
    correct_fields: int
    accuracy: float
    missing_fields: List[str]
    incorrect_fields: List[str]


class SimpleModelEvaluator:
    """Simple evaluator for comparing model extraction performance."""

    def __init__(self):
        """Initialize simple evaluator - no complex configuration needed."""
        pass

    def evaluate_extraction(
        self,
        extracted_data: Dict[str, str],
        ground_truth: Dict[str, str],
        image_name: str = ""
    ) -> SimpleEvaluationResult:
        """
        Compare extracted fields vs ground truth fields.

        Args:
            extracted_data: Fields extracted by model
            ground_truth: Expected field values
            image_name: Optional image identifier for logging

        Returns:
            SimpleEvaluationResult with accuracy metrics
        """
        # Get all fields that should be evaluated
        all_fields = set(ground_truth.keys())

        correct_fields = []
        incorrect_fields = []
        missing_fields = []

        for field in all_fields:
            ground_truth_value = ground_truth.get(field, "").strip()
            extracted_value = extracted_data.get(field, "").strip()

            # Skip empty ground truth fields
            if not ground_truth_value:
                continue

            if not extracted_value:
                missing_fields.append(field)
            elif self._values_match(extracted_value, ground_truth_value):
                correct_fields.append(field)
            else:
                incorrect_fields.append(field)

        total_evaluated = len(correct_fields) + len(incorrect_fields) + len(missing_fields)
        accuracy = len(correct_fields) / total_evaluated if total_evaluated > 0 else 0.0

        return SimpleEvaluationResult(
            total_fields=total_evaluated,
            correct_fields=len(correct_fields),
            accuracy=accuracy,
            missing_fields=missing_fields,
            incorrect_fields=incorrect_fields
        )

    def _values_match(self, extracted: str, ground_truth: str) -> bool:
        """
        Enhanced value matching with business document-specific normalization.

        Args:
            extracted: Value extracted by model
            ground_truth: Expected value

        Returns:
            True if values match (with appropriate normalization)
        """
        # Basic normalization
        extracted_clean = extracted.lower().strip().strip('"')
        ground_truth_clean = ground_truth.lower().strip().strip('"')

        # Exact match after normalization
        if extracted_clean == ground_truth_clean:
            return True

        # For list fields (pipe-separated), check FIRST before other types
        if '|' in extracted_clean or '|' in ground_truth_clean:
            return self._list_match(extracted_clean, ground_truth_clean)

        # For monetary values, try to normalize currency formatting
        if self._is_monetary(ground_truth_clean) or self._is_monetary(extracted_clean):
            return self._monetary_match(extracted_clean, ground_truth_clean)

        # For ABN/numeric IDs, normalize whitespace and formatting
        if self._is_abn_or_numeric_id(ground_truth_clean) or self._is_abn_or_numeric_id(extracted_clean):
            return self._numeric_id_match(extracted_clean, ground_truth_clean)

        # For addresses and text fields, use fuzzy matching
        return self._fuzzy_text_match(extracted_clean, ground_truth_clean)

    def _is_monetary(self, value: str) -> bool:
        """Check if value appears to be a monetary amount."""
        return any(char in value for char in ['$', '£', '€', '.', ',']) and any(char.isdigit() for char in value)

    def _monetary_match(self, extracted: str, ground_truth: str) -> bool:
        """Compare monetary values with flexible formatting."""
        import re

        # Extract just the numeric parts
        extracted_nums = re.sub(r'[^\d.]', '', extracted)
        ground_truth_nums = re.sub(r'[^\d.]', '', ground_truth)

        try:
            return abs(float(extracted_nums) - float(ground_truth_nums)) < 0.01
        except (ValueError, TypeError):
            return False

    def _is_abn_or_numeric_id(self, value: str) -> bool:
        """Check if value appears to be an ABN or numeric ID."""
        # Remove spaces and check if it's mostly digits
        digits_only = ''.join(c for c in value if c.isdigit())
        return len(digits_only) >= 8 and len(digits_only) / len(value.replace(' ', '')) > 0.7

    def _numeric_id_match(self, extracted: str, ground_truth: str) -> bool:
        """Compare numeric IDs (like ABN) ignoring whitespace."""
        import re

        # Extract just digits and normalize
        extracted_digits = re.sub(r'\D', '', extracted)
        ground_truth_digits = re.sub(r'\D', '', ground_truth)

        return extracted_digits == ground_truth_digits

    def _list_match(self, extracted: str, ground_truth: str) -> bool:
        """Compare pipe-separated lists with element-wise normalization."""
        # Split by pipe and normalize each element
        extracted_items = [item.strip() for item in extracted.split('|')]
        ground_truth_items = [item.strip() for item in ground_truth.split('|')]

        if len(extracted_items) != len(ground_truth_items):
            return False

        # Compare each corresponding pair
        for ext_item, gt_item in zip(extracted_items, ground_truth_items, strict=False):
            if not self._single_item_match(ext_item.strip(), gt_item.strip()):
                return False

        return True

    def _single_item_match(self, extracted: str, ground_truth: str) -> bool:
        """Match single items with monetary and text normalization."""
        # Try monetary match first
        if self._is_monetary(extracted) or self._is_monetary(ground_truth):
            return self._monetary_match(extracted, ground_truth)

        # Try exact match
        if extracted.lower().strip() == ground_truth.lower().strip():
            return True

        # Try fuzzy match for text
        return self._fuzzy_text_match(extracted, ground_truth)

    def _fuzzy_text_match(self, extracted: str, ground_truth: str) -> bool:
        """Fuzzy matching for text fields using simple similarity."""
        # For short strings, require exact match
        if len(ground_truth) <= 3:
            return extracted == ground_truth

        # Simple character-based similarity
        if len(extracted) == 0 or len(ground_truth) == 0:
            return False

        # Calculate simple overlap ratio
        common_chars = sum(1 for c in extracted if c in ground_truth)
        similarity = common_chars / max(len(extracted), len(ground_truth))

        return similarity >= 0.85

    def calculate_batch_metrics(self, results: List[SimpleEvaluationResult]) -> Dict[str, Any]:
        """
        Calculate overall metrics for a batch of evaluations.

        Args:
            results: List of individual evaluation results

        Returns:
            Dictionary with batch-level metrics
        """
        if not results:
            return {"overall_accuracy": 0.0, "total_images": 0}

        total_fields = sum(r.total_fields for r in results)
        total_correct = sum(r.correct_fields for r in results)

        return {
            "overall_accuracy": total_correct / total_fields if total_fields > 0 else 0.0,
            "total_images": len(results),
            "total_fields_evaluated": total_fields,
            "total_correct_fields": total_correct,
            "average_accuracy_per_image": sum(r.accuracy for r in results) / len(results)
        }


# Testing
if __name__ == "__main__":
    print("🧪 Testing SimpleModelEvaluator\n")

    evaluator = SimpleModelEvaluator()

    # Test case 1: Perfect match
    extracted = {"SUPPLIER_NAME": "Test Company", "TOTAL_AMOUNT": "$100.00"}
    ground_truth = {"SUPPLIER_NAME": "Test Company", "TOTAL_AMOUNT": "$100.00"}

    result = evaluator.evaluate_extraction(extracted, ground_truth, "test1.jpg")
    print(f"✅ Perfect match - Accuracy: {result.accuracy:.1%}")

    # Test case 2: Partial match with monetary normalization
    extracted2 = {"SUPPLIER_NAME": "Test Company", "TOTAL_AMOUNT": "100.00"}
    ground_truth2 = {"SUPPLIER_NAME": "Test Company", "TOTAL_AMOUNT": "$100.00"}

    result2 = evaluator.evaluate_extraction(extracted2, ground_truth2, "test2.jpg")
    print(f"✅ Monetary normalization - Accuracy: {result2.accuracy:.1%}")

    # Test batch metrics
    batch_metrics = evaluator.calculate_batch_metrics([result, result2])
    print(f"✅ Batch metrics - Overall accuracy: {batch_metrics['overall_accuracy']:.1%}")

    print("\n✅ SimpleModelEvaluator test complete!")