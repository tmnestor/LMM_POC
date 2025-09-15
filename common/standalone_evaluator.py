"""
Standalone Document Type Evaluator

A simplified, standalone version of DocumentTypeEvaluator designed for the clean architecture.
Avoids complex dependencies and infinite recursion while preserving accuracy-critical features.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class DocumentMetrics:
    """Simple metrics configuration for document types."""

    document_type: str
    required_fields: List[str]
    critical_fields: List[str]
    accuracy_threshold: float


class StandaloneEvaluator:
    """Standalone evaluator with accuracy-critical features from DocumentTypeEvaluator."""

    def __init__(self):
        """Initialize with hardcoded metrics to avoid complex dependencies."""
        self.metrics_config = self._get_metrics_config()

    def _get_metrics_config(self) -> Dict[str, DocumentMetrics]:
        """Get hardcoded metrics configuration to avoid schema dependencies."""
        return {
            "invoice": DocumentMetrics(
                document_type="invoice",
                required_fields=[
                    'DOCUMENT_TYPE', 'BUSINESS_ABN', 'SUPPLIER_NAME', 'BUSINESS_ADDRESS',
                    'PAYER_NAME', 'PAYER_ADDRESS', 'INVOICE_DATE', 'LINE_ITEM_DESCRIPTIONS',
                    'LINE_ITEM_QUANTITIES', 'LINE_ITEM_PRICES', 'LINE_ITEM_TOTAL_PRICES',
                    'IS_GST_INCLUDED', 'GST_AMOUNT', 'TOTAL_AMOUNT'
                ],
                critical_fields=['DOCUMENT_TYPE', 'SUPPLIER_NAME', 'TOTAL_AMOUNT'],
                accuracy_threshold=0.75
            ),
            "receipt": DocumentMetrics(
                document_type="receipt",
                required_fields=[
                    'DOCUMENT_TYPE', 'BUSINESS_ABN', 'SUPPLIER_NAME', 'BUSINESS_ADDRESS',
                    'PAYER_NAME', 'PAYER_ADDRESS', 'INVOICE_DATE', 'LINE_ITEM_DESCRIPTIONS',
                    'LINE_ITEM_QUANTITIES', 'LINE_ITEM_PRICES', 'LINE_ITEM_TOTAL_PRICES',
                    'IS_GST_INCLUDED', 'GST_AMOUNT', 'TOTAL_AMOUNT'
                ],
                critical_fields=['DOCUMENT_TYPE', 'SUPPLIER_NAME', 'TOTAL_AMOUNT'],
                accuracy_threshold=0.75
            ),
            "bank_statement": DocumentMetrics(
                document_type="bank_statement",
                required_fields=[
                    'DOCUMENT_TYPE', 'STATEMENT_DATE_RANGE', 'LINE_ITEM_DESCRIPTIONS',
                    'TRANSACTION_DATES', 'TRANSACTION_AMOUNTS_PAID', 'TRANSACTION_AMOUNTS_RECEIVED',
                    'ACCOUNT_BALANCE'
                ],
                critical_fields=['DOCUMENT_TYPE', 'STATEMENT_DATE_RANGE', 'ACCOUNT_BALANCE'],
                accuracy_threshold=0.70
            )
        }

    def evaluate_extraction(
        self,
        extracted_data: Dict[str, Any],
        ground_truth: Dict[str, Any],
        document_type: str,
    ) -> Dict[str, Any]:
        """
        Evaluate extraction results with enhanced accuracy scoring.

        This includes the key accuracy features from DocumentTypeEvaluator:
        - Fuzzy matching for text fields
        - Currency normalization for financial fields
        - Document-type-specific thresholds
        """

        # Normalize document type
        doc_type = self._normalize_document_type(document_type)

        if doc_type not in self.metrics_config:
            return self._generic_evaluation(extracted_data, ground_truth)

        metrics = self.metrics_config[doc_type]
        results = {
            "document_type": doc_type,
            "timestamp": datetime.now().isoformat(),
            "field_scores": {},
            "overall_metrics": {},
        }

        # Evaluate required fields with enhanced scoring
        all_scores = []
        fields_correct = 0
        fields_perfect = 0
        fields_matched = 0  # Count fields with any positive match

        for field in metrics.required_fields:
            extracted_value = extracted_data.get(field, "NOT_FOUND")
            truth_value = ground_truth.get(field, "NOT_FOUND")

            score = self._calculate_enhanced_field_score(
                extracted_value, truth_value, field in metrics.critical_fields, field
            )

            results["field_scores"][field] = score
            accuracy = score["accuracy"]
            all_scores.append(accuracy)

            # Count matches consistently with field-level analysis
            if accuracy > 0.0:  # Any positive match (exact, fuzzy, currency)
                fields_matched += 1
            if accuracy >= 0.8:  # High confidence matches
                fields_correct += 1
            if accuracy == 1.0:  # Perfect matches
                fields_perfect += 1

        # Calculate overall metrics with enhanced logic
        overall_accuracy = sum(all_scores) / len(all_scores) if all_scores else 0

        results["overall_metrics"] = {
            "overall_accuracy": overall_accuracy,
            "total_fields_evaluated": len(all_scores),
            "fields_matched": fields_matched,  # Any positive match (>0.0 accuracy)
            "fields_correct": fields_correct,  # High confidence (>=0.8 accuracy)
            "fields_perfect": fields_perfect,  # Perfect matches (1.0 accuracy)
            "critical_fields_perfect": all(
                results["field_scores"].get(field, {}).get("accuracy", 0) == 1.0
                for field in metrics.critical_fields
            ),
            "meets_threshold": overall_accuracy >= metrics.accuracy_threshold,
            "document_type_threshold": metrics.accuracy_threshold,
        }

        return results

    def _calculate_enhanced_field_score(
        self, extracted: Any, truth: Any, is_critical: bool, field_name: str = None
    ) -> Dict[str, Any]:
        """Calculate enhanced field score with fuzzy matching and currency normalization."""

        # Convert to strings for comparison
        extracted_str = str(extracted).strip()
        truth_str = str(truth).strip()

        # Handle NOT_FOUND cases
        if truth_str == "NOT_FOUND":
            if extracted_str == "NOT_FOUND":
                return {"accuracy": 1.0, "match_type": "correct_not_found", "status": "MATCH"}
            else:
                return {"accuracy": 0.0, "match_type": "false_positive", "status": "MISS"}

        if extracted_str == "NOT_FOUND":
            return {
                "accuracy": 0.0,
                "match_type": "false_negative",
                "status": "MISS",
                "is_critical": is_critical,
            }

        # Currency field handling with normalization
        currency_fields = [
            "TOTAL_AMOUNT", "GST_AMOUNT", "SUBTOTAL_AMOUNT", "LINE_ITEM_PRICES",
            "LINE_ITEM_TOTAL_PRICES", "ACCOUNT_BALANCE", "TRANSACTION_AMOUNTS_PAID",
            "TRANSACTION_AMOUNTS_RECEIVED"
        ]

        if field_name and field_name in currency_fields:
            extracted_normalized = self._normalize_currency(extracted_str)
            truth_normalized = self._normalize_currency(truth_str)

            if extracted_normalized == truth_normalized:
                return {"accuracy": 1.0, "match_type": "exact_currency", "status": "MATCH"}

        # Exact match
        if extracted_str.lower() == truth_str.lower():
            return {"accuracy": 1.0, "match_type": "exact", "status": "MATCH"}

        # Enhanced fuzzy match for text fields
        if self._enhanced_fuzzy_match(extracted_str, truth_str):
            return {"accuracy": 0.8, "match_type": "fuzzy", "status": "MATCH"}

        return {"accuracy": 0.0, "match_type": "mismatch", "status": "MISS", "is_critical": is_critical}

    def _normalize_currency(self, value: str) -> str:
        """Enhanced currency normalization from DocumentTypeEvaluator."""
        import re

        # Remove currency symbols, whitespace, and commas
        normalized = re.sub(r"[$,\s]", "", value)

        # Handle negative values in parentheses
        if "(" in normalized and ")" in normalized:
            normalized = "-" + re.sub(r"[()]", "", normalized)

        # Ensure it's a valid number format
        try:
            # Convert to float and back to string to normalize decimal places
            float_val = float(normalized)
            # Format to 2 decimal places for currency
            return f"{float_val:.2f}"
        except (ValueError, TypeError):
            # If not a valid number, return the cleaned string
            return normalized

    def _enhanced_fuzzy_match(self, str1: str, str2: str, threshold: float = 0.8) -> bool:
        """Enhanced fuzzy matching logic from DocumentTypeEvaluator."""

        # Normalize strings
        s1 = str1.lower().replace(" ", "").replace("-", "").replace(".", "")
        s2 = str2.lower().replace(" ", "").replace("-", "").replace(".", "")

        # Quick exact match after normalization
        if s1 == s2:
            return True

        # Check if one contains the other (good for partial matches)
        if s1 in s2 or s2 in s1:
            return True

        # Character overlap ratio (handles typos and variations)
        common_chars = set(s1) & set(s2)
        total_chars = set(s1) | set(s2)

        if len(total_chars) > 0:
            overlap = len(common_chars) / len(total_chars)
            return overlap >= threshold

        return False

    def _normalize_document_type(self, doc_type: str) -> str:
        """Normalize document type string."""
        doc_type_lower = doc_type.lower().strip()

        if "invoice" in doc_type_lower or "tax" in doc_type_lower:
            return "invoice"
        elif "bank" in doc_type_lower or "statement" in doc_type_lower:
            return "bank_statement"
        elif "receipt" in doc_type_lower:
            return "receipt"

        return doc_type_lower

    def _generic_evaluation(self, extracted_data: Dict, ground_truth: Dict) -> Dict:
        """Fallback evaluation for unknown document types."""

        results = {
            "document_type": "unknown",
            "timestamp": datetime.now().isoformat(),
            "field_scores": {},
            "overall_metrics": {},
        }

        # Evaluate all fields present in ground truth
        scores = []
        fields_correct = 0
        fields_perfect = 0
        fields_matched = 0

        for field, truth_value in ground_truth.items():
            if field == "image_file":
                continue

            score = self._calculate_enhanced_field_score(
                extracted_data.get(field, "NOT_FOUND"), truth_value, False, field
            )
            results["field_scores"][field] = score
            accuracy = score["accuracy"]
            scores.append(accuracy)

            # Count matches consistently with field-level analysis
            if accuracy > 0.0:  # Any positive match
                fields_matched += 1
            if accuracy >= 0.8:  # High confidence matches
                fields_correct += 1
            if accuracy == 1.0:  # Perfect matches
                fields_perfect += 1

        results["overall_metrics"] = {
            "overall_accuracy": sum(scores) / len(scores) if scores else 0,
            "total_fields_evaluated": len(scores),
            "fields_matched": fields_matched,  # Any positive match (>0.0 accuracy)
            "fields_correct": fields_correct,  # High confidence (>=0.8 accuracy)
            "fields_perfect": fields_perfect,  # Perfect matches (1.0 accuracy)
        }

        return results


def main():
    """Test the standalone evaluator."""
    evaluator = StandaloneEvaluator()

    # Sample test data
    extracted = {
        "DOCUMENT_TYPE": "TAX INVOICE",
        "SUPPLIER_NAME": "Test Company Pty Ltd",
        "TOTAL_AMOUNT": "$110.00",
    }

    ground_truth = {
        "DOCUMENT_TYPE": "TAX INVOICE",
        "SUPPLIER_NAME": "Test Company Pty Ltd",
        "TOTAL_AMOUNT": "$110.00",
    }

    result = evaluator.evaluate_extraction(extracted, ground_truth, "invoice")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()