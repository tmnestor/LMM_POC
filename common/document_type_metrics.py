#!/usr/bin/env python3
"""
Document Type Specific Evaluation Metrics Module - DOCUMENT AWARE REDUCTION

Provides targeted evaluation metrics for each document type (invoice, bank_statement, receipt)
with appropriate thresholds and validation rules for Phase 4 integration.

DOCUMENT AWARE REDUCTION UPDATES:
- Updated ATO compliance checking for invoice basic fields only
- Hardcoded fallbacks use reduced field schema (11 invoice, 5 bank statement)
- Metrics calculations work with reduced field counts
- Receipt documents use same schema as invoices per boss requirement
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DocumentTypeMetrics:
    """Metrics configuration for specific document types."""

    document_type: str
    required_fields: List[str]
    optional_fields: List[str]
    accuracy_threshold: float
    critical_fields: List[str]  # Fields that must be 100% accurate
    ato_compliance_fields: List[str]  # ATO mandatory fields for invoices


class DocumentTypeEvaluator:
    """Evaluator that provides document-type-specific metrics and scoring."""

    def __init__(self, config_file: Optional[str] = None):
        """Initialize with document-type-specific metric definitions from unified schema."""

        # Load metrics from unified schema (single source of truth)
        self.metrics_config = self._load_metrics_from_unified_schema()
        self.field_categories = self._load_field_categories_from_unified_schema()
        self.evaluation_config = self._load_evaluation_config_from_unified_schema()

    def _load_metrics_from_unified_schema(self) -> Dict[str, DocumentTypeMetrics]:
        """Load document metrics from unified schema (single source of truth)."""
        from .unified_schema import DocumentTypeFieldSchema

        schema = DocumentTypeFieldSchema()
        metrics_config = {}

        # Get supported document types from unified schema
        supported_types = ["invoice", "receipt", "bank_statement"]

        for doc_type in supported_types:
            # Get fields for this document type from unified schema
            required_fields = schema.get_document_fields(doc_type)

            # Create simplified DocumentTypeMetrics with only essential info
            metrics_config[doc_type] = DocumentTypeMetrics(
                document_type=doc_type,
                required_fields=required_fields,
                optional_fields=[],  # Simplified - no optional fields
                critical_fields=self._get_critical_fields(doc_type),
                ato_compliance_fields=self._get_ato_fields(doc_type)
                if doc_type == "invoice"
                else [],
                accuracy_threshold=self._get_accuracy_threshold(doc_type),
            )

        return metrics_config

    def _get_critical_fields(self, doc_type: str) -> List[str]:
        """Get critical fields for document type from unified schema."""
        from .unified_schema import DocumentTypeFieldSchema

        schema = DocumentTypeFieldSchema()
        # Access document_types configuration from unified schema
        doc_types_config = schema.unified_schema.get("document_types", {})
        doc_config = doc_types_config.get(doc_type, {})
        return doc_config.get("critical_fields", [])

    def _get_ato_fields(self, doc_type: str) -> List[str]:
        """Get ATO compliance fields from unified schema."""
        from .unified_schema import DocumentTypeFieldSchema

        schema = DocumentTypeFieldSchema()
        # Access document_types configuration from unified schema
        doc_types_config = schema.unified_schema.get("document_types", {})
        doc_config = doc_types_config.get(doc_type, {})
        return doc_config.get("ato_compliance_fields", [])

    def _get_accuracy_threshold(self, doc_type: str) -> float:
        """Get accuracy threshold for document type from unified schema."""
        from .unified_schema import DocumentTypeFieldSchema

        schema = DocumentTypeFieldSchema()
        # Access document_types configuration from unified schema
        doc_types_config = schema.unified_schema.get("document_types", {})
        doc_config = doc_types_config.get(doc_type, {})
        return doc_config.get("accuracy_threshold", 0.75)

    def _load_field_categories_from_unified_schema(self) -> Dict[str, List[str]]:
        """Load field categories from unified schema (single source of truth)."""
        from .unified_schema import DocumentTypeFieldSchema

        schema = DocumentTypeFieldSchema()
        # Access field_categories configuration from unified schema
        return schema.unified_schema.get("field_categories", {})

    def _load_evaluation_config_from_unified_schema(self) -> Dict[str, Any]:
        """Load evaluation configuration from unified schema (single source of truth)."""
        return {"field_accuracy_threshold": 0.8}

    # REMOVED: Old complex _load_metrics_config method - replaced with unified schema approach

    # REMOVED: Old complex _load_field_categories method - replaced with unified schema approach

    # REMOVED: Old complex _load_evaluation_config method - replaced with unified schema approach

    def evaluate_extraction(
        self,
        extracted_data: Dict[str, Any],
        ground_truth: Dict[str, Any],
        document_type: str,
    ) -> Dict[str, Any]:
        """
        Evaluate extraction results with document-type-specific metrics.

        Args:
            extracted_data: Model extraction results
            ground_truth: Ground truth data
            document_type: Type of document (invoice, bank_statement, receipt)

        Returns:
            Evaluation results with type-specific metrics
        """

        # Normalize document type
        doc_type = self._normalize_document_type(document_type)

        if doc_type not in self.metrics_config:
            # Fallback to generic evaluation
            return self._generic_evaluation(extracted_data, ground_truth)

        metrics = self.metrics_config[doc_type]
        results = {
            "document_type": doc_type,
            "timestamp": datetime.now().isoformat(),
            "field_scores": {},
            "category_scores": {},
            "overall_metrics": {},
        }

        # Evaluate required fields
        required_scores = []
        for field in metrics.required_fields:
            score = self._calculate_field_score(
                extracted_data.get(field, "NOT_FOUND"),
                ground_truth.get(field, "NOT_FOUND"),
                field in metrics.critical_fields,
                field_name=field,
            )
            results["field_scores"][field] = score
            required_scores.append(score["accuracy"])

        # Evaluate optional fields (if present in ground truth)
        optional_scores = []
        for field in metrics.optional_fields:
            if ground_truth.get(field) and ground_truth.get(field) != "NOT_FOUND":
                score = self._calculate_field_score(
                    extracted_data.get(field, "NOT_FOUND"),
                    ground_truth.get(field, "NOT_FOUND"),
                    False,  # Optional fields are not critical
                    field_name=field,
                )
                results["field_scores"][field] = score
                optional_scores.append(score["accuracy"])

        # Calculate category scores
        results["category_scores"] = self._calculate_category_scores(
            results["field_scores"], doc_type
        )

        # Calculate overall metrics
        all_scores = required_scores + optional_scores
        results["overall_metrics"] = {
            "required_fields_accuracy": sum(required_scores) / len(required_scores)
            if required_scores
            else 0,
            "optional_fields_accuracy": sum(optional_scores) / len(optional_scores)
            if optional_scores
            else 0,
            "overall_accuracy": sum(all_scores) / len(all_scores) if all_scores else 0,
            "critical_fields_perfect": all(
                results["field_scores"].get(field, {}).get("accuracy", 0) == 1.0
                for field in metrics.critical_fields
            ),
            "meets_threshold": (sum(all_scores) / len(all_scores) if all_scores else 0)
            >= metrics.accuracy_threshold,
            "document_type_threshold": metrics.accuracy_threshold,
            "total_fields_evaluated": len(all_scores),
            "fields_correct": sum(1 for s in all_scores if s >= 0.8),
            "fields_perfect": sum(1 for s in all_scores if s == 1.0),
        }

        # Add ATO compliance check for invoices
        if doc_type == "invoice":
            results["overall_metrics"]["ato_compliant"] = self._check_ato_compliance(
                results["field_scores"], metrics.ato_compliance_fields
            )

        return results

    def _calculate_field_score(
        self, extracted: Any, truth: Any, is_critical: bool, field_name: str = None
    ) -> Dict[str, Any]:
        """Calculate score for a single field."""

        # Convert to strings for comparison
        extracted_str = str(extracted).strip()
        truth_str = str(truth).strip()

        # Handle NOT_FOUND cases
        if truth_str == "NOT_FOUND":
            if extracted_str == "NOT_FOUND":
                return {"accuracy": 1.0, "match_type": "correct_not_found"}
            else:
                return {"accuracy": 0.0, "match_type": "false_positive"}

        if extracted_str == "NOT_FOUND":
            return {
                "accuracy": 0.0,
                "match_type": "false_negative",
                "is_critical": is_critical,
            }

        # Check if this is a currency/numeric field
        currency_fields = [
            "TOTAL_AMOUNT",
            "GST_AMOUNT",
            "SUBTOTAL_AMOUNT",
            "LINE_ITEM_PRICES",
            "LINE_ITEM_TOTAL_PRICES",
            "ACCOUNT_CLOSING_BALANCE",
            "ACCOUNT_OPENING_BALANCE",
        ]

        if field_name and field_name in currency_fields:
            # Normalize currency values for comparison
            extracted_normalized = self._normalize_currency(extracted_str)
            truth_normalized = self._normalize_currency(truth_str)

            if extracted_normalized == truth_normalized:
                return {"accuracy": 1.0, "match_type": "exact"}

        # Exact match
        if extracted_str.lower() == truth_str.lower():
            return {"accuracy": 1.0, "match_type": "exact"}

        # Fuzzy match for text fields
        if self._fuzzy_match(extracted_str, truth_str):
            return {"accuracy": 0.8, "match_type": "fuzzy"}

        return {"accuracy": 0.0, "match_type": "mismatch", "is_critical": is_critical}

    def _normalize_currency(self, value: str) -> str:
        """Normalize currency values for comparison."""
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

    def _fuzzy_match(self, str1: str, str2: str, threshold: float = 0.8) -> bool:
        """Simple fuzzy matching for text fields."""

        # Normalize strings
        s1 = str1.lower().replace(" ", "").replace("-", "").replace(".", "")
        s2 = str2.lower().replace(" ", "").replace("-", "").replace(".", "")

        # Check if one contains the other
        if s1 in s2 or s2 in s1:
            return True

        # Check character overlap ratio
        common_chars = set(s1) & set(s2)
        total_chars = set(s1) | set(s2)

        if len(total_chars) > 0:
            overlap = len(common_chars) / len(total_chars)
            return overlap >= threshold

        return False

    def _calculate_category_scores(
        self, field_scores: Dict, doc_type: str
    ) -> Dict[str, float]:
        """Calculate scores by category (metadata, financial, etc.)."""

        categories = {
            "metadata": ["DOCUMENT_TYPE", "INVOICE_NUMBER", "INVOICE_DATE", "DUE_DATE"],
            "business_info": [
                "SUPPLIER_NAME",
                "BUSINESS_ABN",
                "BUSINESS_ADDRESS",
                "BUSINESS_PHONE",
            ],
            "customer_info": [
                "PAYER_NAME",
                "PAYER_ABN",
                "PAYER_ADDRESS",
                "PAYER_PHONE",
            ],
            "financial": ["SUBTOTAL_AMOUNT", "GST_AMOUNT", "TOTAL_AMOUNT"],
            "line_items": [
                "LINE_ITEM_DESCRIPTIONS",
                "LINE_ITEM_QUANTITIES",
                "LINE_ITEM_PRICES",
            ],
            "banking": [
                "BANK_NAME",
                "BANK_BSB_NUMBER",
                "BANK_ACCOUNT_NUMBER",
                "ACCOUNT_CLOSING_BALANCE",
            ],
        }

        category_scores = {}
        for category, fields in categories.items():
            scores = [field_scores[f]["accuracy"] for f in fields if f in field_scores]
            if scores:
                category_scores[category] = sum(scores) / len(scores)

        return category_scores

    def _check_ato_compliance(
        self, field_scores: Dict, compliance_fields: List[str]
    ) -> bool:
        """Check if ATO compliance requirements are met."""

        for field in compliance_fields:
            if field not in field_scores or field_scores[field]["accuracy"] < 1.0:
                return False
        return True

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
        """Fallback generic evaluation for unknown document types."""

        results = {
            "document_type": "unknown",
            "timestamp": datetime.now().isoformat(),
            "field_scores": {},
            "overall_metrics": {},
        }

        # Evaluate all fields present in ground truth
        scores = []
        for field, truth_value in ground_truth.items():
            if field == "image_file":
                continue

            score = self._calculate_field_score(
                extracted_data.get(field, "NOT_FOUND"), truth_value, False
            )
            results["field_scores"][field] = score
            scores.append(score["accuracy"])

        results["overall_metrics"] = {
            "overall_accuracy": sum(scores) / len(scores) if scores else 0,
            "total_fields": len(scores),
            "fields_correct": sum(1 for s in scores if s >= 0.8),
            "fields_perfect": sum(1 for s in scores if s == 1.0),
        }

        return results

    def generate_metrics_report(
        self, evaluations: List[Dict], output_path: str = None
    ) -> str:
        """Generate a comprehensive metrics report for multiple evaluations."""

        report = []
        report.append("=" * 80)
        report.append("DOCUMENT TYPE SPECIFIC EVALUATION METRICS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Evaluations: {len(evaluations)}")
        report.append("")

        # Group by document type
        by_type = {}
        for eval_result in evaluations:
            doc_type = eval_result.get("document_type", "unknown")
            if doc_type not in by_type:
                by_type[doc_type] = []
            by_type[doc_type].append(eval_result)

        # Report for each document type
        for doc_type, type_evals in by_type.items():
            report.append(f"\n{'=' * 40}")
            report.append(f"Document Type: {doc_type.upper()}")
            report.append(f"{'=' * 40}")

            metrics = self.metrics_config.get(doc_type)
            if metrics:
                report.append(
                    f"Accuracy Threshold: {metrics.accuracy_threshold * 100:.0f}%"
                )
                report.append(f"Critical Fields: {', '.join(metrics.critical_fields)}")
                if metrics.ato_compliance_fields:
                    report.append(
                        f"ATO Required: {', '.join(metrics.ato_compliance_fields)}"
                    )

            # Calculate aggregate metrics
            overall_accuracies = [
                e["overall_metrics"]["overall_accuracy"] for e in type_evals
            ]
            meets_threshold = [
                e["overall_metrics"].get("meets_threshold", False) for e in type_evals
            ]
            critical_perfect = [
                e["overall_metrics"].get("critical_fields_perfect", False)
                for e in type_evals
            ]

            report.append("\nSummary Statistics:")
            report.append(f"  • Documents Evaluated: {len(type_evals)}")
            report.append(
                f"  • Average Accuracy: {sum(overall_accuracies) / len(overall_accuracies) * 100:.1f}%"
            )
            report.append(
                f"  • Meeting Threshold: {sum(meets_threshold)} / {len(meets_threshold)} ({sum(meets_threshold) / len(meets_threshold) * 100:.0f}%)"
            )
            report.append(
                f"  • Critical Fields Perfect: {sum(critical_perfect)} / {len(critical_perfect)} ({sum(critical_perfect) / len(critical_perfect) * 100:.0f}%)"
            )

            # Category breakdown
            if type_evals[0].get("category_scores"):
                report.append("\nCategory Performance:")
                category_totals = {}
                for eval_result in type_evals:
                    for cat, score in eval_result.get("category_scores", {}).items():
                        if cat not in category_totals:
                            category_totals[cat] = []
                        category_totals[cat].append(score)

                for cat, scores in sorted(category_totals.items()):
                    avg_score = sum(scores) / len(scores) * 100
                    report.append(f"  • {cat:20s}: {avg_score:5.1f}%")

            # ATO Compliance for invoices
            if doc_type == "invoice":
                ato_compliant = [
                    e["overall_metrics"].get("ato_compliant", False) for e in type_evals
                ]
                report.append(
                    f"\nATO Compliance: {sum(ato_compliant)} / {len(ato_compliant)} ({sum(ato_compliant) / len(ato_compliant) * 100:.0f}%)"
                )

        report.append("\n" + "=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)

        report_text = "\n".join(report)

        # Save to file if path provided
        if output_path:
            output_file = Path(output_path)
            output_file.write_text(report_text)
            print(f"✅ Metrics report saved to: {output_path}")

        return report_text


def main():
    """Test the document type evaluator with sample data."""

    evaluator = DocumentTypeEvaluator()

    # Sample test data
    extracted = {
        "DOCUMENT_TYPE": "TAX INVOICE",
        "INVOICE_NUMBER": "INV-12345",
        "BUSINESS_ABN": "12 345 678 901",
        "SUPPLIER_NAME": "Test Company Pty Ltd",
        "GST_AMOUNT": "$10.00",
        "TOTAL_AMOUNT": "$110.00",
        "INVOICE_DATE": "01/01/2024",
    }

    ground_truth = {
        "DOCUMENT_TYPE": "TAX INVOICE",
        "INVOICE_NUMBER": "INV-12345",
        "BUSINESS_ABN": "12 345 678 901",
        "SUPPLIER_NAME": "Test Company Pty Ltd",
        "GST_AMOUNT": "$10.00",
        "TOTAL_AMOUNT": "$110.00",
        "INVOICE_DATE": "01/01/2024",
    }

    result = evaluator.evaluate_extraction(extracted, ground_truth, "invoice")
    print(json.dumps(result, indent=2))

    # Generate sample report
    report = evaluator.generate_metrics_report([result])
    print("\n" + report)


if __name__ == "__main__":
    main()
