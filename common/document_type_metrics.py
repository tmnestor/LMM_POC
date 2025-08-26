#!/usr/bin/env python3
"""
Document Type Specific Evaluation Metrics Module

Provides targeted evaluation metrics for each document type (invoice, bank_statement, receipt)
with appropriate thresholds and validation rules for Phase 4 integration.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


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
    
    def __init__(self):
        """Initialize with document-type-specific metric definitions."""
        
        # Define metrics for each document type
        self.metrics_config = {
            "invoice": DocumentTypeMetrics(
                document_type="invoice",
                required_fields=[
                    "DOCUMENT_TYPE", "INVOICE_NUMBER", "INVOICE_DATE", "DUE_DATE",
                    "SUPPLIER_NAME", "BUSINESS_ABN", "BUSINESS_ADDRESS", "BUSINESS_PHONE",
                    "LINE_ITEM_DESCRIPTIONS", "LINE_ITEM_QUANTITIES", "LINE_ITEM_PRICES",
                    "SUBTOTAL_AMOUNT", "GST_AMOUNT", "TOTAL_AMOUNT"
                ],
                optional_fields=[
                    "PAYER_NAME", "PAYER_ABN", "PAYER_ADDRESS", "PAYER_PHONE", 
                    "PAYER_EMAIL", "SUPPLIER_WEBSITE"
                ],
                critical_fields=["BUSINESS_ABN", "GST_AMOUNT", "TOTAL_AMOUNT", "INVOICE_NUMBER"],
                ato_compliance_fields=["BUSINESS_ABN", "GST_AMOUNT", "INVOICE_DATE", "SUPPLIER_NAME"],
                accuracy_threshold=0.95  # 95% accuracy required for invoices
            ),
            "bank_statement": DocumentTypeMetrics(
                document_type="bank_statement",
                required_fields=[
                    "DOCUMENT_TYPE", "BANK_NAME", "BANK_BSB_NUMBER", "BANK_ACCOUNT_NUMBER",
                    "BANK_ACCOUNT_HOLDER", "STATEMENT_DATE_RANGE", 
                    "ACCOUNT_OPENING_BALANCE", "ACCOUNT_CLOSING_BALANCE"
                ],
                optional_fields=[
                    "TOTAL_CREDITS", "TOTAL_DEBITS", "LINE_ITEM_DESCRIPTIONS"
                ],
                critical_fields=["BANK_ACCOUNT_NUMBER", "ACCOUNT_CLOSING_BALANCE"],
                ato_compliance_fields=[],  # Not applicable for bank statements
                accuracy_threshold=0.90  # 90% accuracy for bank statements
            ),
            "receipt": DocumentTypeMetrics(
                document_type="receipt",
                required_fields=[
                    "DOCUMENT_TYPE", "SUPPLIER_NAME", "BUSINESS_ABN",
                    "LINE_ITEM_DESCRIPTIONS", "LINE_ITEM_QUANTITIES", "LINE_ITEM_PRICES",
                    "SUBTOTAL_AMOUNT", "GST_AMOUNT", "TOTAL_AMOUNT", "INVOICE_DATE"
                ],
                optional_fields=[
                    "BUSINESS_ADDRESS", "BUSINESS_PHONE", "SUPPLIER_WEBSITE",
                    "PAYER_NAME", "PAYER_ADDRESS"
                ],
                critical_fields=["TOTAL_AMOUNT", "GST_AMOUNT"],
                ato_compliance_fields=["BUSINESS_ABN", "GST_AMOUNT"],
                accuracy_threshold=0.85  # 85% accuracy for receipts
            )
        }
    
    def evaluate_extraction(self, 
                           extracted_data: Dict[str, Any], 
                           ground_truth: Dict[str, Any],
                           document_type: str) -> Dict[str, Any]:
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
            "overall_metrics": {}
        }
        
        # Evaluate required fields
        required_scores = []
        for field in metrics.required_fields:
            score = self._calculate_field_score(
                extracted_data.get(field, "NOT_FOUND"),
                ground_truth.get(field, "NOT_FOUND"),
                field in metrics.critical_fields
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
                    False  # Optional fields are not critical
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
            "required_fields_accuracy": sum(required_scores) / len(required_scores) if required_scores else 0,
            "optional_fields_accuracy": sum(optional_scores) / len(optional_scores) if optional_scores else 0,
            "overall_accuracy": sum(all_scores) / len(all_scores) if all_scores else 0,
            "critical_fields_perfect": all(
                results["field_scores"].get(field, {}).get("accuracy", 0) == 1.0 
                for field in metrics.critical_fields
            ),
            "meets_threshold": (sum(all_scores) / len(all_scores) if all_scores else 0) >= metrics.accuracy_threshold,
            "document_type_threshold": metrics.accuracy_threshold,
            "total_fields_evaluated": len(all_scores),
            "fields_correct": sum(1 for s in all_scores if s >= 0.8),
            "fields_perfect": sum(1 for s in all_scores if s == 1.0)
        }
        
        # Add ATO compliance check for invoices
        if doc_type == "invoice":
            results["overall_metrics"]["ato_compliant"] = self._check_ato_compliance(
                results["field_scores"], metrics.ato_compliance_fields
            )
        
        return results
    
    def _calculate_field_score(self, extracted: Any, truth: Any, is_critical: bool) -> Dict[str, Any]:
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
            return {"accuracy": 0.0, "match_type": "false_negative", "is_critical": is_critical}
        
        # Exact match
        if extracted_str.lower() == truth_str.lower():
            return {"accuracy": 1.0, "match_type": "exact"}
        
        # Fuzzy match for text fields
        if self._fuzzy_match(extracted_str, truth_str):
            return {"accuracy": 0.8, "match_type": "fuzzy"}
        
        return {"accuracy": 0.0, "match_type": "mismatch", "is_critical": is_critical}
    
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
    
    def _calculate_category_scores(self, field_scores: Dict, doc_type: str) -> Dict[str, float]:
        """Calculate scores by category (metadata, financial, etc.)."""
        
        categories = {
            "metadata": ["DOCUMENT_TYPE", "INVOICE_NUMBER", "INVOICE_DATE", "DUE_DATE"],
            "business_info": ["SUPPLIER_NAME", "BUSINESS_ABN", "BUSINESS_ADDRESS", "BUSINESS_PHONE"],
            "customer_info": ["PAYER_NAME", "PAYER_ABN", "PAYER_ADDRESS", "PAYER_PHONE"],
            "financial": ["SUBTOTAL_AMOUNT", "GST_AMOUNT", "TOTAL_AMOUNT"],
            "line_items": ["LINE_ITEM_DESCRIPTIONS", "LINE_ITEM_QUANTITIES", "LINE_ITEM_PRICES"],
            "banking": ["BANK_NAME", "BANK_BSB_NUMBER", "BANK_ACCOUNT_NUMBER", "ACCOUNT_CLOSING_BALANCE"]
        }
        
        category_scores = {}
        for category, fields in categories.items():
            scores = [field_scores[f]["accuracy"] for f in fields if f in field_scores]
            if scores:
                category_scores[category] = sum(scores) / len(scores)
        
        return category_scores
    
    def _check_ato_compliance(self, field_scores: Dict, compliance_fields: List[str]) -> bool:
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
            "overall_metrics": {}
        }
        
        # Evaluate all fields present in ground truth
        scores = []
        for field, truth_value in ground_truth.items():
            if field == "image_file":
                continue
                
            score = self._calculate_field_score(
                extracted_data.get(field, "NOT_FOUND"),
                truth_value,
                False
            )
            results["field_scores"][field] = score
            scores.append(score["accuracy"])
        
        results["overall_metrics"] = {
            "overall_accuracy": sum(scores) / len(scores) if scores else 0,
            "total_fields": len(scores),
            "fields_correct": sum(1 for s in scores if s >= 0.8),
            "fields_perfect": sum(1 for s in scores if s == 1.0)
        }
        
        return results
    
    def generate_metrics_report(self, evaluations: List[Dict], output_path: str = None) -> str:
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
                report.append(f"Accuracy Threshold: {metrics.accuracy_threshold * 100:.0f}%")
                report.append(f"Critical Fields: {', '.join(metrics.critical_fields)}")
                if metrics.ato_compliance_fields:
                    report.append(f"ATO Required: {', '.join(metrics.ato_compliance_fields)}")
            
            # Calculate aggregate metrics
            overall_accuracies = [e["overall_metrics"]["overall_accuracy"] for e in type_evals]
            meets_threshold = [e["overall_metrics"].get("meets_threshold", False) for e in type_evals]
            critical_perfect = [e["overall_metrics"].get("critical_fields_perfect", False) for e in type_evals]
            
            report.append("\nSummary Statistics:")
            report.append(f"  • Documents Evaluated: {len(type_evals)}")
            report.append(f"  • Average Accuracy: {sum(overall_accuracies) / len(overall_accuracies) * 100:.1f}%")
            report.append(f"  • Meeting Threshold: {sum(meets_threshold)} / {len(meets_threshold)} ({sum(meets_threshold) / len(meets_threshold) * 100:.0f}%)")
            report.append(f"  • Critical Fields Perfect: {sum(critical_perfect)} / {len(critical_perfect)} ({sum(critical_perfect) / len(critical_perfect) * 100:.0f}%)")
            
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
                ato_compliant = [e["overall_metrics"].get("ato_compliant", False) for e in type_evals]
                report.append(f"\nATO Compliance: {sum(ato_compliant)} / {len(ato_compliant)} ({sum(ato_compliant) / len(ato_compliant) * 100:.0f}%)")
        
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
        "INVOICE_DATE": "01/01/2024"
    }
    
    ground_truth = {
        "DOCUMENT_TYPE": "TAX INVOICE",
        "INVOICE_NUMBER": "INV-12345", 
        "BUSINESS_ABN": "12 345 678 901",
        "SUPPLIER_NAME": "Test Company Pty Ltd",
        "GST_AMOUNT": "$10.00",
        "TOTAL_AMOUNT": "$110.00",
        "INVOICE_DATE": "01/01/2024"
    }
    
    result = evaluator.evaluate_extraction(extracted, ground_truth, "invoice")
    print(json.dumps(result, indent=2))
    
    # Generate sample report
    report = evaluator.generate_metrics_report([result])
    print("\n" + report)


if __name__ == "__main__":
    main()