"""Standalone evaluator for document extraction results.

Extracted from BatchDocumentProcessor to enable independent testing
without GPU or model dependencies. Owns ground truth loading,
bank math enhancement, debit filtering, F1 scoring, and metric aggregation.
"""

import logging
import os
from pathlib import Path
from typing import Any

from .batch_types import ExtractionOutput
from .evaluation_metrics import (
    calculate_correlation_aware_f1,
    calculate_field_accuracy_with_method,
    load_ground_truth,
)
from .simple_model_evaluator import SimpleModelEvaluator

logger = logging.getLogger(__name__)


class ExtractionEvaluator:
    """Evaluates extracted fields against ground truth.

    Encapsulates: ground truth loading, bank math enhancement,
    debit transaction filtering, field-level F1 scoring
    (standard or correlation-aware), and metric aggregation.

    Stateless after construction -- safe to share across threads.
    """

    def __init__(
        self,
        ground_truth_csv: str | None,
        field_definitions: dict[str, list[str]],
        *,
        enable_math_enhancement: bool = False,
        evaluation_method: str | None = None,
        verbose: bool = False,
    ) -> None:
        """Load ground truth once; configure evaluation strategy.

        Args:
            ground_truth_csv: Path to CSV or JSONL. None = inference-only
                (evaluate returns empty dict).
            field_definitions: Map of doc_type (lowercase) -> field names
                for evaluation filtering. Ignored in JSONL mode (GT record
                keys are used instead).
            enable_math_enhancement: Apply bank statement balance calculations.
            evaluation_method: "order_aware_f1" or "correlation" / "correlation_aware_f1".
                None = read from EVALUATION_METHOD env var, default "order_aware_f1".
            verbose: Debug logging.
        """
        self.field_definitions = field_definitions
        self.enable_math_enhancement = enable_math_enhancement
        self.evaluation_method = evaluation_method or os.environ.get(
            "EVALUATION_METHOD", "order_aware_f1"
        )
        self._verbose = verbose
        self._model_evaluator = SimpleModelEvaluator()

        # JSONL mode: evaluate exactly the fields in each GT record
        self._jsonl_mode = (
            ground_truth_csv is not None and Path(ground_truth_csv).suffix == ".jsonl"
        )

        # Load ground truth once — fail fast if path given but missing
        self._ground_truth_data: dict[str, dict] = {}
        if ground_truth_csv:
            self._ground_truth_data = load_ground_truth(
                ground_truth_csv, verbose=verbose
            )
            if not self._ground_truth_data:
                msg = (
                    f"Ground truth loaded from {ground_truth_csv} but contains "
                    f"0 records. Check file format and contents."
                )
                raise ValueError(msg)
            logger.info(
                "Loaded ground truth for %d images%s",
                len(self._ground_truth_data),
                " (JSONL mode)" if self._jsonl_mode else "",
            )
        else:
            logger.info("No ground truth configured -- inference-only mode")

    @property
    def has_ground_truth(self) -> bool:
        """Whether ground truth data is available."""
        return bool(self._ground_truth_data)

    def evaluate(self, extraction: ExtractionOutput) -> dict[str, Any]:
        """Score one image's extraction against ground truth.

        Returns empty dict when no ground truth is configured or available
        for this image. Pure computation -- no model calls, no GPU.

        Args:
            extraction: The extraction output to evaluate.

        Returns:
            Evaluation dict with accuracy, F1 scores, and field-level metrics.
            Empty dict with 'error' key when no ground truth available.
        """
        ground_truth = self._lookup_ground_truth(extraction.image_path)
        if not ground_truth:
            return {
                "error": f"No ground truth for {extraction.image_name}",
                "overall_accuracy": 0,
            }

        return self._evaluate_extraction(
            extracted_data=dict(extraction.extracted_data),
            ground_truth=ground_truth,
            document_type=extraction.document_type,
            image_name=extraction.image_name,
        )

    def _lookup_ground_truth(self, image_path: str) -> dict:
        """Look up ground truth for an image with fuzzy matching."""
        if not self._ground_truth_data:
            return {}

        image_name = Path(image_path).name

        # Try exact match first
        ground_truth = self._ground_truth_data.get(image_name, {})
        if ground_truth:
            return ground_truth

        # Try without extension
        image_stem = Path(image_path).stem
        for gt_key in self._ground_truth_data:
            if Path(gt_key).stem == image_stem or gt_key == image_stem:
                return self._ground_truth_data[gt_key]

        return {}

    def _evaluate_extraction(
        self,
        extracted_data: dict[str, Any],
        ground_truth: dict[str, Any],
        document_type: str,
        image_name: str,
    ) -> dict[str, Any]:
        """Evaluate extraction results against ground truth.

        Handles mathematical enhancement for bank statements, field filtering,
        F1 scoring (standard or correlation), and metric aggregation.
        """
        # Apply mathematical enhancement for bank statements.
        # The calculator is idempotent: when BalanceCorrector already corrected
        # during extraction, amounts and balance deltas already agree so
        # the calculator makes zero corrections.
        mathematical_analysis = None
        if document_type.upper() == "BANK_STATEMENT" and self.enable_math_enhancement:
            from .bank_statement_calculator import enhance_bank_statement_extraction

            logger.debug("Applying mathematical enhancement for bank statement")

            enhanced_result = enhance_bank_statement_extraction(
                extracted_data, verbose=self._verbose
            )

            extracted_data = {
                k: v
                for k, v in enhanced_result.items()
                if k != "_mathematical_analysis"
            }

            mathematical_analysis = enhanced_result.get("_mathematical_analysis", {})

            logger.debug("Filtering to debit-only transactions for evaluation")
            extracted_data = self._filter_debit_transactions(extracted_data)

        found_fields = [k for k, v in extracted_data.items() if v != "NOT_FOUND"]
        logger.debug("Extracted %d fields from %s", len(found_fields), image_name)

        if (
            document_type.upper() == "BANK_STATEMENT"
            and mathematical_analysis is not None
        ):
            if mathematical_analysis.get("calculation_success"):
                logger.debug(
                    "Mathematical analysis: %d transactions calculated",
                    mathematical_analysis.get("transaction_count", 0),
                )
            else:
                logger.debug("Mathematical analysis failed")

        # Filter ground truth to document-specific fields for accurate evaluation.
        # JSONL mode: each GT record carries only its type's fields -- use them
        # directly instead of looking up field_definitions (handles new types
        # like TRAVEL/LOGBOOK without config changes).
        if self._jsonl_mode:
            _skip = {"filename", "image_name", "image_file", "file"}
            evaluation_fields = [k for k in ground_truth if k not in _skip]
        else:
            document_type_lower_eval = document_type.lower()
            evaluation_fields = self.field_definitions.get(
                document_type_lower_eval, self.field_definitions.get("invoice", [])
            )

        filtered_ground_truth = {
            field: ground_truth[field]
            for field in evaluation_fields
            if field in ground_truth
        }

        if document_type.upper() == "BANK_STATEMENT":
            logger.debug(
                "Evaluating using mathematically corrected values (not raw VLM output)"
            )

        evaluation_result = self._model_evaluator.evaluate_extraction(
            extracted_data, filtered_ground_truth, image_name
        )

        fields_extracted = len(
            [k for k, v in extracted_data.items() if v != "NOT_FOUND"]
        )

        # Build field-level F1 scores
        field_scores: dict[str, dict] = {}
        total_f1_score = 0.0
        total_precision = 0.0
        total_recall = 0.0

        if self.evaluation_method in ["correlation", "correlation_aware_f1"]:
            correlation_result = calculate_correlation_aware_f1(
                extracted_data,
                filtered_ground_truth,
                document_type,
                debug=False,
            )

            for field in filtered_ground_truth:
                field_scores[field] = correlation_result
                total_f1_score += correlation_result["f1_score"]
                total_precision += correlation_result["precision"]
                total_recall += correlation_result["recall"]
        else:
            for field in filtered_ground_truth:
                extracted_val = extracted_data.get(field, "NOT_FOUND")
                ground_val = filtered_ground_truth.get(field, "NOT_FOUND")

                is_debug = field == "IS_GST_INCLUDED" and self._verbose
                if is_debug:
                    logger.debug("BEFORE EVALUATION:")
                    logger.debug(
                        "  extracted_val = '%s' (type: %s)",
                        extracted_val,
                        type(extracted_val).__name__,
                    )
                    logger.debug(
                        "  ground_val = '%s' (type: %s)",
                        ground_val,
                        type(ground_val).__name__,
                    )
                    logger.debug("  Are they equal? %s", extracted_val == ground_val)

                f1_metrics = calculate_field_accuracy_with_method(
                    extracted_val,
                    ground_val,
                    field,
                    method=self.evaluation_method,
                    debug=is_debug,
                )

                if is_debug:
                    logger.debug("AFTER EVALUATION (%s):", field)
                    logger.debug(
                        "  Field '%s' f1_score = %s",
                        field,
                        f1_metrics["f1_score"],
                    )

                field_scores[field] = f1_metrics
                total_f1_score += f1_metrics["f1_score"]
                total_precision += f1_metrics["precision"]
                total_recall += f1_metrics["recall"]

        # Calculate overall metrics from F1 scores
        num_fields = len(field_scores)
        overall_accuracy = total_f1_score / num_fields if num_fields else 0.0
        overall_precision = total_precision / num_fields if num_fields else 0.0
        overall_recall = total_recall / num_fields if num_fields else 0.0

        # Calculate median F1 (more robust to outliers than mean)
        f1_values = [s["f1_score"] for s in field_scores.values()]
        if f1_values:
            sorted_f1 = sorted(f1_values)
            mid = len(sorted_f1) // 2
            if len(sorted_f1) % 2 == 0:
                median_f1 = (sorted_f1[mid - 1] + sorted_f1[mid]) / 2
            else:
                median_f1 = sorted_f1[mid]
        else:
            median_f1 = 0.0

        perfect_matches = sum(
            1 for score in field_scores.values() if score["f1_score"] == 1.0
        )

        evaluation = {
            "overall_accuracy": overall_accuracy,
            "median_f1": median_f1,
            "overall_precision": overall_precision,
            "overall_recall": overall_recall,
            "total_fields": len(field_scores),
            "correct_fields": perfect_matches,
            "missing_fields": evaluation_result.missing_fields,
            "incorrect_fields": evaluation_result.incorrect_fields,
            "fields_extracted": fields_extracted,
            "fields_matched": perfect_matches,
            "field_scores": field_scores,
            "overall_metrics": {
                "overall_accuracy": overall_accuracy,
                "median_f1": median_f1,
                "overall_precision": overall_precision,
                "overall_recall": overall_recall,
                "meets_threshold": median_f1 >= 0.8,
                "document_type_threshold": 0.8,
            },
        }

        logger.debug(
            "Median F1: %.1f%% | Mean F1: %.1f%% for %s",
            median_f1 * 100,
            overall_accuracy * 100,
            image_name,
        )
        logger.debug(
            "Precision: %.1f%% | Recall: %.1f%%",
            overall_precision * 100,
            overall_recall * 100,
        )
        logger.debug("has_field_scores=True, field_count=%d", len(field_scores))

        return evaluation

    def _filter_debit_transactions(self, extracted_data: dict) -> dict:
        """Filter bank statement data to keep only debit transactions.

        Removes credit transactions from all transaction arrays to match ground truth
        which only contains debit transactions.
        """
        if extracted_data.get("DOCUMENT_TYPE") != "BANK_STATEMENT":
            return extracted_data

        try:
            import pandas as pd

            # Get transaction arrays
            descriptions = extracted_data.get("LINE_ITEM_DESCRIPTIONS", "")
            dates = extracted_data.get("TRANSACTION_DATES", "")
            paid = extracted_data.get("TRANSACTION_AMOUNTS_PAID", "")
            received = extracted_data.get("TRANSACTION_AMOUNTS_RECEIVED", "")
            balances = extracted_data.get("ACCOUNT_BALANCE", "")

            # Check for missing required fields
            if any(
                field == "" or field == "NOT_FOUND"
                for field in [descriptions, dates, paid]
            ):
                logger.warning("Missing transaction data - skipping debit filtering")
                return extracted_data

            # Check if balances are all NOT_FOUND
            balance_values = (
                [b.strip() for b in balances.split(" | ")] if balances else []
            )
            all_balances_missing = all(
                b == "NOT_FOUND" or b == "" for b in balance_values
            )

            if balances == "" or balances == "NOT_FOUND" or all_balances_missing:
                logger.warning("No balance data available - skipping debit filtering")
                return extracted_data

            # Split arrays
            desc_list = descriptions.split(" | ")
            date_list = dates.split(" | ")
            paid_list = paid.split(" | ")
            balance_list = balances.split(" | ")
            received_list = (
                received.split(" | ") if received and received != "NOT_FOUND" else None
            )

            logger.debug(
                "Array lengths: desc=%d, date=%d, paid=%d, balance=%d",
                len(desc_list),
                len(date_list),
                len(paid_list),
                len(balance_list),
            )
            if received_list:
                logger.debug("received=%d", len(received_list))

            # Verify arrays have same length
            lengths = [
                len(desc_list),
                len(date_list),
                len(paid_list),
                len(balance_list),
            ]
            if len(set(lengths)) > 1:
                logger.warning(
                    "Array length mismatch: %s - skipping debit filtering",
                    lengths,
                )
                return extracted_data

            # Create DataFrame from transaction data
            transactions_df = pd.DataFrame(
                {
                    "description": desc_list,
                    "date": date_list,
                    "paid": paid_list,
                    "received": received_list,
                    "balance": balance_list,
                }
            )

            logger.debug("Pre-filter: %d transactions", len(transactions_df))

            # Filter to keep only debit transactions (where paid != 'NOT_FOUND')
            debit_df = transactions_df[transactions_df["paid"] != "NOT_FOUND"].copy()

            logger.debug(
                "Debit transactions found: %d/%d",
                len(debit_df),
                len(transactions_df),
            )

            # Convert back to pipe-separated strings
            filtered_data = extracted_data.copy()
            filtered_data["LINE_ITEM_DESCRIPTIONS"] = " | ".join(
                debit_df["description"].tolist()
            )
            filtered_data["TRANSACTION_DATES"] = " | ".join(debit_df["date"].tolist())
            filtered_data["TRANSACTION_AMOUNTS_PAID"] = " | ".join(
                debit_df["paid"].tolist()
            )
            filtered_data["TRANSACTION_AMOUNTS_RECEIVED"] = (
                "NOT_FOUND"  # No credits in debit-only
            )
            filtered_data["ACCOUNT_BALANCE"] = " | ".join(debit_df["balance"].tolist())

            logger.debug("Pandas filtered to %d debit transactions", len(debit_df))

            return filtered_data

        except Exception as e:
            logger.error("Pandas filtering failed: %s", e)
            logger.warning("Falling back to original data")
            return extracted_data
