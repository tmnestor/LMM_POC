"""
Batch Processing Module for Document-Aware Extraction

Handles batch processing of images through document detection and extraction pipeline.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich import print as rprint
from rich.console import Console
from rich.progress import track

from .evaluation_metrics import load_ground_truth

# Import Rich content sanitization to prevent recursion errors and ExtractionCleaner
from .simple_model_evaluator import SimpleModelEvaluator


def load_document_field_definitions() -> Dict[str, List[str]]:
    """
    Load document-aware field definitions from field_definitions.yaml.

    CRITICAL: This function will raise an exception if YAML loading fails.
    NO FALLBACKS - fail fast with clear diagnostics.

    Returns:
        Dictionary mapping document types (lowercase) to field lists

    Raises:
        FileNotFoundError: If field_definitions.yaml does not exist
        ValueError: If YAML structure is invalid or missing required fields
    """
    import yaml

    field_def_path = Path(__file__).parent.parent / "config" / "field_definitions.yaml"

    # Check file exists first for clear error message
    if not field_def_path.exists():
        raise FileNotFoundError(
            f"❌ FATAL: Field definitions file not found\n"
            f"Expected location: {field_def_path.absolute()}\n"
            f"This file is REQUIRED for document-aware field filtering.\n"
            f"Ensure config/field_definitions.yaml exists in the project root."
        )

    try:
        with field_def_path.open("r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(
            f"❌ FATAL: Invalid YAML syntax in field_definitions.yaml\n"
            f"File: {field_def_path.absolute()}\n"
            f"Error: {e}\n"
            f"Fix the YAML syntax errors before proceeding."
        ) from e
    except Exception as e:
        raise ValueError(
            f"❌ FATAL: Could not read field_definitions.yaml\n"
            f"File: {field_def_path.absolute()}\n"
            f"Error: {e}"
        ) from e

    # Validate structure
    if "document_fields" not in config:
        raise ValueError(
            f"❌ FATAL: Missing 'document_fields' section in field_definitions.yaml\n"
            f"File: {field_def_path.absolute()}\n"
            f"Required structure:\n"
            f"document_fields:\n"
            f"  invoice:\n"
            f"    fields: [list of fields]\n"
            f"  receipt:\n"
            f"    fields: [list of fields]\n"
            f"  bank_statement:\n"
            f"    fields: [list of fields]"
        )

    doc_fields = config["document_fields"]

    # Validate each required document type
    for doc_type in ["invoice", "receipt", "bank_statement"]:
        if doc_type not in doc_fields:
            raise ValueError(
                f"❌ FATAL: Missing '{doc_type}' definition in field_definitions.yaml\n"
                f"File: {field_def_path.absolute()}\n"
                f"Each document type must be defined with a 'fields' list."
            )
        if "fields" not in doc_fields[doc_type]:
            raise ValueError(
                f"❌ FATAL: Missing 'fields' list for '{doc_type}' in field_definitions.yaml\n"
                f"File: {field_def_path.absolute()}\n"
                f"Each document type must have a 'fields' list."
            )
        if not doc_fields[doc_type]["fields"]:
            raise ValueError(
                f"❌ FATAL: Empty 'fields' list for '{doc_type}' in field_definitions.yaml\n"
                f"File: {field_def_path.absolute()}\n"
                f"Each document type must have at least one field defined."
            )

    # Build field definitions dynamically from YAML
    result = {
        "invoice": doc_fields["invoice"]["fields"],
        "receipt": doc_fields["receipt"]["fields"],
        "bank_statement": doc_fields["bank_statement"]["fields"],
    }

    # Add travel_expense if defined in YAML
    if "travel_expense" in doc_fields and "fields" in doc_fields["travel_expense"]:
        result["travel_expense"] = doc_fields["travel_expense"]["fields"]

    return result


class BatchDocumentProcessor:
    """Handles batch processing of documents with extraction and evaluation."""

    def __init__(
        self,
        model,
        processor,
        prompt_config: Dict,
        ground_truth_csv: str,
        console: Optional[Console] = None,
        enable_math_enhancement: bool = True,
    ):
        """
        Initialize batch processor for InternVL3 document extraction.

        Args:
            model: InternVL3 handler (DocumentAwareInternVL3HybridProcessor)
            processor: Not used (kept for API compatibility), pass None
            prompt_config: Dictionary with prompt file paths and keys
            ground_truth_csv: Path to ground truth CSV file
            console: Rich console for output
            enable_math_enhancement: Whether to apply mathematical enhancement for bank statements
        """
        # Store InternVL3 handler
        self.internvl3_handler = model

        self.prompt_config = prompt_config
        self.ground_truth_csv = ground_truth_csv
        self.console = console or Console()
        self.enable_math_enhancement = enable_math_enhancement

        # Initialize file-based trace logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._trace_file = f"batch_processor_trace_{timestamp}.log"

        # Use SimpleModelEvaluator for model comparison
        self.model_evaluator = SimpleModelEvaluator()
        self.ground_truth_data = None

    def _trace_log(self, message: str):
        """Log message to both console and file"""
        print(message)
        with Path(self._trace_file).open("a", encoding="utf-8") as f:
            f.write(f"{message}\n")

    def process_batch(
        self, image_paths: List[str], verbose: bool = True, progress_interval: int = 5
    ) -> Tuple[List[Dict], List[float], Dict[str, int]]:
        """
        Process a batch of images through the extraction pipeline.

        Args:
            image_paths: List of image file paths
            verbose: Whether to show progress updates
            progress_interval: How often to show detailed progress

        Returns:
            Tuple of (batch_results, processing_times, document_types_found)
        """
        start_time = time.time()

        batch_results = []
        processing_times = []
        document_types_found = {}

        # Load ground truth data once for the batch
        try:
            self.ground_truth_data = load_ground_truth(self.ground_truth_csv)
            if verbose:
                rprint(
                    f"[green]✅ Loaded ground truth for {len(self.ground_truth_data)} images[/green]"
                )
                # DEBUG: Show sample GT keys to verify loading
                sample_keys = list(self.ground_truth_data.keys())[:3]
                rprint(f"[cyan]📋 Sample GT keys: {sample_keys}[/cyan]")
        except Exception as e:
            if verbose:
                rprint(f"[red]❌ Error loading ground truth: {e}[/red]")
            self.ground_truth_data = {}

        if verbose:
            rprint("\n[bold blue]🚀 Starting Batch Processing[/bold blue]")
            self.console.rule("[bold green]Batch Extraction[/bold green]")

        # Process each image - always show progress bar, even when not verbose
        iterator = track(image_paths, description="Processing images...")

        for idx, image_path in enumerate(iterator, 1):
            image_name = Path(image_path).name

            if verbose:
                rprint(
                    f"\n[bold blue]Processing [{idx}/{len(image_paths)}]: {image_name}[/bold blue]"
                )

            try:
                # Record start time
                start_time = time.time()

                # Process image with InternVL3 handler
                if verbose:
                    rprint(
                        f"[dim]🔍 TRACE: Processing image {idx}/{len(image_paths)}: {image_name}[/dim]"
                    )

                document_type, extraction_result, prompt_name = (
                    self._process_internvl3_image(image_path, verbose)
                )
                document_types_found[document_type] = (
                    document_types_found.get(document_type, 0) + 1
                )

                if verbose:
                    rprint(
                        f"[dim]🔍 TRACE: Processing complete for {image_name}, doc_type={document_type}[/dim]"
                    )

                # Step 4: Evaluate against ground truth using working DocumentTypeEvaluator approach
                image_name = Path(image_path).name

                # Fuzzy ground truth lookup - try exact match first, then without extension
                ground_truth = {}
                if self.ground_truth_data:
                    # Try exact match first
                    ground_truth = self.ground_truth_data.get(image_name, {})

                    # If not found, try without extension
                    if not ground_truth:
                        image_stem = Path(image_path).stem  # name without extension
                        # Look for any GT key that matches the stem
                        for gt_key in self.ground_truth_data.keys():
                            if Path(gt_key).stem == image_stem or gt_key == image_stem:
                                ground_truth = self.ground_truth_data[gt_key]
                                break

                if ground_truth:
                    # Extract data using working DocumentAwareLlamaProcessor format
                    # DocumentAwareLlamaProcessor returns extracted_data at top level
                    extracted_data = extraction_result.get("extracted_data", {})

                    # Apply mathematical enhancement for bank statements
                    # Skip if already handled by UnifiedBankExtractor (V2 sophisticated extraction)
                    skip_math = extraction_result.get("skip_math_enhancement", False)
                    if (
                        document_type.upper() == "BANK_STATEMENT"
                        and self.enable_math_enhancement
                        and not skip_math
                    ):
                        from .bank_statement_calculator import (
                            enhance_bank_statement_extraction,
                        )

                        if verbose:
                            rprint(
                                "[blue]🧮 Applying mathematical enhancement for bank statement[/blue]"
                            )

                        # Get enhanced data with mathematical corrections
                        enhanced_result = enhance_bank_statement_extraction(
                            extracted_data, verbose=verbose
                        )

                        # Separate the corrected extraction data from analysis metadata
                        extracted_data = {
                            k: v
                            for k, v in enhanced_result.items()
                            if k != "_mathematical_analysis"
                        }

                        # Store analysis metadata for reporting but don't include in evaluation
                        mathematical_analysis = enhanced_result.get(
                            "_mathematical_analysis", {}
                        )

                        # CRITICAL: Filter to debit-only transactions for evaluation
                        if verbose:
                            rprint(
                                "[blue]🎯 Filtering to debit-only transactions for evaluation[/blue]"
                            )

                        extracted_data = self._filter_debit_transactions(
                            extracted_data, verbose
                        )
                    elif skip_math and verbose:
                        rprint(
                            "[dim]⏭️  Skipping batch_processor math enhancement (handled by UnifiedBankExtractor)[/dim]"
                        )

                    if verbose:
                        found_fields = [
                            k for k, v in extracted_data.items() if v != "NOT_FOUND"
                        ]
                        rprint(
                            f"[cyan]✓ Extracted {len(found_fields)} fields from {image_name}[/cyan]"
                        )

                        # Show mathematical enhancement results if applied
                        if (
                            document_type.upper() == "BANK_STATEMENT"
                            and "mathematical_analysis" in locals()
                        ):
                            if mathematical_analysis.get("calculation_success"):
                                rprint(
                                    f"[green]✓ Mathematical analysis: {mathematical_analysis.get('transaction_count', 0)} transactions calculated[/green]"
                                )
                            else:
                                rprint(
                                    "[yellow]⚠️ Mathematical analysis failed[/yellow]"
                                )

                    # Filter ground truth to document-specific fields for accurate evaluation
                    # Load document-specific field lists from YAML configuration
                    doc_type_fields = load_document_field_definitions()

                    # Ensure case-insensitive document type matching for evaluation fields
                    document_type_lower_eval = document_type.lower()
                    evaluation_fields = doc_type_fields.get(
                        document_type_lower_eval, doc_type_fields["invoice"]
                    )

                    filtered_ground_truth = {
                        field: ground_truth[field]
                        for field in evaluation_fields
                        if field in ground_truth
                    }

                    # Use SimpleModelEvaluator for clean model comparison with filtered data
                    if verbose and document_type.upper() == "BANK_STATEMENT":
                        rprint(
                            "[blue]🎯 Evaluating using mathematically corrected values (not raw VLM output)[/blue]"
                        )

                    evaluation_result = self.model_evaluator.evaluate_extraction(
                        extracted_data, filtered_ground_truth, image_path
                    )

                    # Convert to expected format for compatibility
                    # Calculate fields_extracted and fields_matched for notebook compatibility
                    fields_extracted = len(
                        [k for k, v in extracted_data.items() if v != "NOT_FOUND"]
                    )

                    # Build field-level scores for detailed comparison display
                    # Use configurable F1-based evaluation from evaluation_metrics.py
                    # Get evaluation method from environment or use default
                    # Users can set EVALUATION_METHOD environment variable to change metrics
                    import os

                    from common.evaluation_metrics import (
                        calculate_correlation_aware_f1,
                        calculate_field_accuracy_with_method,
                    )

                    evaluation_method = os.environ.get(
                        "EVALUATION_METHOD", "order_aware_f1"
                    )

                    field_scores = {}
                    total_f1_score = 0.0
                    total_precision = 0.0
                    total_recall = 0.0

                    # Correlation method needs to be called once for entire document
                    if evaluation_method in ["correlation", "correlation_aware_f1"]:
                        # Get overall correlation metrics for the document
                        correlation_result = calculate_correlation_aware_f1(
                            extracted_data,
                            filtered_ground_truth,
                            document_type,
                            debug=False,
                        )

                        # Use the correlation metrics for all field scores
                        # Each field gets the same combined F1/precision/recall
                        for field in filtered_ground_truth.keys():
                            field_scores[field] = correlation_result
                            total_f1_score += correlation_result["f1_score"]
                            total_precision += correlation_result["precision"]
                            total_recall += correlation_result["recall"]
                    else:
                        # Standard per-field evaluation
                        for field in filtered_ground_truth.keys():
                            extracted_val = extracted_data.get(field, "NOT_FOUND")
                            ground_val = filtered_ground_truth.get(field, "NOT_FOUND")

                            # Use configurable F1-based scoring (default: order_aware_f1)
                            # Available methods: 'order_aware_f1', 'f1', 'kieval', 'correlation'
                            # DEBUG: Enable debug for IS_GST_INCLUDED
                            is_debug = field == "IS_GST_INCLUDED" and verbose
                            if is_debug:
                                rprint("[yellow]🔍 BEFORE EVALUATION:[/yellow]")
                                rprint(
                                    f"  extracted_val = '{extracted_val}' (type: {type(extracted_val).__name__})"
                                )
                                rprint(
                                    f"  ground_val = '{ground_val}' (type: {type(ground_val).__name__})"
                                )
                                rprint(
                                    f"  Are they equal? {extracted_val == ground_val}"
                                )

                            f1_metrics = calculate_field_accuracy_with_method(
                                extracted_val,
                                ground_val,
                                field,
                                method=evaluation_method,
                                debug=is_debug,
                            )

                            if is_debug:
                                rprint(f"[yellow]🔍 AFTER EVALUATION ({field}):[/yellow]")
                                rprint(f"  Field '{field}' f1_score = {f1_metrics['f1_score']}")

                            field_scores[field] = f1_metrics
                            total_f1_score += f1_metrics["f1_score"]
                            total_precision += f1_metrics["precision"]
                            total_recall += f1_metrics["recall"]

                    # Calculate overall metrics from F1 scores
                    num_fields = len(field_scores)
                    overall_accuracy = (
                        total_f1_score / num_fields if num_fields else 0.0
                    )
                    overall_precision = (
                        total_precision / num_fields if num_fields else 0.0
                    )
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

                    # Count perfect matches for compatibility
                    perfect_matches = sum(
                        1 for score in field_scores.values() if score["f1_score"] == 1.0
                    )
                    fields_matched = perfect_matches

                    evaluation = {
                        "overall_accuracy": overall_accuracy,  # F1-based average (mean)
                        "median_f1": median_f1,  # Median F1 (robust to outliers)
                        "overall_precision": overall_precision,  # Average precision
                        "overall_recall": overall_recall,  # Average recall
                        "total_fields": len(field_scores),
                        "correct_fields": perfect_matches,  # Perfect F1=1.0 matches
                        "missing_fields": evaluation_result.missing_fields,
                        "incorrect_fields": evaluation_result.incorrect_fields,
                        # Add notebook-expected keys
                        "fields_extracted": fields_extracted,
                        "fields_matched": fields_matched,
                        # Add field-level scores for detailed comparison (now with F1 metrics)
                        "field_scores": field_scores,
                        "overall_metrics": {
                            "overall_accuracy": overall_accuracy,  # F1-based mean
                            "median_f1": median_f1,  # Median F1
                            "overall_precision": overall_precision,
                            "overall_recall": overall_recall,
                            "meets_threshold": median_f1 >= 0.8,  # Use median for threshold
                            "document_type_threshold": 0.8,
                        },
                    }

                    # SimpleModelEvaluator already provides data in correct format - no flattening needed

                    # Show evaluation summary
                    if verbose and evaluation:
                        mean_f1 = evaluation.get("overall_accuracy", 0) * 100
                        median_f1_pct = evaluation.get("median_f1", 0) * 100
                        precision = evaluation.get("overall_precision", 0) * 100
                        recall = evaluation.get("overall_recall", 0) * 100
                        rprint(
                            f"[cyan]✓ Median F1: {median_f1_pct:.1f}% | Mean F1: {mean_f1:.1f}% for {image_name}[/cyan]"
                        )
                        rprint(
                            f"[dim]  Precision: {precision:.1f}% | Recall: {recall:.1f}%[/dim]"
                        )

                    # Debug: Show field score count
                    if verbose:
                        has_field_scores = "field_scores" in evaluation
                        rprint(
                            f"[dim]🔍 DEBUG: has_field_scores={has_field_scores}, field_count={len(evaluation.get('field_scores', {}))}[/dim]"
                        )
                else:
                    evaluation = {
                        "error": f"No ground truth for {image_name}",
                        "overall_accuracy": 0,
                    }

                # Record processing time
                processing_time = time.time() - start_time
                processing_times.append(processing_time)

                # Store results
                result = {
                    "image_name": image_name,
                    "image_path": image_path,
                    "document_type": document_type,
                    "extraction_result": extraction_result,
                    "evaluation": evaluation,
                    "processing_time": processing_time,
                    "prompt_used": prompt_name,
                    "timestamp": datetime.now().isoformat(),
                }
                batch_results.append(result)

                # Always show compact F1 score for each document (regardless of verbose)
                if evaluation and "median_f1" in evaluation:
                    median_f1_pct = evaluation.get("median_f1", 0) * 100
                    mean_f1_pct = evaluation.get("overall_accuracy", 0) * 100
                    rprint(
                        f"  [green]✓[/green] {image_name}: "
                        f"[cyan]Median {median_f1_pct:.1f}%[/cyan] | "
                        f"Mean {mean_f1_pct:.1f}% | {processing_time:.1f}s"
                    )
                elif evaluation:
                    # Fallback if median not available
                    mean_f1_pct = evaluation.get("overall_accuracy", 0) * 100
                    rprint(
                        f"  [green]✓[/green] {image_name}: "
                        f"[cyan]F1 {mean_f1_pct:.1f}%[/cyan] | {processing_time:.1f}s"
                    )

                # Additional verbose progress update
                if verbose and (
                    idx % progress_interval == 0 or idx == len(image_paths)
                ):
                    accuracy = (
                        evaluation.get("overall_accuracy", 0) * 100 if evaluation else 0
                    )
                    rprint(
                        f"  [{idx}/{len(image_paths)}] {image_name}: {document_type} - "
                        f"Accuracy: {accuracy:.1f}% - Time: {processing_time:.2f}s"
                    )

            except Exception as e:
                if verbose:
                    rprint(f"[red]❌ Error processing {image_name}: {e}[/red]")

                # Store error result
                batch_results.append(
                    {
                        "image_name": image_name,
                        "image_path": image_path,
                        "error": str(e),
                        "processing_time": time.time() - start_time
                        if "start_time" in locals()
                        else 0,
                    }
                )

        if verbose:
            self.console.rule("[bold green]Batch Processing Complete[/bold green]")

        end_time = time.time()

        return batch_results, processing_times, document_types_found

    def _filter_debit_transactions(
        self, extracted_data: dict, verbose: bool = False
    ) -> dict:
        """
        Filter bank statement data to keep only debit transactions using pandas.

        This removes credit transactions from all transaction arrays to match ground truth
        which only contains debit transactions.
        """
        if extracted_data.get("DOCUMENT_TYPE") != "BANK_STATEMENT":
            return extracted_data  # Only filter bank statements

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
                if verbose:
                    rprint(
                        "[yellow]⚠️ Missing transaction data - skipping debit filtering[/yellow]"
                    )
                return extracted_data

            # Check if balances are all NOT_FOUND (e.g., from DEBIT_CREDIT_DESCRIPTION strategy)
            # In this case, balances is something like "NOT_FOUND | NOT_FOUND | ..."
            balance_values = [b.strip() for b in balances.split(" | ")] if balances else []
            all_balances_missing = all(b == "NOT_FOUND" or b == "" for b in balance_values)

            if balances == "" or balances == "NOT_FOUND" or all_balances_missing:
                if verbose:
                    rprint(
                        "[yellow]⚠️ No balance data available - skipping debit filtering[/yellow]"
                    )
                return extracted_data

            # Split arrays
            desc_list = descriptions.split(" | ")
            date_list = dates.split(" | ")
            paid_list = paid.split(" | ")
            balance_list = balances.split(" | ")
            received_list = received.split(" | ") if received and received != "NOT_FOUND" else None

            # DEBUG: Show array lengths before DataFrame creation
            if verbose:
                rprint(f"[dim]Array lengths: desc={len(desc_list)}, date={len(date_list)}, paid={len(paid_list)}, balance={len(balance_list)}[/dim]")
                if received_list:
                    rprint(f"[dim]  received={len(received_list)}[/dim]")

            # Verify arrays have same length
            lengths = [len(desc_list), len(date_list), len(paid_list), len(balance_list)]
            if len(set(lengths)) > 1:
                if verbose:
                    rprint(f"[yellow]⚠️ Array length mismatch: {lengths} - skipping debit filtering[/yellow]")
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

            if verbose:
                rprint(f"[dim]Pre-filter: {len(transactions_df)} transactions[/dim]")

            # Filter to keep only debit transactions (where paid != 'NOT_FOUND')
            debit_df = transactions_df[transactions_df["paid"] != "NOT_FOUND"].copy()

            if verbose:
                rprint(
                    f"[dim]Debit transactions found: {len(debit_df)}/{len(transactions_df)}[/dim]"
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

            if verbose:
                rprint(
                    f"[green]✅ Pandas filtered to {len(debit_df)} debit transactions[/green]"
                )

            return filtered_data

        except Exception as e:
            if verbose:
                rprint(f"[red]❌ Pandas filtering failed: {e}[/red]")
                rprint("[yellow]⚠️ Falling back to original data[/yellow]")
            return extracted_data

    def _process_internvl3_image(
        self, image_path: str, verbose: bool
    ) -> Tuple[str, Dict, str]:
        """
        Process single image using InternVL3 handler.

        Args:
            image_path: Path to image
            verbose: Whether to show verbose output

        Returns:
            Tuple of (document_type, extraction_result, prompt_name)
        """

        if verbose:
            rprint("\n[bold cyan]📋 INTERNVL3 DOCUMENT TYPE DETECTION[/bold cyan]")
            rprint("[cyan]━" * 80 + "[/cyan]")

        # Step 1: Detect and classify document
        classification_info = self.internvl3_handler.detect_and_classify_document(
            image_path, verbose=verbose
        )
        document_type = classification_info["document_type"]

        if verbose:
            rprint(f"[green]✅ Detected Document Type: {document_type}[/green]")
            rprint("[cyan]━" * 80 + "[/cyan]\n")

            rprint(
                f"[bold cyan]📊 INTERNVL3 DOCUMENT-AWARE EXTRACTION ({document_type.upper()})[/bold cyan]"
            )
            rprint("[cyan]━" * 80 + "[/cyan]")

        # Step 2: Process with document-aware extraction
        extraction_result = self.internvl3_handler.process_document_aware(
            image_path, classification_info, verbose=verbose
        )

        if verbose:
            rprint("[cyan]━" * 80 + "[/cyan]\n")

        # Extract the actual extracted_data for evaluation
        extracted_data = extraction_result.get("extracted_data", {})

        # Create extraction_result in the format expected by batch processor
        formatted_result = {
            "extracted_data": extracted_data,
            "document_type": document_type,
            "image_file": Path(image_path).name,
            "processing_time": extraction_result.get("processing_time", 0),
        }

        # Prompt name for InternVL3
        prompt_name = f"internvl3_{document_type.lower()}"

        return document_type, formatted_result, prompt_name



def print_accuracy_by_document_type(
    batch_results: list[dict],
    console: Console | None = None,
) -> dict:
    """
    Print accuracy summary separated by document type.

    Invoice/Receipt have 14 fields, Bank Statements have 5 fields.
    This provides a fair comparison by reporting metrics separately.

    Args:
        batch_results: List of result dictionaries from batch processing
        console: Rich console for output (optional)

    Returns:
        dict: Summary statistics by document type
    """
    if console is None:
        console = Console()

    # Group results by document type
    doc_type_results: dict[str, list[dict]] = {
        "invoice_receipt": [],
        "bank_statement": [],
    }

    for result in batch_results:
        if "error" in result:
            continue

        doc_type = result.get("document_type", "").upper()
        evaluation = result.get("evaluation", {})

        if not evaluation or "overall_accuracy" not in evaluation:
            continue

        # Group invoice and receipt together
        if doc_type in ["INVOICE", "RECEIPT"]:
            doc_type_results["invoice_receipt"].append(result)
        elif doc_type == "BANK_STATEMENT":
            doc_type_results["bank_statement"].append(result)

    # Calculate and display metrics for each document type
    console.rule("[bold cyan]Accuracy by Document Type[/bold cyan]")

    summary = {}

    for doc_type_key, results in doc_type_results.items():
        if not results:
            continue

        # Extract metrics
        mean_f1_scores = []
        median_f1_scores = []

        for r in results:
            eval_data = r.get("evaluation", {})
            mean_f1_scores.append(eval_data.get("overall_accuracy", 0))
            median_f1_scores.append(eval_data.get("median_f1", 0))

        # Calculate aggregates
        n_docs = len(results)
        avg_mean_f1 = sum(mean_f1_scores) / n_docs if n_docs else 0
        avg_median_f1 = sum(median_f1_scores) / n_docs if n_docs else 0

        # Calculate median of medians (most robust)
        sorted_medians = sorted(median_f1_scores)
        mid = len(sorted_medians) // 2
        if len(sorted_medians) % 2 == 0 and len(sorted_medians) > 0:
            median_of_medians = (sorted_medians[mid - 1] + sorted_medians[mid]) / 2
        elif len(sorted_medians) > 0:
            median_of_medians = sorted_medians[mid]
        else:
            median_of_medians = 0

        # Display
        display_name = "Invoice/Receipt (14 fields)" if doc_type_key == "invoice_receipt" else "Bank Statement (5 fields)"
        field_count = 14 if doc_type_key == "invoice_receipt" else 5

        rprint(f"\n[bold blue]{display_name}[/bold blue]")
        rprint(f"  Documents: {n_docs}")
        rprint(f"  [cyan]Median F1 (avg): {avg_median_f1 * 100:.1f}%[/cyan] ← typical field performance")
        rprint(f"  Mean F1 (avg):   {avg_mean_f1 * 100:.1f}%")
        rprint(f"  Median of Medians: {median_of_medians * 100:.1f}% ← most robust")

        summary[doc_type_key] = {
            "count": n_docs,
            "field_count": field_count,
            "avg_mean_f1": avg_mean_f1,
            "avg_median_f1": avg_median_f1,
            "median_of_medians": median_of_medians,
        }

    # Overall summary (weighted by document count, not field count)
    total_docs = sum(s["count"] for s in summary.values())
    if total_docs > 0:
        weighted_median = sum(
            s["avg_median_f1"] * s["count"] for s in summary.values()
        ) / total_docs
        weighted_mean = sum(
            s["avg_mean_f1"] * s["count"] for s in summary.values()
        ) / total_docs

        rprint("\n[bold green]Overall (weighted by document count)[/bold green]")
        rprint(f"  Total Documents: {total_docs}")
        rprint(f"  [cyan]Weighted Median F1: {weighted_median * 100:.1f}%[/cyan]")
        rprint(f"  Weighted Mean F1:   {weighted_mean * 100:.1f}%")

        summary["overall"] = {
            "count": total_docs,
            "weighted_median_f1": weighted_median,
            "weighted_mean_f1": weighted_mean,
        }

    console.rule()

    return summary
