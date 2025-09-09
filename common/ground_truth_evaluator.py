"""Ground truth evaluation utilities.

This module provides comprehensive functionality for evaluating extraction results
against ground truth data with document-type-specific handling and rich formatting.
"""

from pathlib import Path

import pandas as pd
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from .evaluation_metrics import calculate_field_accuracy
from .extraction_parser import parse_extraction_response
from .response_preprocessing import clean_markdown_response, map_fields_to_universal


class GroundTruthEvaluator:
    """Handles comprehensive ground truth evaluation for document extraction."""

    def __init__(self, ground_truth_csv: str):
        """Initialize the ground truth evaluator.

        Args:
            ground_truth_csv: Path to the ground truth CSV file
        """
        self.ground_truth_csv = ground_truth_csv
        self.console = Console()

    def evaluate_extraction(
        self, test_result: dict, document_type: str, image_path: str
    ) -> dict:
        """Perform comprehensive field-level ground truth evaluation.

        Args:
            test_result: Result dictionary from extraction test
            document_type: Type of document (INVOICE, RECEIPT, BANK_STATEMENT)
            image_path: Path to the processed image

        Returns:
            Dictionary containing evaluation metrics and results
        """
        rprint("[bold cyan]📊 FIELD-LEVEL GROUND TRUTH EVALUATION[/bold cyan]")
        self.console.rule(
            "[bold blue]Comprehensive Document-Aware Field Evaluation[/bold blue]"
        )

        # Get the raw response from the test
        raw_response = test_result["raw_result"]["raw_response"]

        # Clean the response before parsing
        cleaned_response = clean_markdown_response(raw_response)

        # Map document-specific fields to universal field names for evaluation
        mapped_response = map_fields_to_universal(cleaned_response, document_type)

        image_filename = Path(image_path).name

        rprint(f"[yellow]🔍 Evaluating extraction for: {image_filename}[/yellow]")
        rprint(f"[dim]Document Type: {document_type}[/dim]")
        rprint(
            f"[dim]Using {document_type.lower()}-specific extraction with universal field mapping for evaluation[/dim]"
        )

        # Parse the mapped extraction response to get field-level data
        extracted_fields = parse_extraction_response(mapped_response)

        # Load ground truth for this specific image
        try:
            ground_truth = self._load_ground_truth_for_image(image_filename)

            if ground_truth:
                # Calculate accuracy metrics
                metrics = self._calculate_field_accuracy_metrics(
                    extracted_fields, ground_truth, document_type
                )

                # Display results
                self._display_field_comparison_table(
                    metrics["comparison_data"], document_type
                )
                self._display_summary_statistics(metrics, document_type)
                self._display_document_type_accuracy(extracted_fields, ground_truth)
                self._display_performance_assessment(
                    metrics, test_result, document_type
                )

                return metrics
            else:
                rprint(f"[red]❌ No ground truth found for {image_filename}[/red]")
                rprint(
                    "[yellow]💡 Make sure the image filename matches exactly in ground_truth.csv[/yellow]"
                )
                return {}

        except Exception as e:
            rprint(f"[red]❌ Error loading ground truth: {e}[/red]")
            import traceback

            traceback.print_exc()
            return {}

        finally:
            self.console.rule(
                "[bold green]Field-Level Evaluation Complete[/bold green]"
            )

    def _load_ground_truth_for_image(self, image_filename: str) -> dict:
        """Load ground truth data for a specific image.

        Args:
            image_filename: Name of the image file

        Returns:
            Dictionary containing ground truth data or empty dict if not found
        """
        try:
            ground_truth_df = pd.read_csv(self.ground_truth_csv)
            image_row = ground_truth_df[ground_truth_df["image_file"] == image_filename]

            if not image_row.empty:
                ground_truth = image_row.iloc[0].to_dict()
                rprint(f"[green]✅ Ground truth loaded for {image_filename}[/green]")
                valid_fields = len(
                    [
                        v
                        for v in ground_truth.values()
                        if v != "NOT_FOUND" and pd.notna(v)
                    ]
                )
                rprint(f"[dim]Found {valid_fields} ground truth fields[/dim]")
                return ground_truth
            else:
                return {}

        except Exception as e:
            rprint(f"[red]❌ Error loading ground truth CSV: {e}[/red]")
            return {}

    def _get_focus_fields_for_document_type(self, document_type: str) -> list:
        """Get focus fields for a specific document type.

        Args:
            document_type: Type of document

        Returns:
            List of field names that are key for this document type
        """
        focus_fields_map = {
            "BANK_STATEMENT": [
                "DOCUMENT_TYPE",
                "STATEMENT_DATE_RANGE",
                "LINE_ITEM_DESCRIPTIONS",
                "TRANSACTION_DATES",
                "TRANSACTION_AMOUNTS_PAID",
            ],
            "INVOICE": [
                "DOCUMENT_TYPE",
                "BUSINESS_ABN",
                "SUPPLIER_NAME",
                "INVOICE_DATE",
                "LINE_ITEM_DESCRIPTIONS",
                "GST_AMOUNT",
                "TOTAL_AMOUNT",
            ],
            "RECEIPT": [
                "DOCUMENT_TYPE",
                "SUPPLIER_NAME",
                "INVOICE_DATE",
                "LINE_ITEM_DESCRIPTIONS",
                "TOTAL_AMOUNT",
            ],
        }
        return focus_fields_map.get(document_type, [])

    def _calculate_field_accuracy_metrics(
        self, extracted_fields: dict, ground_truth: dict, document_type: str
    ) -> dict:
        """Calculate comprehensive field accuracy metrics.

        Args:
            extracted_fields: Dictionary of extracted field values
            ground_truth: Dictionary of ground truth values
            document_type: Type of document

        Returns:
            Dictionary containing metrics and comparison data
        """
        focus_fields = self._get_focus_fields_for_document_type(document_type)

        total_fields = 0
        fields_found = 0
        exact_matches = 0
        partial_matches = 0
        field_scores = {}
        comparison_data = []

        # Compare each field
        for field_name in ground_truth.keys():
            if field_name == "image_file":  # Skip the image filename column
                continue

            ground_value = ground_truth.get(field_name, "NOT_FOUND")
            extracted_value = extracted_fields.get(field_name, "NOT_FOUND")

            # Convert pandas NaN to NOT_FOUND
            if pd.isna(ground_value):
                ground_value = "NOT_FOUND"
            if pd.isna(extracted_value):
                extracted_value = "NOT_FOUND"

            # Skip if both are NOT_FOUND (field not applicable)
            if str(ground_value) == "NOT_FOUND" and str(extracted_value) == "NOT_FOUND":
                continue

            total_fields += 1

            # Calculate field accuracy
            accuracy = calculate_field_accuracy(
                extracted_value, ground_value, field_name, debug=False
            )

            field_scores[field_name] = accuracy

            # Determine status
            if accuracy == 1.0:
                status = "✅"
                exact_matches += 1
                if str(extracted_value) != "NOT_FOUND":
                    fields_found += 1
            elif accuracy >= 0.8:
                status = "≈"
                partial_matches += 1
                if str(extracted_value) != "NOT_FOUND":
                    fields_found += 1
            else:
                status = "❌"
                if str(extracted_value) != "NOT_FOUND":
                    fields_found += 1

            # Highlight focus fields for this document type
            display_field_name = field_name
            if field_name in focus_fields:
                display_field_name = f"🔑 {field_name}"

            # Truncate long values for display
            extracted_display = str(extracted_value)[:38] + (
                "..." if len(str(extracted_value)) > 38 else ""
            )
            ground_display = str(ground_value)[:38] + (
                "..." if len(str(ground_value)) > 38 else ""
            )

            comparison_data.append(
                {
                    "status": status,
                    "field_name": display_field_name,
                    "extracted_value": extracted_display,
                    "ground_value": ground_display,
                    "accuracy": accuracy,
                }
            )

        # Calculate overall accuracy
        overall_accuracy = (
            sum(field_scores.values()) / len(field_scores) if field_scores else 0
        )

        return {
            "total_fields": total_fields,
            "fields_found": fields_found,
            "exact_matches": exact_matches,
            "partial_matches": partial_matches,
            "overall_accuracy": overall_accuracy,
            "field_scores": field_scores,
            "comparison_data": comparison_data,
        }

    def _display_field_comparison_table(
        self, comparison_data: list, document_type: str
    ):
        """Display field-by-field comparison table.

        Args:
            comparison_data: List of comparison dictionaries
            document_type: Type of document
        """
        comparison_table = Table(
            title=f"📋 Field-by-Field Comparison ({document_type}→Universal Mapping)",
            border_style="green",
        )
        comparison_table.add_column("Status", style="bold", width=8)
        comparison_table.add_column("Field", style="cyan", width=25)
        comparison_table.add_column("Extracted", style="yellow", max_width=40)
        comparison_table.add_column("Ground Truth", style="magenta", max_width=40)
        comparison_table.add_column("Score", style="blue", justify="right")

        for item in comparison_data:
            comparison_table.add_row(
                item["status"],
                item["field_name"],
                item["extracted_value"],
                item["ground_value"],
                f"{item['accuracy']:.2f}",
            )

        self.console.print(comparison_table)

    def _display_summary_statistics(self, metrics: dict, document_type: str):
        """Display summary statistics table.

        Args:
            metrics: Dictionary containing calculated metrics
            document_type: Type of document
        """
        summary_table = Table(
            title=f"📈 Evaluation Summary ({document_type} Extraction)",
            border_style="blue",
        )
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="magenta")
        summary_table.add_column("Percentage", style="yellow")

        total_fields = metrics["total_fields"]
        fields_found = metrics["fields_found"]
        exact_matches = metrics["exact_matches"]
        partial_matches = metrics["partial_matches"]
        overall_accuracy = metrics["overall_accuracy"]

        summary_table.add_row("Document Type", document_type, "-")
        summary_table.add_row("Total Fields Evaluated", str(total_fields), "100%")
        summary_table.add_row(
            "Fields Found",
            str(fields_found),
            f"{(fields_found / total_fields * 100):.1f}%" if total_fields > 0 else "0%",
        )
        summary_table.add_row(
            "Exact Matches",
            str(exact_matches),
            f"{(exact_matches / total_fields * 100):.1f}%"
            if total_fields > 0
            else "0%",
        )
        summary_table.add_row(
            "Partial Matches (≥0.8)",
            str(partial_matches),
            f"{(partial_matches / total_fields * 100):.1f%}"
            if total_fields > 0
            else "0%",
        )
        summary_table.add_row(
            "Overall Accuracy",
            f"{overall_accuracy:.3f}",
            f"{(overall_accuracy * 100):.1f}%",
        )

        self.console.print(summary_table)

    def _display_document_type_accuracy(
        self, extracted_fields: dict, ground_truth: dict
    ):
        """Display document type detection accuracy.

        Args:
            extracted_fields: Dictionary of extracted field values
            ground_truth: Dictionary of ground truth values
        """
        if "DOCUMENT_TYPE" in extracted_fields:
            extracted_doc_type = extracted_fields["DOCUMENT_TYPE"]
            ground_doc_type = ground_truth.get("DOCUMENT_TYPE", "NOT_FOUND")

            if extracted_doc_type != "NOT_FOUND" and ground_doc_type != "NOT_FOUND":
                doc_type_match = (
                    extracted_doc_type.upper() == str(ground_doc_type).upper()
                )
                rprint("\n[bold]Document Type Detection:[/bold]")
                rprint(f"  Extracted: {extracted_doc_type}")
                rprint(f"  Ground Truth: {ground_doc_type}")
                rprint(
                    f"  Status: {'✅ Correct' if doc_type_match else '❌ Incorrect'}"
                )

    def _display_performance_assessment(
        self, metrics: dict, test_result: dict, document_type: str
    ):
        """Display comprehensive performance assessment.

        Args:
            metrics: Dictionary containing calculated metrics
            test_result: Original test result dictionary
            document_type: Type of document
        """
        overall_accuracy = metrics["overall_accuracy"]

        rprint("\n[bold cyan]📊 Performance Assessment:[/bold cyan]")
        rprint(
            f"[dim]Note: Using {document_type.lower()}-specific extraction with field mapping[/dim]"
        )

        # Accuracy assessment
        if overall_accuracy >= 0.95:
            rprint(
                "[bold green]🎉 EXCELLENT - Production Ready (≥95% accuracy)[/bold green]"
            )
        elif overall_accuracy >= 0.85:
            rprint(
                "[yellow]✅ GOOD - Minor improvements needed (85-94% accuracy)[/yellow]"
            )
        elif overall_accuracy >= 0.75:
            rprint(
                "[yellow]⚠️ FAIR - Significant improvements needed (75-84% accuracy)[/yellow]"
            )
        else:
            rprint("[red]❌ POOR - Major improvements required (<75% accuracy)[/red]")

        # Processing performance
        if "processing_time" in test_result:
            proc_time = test_result["processing_time"]
            rprint(f"\n⏱️  Processing Time: {proc_time:.2f}s")
            if proc_time < 5:
                rprint("[green]  Speed: Excellent (<5s)[/green]")
            elif proc_time < 10:
                rprint("[yellow]  Speed: Good (5-10s)[/yellow]")
            else:
                rprint("[red]  Speed: Needs optimization (>10s)[/red]")


def evaluate_extraction_against_ground_truth(
    test_result: dict, document_type: str, image_path: str, ground_truth_csv: str
) -> dict:
    """Convenience function for ground truth evaluation.

    Args:
        test_result: Result dictionary from extraction test
        document_type: Type of document (INVOICE, RECEIPT, BANK_STATEMENT)
        image_path: Path to the processed image
        ground_truth_csv: Path to ground truth CSV file

    Returns:
        Dictionary containing evaluation metrics and results
    """
    evaluator = GroundTruthEvaluator(ground_truth_csv)
    return evaluator.evaluate_extraction(test_result, document_type, image_path)
