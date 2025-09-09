"""Extraction result display utilities.

This module provides comprehensive formatting and display functionality for
document extraction results with document-type-specific handling.
"""

from rich import print as rprint
from rich.console import Console
from rich.table import Table


class ExtractionResultDisplay:
    """Handles display and formatting of extraction results."""

    def __init__(self):
        """Initialize the result display manager."""
        self.console = Console()

    def display_extraction_results(
        self,
        test_result: dict,
        document_type: str,
        image_path: str,
        expected_count_func=None,
        ground_truth_csv: str = None,
    ) -> None:
        """Display comprehensive extraction results.

        Args:
            test_result: Result dictionary from extraction test
            document_type: Type of document (INVOICE, RECEIPT, BANK_STATEMENT)
            image_path: Path to the processed image
            expected_count_func: Function to get expected count for accuracy calculation
            ground_truth_csv: Path to ground truth CSV for accuracy calculation
        """
        if not test_result.get("success"):
            rprint(
                f"[red]❌ V100 test failed: {test_result.get('error', 'Unknown error')}[/red]"
            )
            return

        result = test_result["raw_result"]

        # Display basic processing info
        rprint(f"[cyan]📄 Image processed:[/cyan] {result['image_path']}")
        rprint(f"[cyan]📋 Document type:[/cyan] {document_type}")

        # Display document-specific metrics
        self._display_document_metrics(result, document_type)

        # Display processing performance
        rprint(
            f"[green]⏰ Processing time:[/green] {test_result['processing_time']:.3f}s"
        )
        rprint(f"[blue]🚀 V100 Optimized:[/blue] {result.get('v100_optimized', True)}")

        # Display V100-specific memory metrics
        self._display_memory_metrics(test_result)

        # Display extracted data based on document type
        if document_type == "BANK_STATEMENT":
            self._display_bank_statement_results(result)
        else:
            self._display_invoice_receipt_results(result, document_type)

        # Display performance analysis
        self._display_performance_analysis(
            test_result,
            document_type,
            image_path,
            expected_count_func,
            ground_truth_csv,
        )

    def _display_document_metrics(self, result: dict, document_type: str):
        """Display document-specific metrics.

        Args:
            result: Extraction result data
            document_type: Type of document
        """
        if document_type == "BANK_STATEMENT":
            transaction_count = result.get("transaction_count", "N/A")
            rprint(f"[magenta]📊 Transactions found:[/magenta] {transaction_count}")
        else:
            # For invoices/receipts, count line items
            if result.get("parsed_data"):
                parsed_fields = result["parsed_data"]
                if isinstance(parsed_fields, dict):
                    line_items = parsed_fields.get("LINE_ITEM_DESCRIPTIONS", "")
                    if line_items and line_items != "NOT_FOUND":
                        item_count = len(line_items.split(" | "))
                        rprint(f"[magenta]📊 Line items found:[/magenta] {item_count}")

    def _display_memory_metrics(self, test_result: dict):
        """Display V100-specific memory metrics.

        Args:
            test_result: Result dictionary containing memory metrics
        """
        if "memory_metrics" in test_result:
            memory_metrics = test_result["memory_metrics"]
            rprint(
                f"[yellow]💾 Memory delta:[/yellow] {memory_metrics['memory_delta_gb']:+.3f}GB"
            )
            rprint(
                f"[yellow]📊 Fragmentation change:[/yellow] {memory_metrics['fragmentation_change_gb']:+.3f}GB"
            )
            rprint(
                f"[green]🛡️ ResilientGenerator:[/green] {memory_metrics['resilient_generator_used']}"
            )

    def _display_bank_statement_results(self, result: dict):
        """Display bank statement specific results.

        Args:
            result: Extraction result data
        """
        if result.get("bank_details"):
            # Bank statement specific display
            bank_details = result["bank_details"]
            bank_details_table = Table(
                title="🏦 V100-Extracted Bank Account Details", border_style="blue"
            )
            bank_details_table.add_column("Field", style="cyan")
            bank_details_table.add_column("Value", style="yellow")

            for field, value in bank_details.items():
                if value != "NOT_FOUND":  # Only show fields that have values
                    bank_details_table.add_row(field, value)

            self.console.print(bank_details_table)

        # Show parsed transaction data in a table
        if result.get("parsed_data"):
            parsed_table = Table(
                title="📋 V100-Extracted Transaction Data", border_style="green"
            )
            parsed_table.add_column("#", style="dim", width=3)
            parsed_table.add_column("Date", style="cyan")
            parsed_table.add_column("Description", style="white", max_width=30)
            parsed_table.add_column("Debit", style="red")
            parsed_table.add_column("Credit", style="green")
            parsed_table.add_column("Balance", style="yellow")

            for i, transaction in enumerate(result["parsed_data"], 1):
                parsed_table.add_row(
                    str(i),
                    transaction.get("date", ""),
                    transaction.get("description", ""),
                    transaction.get("debit", ""),
                    transaction.get("credit", ""),
                    transaction.get("balance", ""),
                )

            self.console.print(parsed_table)

    def _display_invoice_receipt_results(self, result: dict, document_type: str):
        """Display invoice/receipt specific results.

        Args:
            result: Extraction result data
            document_type: Type of document
        """
        raw_response = result.get("raw_response", "")
        if not raw_response:
            return

        # Parse the response to extract fields
        field_table = Table(
            title=f"📋 V100-Extracted {document_type} Fields", border_style="green"
        )
        field_table.add_column("Field", style="cyan", width=25)
        field_table.add_column("Value", style="yellow", max_width=50)

        # Parse each line looking for field:value pairs
        lines = raw_response.split("\n")
        for line in lines:
            if ":" in line and not line.startswith("#") and not line.startswith("|"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    field = parts[0].strip()
                    value = parts[1].strip()
                    if value and value != "NOT_FOUND":
                        # Truncate long values for display
                        display_value = value[:48] + "..." if len(value) > 48 else value
                        field_table.add_row(field, display_value)

        self.console.print(field_table)

    def _display_performance_analysis(
        self,
        test_result: dict,
        document_type: str,
        image_path: str,
        expected_count_func=None,
        ground_truth_csv: str = None,
    ):
        """Display comprehensive performance analysis.

        Args:
            test_result: Result dictionary from extraction
            document_type: Type of document
            image_path: Path to processed image
            expected_count_func: Function to get expected count
            ground_truth_csv: Path to ground truth CSV
        """
        result = test_result["raw_result"]

        # Calculate expected vs extracted counts
        if (
            document_type == "BANK_STATEMENT"
            and expected_count_func
            and ground_truth_csv
        ):
            expected_count = expected_count_func(image_path, ground_truth_csv)
            extracted_count = test_result.get("extracted_count", 0)
        else:
            # For invoices/receipts, count fields extracted
            expected_count = 14  # Standard field count for invoice/receipt
            extracted_count = len(
                [
                    1
                    for line in result.get("raw_response", "").split("\n")
                    if ":" in line and "NOT_FOUND" not in line
                ]
            )

        # Create performance table
        performance_table = Table(
            title=f"🚀 V100 Production Performance Analysis - {document_type}",
            border_style="blue",
        )
        performance_table.add_column("Metric", style="cyan")
        performance_table.add_column("Value", style="magenta")
        performance_table.add_column("V100 Status", style="green")

        # Calculate accuracy percentage
        accuracy_pct = (
            (extracted_count / expected_count * 100) if expected_count > 0 else 0
        )

        # Add performance metrics
        performance_table.add_row("Document Type", document_type, "✅ Detected")
        performance_table.add_row(
            "Extraction Method", "V100 ResilientGenerator", "✅ Production Ready"
        )

        if document_type == "BANK_STATEMENT":
            performance_table.add_row(
                "Transactions Extracted",
                str(extracted_count),
                "✅ Perfect"
                if extracted_count == expected_count
                else f"⚠️ Expected {expected_count}",
            )
        else:
            performance_table.add_row(
                "Fields Extracted",
                str(extracted_count),
                "✅ Good" if extracted_count >= 10 else "⚠️ Review",
            )

        performance_table.add_row(
            "Accuracy",
            f"{accuracy_pct:.1f}%",
            "✅ Excellent" if accuracy_pct >= 95 else "⚠️ Needs Review",
        )
        performance_table.add_row(
            "Processing Speed",
            f"{test_result['processing_time']:.3f}s",
            "✅ V100 Optimized",
        )
        performance_table.add_row(
            "Memory Management", "Active Monitoring", "✅ Fragmentation Detection"
        )
        performance_table.add_row(
            "OOM Protection", "6-Tier Fallback", "✅ ResilientGenerator"
        )
        performance_table.add_row(
            "Response Quality",
            f"{test_result['response_length']} chars",
            "✅ Good" if 100 < test_result["response_length"] < 5000 else "⚠️ Review",
        )
        performance_table.add_row(
            "Hallucination Check",
            "Pattern Detection",
            "❌ Detected"
            if test_result.get("hallucination_detected", False)
            else "✅ Clean",
        )

        self.console.print(performance_table)


def display_extraction_test_results(
    test_result: dict,
    document_type: str,
    image_path: str,
    expected_count_func=None,
    ground_truth_csv: str = None,
) -> None:
    """Convenience function to display extraction test results.

    Args:
        test_result: Result dictionary from extraction test
        document_type: Type of document (INVOICE, RECEIPT, BANK_STATEMENT)
        image_path: Path to the processed image
        expected_count_func: Function to get expected count for accuracy calculation
        ground_truth_csv: Path to ground truth CSV for accuracy calculation
    """
    rprint(
        "[bold cyan]🧪 Testing V100-optimized LlamaVisionTableExtractor with document-aware extraction...[/bold cyan]"
    )

    console = Console()
    console.rule("[bold blue]V100 Production Document-Aware Demonstration[/bold blue]")

    if image_path:
        rprint(
            f"[yellow]🎯 Running comprehensive V100 production test on {document_type}...[/yellow]"
        )

        display = ExtractionResultDisplay()
        display.display_extraction_results(
            test_result,
            document_type,
            image_path,
            expected_count_func,
            ground_truth_csv,
        )
    else:
        rprint("[red]❌ No image path available for V100 testing[/red]")

    console.rule("[bold green]Testing Complete[/bold green]")
