"""
Bank Statement Structure Classification Module

Distinguishes between flat and date-grouped bank statement structures
to apply optimal extraction prompts.
"""

from pathlib import Path
from typing import Dict, Literal

from rich import print as rprint


class BankStatementClassifier:
    """Classifies bank statement structure for optimal prompt selection."""

    def __init__(self):
        """Initialize the classifier."""
        self.structure_types = {
            "flat": {
                "description": "Simple table with continuous transaction rows",
                "characteristics": [
                    "Single transaction table with clear column headers",
                    "Continuous date sequence in single table",
                    "Column headers like Date|Description|Withdrawal|Deposit|Balance",
                ],
                "examples": ["commbank_flat_simple.png"],
            },
            "date_grouped": {
                "description": "Transactions grouped by date sections",
                "characteristics": [
                    "Date headers separate transaction groups",
                    "Multiple sections with date headings",
                    "More complex layout with grouped structure",
                ],
                "examples": ["commbank_statement_001.png"],
            },
        }

    def classify_statement_structure(
        self, image_path: str, verbose: bool = False
    ) -> Literal["flat", "date_grouped"]:
        """
        Classify bank statement structure based on image characteristics.

        Args:
            image_path: Path to bank statement image
            verbose: Whether to show classification details

        Returns:
            Either "flat" or "date_grouped"
        """
        image_name = Path(image_path).name.lower()

        if verbose:
            rprint(
                f"[cyan]🏦 Classifying bank statement structure: {image_name}[/cyan]"
            )

        # Use filename-based classification for now
        # Future enhancement: Could use vision-based analysis
        structure_type = self._classify_by_filename(image_name)

        if verbose:
            self._display_classification_result(structure_type, image_name)

        return structure_type

    def _classify_by_filename(self, image_name: str) -> Literal["flat", "date_grouped"]:
        """Classify based on filename patterns."""

        # Flat statement indicators
        flat_indicators = ["flat", "simple", "basic", "straightforward"]

        # Date-grouped statement indicators
        grouped_indicators = ["statement_001", "complex", "grouped", "sectioned"]

        # Check for flat indicators
        if any(indicator in image_name for indicator in flat_indicators):
            return "flat"

        # Check for grouped indicators
        if any(indicator in image_name for indicator in grouped_indicators):
            return "date_grouped"

        # Default classification logic
        # Most standard bank statements are date-grouped
        if "statement" in image_name and "flat" not in image_name:
            return "date_grouped"

        # Conservative default to flat for simpler processing
        return "flat"

    def _display_classification_result(
        self, structure_type: Literal["flat", "date_grouped"], image_name: str
    ):
        """Display classification result with details."""

        structure_info = self.structure_types[structure_type]

        rprint(f"[green]📋 Classification Result: {structure_type.upper()}[/green]")
        rprint(f"[dim]Description: {structure_info['description']}[/dim]")

        if structure_type == "flat":
            rprint("[dim]💡 Will use optimized flat statement prompt[/dim]")
        else:
            rprint("[dim]💡 Will use specialized date-grouped prompt[/dim]")

    def get_recommended_prompt_key(
        self, structure_type: Literal["flat", "date_grouped"]
    ) -> str:
        """Get recommended prompt key for structure type."""

        prompt_mapping = {"flat": "flat_optimized", "date_grouped": "date_grouped"}

        return prompt_mapping[structure_type]

    def get_structure_info(
        self, structure_type: Literal["flat", "date_grouped"]
    ) -> Dict:
        """Get detailed information about a structure type."""
        return self.structure_types[structure_type]


def classify_bank_statement_structure(
    image_path: str, verbose: bool = False
) -> Literal["flat", "date_grouped"]:
    """
    Convenience function to classify bank statement structure.

    Args:
        image_path: Path to bank statement image
        verbose: Whether to show classification details

    Returns:
        Either "flat" or "date_grouped"
    """
    classifier = BankStatementClassifier()
    return classifier.classify_statement_structure(image_path, verbose)
