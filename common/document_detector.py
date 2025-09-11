"""Document type detection utilities.

This module provides functionality to automatically detect document types
(invoice, receipt, bank statement) using vision-language models with static YAML prompts.
"""

from pathlib import Path

import yaml
from rich import print as rprint
from rich.console import Console
from rich.table import Table


class DocumentDetector:
    """Handles document type detection using YAML-based prompts."""

    def __init__(
        self, detection_prompt_file: str, detection_prompt_key: str = "detection"
    ):
        """Initialize the document detector.

        Args:
            detection_prompt_file: Path to YAML file containing detection prompts
            detection_prompt_key: Key of the prompt to use from the YAML file
        """
        self.detection_prompt_file = detection_prompt_file
        self.detection_prompt_key = detection_prompt_key
        self.console = Console()

        # Load detection configuration
        self._load_detection_config()

    def _load_detection_config(self):
        """Load detection prompt and configuration from YAML file."""
        try:
            detection_path = Path(self.detection_prompt_file)

            if not detection_path.exists():
                rprint(
                    f"[red]❌ Detection prompt file not found at {self.detection_prompt_file}[/red]"
                )
                raise FileNotFoundError(
                    f"Required prompt file not found: {self.detection_prompt_file}"
                )

            # Load the YAML file
            with detection_path.open("r") as f:
                detection_config = yaml.safe_load(f)

            # Extract the detection prompt
            if self.detection_prompt_key in detection_config.get("prompts", {}):
                detection_prompt_data = detection_config["prompts"][
                    self.detection_prompt_key
                ]
                self.detection_prompt = detection_prompt_data["prompt"]
                self.detection_name = detection_prompt_data.get(
                    "name", self.detection_prompt_key
                )

                rprint(
                    f"[green]✅ Loaded detection prompt: {self.detection_name}[/green]"
                )
            else:
                available_keys = list(detection_config.get("prompts", {}).keys())
                rprint(
                    f"[red]❌ Detection prompt key '{self.detection_prompt_key}' not found[/red]"
                )
                raise KeyError(
                    f"Prompt '{self.detection_prompt_key}' not found. Available: {available_keys}"
                )

            # Get type mappings for normalization
            self.type_mappings = detection_config.get("type_mappings", {})

        except Exception as e:
            rprint(f"[red]❌ Error loading detection prompt: {e}[/red]")
            raise

    def detect_document_type(
        self, image_path: str, extractor, prompt_files: dict
    ) -> str:
        """Detect document type from an image.

        Args:
            image_path: Path to the image file
            extractor: Vision model extractor instance
            prompt_files: Dictionary mapping document types to prompt files

        Returns:
            Detected document type (INVOICE, RECEIPT, BANK_STATEMENT, or ESTIMATE)
        """
        if not image_path:
            rprint("[red]❌ No image path available for detection[/red]")
            return "INVOICE"

        image_name = Path(image_path).name
        rprint(f"[yellow]🔍 Detecting document type for: {image_name}[/yellow]")

        try:
            # Use the extractor to run detection prompt
            detection_result = extractor.test_extraction(
                image_path,
                self.detection_prompt,
                max_new_tokens=50,  # Short response for detection
            )

            if detection_result.get("success"):
                raw_detection = detection_result["raw_result"]["raw_response"]

                # Parse and normalize the detection response
                detected_type = self._parse_detection_response(raw_detection)

                # Display detection result
                self._display_detection_result(
                    detected_type, raw_detection, prompt_files
                )

                rprint(
                    f"[bold green]✅ Document type detected: {detected_type}[/bold green]"
                )
                return detected_type

            else:
                rprint(
                    f"[red]❌ Detection failed: {detection_result.get('error', 'Unknown error')}[/red]"
                )
                rprint("[yellow]⚠️ Defaulting to INVOICE[/yellow]")
                return "INVOICE"

        except Exception as e:
            rprint(f"[red]❌ Error during detection: {e}[/red]")
            rprint("[yellow]⚠️ Defaulting to INVOICE[/yellow]")
            return "INVOICE"

    def _parse_detection_response(self, raw_detection: str) -> str:
        """Parse and normalize the raw detection response.

        Args:
            raw_detection: Raw response from the model

        Returns:
            Normalized document type
        """
        # Parse the detection response
        detected_type = raw_detection.strip().upper()

        # Apply type mappings to normalize variations
        for variation, canonical in self.type_mappings.items():
            if variation.lower() in raw_detection.lower():
                detected_type = canonical
                break

        # Ensure we have a valid document type
        if detected_type not in ["INVOICE", "RECEIPT", "BANK_STATEMENT", "STATEMENT", "ESTIMATE"]:
            # Try to extract from response
            if "INVOICE" in detected_type or "BILL" in detected_type:
                detected_type = "INVOICE"
            elif "ESTIMATE" in detected_type or "QUOTE" in detected_type:
                detected_type = "ESTIMATE"
            elif "RECEIPT" in detected_type:
                detected_type = "RECEIPT"
            elif "STATEMENT" in detected_type or "BANK" in detected_type:
                detected_type = "BANK_STATEMENT"
            else:
                rprint(
                    f"[yellow]⚠️ Unknown document type: {detected_type}, defaulting to INVOICE[/yellow]"
                )
                detected_type = "INVOICE"

        # Normalize STATEMENT to BANK_STATEMENT
        if detected_type == "STATEMENT":
            detected_type = "BANK_STATEMENT"

        return detected_type

    def _display_detection_result(
        self, detected_type: str, raw_detection: str, prompt_files: dict
    ):
        """Display the detection result in a formatted table.

        Args:
            detected_type: Normalized document type
            raw_detection: Raw response from model
            prompt_files: Dictionary mapping document types to prompt files
        """
        detection_table = Table(
            title="🔍 Document Type Detection Result", border_style="blue"
        )
        detection_table.add_column("Property", style="cyan")
        detection_table.add_column("Value", style="yellow")

        detection_table.add_row("Detected Type", detected_type)
        detection_table.add_row(
            "Prompt File", prompt_files.get(detected_type, "Unknown")
        )
        detection_table.add_row("Raw Response", raw_detection[:100])

        self.console.print(detection_table)


def detect_document_type(
    image_path: str,
    extractor,
    detection_prompt_file: str,
    prompt_files: dict,
    detection_prompt_key: str = "detection",
) -> str:
    """Convenience function for document type detection.

    Args:
        image_path: Path to the image file
        extractor: Vision model extractor instance
        detection_prompt_file: Path to YAML file containing detection prompts
        prompt_files: Dictionary mapping document types to prompt files
        detection_prompt_key: Key of the prompt to use from the YAML file

    Returns:
        Detected document type (INVOICE, RECEIPT, BANK_STATEMENT, or ESTIMATE)
    """
    detector = DocumentDetector(detection_prompt_file, detection_prompt_key)
    return detector.detect_document_type(image_path, extractor, prompt_files)
