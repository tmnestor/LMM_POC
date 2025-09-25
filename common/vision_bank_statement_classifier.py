"""
Vision-Based Bank Statement Structure Classification

Uses Vision-Language Models to analyze bank statement images and determine
their structural layout: flat table vs date-grouped sections.

This replaces inappropriate filename-based classification with content-driven analysis.
"""

from pathlib import Path
from typing import Literal

from rich import print as rprint


class VisionBankStatementClassifier:
    """Vision-based classifier for bank statement structural layout."""

    def __init__(self, model=None, processor=None, verbose: bool = False):
        """
        Initialize vision-based classifier.

        Args:
            model: Pre-loaded VLM model (Llama or InternVL3)
            processor: Pre-loaded processor (for Llama, None for InternVL3)
            verbose: Whether to show classification details
        """
        self.model = model
        self.processor = processor
        self.verbose = verbose

        # Classification prompt for VLM
        self.classification_prompt = """
        Analyze this bank statement image and classify its structural layout.

        Look at how transactions are organized:

        FLAT TABLE: Transactions are in a continuous table format with column headers like:
        Date | Description | Withdrawal | Deposit | Balance
        - All transactions are in one continuous table
        - Clear column structure throughout

        DATE-GROUPED: Transactions are grouped under date section headers like:
        "Thu 04 Sep 2025"
        [transactions for that date]
        "Mon 01 Sep 2025"
        [transactions for that date]
        - Date headers separate different sections
        - Transactions are grouped by date sections

        Respond with only: FLAT or DATE_GROUPED
        """

    def classify_structure_vision(
        self, image_path: str
    ) -> Literal["flat", "date_grouped"]:
        """
        Classify bank statement structure using vision analysis.

        Args:
            image_path: Path to bank statement image

        Returns:
            Either "flat" or "date_grouped" based on visual content analysis
        """
        image_name = Path(image_path).name

        if self.verbose:
            rprint(f"[cyan]🔍 Analyzing bank statement structure: {image_name}[/cyan]")

        try:
            # Use VLM to analyze the image structure
            if self.model is not None:
                classification_result = self._analyze_with_vlm(image_path)
            else:
                # Fallback: Conservative default if no model available
                if self.verbose:
                    rprint("[yellow]⚠️ No VLM model provided, using conservative default[/yellow]")
                classification_result = "flat"

            # Normalize result
            if "DATE" in classification_result.upper() or "GROUPED" in classification_result.upper():
                structure_type = "date_grouped"
            else:
                structure_type = "flat"

            if self.verbose:
                self._display_classification_result(structure_type, image_name)

            return structure_type

        except Exception as e:
            if self.verbose:
                rprint(f"[red]❌ Classification error: {e}[/red]")
                rprint("[yellow]Using fallback: flat[/yellow]")
            return "flat"

    def _analyze_with_vlm(self, image_path: str) -> str:
        """
        Use Vision-Language Model to analyze bank statement structure.

        Args:
            image_path: Path to bank statement image

        Returns:
            Classification response from VLM
        """
        # This would be implemented based on the specific VLM being used
        # For now, return a placeholder - this will be connected to the actual VLM

        # Detect model type and use appropriate processing
        if hasattr(self.model, 'chat'):
            # InternVL3-style processing
            return self._analyze_with_internvl3(image_path)
        elif self.processor is not None:
            # Llama-style processing
            return self._analyze_with_llama(image_path)
        else:
            raise ValueError("No suitable VLM model configuration found")

    def _analyze_with_llama(self, image_path: str) -> str:
        """Analyze using Llama Vision model."""
        import torch
        from PIL import Image

        # Load and process image
        image = Image.open(image_path).convert('RGB')

        # Create conversation format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.classification_prompt}
                ]
            }
        ]

        # Process with Llama processor
        input_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(image, input_text, return_tensors="pt").to(self.model.device)

        # Generate classification
        with torch.no_grad():
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.0,
                do_sample=False,
            )

        # Decode response
        output_text = self.processor.batch_decode(
            generate_ids[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return output_text.strip()

    def _analyze_with_internvl3(self, image_path: str) -> str:
        """Analyze using InternVL3 model."""
        from PIL import Image

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Use InternVL3 chat method
        response = self.model.chat(
            tokenizer=None,  # InternVL3 uses internal tokenizer
            pixel_values=image,
            question=self.classification_prompt,
            generation_config=dict(
                max_new_tokens=50,
                temperature=0.0,
                do_sample=False,
            )
        )

        return response.strip()

    def _display_classification_result(
        self, structure_type: Literal["flat", "date_grouped"], image_name: str
    ):
        """Display classification result with details."""

        structure_descriptions = {
            "flat": "Continuous table format with column headers",
            "date_grouped": "Transactions grouped under date section headers"
        }

        description = structure_descriptions[structure_type]

        rprint(f"[green]📋 Classification Result: {structure_type.upper()}[/green]")
        rprint(f"[dim]Description: {description}[/dim]")

        if structure_type == "flat":
            rprint("[dim]💡 Will use flat table extraction prompt[/dim]")
        else:
            rprint("[dim]💡 Will use date-grouped extraction prompt[/dim]")


def classify_bank_statement_structure_vision(
    image_path: str,
    model=None,
    processor=None,
    verbose: bool = False
) -> Literal["flat", "date_grouped"]:
    """
    Convenience function for vision-based bank statement structure classification.

    Args:
        image_path: Path to bank statement image
        model: Pre-loaded VLM model (Llama or InternVL3)
        processor: Pre-loaded processor (for Llama, None for InternVL3)
        verbose: Whether to show classification details

    Returns:
        Either "flat" or "date_grouped" based on visual content analysis
    """
    classifier = VisionBankStatementClassifier(model, processor, verbose)
    return classifier.classify_structure_vision(image_path)