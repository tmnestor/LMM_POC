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

        # Load classification prompt from YAML file
        self.classification_prompt = self._load_classification_prompt()

    def _load_classification_prompt(self) -> str:
        """Load classification prompt from YAML file."""
        try:
            from .simple_prompt_loader import SimplePromptLoader

            prompt_loader = SimplePromptLoader()
            classification_prompt = prompt_loader.load_prompt(
                "bank_statement_classification.yaml",
                "structure_classification"
            )
            return classification_prompt
        except Exception as e:
            # Fallback prompt if YAML loading fails
            if self.verbose:
                rprint(f"[yellow]⚠️ Could not load classification prompt from YAML: {e}[/yellow]")
                rprint("[yellow]Using fallback prompt[/yellow]")

            return """
            Analyze this bank statement and classify its layout.

            If you see date headers that separate transaction groups, respond: DATE_GROUPED
            If transactions are in one continuous table, respond: FLAT

            Response:"""

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
                if self.verbose:
                    rprint("[dim]🤖 Using VLM for structure analysis...[/dim]")
                classification_result = self._analyze_with_vlm(image_path)

                if self.verbose:
                    rprint(f"[dim]📋 VLM response: '{classification_result}'[/dim]")
            else:
                # Fallback: Conservative default if no model available
                if self.verbose:
                    rprint("[yellow]⚠️ No VLM model provided, using conservative default[/yellow]")
                classification_result = "flat"

            # Normalize result with better detection
            classification_upper = classification_result.upper()

            # Look for generic classification indicators (no specific data from actual images)
            date_indicators = [
                "DATE", "GROUPED", "SECTION", "HEADER",  # Original core terms
                "ORGANIZE", "SEPARATE", "UNDER", "LABELS",  # Action/relationship words
                "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"  # Generic day names
            ]
            flat_indicators = [
                "FLAT", "TABLE", "COLUMN", "CONTINUOUS", "ROW",  # Original core terms
                "SINGLE", "UNBROKEN", "CONSISTENT"  # Layout continuity terms
            ]

            has_date_indicators = any(indicator in classification_upper for indicator in date_indicators)
            has_flat_indicators = any(indicator in classification_upper for indicator in flat_indicators)

            if has_date_indicators and not has_flat_indicators:
                structure_type = "date_grouped"
            elif has_flat_indicators and not has_date_indicators:
                structure_type = "flat"
            elif "DATE" in classification_upper or "GROUPED" in classification_upper:
                structure_type = "date_grouped"
            else:
                # Default fallback
                structure_type = "flat"

            if self.verbose:
                rprint(f"[dim]🧠 Analysis: date_indicators={has_date_indicators}, flat_indicators={has_flat_indicators}[/dim]")

                # Show which specific indicators were triggered for debugging
                triggered_date = [ind for ind in date_indicators if ind in classification_upper]
                triggered_flat = [ind for ind in flat_indicators if ind in classification_upper]
                if triggered_date:
                    rprint(f"[dim]📅 Triggered date indicators: {triggered_date}[/dim]")
                if triggered_flat:
                    rprint(f"[dim]📊 Triggered flat indicators: {triggered_flat}[/dim]")

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
            # Direct InternVL3 model with chat method
            return self._analyze_with_internvl3(image_path)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'chat'):
            # InternVL3 processor wrapping a model with chat method
            return self._analyze_with_internvl3(image_path)
        elif hasattr(self.model, 'load_image'):
            # InternVL3 processor (has load_image method)
            return self._analyze_with_internvl3(image_path)
        elif self.processor is not None:
            # Llama-style processing with separate processor
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
        try:
            # Check if model has the InternVL3-specific load_image method
            if hasattr(self.model, 'load_image'):
                # Model is wrapped in a processor that has load_image
                pixel_values = self.model.load_image(image_path)
            elif hasattr(self.model, '__class__') and 'InternVL3' in str(self.model.__class__):
                # Direct InternVL3 model - need to preprocess manually
                from models.document_aware_internvl3_processor import (
                    DocumentAwareInternVL3Processor,
                )
                # Create a temporary processor instance to use its load_image method
                temp_processor = DocumentAwareInternVL3Processor(self.model, debug=False)
                pixel_values = temp_processor.load_image(image_path)
            else:
                # Fallback: Try to load and preprocess the image directly
                from PIL import Image
                image = Image.open(image_path).convert('RGB')

                # Try to find the model's preprocessing method
                if hasattr(self.model, 'img_processor'):
                    pixel_values = self.model.img_processor([image])
                elif hasattr(self.model, 'image_processor'):
                    pixel_values = self.model.image_processor([image])
                else:
                    # Last resort - pass raw image and hope model handles it
                    pixel_values = image

            # Use the same proven method as document detection and extraction
            if hasattr(self.model, '_resilient_generate'):
                # It's a processor with the working _resilient_generate method
                response = self.model._resilient_generate(
                    pixel_values=pixel_values,
                    question=self.classification_prompt,
                    max_new_tokens=50,
                    temperature=0.0,
                    do_sample=False,
                    is_detection=True  # Classification is like detection - short response
                )
            else:
                # Fallback to manual interface (should not happen with proper processor)
                response = self.model.model.chat(
                    self.model.tokenizer,
                    pixel_values,
                    self.classification_prompt,
                    generation_config=dict(
                        max_new_tokens=50,
                        temperature=0.0,
                        do_sample=False,
                        top_p=0.9,
                        use_cache=True,
                    ),
                    history=None,
                    return_history=False
                )

            return response.strip() if isinstance(response, str) else str(response).strip()

        except Exception as e:
            if self.verbose:
                rprint(f"[red]❌ InternVL3 analysis failed: {e}[/red]")
            raise

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