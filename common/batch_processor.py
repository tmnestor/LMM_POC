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

from .document_type_metrics import DocumentTypeEvaluator
from .evaluation_metrics import load_ground_truth
from .prompt_loader import load_document_prompt


class BatchDocumentProcessor:
    """Handles batch processing of documents with extraction and evaluation."""

    def __init__(
        self,
        model,
        processor,
        prompt_config: Dict,
        ground_truth_csv: str,
        console: Optional[Console] = None,
    ):
        """
        Initialize batch processor with support for both Llama and InternVL3 models.

        Args:
            model: Loaded model instance (Llama model or InternVL3 handler)
            processor: Loaded processor instance (Llama processor or None for InternVL3)
            prompt_config: Dictionary with prompt file paths and keys
            ground_truth_csv: Path to ground truth CSV file
            console: Rich console for output
        """
        # Detect model type and store components appropriately
        self.model_type = self._detect_model_type(model, processor)

        if self.model_type == "internvl3":
            # For InternVL3, model param is actually the handler
            self.internvl3_handler = model
            self.model = None
            self.processor = None
        else:
            # For Llama, use traditional model/processor
            self.model = model
            self.processor = processor
            self.internvl3_handler = None

        self.prompt_config = prompt_config
        self.ground_truth_csv = ground_truth_csv
        self.console = console or Console()

        # Initialize file-based trace logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._trace_file = f"batch_processor_trace_{timestamp}.log"

        # Use DocumentTypeEvaluator for all evaluation
        self.document_evaluator = DocumentTypeEvaluator()
        self.ground_truth_data = None

    def _trace_log(self, message: str):
        """Log message to both console and file"""
        print(message)
        with Path(self._trace_file).open("a", encoding="utf-8") as f:
            f.write(f"{message}\n")

    def _detect_model_type(self, model, processor) -> str:
        """
        Detect whether we're working with Llama or InternVL3 components.

        Args:
            model: Model or handler object
            processor: Processor object (None for InternVL3)

        Returns:
            String indicating model type: 'llama' or 'internvl3'
        """
        # InternVL3 handler has these specific methods
        if hasattr(model, "detect_and_classify_document") and hasattr(
            model, "process_document_aware"
        ):
            return "internvl3"
        else:
            return "llama"

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

                # Route processing based on model type
                if self.model_type == "internvl3":
                    if verbose:
                        rprint(
                            f"[dim]🔍 TRACE: Processing InternVL3 image {idx}/{len(image_paths)}: {image_name}[/dim]"
                        )

                    # InternVL3 processing path
                    document_type, extraction_result, prompt_name = (
                        self._process_internvl3_image(image_path, verbose)
                    )
                    document_types_found[document_type] = (
                        document_types_found.get(document_type, 0) + 1
                    )

                    if verbose:
                        rprint(
                            f"[dim]🔍 TRACE: InternVL3 processing complete for {image_name}, doc_type={document_type}[/dim]"
                        )

                else:
                    # Llama processing path (original logic)
                    document_type, extraction_result, prompt_name = (
                        self._process_llama_image(image_path, verbose)
                    )
                    document_types_found[document_type] = (
                        document_types_found.get(document_type, 0) + 1
                    )

                # Step 4: Evaluate against ground truth using working DocumentTypeEvaluator approach
                image_name = Path(image_path).name
                ground_truth = (
                    self.ground_truth_data.get(image_name, {})
                    if self.ground_truth_data
                    else {}
                )

                if ground_truth:
                    # Extract data using working DocumentAwareLlamaProcessor format
                    # DocumentAwareLlamaProcessor returns extracted_data at top level
                    extracted_data = extraction_result.get("extracted_data", {})

                    # Apply mathematical enhancement for bank statements
                    if document_type.upper() == "BANK_STATEMENT":
                        from .bank_statement_calculator import enhance_bank_statement_extraction

                        if verbose:
                            rprint(f"[blue]🧮 Applying mathematical enhancement for bank statement[/blue]")

                        extracted_data = enhance_bank_statement_extraction(
                            extracted_data, verbose=verbose
                        )

                    if verbose:
                        found_fields = [
                            k for k, v in extracted_data.items() if v != "NOT_FOUND"
                        ]
                        rprint(
                            f"[cyan]✓ Extracted {len(found_fields)} fields from {image_name}[/cyan]"
                        )

                        # Show mathematical enhancement results if applied
                        if document_type.upper() == "BANK_STATEMENT" and '_mathematical_analysis' in extracted_data:
                            analysis = extracted_data['_mathematical_analysis']
                            if analysis.get('calculation_success'):
                                rprint(f"[green]✓ Mathematical analysis: {analysis.get('transaction_count', 0)} transactions calculated[/green]")
                            else:
                                rprint(f"[yellow]⚠️ Mathematical analysis failed[/yellow]")

                    # Use the working DocumentTypeEvaluator approach that succeeds
                    evaluation = self.document_evaluator.evaluate_extraction(
                        extracted_data, ground_truth, document_type
                    )

                    # Fix structure mismatch: BatchAnalytics expects overall_accuracy at top level
                    # but DocumentTypeEvaluator puts it in overall_metrics
                    if evaluation and "overall_metrics" in evaluation:
                        # Flatten the structure for BatchAnalytics compatibility
                        evaluation["overall_accuracy"] = evaluation[
                            "overall_metrics"
                        ].get("overall_accuracy", 0)
                        evaluation["fields_extracted"] = evaluation[
                            "overall_metrics"
                        ].get("total_fields_evaluated", 0)
                        evaluation["fields_matched"] = evaluation[
                            "overall_metrics"
                        ].get("fields_correct", 0)
                        evaluation["total_fields"] = evaluation["overall_metrics"].get(
                            "total_fields_evaluated", 0
                        )

                    # Show evaluation summary
                    if verbose and evaluation:
                        accuracy = evaluation.get("overall_accuracy", 0) * 100
                        rprint(
                            f"[cyan]✓ Accuracy: {accuracy:.1f}% for {image_name}[/cyan]"
                        )

                    # Only show detailed comparison for Llama - InternVL3 has its own display logic
                    if (
                        verbose
                        and evaluation
                        and "field_scores" in evaluation
                        and self.model_type == "llama"
                    ):
                        self._display_detailed_field_comparison(
                            image_name,
                            extracted_data,
                            ground_truth,
                            evaluation,
                            document_type,
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

                # Progress update
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

    def _parse_document_type_response(
        self, response: str, detection_config: dict
    ) -> str:
        """Parse document type response using YAML-configured type mappings."""
        if not response:
            return detection_config.get("settings", {}).get("fallback_type", "INVOICE")

        response_lower = response.lower().strip()

        # First try direct match for standard types
        standard_types = ["INVOICE", "RECEIPT", "BANK_STATEMENT"]
        for doc_type in standard_types:
            if doc_type.lower() in response_lower:
                return doc_type

        # Look in type mappings for variations
        type_mappings = detection_config.get("type_mappings", {})
        for variant, canonical in type_mappings.items():
            if variant.lower() in response_lower:
                return canonical

        # Final fallback
        fallback = detection_config.get("settings", {}).get("fallback_type", "INVOICE")
        return fallback

    def _classify_bank_statement_structure(
        self, image_path: str, verbose: bool = False
    ):
        """Classify bank statement structure for optimal prompt selection."""
        from .bank_statement_classifier import classify_bank_statement_structure

        return classify_bank_statement_structure(image_path, verbose)

    def _display_detailed_field_comparison(
        self,
        image_name: str,
        extracted_data: dict,
        ground_truth: dict,
        evaluation: dict,
        document_type: str,
    ):
        """Display detailed field-by-field comparison like in document-aware system."""

        rprint(f"\n{'=' * 120}")
        rprint("📋 STEP 4: Extracted Data Results with Ground Truth Comparison")
        rprint("=" * 120)

        # Display extracted data first
        rprint("\n🔍 EXTRACTED DATA:")
        for field, value in extracted_data.items():
            if value != "NOT_FOUND":
                rprint(f"✅ {field}: {value}")
            else:
                rprint(f"❌ {field}: {value}")

        # Ground truth comparison table
        rprint(f"\n📊 Ground truth loaded for {image_name}")
        rprint("-" * 120)

        field_scores = evaluation.get("field_scores", {})

        # Table header with consistent spacing like document-aware system
        rprint(f"{'STATUS':<8} {'FIELD':<25} {'EXTRACTED':<40} {'GROUND TRUTH':<40}")
        rprint("=" * 120)

        # Field-by-field comparison
        fields_found = 0
        exact_matches = 0
        total_fields = len(field_scores)

        for field, score in field_scores.items():
            extracted_val = extracted_data.get(field, "NOT_FOUND")
            ground_val = ground_truth.get(field, "NOT_FOUND")

            # Determine status symbol (same logic as document-aware system)
            if score.get("accuracy", 0) == 1.0:
                status = "✅"
                exact_matches += 1
            elif score.get("accuracy", 0) >= 0.8:
                status = "≈"
            else:
                status = "❌"

            if extracted_val != "NOT_FOUND":
                fields_found += 1

            # Truncate long values for display (same as document-aware system)
            extracted_display = str(extracted_val)[:38] + (
                "..." if len(str(extracted_val)) > 38 else ""
            )
            ground_display = str(ground_val)[:38] + (
                "..." if len(str(ground_val)) > 38 else ""
            )

            rprint(
                f"{status:<8} {field:<25} {extracted_display:<40} {ground_display:<40}"
            )

        # Summary section (same format as document-aware system)
        overall_accuracy = evaluation.get("overall_metrics", {}).get(
            "overall_accuracy", 0
        )

        rprint("\n📊 EXTRACTION SUMMARY:")
        rprint(
            f"✅ Fields Found: {fields_found}/{total_fields} ({fields_found / total_fields * 100:.1f}%)"
        )
        rprint(
            f"🎯 Exact Matches: {exact_matches}/{total_fields} ({exact_matches / total_fields * 100:.1f}%)"
        )
        rprint(f"📈 Extraction Success Rate: {overall_accuracy * 100:.1f}%")
        rprint(f"⏱️ Accuracy (matches/total): {overall_accuracy * 100:.1f}%")
        rprint(f"🤖 Document Type: {document_type}")
        rprint("🔧 Model: Llama-3.2-11B-Vision-Instruct")

        # Additional metrics (same as document-aware system)
        meets_threshold = evaluation.get("overall_metrics", {}).get(
            "meets_threshold", False
        )
        threshold = evaluation.get("overall_metrics", {}).get(
            "document_type_threshold", 0.8
        )
        rprint("\n≈ = Partial match")
        rprint("✗ = No match")
        rprint(
            f"Note: Meets accuracy threshold ({threshold * 100:.0f}%): {'✅ Yes' if meets_threshold else '❌ No'}"
        )
        rprint("=" * 120)

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

        # Step 1: Detect and classify document
        classification_info = self.internvl3_handler.detect_and_classify_document(
            image_path
        )
        document_type = classification_info["document_type"]

        if verbose:
            rprint(f"[cyan]📄 Document type detected: {document_type}[/cyan]")

        # Step 2: Process with document-aware extraction
        extraction_result = self.internvl3_handler.process_document_aware(
            image_path, classification_info
        )

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

    def _process_llama_image(
        self, image_path: str, verbose: bool
    ) -> Tuple[str, Dict, str]:
        """
        Process single image using Llama model (original logic).

        Args:
            image_path: Path to image
            verbose: Whether to show verbose output

        Returns:
            Tuple of (document_type, extraction_result, prompt_name)
        """
        # Step 1: Detect document type using YAML-first approach
        import yaml

        from common.unified_schema import DocumentTypeFieldSchema

        # Load detection config from YAML
        detection_path = Path(self.prompt_config["detection_file"])
        with detection_path.open("r") as f:
            detection_config = yaml.safe_load(f)

        # Get detection prompt and settings
        detection_prompt_key = detection_config.get("settings", {}).get(
            "default_prompt", "detection"
        )
        doc_type_prompt = detection_config["prompts"][detection_prompt_key]["prompt"]
        max_tokens = detection_config.get("settings", {}).get("max_new_tokens", 50)

        # Create simple processor for detection only
        from models.document_aware_llama_processor import DocumentAwareLlamaProcessor

        detection_processor = DocumentAwareLlamaProcessor(
            field_list=["DOCUMENT_TYPE"],  # Single field for detection
            skip_model_loading=True,
            debug=verbose,
        )
        detection_processor.model = self.model
        detection_processor.processor = self.processor

        # Extract document type using YAML prompt
        response = detection_processor._extract_with_custom_prompt(
            image_path, doc_type_prompt, max_new_tokens=max_tokens
        )

        # Parse document type from response
        document_type = self._parse_document_type_response(response, detection_config)

        # Special handling for bank statements: classify structure type
        if document_type == "BANK_STATEMENT":
            bank_structure = self._classify_bank_statement_structure(
                image_path, verbose
            )

            # Update prompt configuration for bank statement structure
            extraction_files = self.prompt_config["extraction_files"].copy()
            extraction_keys = self.prompt_config["extraction_keys"].copy()

            if bank_structure == "flat":
                extraction_files["BANK_STATEMENT"] = (
                    "prompts/bank_statement_flat_optimized.yaml"
                )
                extraction_keys["BANK_STATEMENT"] = "flat_optimized"
            else:  # date_grouped
                extraction_files["BANK_STATEMENT"] = (
                    "prompts/bank_statement_date_grouped.yaml"
                )
                extraction_keys["BANK_STATEMENT"] = "date_grouped"

            if verbose:
                rprint(f"[cyan]🏦 Bank statement structure: {bank_structure}[/cyan]")
                rprint(
                    f"[cyan]📁 Using prompt: {extraction_files['BANK_STATEMENT']}[/cyan]"
                )
        else:
            extraction_files = self.prompt_config["extraction_files"]
            extraction_keys = self.prompt_config["extraction_keys"]

        # Step 2: Load document-specific prompt
        extraction_prompt, prompt_name, _ = load_document_prompt(
            prompt_files=extraction_files,
            prompt_keys=extraction_keys,
            document_type=document_type,
            verbose=verbose,
        )

        # Step 3: Extract fields using DocumentAwareLlamaProcessor
        schema_loader = DocumentTypeFieldSchema()
        field_list = schema_loader.get_field_names_for_type(document_type)

        # Create document-aware processor with loaded model/processor
        doc_processor = DocumentAwareLlamaProcessor(
            field_list=field_list,
            skip_model_loading=True,  # Use existing model
            debug=verbose,
        )
        doc_processor.model = self.model
        doc_processor.processor = self.processor

        # Load max_tokens from YAML settings
        prompt_file = extraction_files.get(
            document_type, "prompts/invoice_extraction.yaml"
        )
        try:
            with Path(prompt_file).open("r") as f:
                yaml_config = yaml.safe_load(f)
                max_tokens = yaml_config.get("settings", {}).get("max_new_tokens", 600)
        except Exception:
            max_tokens = 600  # fallback

        if verbose:
            rprint(f"[cyan]🔧 Using max_tokens: {max_tokens} from {prompt_file}[/cyan]")

        # Extract data using document-aware approach with loaded YAML prompt and tokens
        extraction_result = doc_processor.process_single_image(
            image_path, custom_prompt=extraction_prompt, custom_max_tokens=max_tokens
        )

        return document_type, extraction_result, prompt_name
