"""
Batch Processing Module for Document-Aware Extraction

Handles batch processing of images through document detection and extraction pipeline.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from rich import print as rprint
from rich.console import Console
from rich.progress import track

from .evaluation_metrics import load_ground_truth

# Import Rich content sanitization to prevent recursion errors
from .extraction_cleaner import sanitize_for_rich
from .simple_model_evaluator import SimpleModelEvaluator
from .simple_prompt_loader import SimplePromptLoader


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

        # Use SimpleModelEvaluator for model comparison
        self.model_evaluator = SimpleModelEvaluator()
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
                    # Define document-specific field lists for evaluation
                    doc_type_fields = {
                        "invoice": [
                            "DOCUMENT_TYPE",
                            "BUSINESS_ABN",
                            "SUPPLIER_NAME",
                            "BUSINESS_ADDRESS",
                            "PAYER_NAME",
                            "PAYER_ADDRESS",
                            "INVOICE_DATE",
                            "LINE_ITEM_DESCRIPTIONS",
                            "LINE_ITEM_QUANTITIES",
                            "LINE_ITEM_PRICES",
                            "LINE_ITEM_TOTAL_PRICES",
                            "IS_GST_INCLUDED",
                            "GST_AMOUNT",
                            "TOTAL_AMOUNT",
                        ],
                        "receipt": [
                            "DOCUMENT_TYPE",
                            "BUSINESS_ABN",
                            "SUPPLIER_NAME",
                            "BUSINESS_ADDRESS",
                            "PAYER_NAME",
                            "PAYER_ADDRESS",
                            "INVOICE_DATE",
                            "LINE_ITEM_DESCRIPTIONS",
                            "LINE_ITEM_QUANTITIES",
                            "LINE_ITEM_PRICES",
                            "LINE_ITEM_TOTAL_PRICES",
                            "IS_GST_INCLUDED",
                            "GST_AMOUNT",
                            "TOTAL_AMOUNT",
                        ],
                        "bank_statement": [
                            "DOCUMENT_TYPE",
                            "STATEMENT_DATE_RANGE",
                            "LINE_ITEM_DESCRIPTIONS",
                            "TRANSACTION_DATES",
                            "TRANSACTION_AMOUNTS_PAID",
                            "TRANSACTION_AMOUNTS_RECEIVED",
                            "ACCOUNT_BALANCE",
                        ],
                    }
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
                    fields_matched = evaluation_result.correct_fields

                    evaluation = {
                        "overall_accuracy": evaluation_result.accuracy,
                        "total_fields": evaluation_result.total_fields,
                        "correct_fields": evaluation_result.correct_fields,
                        "missing_fields": evaluation_result.missing_fields,
                        "incorrect_fields": evaluation_result.incorrect_fields,
                        # Add notebook-expected keys
                        "fields_extracted": fields_extracted,
                        "fields_matched": fields_matched,
                    }

                    # SimpleModelEvaluator already provides data in correct format - no flattening needed

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

            if any(
                field == "" or field == "NOT_FOUND"
                for field in [descriptions, dates, paid, balances]
            ):
                if verbose:
                    rprint(
                        "[yellow]⚠️ Missing transaction data - skipping debit filtering[/yellow]"
                    )
                return extracted_data

            # Create DataFrame from transaction data
            transactions_df = pd.DataFrame(
                {
                    "description": descriptions.split(" | "),
                    "date": dates.split(" | "),
                    "paid": paid.split(" | "),
                    "received": received.split(" | ")
                    if received != "NOT_FOUND"
                    else None,
                    "balance": balances.split(" | "),
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

    def _parse_document_type_response(
        self, response: str, detection_config: dict
    ) -> str:
        """Parse document type response using YAML-configured type mappings."""
        if not response:
            return detection_config.get("settings", {}).get("fallback_type", "INVOICE")

        response_lower = response.lower().strip()

        # Check for bank statement first (matches both "bank_statement" and "bank statement")
        if "bank" in response_lower and "statement" in response_lower:
            return "BANK_STATEMENT"

        # Check for receipt
        if "receipt" in response_lower:
            return "RECEIPT"

        # Check for invoice
        if "invoice" in response_lower or "bill" in response_lower:
            return "INVOICE"

        # Look in type mappings for variations
        type_mappings = detection_config.get("type_mappings", {})
        for variant, canonical in type_mappings.items():
            if variant.lower() in response_lower:
                return canonical

        # Final fallback
        fallback = detection_config.get("settings", {}).get("fallback_type", "INVOICE")
        return fallback

    # Removed: _classify_bank_statement_structure - filename-based classification is inappropriate

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

        # Load detection config from YAML
        detection_path = Path(self.prompt_config["detection_file"])
        with detection_path.open("r") as f:
            detection_config = yaml.safe_load(f)

        # Get detection prompt and settings
        # Use the key specified in prompt_config, falling back to YAML default if not specified
        detection_prompt_key = self.prompt_config.get(
            "detection_key"
        ) or detection_config.get("settings", {}).get("default_prompt", "detection")
        doc_type_prompt = detection_config["prompts"][detection_prompt_key]["prompt"]
        max_tokens = detection_config.get("settings", {}).get("max_new_tokens", 50)

        # Debug: Show detection token configuration
        if verbose:
            configured_model_tokens = getattr(self.model.config, "max_new_tokens", None)
            rprint(
                f"[cyan]🔧 Detection tokens - YAML: {max_tokens}, Model config: {configured_model_tokens}[/cyan]"
            )

        # Show detection prompt when verbose
        if verbose:
            rprint("\n[bold cyan]📋 DOCUMENT TYPE DETECTION[/bold cyan]")
            rprint("[cyan]━" * 80 + "[/cyan]")
            rprint(
                f"[yellow]Detection Prompt (using key: '{detection_prompt_key}'):[/yellow]"
            )
            rprint(f"[dim]{doc_type_prompt}[/dim]")
            rprint("[cyan]━" * 80 + "[/cyan]\n")

        # Use direct model approach (like working llama_single_image.ipynb)
        from PIL import Image

        # Load image directly
        image = Image.open(image_path)

        # Create message structure like working notebook
        messageDataStructure = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": doc_type_prompt},
                ],
            }
        ]

        # Process input like working notebook
        textInput = self.processor.apply_chat_template(
            messageDataStructure, add_generation_prompt=True
        )
        inputs = self.processor(image, textInput, return_tensors="pt")

        # Clear GPU 1 cache before processing since model has layers there
        torch.cuda.empty_cache()
        if torch.cuda.device_count() > 1:
            with torch.cuda.device(1):
                torch.cuda.empty_cache()

        # Move to model device like working notebook
        inputs = inputs.to(self.model.device)

        # Generate response directly like working notebook
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        response = self.processor.decode(output[0], skip_special_tokens=True)

        # Show raw response when verbose (sanitized for Rich console safety)
        if verbose:
            safe_response = sanitize_for_rich(response, max_length=500)
            rprint(f"[yellow]Model Response:[/yellow] {safe_response}")

        # Parse document type from response
        document_type = self._parse_document_type_response(response, detection_config)

        if verbose:
            rprint(f"[green]✅ Detected Document Type: {document_type}[/green]\n")

        # Memory cleanup between detection and extraction
        if verbose:
            rprint("[dim]🧹 Cleaning memory before extraction...[/dim]")
        from common.gpu_optimization import emergency_cleanup

        emergency_cleanup(verbose=False)

        # Initialize bank statement structure (used for prompt selection)
        bank_structure = None

        # Bank statements: Use vision-based structure classification for optimal prompt selection
        if document_type == "BANK_STATEMENT":
            from .vision_bank_statement_classifier import (
                classify_bank_statement_structure_vision,
            )

            # Classify bank statement structure using vision analysis
            bank_structure = classify_bank_statement_structure_vision(
                image_path=image_path,
                model=self.model,
                processor=self.processor,
                verbose=verbose,
            )

            if verbose:
                rprint(f"[cyan]🏦 Bank statement structure: {bank_structure}[/cyan]")
                rprint(
                    f"[cyan]📁 Using prompt: llama_prompts.yaml (bank_statement_{bank_structure})[/cyan]"
                )

        # Step 2: Load document-specific prompt using SimplePromptLoader
        prompt_loader = SimplePromptLoader()
        # Use llama prompts for Llama processor - with structure-specific bank statement support
        try:
            if document_type == "BANK_STATEMENT":
                # Use structure-specific bank statement prompt
                prompt_key = f"bank_statement_{bank_structure}"
                extraction_prompt = prompt_loader.load_prompt(
                    "llama_prompts.yaml", prompt_key
                )
                prompt_name = f"llama_{prompt_key}_prompt"
            else:
                # Use standard document type prompt
                extraction_prompt = prompt_loader.load_prompt(
                    "llama_prompts.yaml", document_type.lower()
                )
                prompt_name = f"llama_{document_type.lower()}_prompt"
        except KeyError:
            # Fall back to universal prompt if specific document type not found
            extraction_prompt = prompt_loader.load_prompt(
                "llama_prompts.yaml", "universal"
            )
            prompt_name = "llama_universal_prompt"

        # Step 3: Extract fields using simplified field mapping
        doc_type_fields = {
            "invoice": [
                "DOCUMENT_TYPE",
                "BUSINESS_ABN",
                "SUPPLIER_NAME",
                "BUSINESS_ADDRESS",
                "PAYER_NAME",
                "PAYER_ADDRESS",
                "INVOICE_DATE",
                "LINE_ITEM_DESCRIPTIONS",
                "LINE_ITEM_QUANTITIES",
                "LINE_ITEM_PRICES",
                "LINE_ITEM_TOTAL_PRICES",
                "IS_GST_INCLUDED",
                "GST_AMOUNT",
                "TOTAL_AMOUNT",
            ],
            "receipt": [
                "DOCUMENT_TYPE",
                "BUSINESS_ABN",
                "SUPPLIER_NAME",
                "BUSINESS_ADDRESS",
                "PAYER_NAME",
                "PAYER_ADDRESS",
                "INVOICE_DATE",
                "LINE_ITEM_DESCRIPTIONS",
                "LINE_ITEM_QUANTITIES",
                "LINE_ITEM_PRICES",
                "LINE_ITEM_TOTAL_PRICES",
                "IS_GST_INCLUDED",
                "GST_AMOUNT",
                "TOTAL_AMOUNT",
            ],
            "bank_statement": [
                "DOCUMENT_TYPE",
                "STATEMENT_DATE_RANGE",
                "LINE_ITEM_DESCRIPTIONS",
                "TRANSACTION_DATES",
                "TRANSACTION_AMOUNTS_PAID",
                "TRANSACTION_AMOUNTS_RECEIVED",
                "ACCOUNT_BALANCE",
            ],
        }
        # Ensure case-insensitive document type matching for field list selection
        document_type_lower = document_type.lower()
        field_list = doc_type_fields.get(
            document_type_lower, doc_type_fields["invoice"]
        )

        # Create document-aware processor with loaded model/processor
        if verbose:
            rprint("\n[bold cyan]📋 FIELD EXTRACTION[/bold cyan]")
            rprint(
                f"[cyan]Creating extraction processor with {len(field_list)} fields for {document_type}[/cyan]"
            )
            rprint(
                f"[dim]Fields: {', '.join(field_list[:3])}... ({len(field_list)} total)[/dim]"
            )

        # Use configured max_tokens from model if available, otherwise calculate
        configured_tokens = getattr(self.model.config, "max_new_tokens", None)
        if configured_tokens:
            max_tokens = configured_tokens
            if verbose:
                rprint(
                    f"[cyan]🔧 Using configured max_tokens: {max_tokens} (from notebook CONFIG)[/cyan]"
                )
        else:
            # Fallback to calculation
            from .config import get_max_new_tokens

            max_tokens = get_max_new_tokens("llama", len(field_list))
            if verbose:
                rprint(
                    f"[cyan]🔧 Calculated max_tokens: {max_tokens} for {len(field_list)} fields[/cyan]"
                )

        # Use direct model approach for extraction (like working notebook)
        image = Image.open(image_path)

        # Create message structure like working notebook
        messageDataStructure = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": extraction_prompt},
                ],
            }
        ]

        # Process input like working notebook
        textInput = self.processor.apply_chat_template(
            messageDataStructure, add_generation_prompt=True
        )
        inputs = self.processor(image, textInput, return_tensors="pt").to(
            self.model.device
        )

        # Generate response directly like working notebook
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        response = self.processor.decode(output[0], skip_special_tokens=True)

        # Show raw extraction response when verbose
        if verbose:
            rprint("\n[bold yellow]📄 RAW MODEL EXTRACTION RESPONSE:[/bold yellow]")
            # Show response length and check if model generated output
            rprint(f"[dim]Total response length: {len(response)} characters[/dim]")

            # Extract only the assistant's response (after the prompt)
            # Llama responses include the full conversation, split at "assistant" marker
            if "assistant" in response:
                # Get everything after the last "assistant" marker
                model_response = response.split("assistant")[-1].strip()
                if not model_response or len(model_response) < 50:
                    rprint(
                        "[red]⚠️ WARNING: Model generated very short or empty response![/red]"
                    )
                    rprint(
                        "[yellow]This likely means max_tokens is too low for this prompt.[/yellow]"
                    )
            else:
                # No assistant marker means model didn't generate anything
                rprint(
                    "[red]⚠️ CRITICAL: No 'assistant' response found in output![/red]"
                )
                rprint(
                    "[yellow]Model may have hit token limit during prompt processing.[/yellow]"
                )
                model_response = response

            # Show only last 3000 chars to see the actual extraction part
            if len(model_response) > 3000:
                display_response = "...[truncated]..." + model_response[-3000:]
            else:
                display_response = model_response

            # Sanitize for Rich console
            safe_response = sanitize_for_rich(display_response, max_length=10000)
            rprint(f"[dim]{safe_response}[/dim]")
            rprint("[cyan]━" * 80 + "[/cyan]\n")

        # Create extraction result in expected format
        from .extraction_parser import parse_extraction_response

        parsed_data = parse_extraction_response(
            response, clean_conversation_artifacts=False, expected_fields=field_list
        )

        # Show parsed/cleaned data when verbose
        if verbose:
            rprint("[bold green]✅ PARSED EXTRACTION DATA:[/bold green]")
            for field, value in parsed_data.items():
                if value != "NOT_FOUND":
                    # Truncate long values for readability
                    display_value = (
                        str(value)[:100] + "..." if len(str(value)) > 100 else value
                    )
                    rprint(f"  [cyan]{field}:[/cyan] {display_value}")
            rprint("[cyan]━" * 80 + "[/cyan]\n")

        extraction_result = {
            "extracted_data": parsed_data,
            "raw_response": response,
            "field_list": field_list,
        }

        return document_type, extraction_result, prompt_name
