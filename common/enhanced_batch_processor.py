"""
Enhanced Batch Processor Module

Provides clean, modular batch processing for InternVL3 with comprehensive analytics
and display features matching the Llama version.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rich import print as rprint
from rich.console import Console

from .document_type_metrics import DocumentTypeEvaluator
from .evaluation_metrics import load_ground_truth
from .gpu_optimization import cleanup_model_handler, clear_gpu_cache
from .internvl3_batch_display import (
    display_field_comparison,
    display_image,
    display_raw_and_cleaned,
)


class EnhancedBatchProcessor:
    """Enhanced batch processor for InternVL3 with comprehensive analytics."""

    def __init__(self, console: Console = None):
        """Initialize the enhanced batch processor."""
        self.console = console or Console()
        self.document_evaluator = DocumentTypeEvaluator()

    def load_ground_truth(self, ground_truth_path: str) -> Dict[str, Dict]:
        """Load ground truth data for evaluation."""
        try:
            ground_truth_data = load_ground_truth(ground_truth_path)
            rprint(f"[green]✅ Ground truth loaded for {len(ground_truth_data)} images[/green]")
            return ground_truth_data
        except Exception as e:
            rprint(f"[red]⚠️ Could not load ground truth: {e}[/red]")
            return {}

    def initialize_processor(self, processor_class, config: Dict, schema_loader) -> Any:
        """Initialize the document processor with V100 optimizations."""
        rprint("[cyan]🔧 STEP 1: Loading model ONCE for all images...[/cyan]")

        # Clean up any existing models
        cleanup_model_handler('processor', globals())

        # Get default fields to start with
        default_fields = schema_loader.get_document_fields("invoice")

        # Create processor instance
        processor = processor_class(
            field_list=default_fields,
            model_path=config['MODEL_PATH'],
            device="cuda",
            debug=False,
            skip_model_loading=False
        )

        rprint("[green]✅ Model loaded successfully![/green]")
        return processor

    def detect_document_type(self, image_path: Path, processor: Any = None, schema_loader: Any = None) -> tuple[str, dict]:
        """
        YAML-first document type detection using unified schema configuration.

        Uses the existing document_type_detection prompts from unified_schema.yaml
        to properly classify documents with model-specific optimized prompts.

        Returns:
            Tuple of (document_type, classification_prompt_info)
        """
        if processor is None or schema_loader is None:
            # Fallback to filename-based detection only if no processor/schema available
            image_name = image_path.name.lower()
            fallback_prompt_info = {
                "prompt": "Filename-based detection (no AI classification)",
                "source": "filename_pattern_matching",
                "field_count": 0,
                "template_type": "fallback_detection",
                "prompt_file": "N/A",
                "prompt_key": "filename_fallback"
            }

            if "statement" in image_name or "bank" in image_name:
                return "bank_statement", fallback_prompt_info
            elif "receipt" in image_name:
                return "receipt", fallback_prompt_info
            else:
                return "invoice", fallback_prompt_info  # Default

        try:
            # Load YAML-first document type detection configuration
            detection_config = schema_loader.load_detection_prompts()
            internvl3_prompts = detection_config.get("detection_prompts", {}).get("internvl3", {})
            type_mappings = detection_config.get("type_mappings", {})

            if not internvl3_prompts:
                raise Exception("No InternVL3 detection prompts found in unified schema")

            # Use the YAML-configured InternVL3 detection prompt
            classification_prompt = internvl3_prompts.get("user_prompt", "")
            max_tokens = internvl3_prompts.get("max_tokens", 20)
            temperature = internvl3_prompts.get("temperature", 0.0)

            # Create prompt info for display
            classification_prompt_info = {
                "prompt": classification_prompt,
                "source": "unified_schema_yaml",
                "field_count": 0,  # Classification doesn't extract fields
                "template_type": "document_type_detection",
                "prompt_file": "config/unified_schema.yaml",
                "prompt_key": "document_type_detection.prompts.internvl3",
                "max_tokens": max_tokens,
                "temperature": temperature
            }

            # Use the processor to classify with YAML prompt
            response = processor._generate_response(
                str(image_path),
                classification_prompt
            )

            if response:
                # Clean and normalize the response
                doc_type = response.strip().lower()

                # Apply YAML-configured type mappings
                if doc_type in type_mappings:
                    doc_type = type_mappings[doc_type]

                # Validate against supported types
                supported_types = detection_config.get("supported_types", ["invoice", "receipt", "bank_statement"])
                if doc_type in supported_types:
                    return doc_type, classification_prompt_info
                else:
                    # Try partial matching for common variations
                    if "receipt" in doc_type:
                        return "receipt", classification_prompt_info
                    elif "statement" in doc_type or "bank" in doc_type:
                        return "bank_statement", classification_prompt_info
                    elif "invoice" in doc_type:
                        return "invoice", classification_prompt_info
                    else:
                        # Final fallback
                        return "invoice", classification_prompt_info  # Default
            else:
                raise Exception("No response from processor")

        except Exception as e:
            rprint(f"[yellow]⚠️ YAML document type detection failed: {e}[/yellow]")
            rprint("[yellow]   Falling back to filename-based detection[/yellow]")

            # Fallback to filename-based detection
            image_name = image_path.name.lower()
            error_prompt_info = {
                "prompt": f"Filename-based detection (YAML failed: {str(e)})",
                "source": "filename_pattern_matching_fallback",
                "field_count": 0,
                "template_type": "error_fallback_detection",
                "prompt_file": "N/A",
                "prompt_key": "error_fallback"
            }

            if "statement" in image_name or "bank" in image_name:
                return "bank_statement", error_prompt_info
            elif "receipt" in image_name:
                return "receipt", error_prompt_info
            else:
                return "invoice", error_prompt_info  # Default

    def process_single_image_enhanced(
        self,
        image_path: Path,
        processor: Any,
        schema_loader: Any,
        ground_truth_data: Dict,
        image_index: int,
        total_images: int,
        show_enhanced_display: bool = True,
        prompt_config: Dict = None
    ) -> Dict[str, Any]:
        """Process a single image with enhanced display and evaluation."""

        rprint(f"[bold blue]📄 Processing [{image_index}/{total_images}]: {image_path.name}[/bold blue]")

        # Display the image being processed (like Llama version)
        if show_enhanced_display:
            rprint("[bold blue]📸 DISPLAYING IMAGE BEING PROCESSED:[/bold blue]")
            display_image(str(image_path), width=800)

        image_start = time.perf_counter()

        try:
            # Step A: Document Type Detection
            rprint("   🔍 Step A: Detecting document type...")
            doc_type, classification_prompt_info = self.detect_document_type(image_path, processor, schema_loader)

            # Get document-specific fields
            field_list = schema_loader.get_document_fields(doc_type)
            rprint(f"   📋 Document type: {doc_type} ({len(field_list)} fields)")

            # Step B: Update processor field list and document type (no model reloading)
            rprint("   🔄 Step B: Updating field list and document type (no model reload)...")
            processor.field_list = field_list
            processor.field_count = len(field_list)
            processor.detected_document_type = doc_type  # Store detected document type for YAML prompts

            # Step C: Process image with shared model
            rprint("   ⚡ Step C: Processing image with shared model...")
            result = processor.process_single_image(str(image_path), prompt_config)

            # Step D: Enhanced metadata and evaluation
            processing_time = time.perf_counter() - image_start
            image_name = image_path.name

            # Create enhanced result with metadata
            enhanced_result = {
                "image_name": image_name,
                "image_file": image_name,
                "image_path": str(image_path),
                "document_type": doc_type,
                "processing_time": processing_time,
                "prompt_used": f"internvl3_{doc_type.lower()}",
                "timestamp": datetime.now().isoformat(),
                "classification_prompt_info": classification_prompt_info,
            }

            # Add original result data
            enhanced_result.update(result)

            # Ground truth evaluation
            ground_truth = ground_truth_data.get(image_name, {})
            if ground_truth:
                rprint("   📊 Step D: Evaluating against ground truth...")

                extracted_data = result.get("extracted_data", {})
                evaluation = self.document_evaluator.evaluate_extraction(
                    extracted_data, ground_truth, doc_type
                )

                # Format evaluation for BatchAnalytics compatibility
                if evaluation and "overall_metrics" in evaluation:
                    evaluation["overall_accuracy"] = evaluation["overall_metrics"].get("overall_accuracy", 0)
                    evaluation["fields_extracted"] = evaluation["overall_metrics"].get("total_fields_evaluated", 0)
                    evaluation["fields_matched"] = evaluation["overall_metrics"].get("fields_correct", 0)
                    evaluation["total_fields"] = evaluation["overall_metrics"].get("total_fields_evaluated", 0)

                enhanced_result["evaluation"] = evaluation

                # Display evaluation summary
                accuracy = evaluation.get("overall_accuracy", 0) * 100 if evaluation else 0
                rprint(f"   🎯 Accuracy: {accuracy:.1f}%")
            else:
                enhanced_result["evaluation"] = {
                    "error": f"No ground truth for {image_name}",
                    "overall_accuracy": 0,
                }
                rprint("   ⚠️ No ground truth available for evaluation")

            # Count found fields
            found_fields = len([v for v in result.get("extracted_data", {}).values() if v != "NOT_FOUND"])
            enhanced_result["found_fields"] = found_fields

            rprint(f"   [green]✅ Completed: {found_fields}/{len(field_list)} fields found in {processing_time:.2f}s[/green]")

            # Enhanced display features (like Llama version)
            if show_enhanced_display:
                rprint("[bold blue]📋 ENHANCED DETAILED DISPLAY (No Truncation):[/bold blue]")

                # Display document classification prompt first
                from common.internvl3_batch_display import display_prompt_info
                rprint("[bold cyan]🔍 STEP 1: DOCUMENT TYPE CLASSIFICATION[/bold cyan]")
                display_prompt_info({"prompt_info": classification_prompt_info}, "classification", show_full=True)

                # Display extraction prompt second
                rprint("[bold cyan]⚡ STEP 2: DOCUMENT-AWARE FIELD EXTRACTION[/bold cyan]")
                display_prompt_info(result, doc_type, show_full=True)
                display_raw_and_cleaned(result, show_full=True)
                display_field_comparison(result, ground_truth, doc_type, show_full=True)
                rprint("=" * 120 + "\n")

            # Light cleanup
            clear_gpu_cache()

            return enhanced_result, processing_time, doc_type

        except Exception as e:
            rprint(f"   [red]❌ Error processing {image_path.name}: {e}[/red]")

            processing_time = time.perf_counter() - image_start
            error_result = {
                "image_name": image_path.name,
                "image_path": str(image_path),
                "document_type": "error",
                "error": str(e),
                "processing_time": processing_time,
            }

            # Light cleanup on error
            clear_gpu_cache()

            return error_result, processing_time, "error"

    def process_batch(
        self,
        image_files: List[Path],
        processor_class: Any,
        schema_loader: Any,
        config: Dict,
        show_enhanced_display: bool = True,
        prompt_config: Dict = None
    ) -> Tuple[List[Dict], List[float], Dict[str, int]]:
        """
        Process a batch of images with enhanced analytics and display.

        Args:
            image_files: List of image file paths
            processor_class: DocumentAwareInternVL3Processor class
            schema_loader: Schema loader instance
            config: Configuration dictionary
            show_enhanced_display: Whether to show enhanced display features

        Returns:
            Tuple of (batch_results, processing_times, document_types_found)
        """

        rprint("[bold green]🚀 STARTING ENHANCED BATCH PROCESSING[/bold green]")
        self.console.rule("[bold green]Enhanced Analytics & Evaluation[/bold green]")
        rprint(f"📊 Processing {len(image_files)} images with comprehensive analytics...")
        rprint("🧠 Model: InternVL3-8B with V100 optimizations")
        rprint("💾 Strategy: SHARED MODEL - Load once, reuse for all images (NO BatchProcessor)")
        rprint()

        # Load ground truth data
        ground_truth_data = self.load_ground_truth(config['GROUND_TRUTH'])

        # Initialize processor
        processor = self.initialize_processor(processor_class, config, schema_loader)

        # Initialize storage
        batch_results = []
        processing_times = []
        document_types_found = {}
        batch_start_time = time.perf_counter()

        # Main processing loop
        rprint("[cyan]🔧 STEP 2: Processing images with enhanced analytics and evaluation...[/cyan]")

        for i, image_path in enumerate(image_files, 1):
            result, proc_time, doc_type = self.process_single_image_enhanced(
                image_path, processor, schema_loader, ground_truth_data,
                i, len(image_files), show_enhanced_display, prompt_config
            )

            batch_results.append(result)
            processing_times.append(proc_time)

            if doc_type != "error":
                document_types_found[doc_type] = document_types_found.get(doc_type, 0) + 1

            rprint()  # Blank line between images

        # Final cleanup
        rprint("[cyan]🧹 STEP 3: Final cleanup...[/cyan]")
        del processor
        clear_gpu_cache()

        # Summary
        total_batch_time = time.perf_counter() - batch_start_time

        rprint("[bold green]🎉 ENHANCED BATCH PROCESSING COMPLETED![/bold green]")
        self.console.rule("[bold green]Processing Complete[/bold green]")
        rprint(f"📊 Processed: {len(batch_results)} images")
        rprint(f"⏱️ Total time: {total_batch_time:.2f}s")
        rprint(f"⚡ Average time per image: {total_batch_time/len(batch_results):.2f}s")
        rprint(f"📈 Throughput: {len(batch_results)/total_batch_time*60:.1f} images/minute")
        rprint("[cyan]💡 OPTIMIZATION: Model loaded once and reused for all images![/cyan]")
        rprint("[green]✅ Enhanced results ready for comprehensive analytics and reporting![/green]")
        rprint()
        rprint("[bold green]🎉 FEATURE PARITY ACHIEVED:[/bold green]")
        rprint("   ✅ Image display for each processed document (like Llama)")
        rprint("   ✅ Full prompts with syntax highlighting (no truncation)")
        rprint("   ✅ Complete raw responses (no truncation)")
        rprint("   ✅ 120-character wide detailed field comparison tables")
        rprint("   ✅ Enhanced field comparison with partial matching")
        rprint("   ✅ Comprehensive evaluation metrics and summaries")
        rprint("   ✅ Professional analytics, visualizations, and reporting")

        return batch_results, processing_times, document_types_found


def run_enhanced_batch_processing(
    image_files: List[Path],
    processor_class: Any,
    schema_loader: Any,
    config: Dict,
    show_enhanced_display: bool = True,
    prompt_config: Dict = None
) -> Tuple[List[Dict], List[float], Dict[str, int]]:
    """
    Convenience function to run enhanced batch processing.

    Args:
        image_files: List of image file paths
        processor_class: DocumentAwareInternVL3Processor class
        schema_loader: Schema loader instance
        config: Configuration dictionary
        show_enhanced_display: Whether to show enhanced display features

    Returns:
        Tuple of (batch_results, processing_times, document_types_found)
    """
    processor = EnhancedBatchProcessor()
    return processor.process_batch(
        image_files, processor_class, schema_loader, config, show_enhanced_display, prompt_config
    )