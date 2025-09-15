"""
InternVL3 Enhanced Batch Processor

A modular batch processing class extracted from the massive notebook cell.
Provides clean interface for InternVL3 batch processing with all enhanced features:
- Early model loading with fallback handling
- Dynamic field processing based on document type
- Enhanced evaluation with fuzzy matching and currency normalization
- Progress bar integration with Rich
- Advanced progress monitoring and memory reporting
- GPU memory fragmentation detection and cleanup
- Performance profiling with detailed timing breakdowns
- Verbose vs minimal output modes
- Error handling and graceful fallbacks
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rich import print as rprint
from rich.console import Console
from rich.progress import track

# Import GPU optimization utilities for advanced monitoring
from .gpu_optimization import (
    clear_gpu_cache,
    comprehensive_memory_cleanup,
    detect_memory_fragmentation,
    get_available_gpu_memory,
    handle_memory_fragmentation,
)


class InternVL3EnhancedBatchProcessor:
    """Enhanced batch processor for InternVL3 with all sophisticated features."""

    def __init__(
        self,
        config: Dict[str, Any],
        prompt_config: Dict[str, Any],
        model_components: Dict[str, Any],
        schema_loader: Any,
        evaluator: Any,
        detector: Any,
        field_processor: Any
    ):
        """
        Initialize the enhanced batch processor.

        Args:
            config: Configuration dictionary with model settings, verbosity, etc.
            prompt_config: Prompt configuration with detection and extraction files
            model_components: Dictionary with pre-loaded model, tokenizer, and metadata
            schema_loader: DocumentTypeFieldSchema instance for field management
            evaluator: StandaloneEvaluator for enhanced accuracy scoring
            detector: StandaloneDocumentDetector for YAML-first detection
            field_processor: DynamicFieldProcessor for document-specific fields
        """
        self.config = config
        self.prompt_config = prompt_config
        self.model_components = model_components
        self.schema_loader = schema_loader
        self.evaluator = evaluator
        self.detector = detector
        self.field_processor = field_processor

        # Initialize processor as None - will be created during processing
        self.processor = None

        # Detect model variant for prompt naming
        model_variant = "8B" if "8B" in config['MODEL_PATH'] else "2B"
        self.model_suffix = f"internvl3_{model_variant.lower()}"

        # Initialize console for advanced reporting
        self.console = Console()

        # Performance tracking
        self.performance_metrics = {
            'total_time': 0,
            'detection_time': 0,
            'processing_time': 0,
            'evaluation_time': 0,
            'memory_cleanup_time': 0
        }

    def _get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics in GB."""
        try:
            allocated, reserved, fragmentation = detect_memory_fragmentation()
            available = get_available_gpu_memory()
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'fragmentation_gb': fragmentation,
                'available_gb': available
            }
        except Exception:
            return {
                'allocated_gb': 0.0,
                'reserved_gb': 0.0,
                'fragmentation_gb': 0.0,
                'available_gb': 0.0
            }

    def _display_memory_report(self, stage: str, verbose: bool = True) -> None:
        """Display detailed memory usage report."""
        if not verbose:
            return

        stats = self._get_memory_stats()
        rprint(f"[bold blue]🧠 Memory Report - {stage}:[/bold blue]")
        rprint(f"   📊 Allocated: {stats['allocated_gb']:.2f}GB")
        rprint(f"   📦 Reserved: {stats['reserved_gb']:.2f}GB")
        rprint(f"   💸 Available: {stats['available_gb']:.2f}GB")

        if stats['fragmentation_gb'] > 0.5:
            rprint(f"   ⚠️ Fragmentation: {stats['fragmentation_gb']:.2f}GB [yellow](cleanup recommended)[/yellow]")
        else:
            rprint(f"   ✅ Fragmentation: {stats['fragmentation_gb']:.2f}GB [green](healthy)[/green]")

    def _monitor_performance_phase(self, phase_name: str, start_time: float, verbose: bool = True) -> None:
        """Monitor and record performance for a specific processing phase."""
        elapsed = time.time() - start_time
        self.performance_metrics[f'{phase_name}_time'] += elapsed

        if verbose:
            rprint(f"   ⏱️ {phase_name.title()} phase: {elapsed:.3f}s")

    def _display_performance_summary(self, total_images: int, verbose: bool = True) -> None:
        """Display comprehensive performance summary."""
        if not verbose:
            return

        total_time = self.performance_metrics['total_time']
        rprint("\n[bold blue]📈 Performance Summary:[/bold blue]")
        rprint(f"   🎯 Total batch time: {total_time:.2f}s")
        rprint(f"   ⚡ Average per image: {total_time/total_images:.2f}s")
        rprint(f"   🚀 Throughput: {total_images/total_time*60:.1f} images/minute")

        # Phase breakdown
        rprint("\n[bold blue]⏱️ Phase Breakdown:[/bold blue]")
        for phase, time_spent in self.performance_metrics.items():
            if phase != 'total_time' and time_spent > 0:
                percentage = (time_spent / total_time) * 100
                rprint(f"   {phase.replace('_', ' ').title()}: {time_spent:.2f}s ({percentage:.1f}%)")

    def _perform_memory_maintenance(self, image_index: int, total_images: int, verbose: bool = True) -> None:
        """Perform intelligent memory maintenance based on progress and fragmentation."""
        stats = self._get_memory_stats()

        # Cleanup every 25% of progress or if fragmentation is high
        cleanup_threshold = max(1, total_images // 4)
        should_cleanup = (
            image_index % cleanup_threshold == 0 or
            stats['fragmentation_gb'] > 1.0 or
            stats['available_gb'] < 2.0
        )

        if should_cleanup:
            cleanup_start = time.time()
            if verbose:
                rprint(f"   🧹 Memory maintenance (image {image_index}/{total_images})...")

            # Use comprehensive cleanup with fragmentation handling
            handle_memory_fragmentation(threshold_gb=0.5, aggressive=True, verbose=False)
            clear_gpu_cache(verbose=False)

            cleanup_time = time.time() - cleanup_start
            self.performance_metrics['memory_cleanup_time'] += cleanup_time

            if verbose:
                new_stats = self._get_memory_stats()
                freed = stats['allocated_gb'] - new_stats['allocated_gb']
                rprint(f"   ✅ Cleanup complete: {freed:.2f}GB freed in {cleanup_time:.2f}s")

    def process_batch(
        self,
        images: List[str],
        verbose: bool = True,
        ground_truth: Optional[Dict[str, Dict]] = None
    ) -> Tuple[List[Dict], List[float], Dict[str, int]]:
        """
        Process a batch of images with enhanced InternVL3 features.

        Args:
            images: List of image file paths to process
            verbose: Whether to show detailed output (True) or minimal output (False)
            ground_truth: Optional ground truth data for evaluation

        Returns:
            Tuple of (batch_results, processing_times, document_types_found)
        """
        batch_results = []
        processing_times = []
        document_types_found = {}

        # Start timing the entire batch
        batch_start_time = time.time()

        rprint("[bold blue]🚀 Starting ENHANCED batch processing with advanced monitoring...[/bold blue]")
        rprint("[cyan]Features: YAML-first detection, dynamic fields, enhanced evaluation, memory monitoring, performance profiling[/cyan]")

        # Display initial memory state
        self._display_memory_report("Batch Start", verbose)

        # Display early loading status
        if self.model_components.get('early_loaded', False):
            rprint("[green]✅ Using early-loaded InternVL3 model with V100-compatible optimizations[/green]")
            if verbose:
                optimization_features = self.model_components.get('optimization_features', [])
                rprint(f"[cyan]🔧 Active optimizations: {', '.join(optimization_features)}[/cyan]")
        else:
            rprint("[yellow]⚠️ Early loading not available, will initialize during processing[/yellow]")

        # Initialize processor with early-loaded model components
        init_start = time.time()
        self._initialize_processor(verbose)
        self._monitor_performance_phase("initialization", init_start, verbose)

        # Process images with progress bar
        progress_description = f"Processing {len(images)} images"
        for i, image_path in enumerate(track(images, description=progress_description), 1):
            image_name = Path(image_path).name

            # Only show detailed processing info if verbose
            if verbose:
                rprint(f"\n[cyan]📄 Processing {i}/{len(images)}: {image_name}[/cyan]")

            start_time = time.time()

            try:
                # Perform memory maintenance if needed
                self._perform_memory_maintenance(i, len(images), verbose)

                # Process single image through the enhanced pipeline
                result = self._process_single_image(
                    image_path, image_name, verbose, ground_truth
                )

                processing_time = time.time() - start_time
                result['processing_time'] = processing_time

                batch_results.append(result)
                processing_times.append(processing_time)

                # Track document types
                doc_type = result.get('document_type', 'unknown')
                document_types_found[doc_type] = document_types_found.get(doc_type, 0) + 1

                # Display results based on verbosity level
                self._display_result(result, verbose, i, len(images))

                # Show memory status for every 5th image in verbose mode
                if verbose and i % 5 == 0:
                    self._display_memory_report(f"Progress {i}/{len(images)}", verbose)

            except Exception as e:
                processing_time = time.time() - start_time

                # Handle errors gracefully
                error_result = self._create_error_result(image_path, image_name, str(e), processing_time)
                batch_results.append(error_result)
                processing_times.append(processing_time)

                # Display error based on verbosity
                self._display_error(e, image_name, verbose)

        # Record total batch time
        self.performance_metrics['total_time'] = time.time() - batch_start_time

        # Final memory cleanup and reporting
        final_cleanup_start = time.time()
        comprehensive_memory_cleanup(verbose=False)
        self._monitor_performance_phase("final_cleanup", final_cleanup_start, verbose)

        # Display final memory state
        self._display_memory_report("Batch Complete", verbose)

        # Display enhanced summary with performance metrics
        self._display_summary(batch_results, processing_times, document_types_found, verbose)

        # Display detailed performance analysis
        self._display_performance_summary(len(images), verbose)

        return batch_results, processing_times, document_types_found

    def _initialize_processor(self, verbose: bool) -> None:
        """Initialize the InternVL3 processor with early-loaded model support and fallbacks."""
        all_fields = self.schema_loader.get_all_fields()

        if self.model_components.get('early_loaded', False):
            # Try to use pre-loaded model components
            if verbose:
                rprint("[yellow]Initializing processor with early-loaded InternVL3 model...[/yellow]")

            try:
                from models.document_aware_internvl3_processor import (
                    DocumentAwareInternVL3Processor,
                )

                self.processor = DocumentAwareInternVL3Processor(
                    field_list=all_fields,
                    model_path=self.config['MODEL_PATH'],
                    device="cuda" if self.config['DEVICE_MAP'] != 'cpu' else 'cpu',
                    debug=False,
                    pre_loaded_model=self.model_components.get('model'),
                    pre_loaded_tokenizer=self.model_components.get('tokenizer')
                )
                rprint("[green]✅ Processor initialized with early-loaded model[/green]")
                return

            except Exception as e:
                if verbose:
                    rprint(f"[yellow]⚠️ Failed to use early-loaded model, falling back: {e}[/yellow]")

                # Fallback to original initialization
                from models.document_aware_internvl3_processor import (
                    DocumentAwareInternVL3Processor,
                )

                self.processor = DocumentAwareInternVL3Processor(
                    field_list=all_fields,
                    model_path=self.config['MODEL_PATH'],
                    device="cuda" if self.config['DEVICE_MAP'] != 'cpu' else 'cpu',
                    debug=False
                )
                rprint("[green]✅ Processor initialized with fallback method[/green]")
        else:
            # Original initialization path
            if verbose:
                rprint("[yellow]Initializing InternVL3 processor (lazy loading)...[/yellow]")

            from models.document_aware_internvl3_processor import (
                DocumentAwareInternVL3Processor,
            )

            self.processor = DocumentAwareInternVL3Processor(
                field_list=all_fields,
                model_path=self.config['MODEL_PATH'],
                device="cuda" if self.config['DEVICE_MAP'] != 'cpu' else 'cpu',
                debug=False
            )
            rprint("[green]✅ Processor initialized successfully[/green]")

    def _process_single_image(
        self,
        image_path: str,
        image_name: str,
        verbose: bool,
        ground_truth: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, Any]:
        """Process a single image through the enhanced pipeline."""

        # ENHANCED STEP 1: YAML-first document type detection
        if verbose:
            rprint("   🔍 Step 1: Enhanced document type detection...")

        detection_start = time.time()
        detected_type, detection_info = self.detector.detect_document_type(
            Path(image_path), self.processor, self.prompt_config
        )
        self._monitor_performance_phase("detection", detection_start, verbose)

        if verbose:
            confidence = detection_info.get('detection_confidence', 'unknown')
            rprint(f"   📋 Detected: {detected_type} (confidence: {confidence})")

            # Show detection prompt if requested
            if self.config.get('SHOW_PROMPTS', False):
                self._display_detection_prompt(detection_info)

        # ENHANCED STEP 2: Dynamic field processing based on detected type
        if verbose:
            rprint("   🔄 Step 2: Updating processor with document-specific fields...")

        update_success = self.field_processor.update_processor_fields(
            self.processor, detected_type, verbose=False  # Never verbose for this step
        )

        if not update_success and verbose:
            rprint("   [yellow]⚠️ Field update failed, using default fields[/yellow]")

        # ENHANCED STEP 3: Extract fields using detected document type and updated fields
        if verbose:
            rprint("   ⚡ Step 3: Enhanced field extraction...")

        processing_start = time.time()
        result = self.processor.process_single_image(
            str(image_path),
            prompt_config=self.prompt_config
        )
        self._monitor_performance_phase("processing", processing_start, verbose)

        # Get document type and extracted data
        doc_type = result.get('document_type', detected_type).lower()
        extracted_data = result.get('extracted_data', {})

        # Create prompt identifier for analytics
        prompt_used = f"{self.model_suffix}_{doc_type}"

        # ENHANCED STEP 4: Advanced evaluation with fuzzy matching and currency normalization
        evaluation = {}
        if ground_truth and image_name in ground_truth:
            if verbose:
                rprint("   📊 Step 4: Enhanced evaluation with fuzzy matching...")

            evaluation_start = time.time()
            evaluation = self._evaluate_extraction(extracted_data, ground_truth[image_name], detected_type)
            self._monitor_performance_phase("evaluation", evaluation_start, verbose)

        # Create comprehensive result
        return {
            'image_path': str(image_path),
            'image_name': image_name,
            'document_type': doc_type,
            'prompt_used': prompt_used,
            'extraction_result': {
                'extracted_data': extracted_data,
                'document_type': doc_type
            },
            'evaluation': evaluation,
            'detection_info': detection_info,
            'enhanced_processing': True,
            'early_model_loading': self.model_components.get('early_loaded', False)
        }

    def _evaluate_extraction(
        self,
        extracted_data: Dict[str, Any],
        gt_data: Dict[str, Any],
        detected_type: str
    ) -> Dict[str, Any]:
        """Evaluate extraction using enhanced evaluator with field details."""

        # Use enhanced evaluator for fuzzy matching and currency normalization
        enhanced_evaluation = self.evaluator.evaluate_extraction(
            extracted_data, gt_data, detected_type
        )

        # Extract metrics for BatchAnalytics compatibility
        overall_metrics = enhanced_evaluation.get('overall_metrics', {})
        field_scores = enhanced_evaluation.get('field_scores', {})

        # Create detailed field comparison with extracted and ground truth values
        field_details = []
        for field, score_info in field_scores.items():
            accuracy = score_info.get('accuracy', 0)
            match_type = score_info.get('match_type', 'unknown')
            extracted_val = score_info.get('extracted_value', 'NOT_FOUND')
            ground_truth_val = score_info.get('ground_truth_value', 'NOT_FOUND')

            # Create status line
            if accuracy == 1.0:
                status_line = f"✅ {field}: MATCH ({match_type})"
            elif accuracy >= 0.8:
                status_line = f"🟡 {field}: FUZZY ({match_type})"
            elif accuracy > 0:
                status_line = f"🟠 {field}: PARTIAL ({match_type})"
            else:
                status_line = f"❌ {field}: MISS ({match_type})"

            # Add extracted and expected values
            value_line = f"   Extracted: '{extracted_val}' | Expected: '{ground_truth_val}'"

            field_details.append(status_line)
            field_details.append(value_line)

        return {
            'overall_accuracy': overall_metrics.get('overall_accuracy', 0),
            'fields_extracted': overall_metrics.get('total_fields_evaluated', 0),
            'fields_matched': overall_metrics.get('fields_correct', 0),
            'total_fields': overall_metrics.get('total_fields_evaluated', 0),
            'field_details': field_details,
            'enhanced_metrics': enhanced_evaluation
        }

    def _display_detection_prompt(self, detection_info: Dict[str, Any]) -> None:
        """Display the document detection prompt if available."""
        detection_prompt = detection_info.get('prompt', 'N/A')
        rprint("   📝 Document Detection Prompt:")
        rprint("   " + "━" * 100)

        if detection_prompt and detection_prompt != 'N/A':
            prompt_lines = detection_prompt.split('\n')
            for line in prompt_lines[:5]:  # Show first 5 lines only
                rprint(f"   {line}")
            if len(prompt_lines) > 5:
                rprint(f"   ... ({len(prompt_lines) - 5} more lines)")
        else:
            rprint("   [yellow]Detection prompt not captured[/yellow]")

        rprint("   " + "━" * 100)

    def _display_result(self, result: Dict[str, Any], verbose: bool, current: int, total: int) -> None:
        """Display processing result based on verbosity level."""
        evaluation = result.get('evaluation', {})
        doc_type = result.get('document_type', 'unknown')
        processing_time = result.get('processing_time', 0)
        image_name = result.get('image_name', 'unknown')

        if evaluation:
            acc_pct = evaluation.get('overall_accuracy', 0) * 100
            fields_found = evaluation.get('fields_extracted', 0)
            fields_matched = evaluation.get('fields_matched', 0)

            if verbose:
                # Detailed output in verbose mode
                enhanced_flag = "🚀" if evaluation.get('enhanced_metrics') else ""
                early_flag = "⚡" if result.get('early_model_loading', False) else ""
                rprint(f"   ✅ Success {enhanced_flag}{early_flag} - Type: {doc_type}, Found: {fields_found}, Matched: {fields_matched}, Accuracy: {acc_pct:.1f}%, Time: {processing_time:.2f}s")

                # Show enhanced field details for first 3 images
                field_details = evaluation.get('field_details', [])
                if field_details and current <= 3:
                    rprint("   📋 Enhanced field analysis with extracted values (first 3 fields):")
                    # Show first 3 fields (each field has 2 lines: status + values)
                    for detail_idx in range(0, min(6, len(field_details)), 2):
                        if detail_idx + 1 < len(field_details):
                            rprint(f"     {field_details[detail_idx]}")
                            rprint(f"     {field_details[detail_idx + 1]}")
                    remaining_fields = (len(field_details) // 2) - 3
                    if remaining_fields > 0:
                        rprint(f"     ... and {remaining_fields} more fields")
            else:
                # Minimal output when not verbose - single line per image
                status_icon = "✅" if acc_pct >= 70 else "⚠️" if acc_pct >= 50 else "❌"
                rprint(f"{status_icon} {image_name}: {doc_type}, Accuracy: {acc_pct:.1f}%, Time: {processing_time:.1f}s")
        else:
            # No evaluation available
            if verbose:
                early_flag = "⚡" if result.get('early_model_loading', False) else ""
                rprint(f"   ✅ Success {early_flag} - Type: {doc_type}, Time: {processing_time:.2f}s")
            else:
                rprint(f"✅ {image_name}: {doc_type}, Time: {processing_time:.1f}s")

    def _display_error(self, error: Exception, image_name: str, verbose: bool) -> None:
        """Display error message based on verbosity level."""
        if verbose:
            rprint(f"   ❌ Error: {str(error)}")
        else:
            rprint(f"❌ {image_name}: Error - {str(error)[:50]}...")

    def _create_error_result(self, image_path: str, image_name: str, error_msg: str, processing_time: float) -> Dict[str, Any]:
        """Create error result structure."""
        return {
            'image_path': str(image_path),
            'image_name': image_name,
            'document_type': 'error',
            'prompt_used': f"{self.model_suffix}_error",
            'error': error_msg,
            'processing_time': processing_time,
            'enhanced_processing': False,
            'early_model_loading': self.model_components.get('early_loaded', False)
        }

    def _display_summary(
        self,
        batch_results: List[Dict],
        processing_times: List[float],
        document_types_found: Dict[str, int],
        verbose: bool
    ) -> None:
        """Display enhanced summary with early loading info."""
        enhanced_count = len([r for r in batch_results if r.get('enhanced_processing', False)])
        early_loading_count = len([r for r in batch_results if r.get('early_model_loading', False)])

        rprint("\n[bold green]🚀 Enhanced processing complete![/bold green]")
        rprint(f"[green]✅ Processed: {len(batch_results)} images[/green]")

        if verbose:
            rprint(f"[cyan]🚀 Enhanced: {enhanced_count}/{len(batch_results)} images[/cyan]")
            rprint(f"[cyan]⚡ Early loading: {early_loading_count}/{len(batch_results)} images[/cyan]")

        if processing_times:
            rprint(f"[cyan]⏱️ Average time: {np.mean(processing_times):.2f}s[/cyan]")
            if verbose:
                rprint(f"[cyan]⏱️ Total time: {sum(processing_times):.2f}s[/cyan]")

        rprint(f"[cyan]📋 Document types found: {document_types_found}[/cyan]")

        if verbose:
            rprint("[cyan]🎯 Improvements: YAML detection + Dynamic fields + Enhanced evaluation + Early model loading + Value display[/cyan]")