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

from .document_detector import detect_document_type
from .document_type_metrics import DocumentTypeEvaluator
from .evaluation_metrics import load_ground_truth
from .prompt_loader import load_document_prompt


class BatchDocumentProcessor:
    """Handles batch processing of documents with extraction and evaluation."""
    
    def __init__(
        self,
        extractor,
        evaluator,
        prompt_config: Dict,
        console: Optional[Console] = None
    ):
        """
        Initialize batch processor.
        
        Args:
            extractor: LlamaVisionTableExtractor instance
            evaluator: GroundTruthEvaluator instance (legacy - will be replaced with DocumentTypeEvaluator)
            prompt_config: Dictionary with prompt file paths and keys
            console: Rich console for output
        """
        self.extractor = extractor
        self.evaluator = evaluator  # Keep for backward compatibility but will use DocumentTypeEvaluator
        self.prompt_config = prompt_config
        self.console = console or Console()
        
        # Use the working evaluation approach from document-aware system
        self.document_evaluator = DocumentTypeEvaluator()
        self.ground_truth_data = None
        
    def process_batch(
        self,
        image_paths: List[str],
        verbose: bool = True,
        progress_interval: int = 5
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
        batch_results = []
        processing_times = []
        document_types_found = {}
        
        # Load ground truth data once for the batch (working approach from document-aware system)
        if hasattr(self.evaluator, 'ground_truth_csv'):
            try:
                self.ground_truth_data = load_ground_truth(self.evaluator.ground_truth_csv)
                if verbose:
                    rprint(f"[green]✅ Loaded ground truth for {len(self.ground_truth_data)} images[/green]")
            except Exception as e:
                if verbose:
                    rprint(f"[red]❌ Error loading ground truth: {e}[/red]")
                self.ground_truth_data = {}
        
        if verbose:
            rprint("\n[bold blue]🚀 Starting Batch Processing[/bold blue]")
            self.console.rule("[bold green]Batch Extraction[/bold green]")
        
        # Process each image
        iterator = track(image_paths, description="Processing images...") if verbose else image_paths
        
        for idx, image_path in enumerate(iterator, 1):
            image_name = Path(image_path).name
            
            try:
                # Record start time
                start_time = time.time()
                
                # Step 1: Detect document type
                document_type = detect_document_type(
                    image_path=image_path,
                    extractor=self.extractor,
                    detection_prompt_file=self.prompt_config['detection_file'],
                    prompt_files=self.prompt_config['extraction_files'],
                    detection_prompt_key=self.prompt_config['detection_key']
                )
                
                # Track document types
                document_types_found[document_type] = document_types_found.get(document_type, 0) + 1
                
                # Step 2: Load document-specific prompt
                extraction_prompt, prompt_name, _ = load_document_prompt(
                    prompt_files=self.prompt_config['extraction_files'],
                    prompt_keys=self.prompt_config['extraction_keys'],
                    document_type=document_type
                )
                
                # Step 3: Extract fields using document-aware processor
                # Use the working DocumentAwareLlamaProcessor approach instead of bank-specific extractor
                from common.unified_schema import DocumentTypeFieldSchema
                from models.document_aware_llama_processor import (
                    DocumentAwareLlamaProcessor,
                )
                
                # Get document-specific fields for this document type
                schema_loader = DocumentTypeFieldSchema()
                field_list = schema_loader.get_field_list_for_document_type(document_type)
                
                # Create document-aware processor with the model/processor from extractor
                doc_processor = DocumentAwareLlamaProcessor(
                    field_list=field_list,
                    skip_model_loading=True,  # Reuse existing model
                    debug=verbose
                )
                # Inject the existing model and processor
                doc_processor.model = self.extractor.model
                doc_processor.processor = self.extractor.processor
                
                # Use the working extraction method
                extraction_result = doc_processor.process_single_image(image_path)
                
                # Step 4: Evaluate against ground truth using working DocumentTypeEvaluator approach
                image_name = Path(image_path).name
                ground_truth = self.ground_truth_data.get(image_name, {}) if self.ground_truth_data else {}
                
                if ground_truth:
                    # Extract data using working DocumentAwareLlamaProcessor format
                    # DocumentAwareLlamaProcessor returns extracted_data at top level
                    extracted_data = extraction_result.get("extracted_data", {})
                    
                    if verbose:
                        rprint(f"[dim]DEBUG: extraction_result keys: {list(extraction_result.keys())}[/dim]")
                        found_fields = [k for k, v in extracted_data.items() if v != "NOT_FOUND"]
                        rprint(f"[dim]DEBUG: Found {len(found_fields)} fields from DocumentAwareLlamaProcessor[/dim]")
                    
                    # Use the working DocumentTypeEvaluator approach that succeeds
                    evaluation = self.document_evaluator.evaluate_extraction(
                        extracted_data, ground_truth, document_type
                    )
                    
                    # Add detailed debug output like document-aware system
                    if verbose and evaluation and "field_scores" in evaluation:
                        self._display_detailed_field_comparison(
                            image_name, extracted_data, ground_truth, evaluation, document_type
                        )
                else:
                    evaluation = {"error": f"No ground truth for {image_name}", "overall_accuracy": 0}
                
                # Record processing time
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                # Store results
                result = {
                    'image_name': image_name,
                    'image_path': image_path,
                    'document_type': document_type,
                    'extraction_result': extraction_result,
                    'evaluation': evaluation,
                    'processing_time': processing_time,
                    'prompt_used': prompt_name,
                    'timestamp': datetime.now().isoformat()
                }
                batch_results.append(result)
                
                # Progress update
                if verbose and (idx % progress_interval == 0 or idx == len(image_paths)):
                    accuracy = evaluation.get('overall_accuracy', 0) * 100 if evaluation else 0
                    rprint(
                        f"  [{idx}/{len(image_paths)}] {image_name}: {document_type} - "
                        f"Accuracy: {accuracy:.1f}% - Time: {processing_time:.2f}s"
                    )
                    
            except Exception as e:
                if verbose:
                    rprint(f"[red]❌ Error processing {image_name}: {e}[/red]")
                
                # Store error result
                batch_results.append({
                    'image_name': image_name,
                    'image_path': image_path,
                    'error': str(e),
                    'processing_time': time.time() - start_time if 'start_time' in locals() else 0
                })
        
        if verbose:
            self.console.rule("[bold green]Batch Processing Complete[/bold green]")
            
        return batch_results, processing_times, document_types_found
    
    def _display_detailed_field_comparison(
        self, image_name: str, extracted_data: dict, ground_truth: dict, 
        evaluation: dict, document_type: str
    ):
        """Display detailed field-by-field comparison like in document-aware system."""
        
        rprint(f"\n{'='*120}")
        rprint("📋 STEP 4: Extracted Data Results with Ground Truth Comparison")  
        rprint("="*120)
        
        # Display extracted data first
        rprint("\n🔍 EXTRACTED DATA:")
        for field, value in extracted_data.items():
            if value != "NOT_FOUND":
                rprint(f"✅ {field}: {value}")
            else:
                rprint(f"❌ {field}: {value}")
        
        # Ground truth comparison table
        rprint(f"\n📊 Ground truth loaded for {image_name}")
        rprint("-"*120)
        
        field_scores = evaluation.get("field_scores", {})
        
        # Table header with consistent spacing like document-aware system
        rprint(f"{'STATUS':<8} {'FIELD':<25} {'EXTRACTED':<40} {'GROUND TRUTH':<40}")
        rprint("="*120)
        
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
            extracted_display = str(extracted_val)[:38] + ("..." if len(str(extracted_val)) > 38 else "")
            ground_display = str(ground_val)[:38] + ("..." if len(str(ground_val)) > 38 else "")
            
            rprint(f"{status:<8} {field:<25} {extracted_display:<40} {ground_display:<40}")
        
        # Summary section (same format as document-aware system)
        overall_accuracy = evaluation.get("overall_metrics", {}).get("overall_accuracy", 0)
        
        rprint("\n📊 EXTRACTION SUMMARY:")
        rprint(f"✅ Fields Found: {fields_found}/{total_fields} ({fields_found/total_fields*100:.1f}%)")
        rprint(f"🎯 Exact Matches: {exact_matches}/{total_fields} ({exact_matches/total_fields*100:.1f}%)")  
        rprint(f"📈 Extraction Success Rate: {overall_accuracy*100:.1f}%")
        rprint(f"⏱️ Accuracy (matches/total): {overall_accuracy*100:.1f}%")
        rprint(f"🤖 Document Type: {document_type}")
        rprint("🔧 Model: Llama-3.2-11B-Vision-Instruct")
        
        # Additional metrics (same as document-aware system)
        meets_threshold = evaluation.get("overall_metrics", {}).get("meets_threshold", False)
        threshold = evaluation.get("overall_metrics", {}).get("document_type_threshold", 0.8)
        rprint("\n≈ = Partial match")  
        rprint("✗ = No match")
        rprint(f"Note: Meets accuracy threshold ({threshold*100:.0f}%): {'✅ Yes' if meets_threshold else '❌ No'}")
        rprint("="*120)