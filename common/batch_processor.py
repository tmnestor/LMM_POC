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
            evaluator: GroundTruthEvaluator instance
            prompt_config: Dictionary with prompt file paths and keys
            console: Rich console for output
        """
        self.extractor = extractor
        self.evaluator = evaluator
        self.prompt_config = prompt_config
        self.console = console or Console()
        
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
                    detection_prompt_key=self.prompt_config['detection_key'],
                    verbose=False
                )
                
                # Track document types
                document_types_found[document_type] = document_types_found.get(document_type, 0) + 1
                
                # Step 2: Load document-specific prompt
                extraction_prompt, prompt_name, _ = load_document_prompt(
                    prompt_files=self.prompt_config['extraction_files'],
                    prompt_keys=self.prompt_config['extraction_keys'],
                    document_type=document_type,
                    verbose=False
                )
                
                # Step 3: Extract fields
                extraction_result = self.extractor.test_extraction(
                    image_path,
                    extraction_prompt,
                    verbose=False
                )
                
                # Step 4: Evaluate against ground truth
                evaluation = self.evaluator.evaluate_extraction(
                    test_result=extraction_result,
                    document_type=document_type,
                    image_path=image_path,
                    verbose=False
                )
                
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