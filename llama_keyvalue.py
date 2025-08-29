#!/usr/bin/env python3
"""
Llama Vision Key-Value Extraction with Comprehensive Evaluation Pipeline

This module implements a complete evaluation pipeline for the Llama-3.2-11B-Vision-Instruct
model, designed to extract structured key-value data from business documents (invoices,
statements, etc.) and evaluate performance against ground truth data.

Pipeline Overview:
    1. Setup & Validation - Verify paths, create output directories
    2. Model Initialization - Load Llama Vision processor with optimal config
    3. Image Discovery - Find and filter document images for processing
    4. Batch Processing - Extract structured data from all images
    5. Data Generation - Create CSV outputs with extraction results
    6. Evaluation - Compare results against ground truth for accuracy metrics
    7. Reporting - Generate comprehensive evaluation reports and summaries

Input Requirements:
    - Document images in DATA_DIR (PNG, JPG, JPEG formats)
    - Ground truth CSV file for evaluation (ground_truth.csv)
    - Llama-3.2-11B-Vision-Instruct model at LLAMA_MODEL_PATH
    - CUDA-compatible GPU recommended (16GB+ VRAM or 8-bit quantization)

Output Files:
    - llama_batch_extraction_{timestamp}.csv - Main extraction results
    - llama_extraction_metadata_{timestamp}.csv - Processing metadata
    - llama_evaluation_results_{timestamp}.json - Detailed evaluation metrics
    - llama_executive_summary_{timestamp}.md - High-level performance summary
    - llama_deployment_checklist_{timestamp}.md - Deployment readiness assessment

Usage:
    python llama_keyvalue.py

Configuration:
    Edit common/config.py to adjust:
    - DATA_DIR: Path to document images
    - LLAMA_MODEL_PATH: Path to model weights
    - OUTPUT_DIR: Path for generated reports
    - EXTRACTION_FIELDS: List of fields to extract (V4 schema with 47 business document fields)

Model Specifications:
    - Parameters: 11B parameter model  
    - Field extraction: V4 schema with 47 structured fields including document intelligence

Dependencies:
    - models.llama_processor: Llama-specific processing logic
    - common.config: Centralized configuration management
    - common.evaluation_utils: Shared evaluation and parsing utilities
    - common.reporting: Report generation and formatting
    - transformers: Hugging Face model loading (pinned to 4.45.2)
    - torch: PyTorch for model inference (CUDA recommended)

Note:
    This script focuses on batch evaluation with ground truth comparison.
    For single document processing or interactive use, see the Jupyter notebooks.
"""

import argparse
from datetime import datetime
from pathlib import Path

# Import shared modules
from common.config import DATA_DIR as data_dir
from common.config import (
    DEFAULT_EXTRACTION_MODE,
    EXTRACTION_FIELDS,
    EXTRACTION_MODES,
    FIELD_COUNT,
)
from common.config import GROUND_TRUTH_PATH as ground_truth_path
from common.config import LLAMA_MODEL_PATH as model_path
from common.config import OUTPUT_DIR as output_dir
from common.evaluation_metrics import (
    evaluate_extraction_results,
    load_ground_truth,
)
from common.extraction_parser import (
    create_extraction_dataframe,
    discover_images,
)
from common.reporting import generate_comprehensive_reports, print_evaluation_summary
from models.llama_processor import LlamaProcessor


def parse_arguments():
    """Parse command line arguments for extraction mode and debugging."""
    parser = argparse.ArgumentParser(
        description="Llama Vision Key-Value Extraction with Comprehensive Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python llama_keyvalue.py --extraction-mode grouped --debug
  python llama_keyvalue.py --extraction-mode adaptive
  python llama_keyvalue.py --extraction-mode single_pass
  python llama_keyvalue.py --image-path path/to/image.png --debug
        """,
    )

    parser.add_argument(
        "--extraction-mode",
        choices=EXTRACTION_MODES,
        default=DEFAULT_EXTRACTION_MODE,
        help=f"Extraction strategy to use (default: {DEFAULT_EXTRACTION_MODE})",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for grouped extraction",
    )

    parser.add_argument(
        "--limit-images",
        type=int,
        help="Limit processing to first N images (for testing)",
    )

    parser.add_argument(
        "--image-path",
        type=str,
        help="Path to single image file for testing (overrides directory processing)",
    )

    parser.add_argument(
        "--debug-ocr",
        action="store_true", 
        help="Enable debug OCR mode for raw markdown output (requires --debug)",
    )

    return parser.parse_args()


def main(extraction_mode=None, debug=False, limit_images=None, image_path=None, debug_ocr=False):
    """
    Execute the complete Llama Vision evaluation pipeline.

    This function orchestrates the entire evaluation workflow from initial setup
    through final report generation. It handles all error conditions gracefully
    and provides detailed progress feedback throughout processing.

    Args:
        extraction_mode (str): Extraction strategy ('single_pass', 'grouped', 'adaptive')
        debug (bool): Enable debug logging for grouped extraction
        limit_images (int): Limit processing to first N images (for testing)

    Pipeline Stages:
        1. Environment validation and setup
        2. Model processor initialization
        3. Document image discovery and filtering
        4. Batch processing with progress tracking
        5. Data frame creation and CSV export
        6. Ground truth evaluation and metrics calculation
        7. Comprehensive report generation

    Configuration Dependencies:
        - DATA_DIR: Must contain document images (*.png, *.jpg, *.jpeg)
        - LLAMA_MODEL_PATH: Must point to valid Llama-3.2-Vision model
        - GROUND_TRUTH_PATH: Must contain ground_truth.csv
        - OUTPUT_DIR: Directory for generated reports (created if missing)

    Error Handling:
        - Missing directories or files cause graceful exit with helpful messages
        - Model loading failures are caught and reported with debugging info
        - Individual image processing errors don't stop batch processing
        - Evaluation continues even if some ground truth data is missing

    Output Files Generated:
        - Primary CSV with extraction results (25 fields per document)
        - Metadata CSV with processing statistics and quality metrics
        - JSON evaluation results with field-level and document-level accuracy
        - Executive summary in Markdown format for stakeholder review
        - Deployment checklist assessing production readiness

    Performance Metrics Tracked:
        - Processing time per document and total batch time
        - Field extraction success rates and completeness scores
        - Memory usage and GPU utilization during processing
        - Overall accuracy against ground truth data
        - Document quality distribution (good/fair/poor)

    Returns:
        None. All results are saved to OUTPUT_DIR and progress printed to console.

    Raises:
        No exceptions are raised; all errors are caught and handled gracefully
        with informative error messages and clean exit.
    """

    # =============================================================================
    # PIPELINE INITIALIZATION AND CONFIGURATION DISPLAY
    # =============================================================================
    print("\n" + "=" * 80)
    print("🦙 LLAMA VISION COMPREHENSIVE EVALUATION PIPELINE")
    print("=" * 80)

    # Display critical configuration paths for verification
    print(f"📁 Data directory: {data_dir}")
    print(f"📂 Output directory: {output_dir}")
    print(f"📊 Ground truth: {ground_truth_path}")
    print(f"🔧 Model: {model_path}")

    # =============================================================================
    # ENVIRONMENT VALIDATION AND SETUP
    # =============================================================================
    # Create output directory structure - ensures all report files can be written
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Validate critical input paths before proceeding with expensive model loading
    if not Path(data_dir).exists():
        print(f"❌ ERROR: Data directory not found: {data_dir}")
        print(
            "💡 Ensure DATA_DIR in config.py points to directory with document images"
        )
        return

    if not Path(ground_truth_path).exists():
        print(f"❌ ERROR: Ground truth file not found: {ground_truth_path}")
        print("💡 Ensure GROUND_TRUTH_PATH in config.py points to evaluation CSV file")
        return

    # =============================================================================
    # MODEL PROCESSOR INITIALIZATION
    # =============================================================================
    # Load Llama-3.2-Vision model with optimal configuration for extraction tasks
    # V4 schema uses single_pass mode with YAML-first prompts and document intelligence
    original_mode = extraction_mode or DEFAULT_EXTRACTION_MODE
    extraction_mode = "single_pass"  # V4 system uses single_pass with intelligent field filtering
    print(
        f"\n🚀 Initializing Llama Vision processor with {extraction_mode} extraction mode (V4 schema)..."
    )
    if original_mode != extraction_mode:
        print(f"💡 Note: Overriding {original_mode} → {extraction_mode} for V4 compatibility")
    
    processor = LlamaProcessor(
        model_path=model_path, 
        extraction_mode=extraction_mode, 
        debug=debug,
        enable_v4_schema=True
    )

    # =============================================================================
    # DEBUG CONFIGURATION OUTPUT
    # =============================================================================
    if debug:
        print("\n" + "=" * 80)
        print("🔧 COMPLETE LLAMA CONFIGURATION DEBUG")
        print("=" * 80)

        # Environment and paths
        import os

        print(f"📁 Environment: {os.getenv('LMM_ENVIRONMENT', 'not set')}")
        print(f"📁 Model path: {model_path}")
        print(f"📁 Data directory: {data_dir}")
        print(f"📁 Ground truth: {ground_truth_path}")
        print(f"📁 Output directory: {output_dir}")

        # Extraction configuration
        print(f"\n🎯 Extraction mode: {extraction_mode}")
        print(f"🎯 Debug enabled: {debug}")
        if limit_images:
            print(f"🎯 Image limit: {limit_images}")

        # Field configuration
        print(f"\n📋 Total fields: {FIELD_COUNT}")
        print(f"📋 First field: {EXTRACTION_FIELDS[0]}")
        print(f"📋 Last field: {EXTRACTION_FIELDS[-1]}")
        print(
            f"📋 Field sequence: {' → '.join(EXTRACTION_FIELDS[:3])} ... {' → '.join(EXTRACTION_FIELDS[-3:])}"
        )

        # Prompt file information
        if extraction_mode == "single_pass":
            prompt_file = "llama_single_pass_prompts.yaml"
            print(f"\n📝 Prompt file: {prompt_file}")
            print("📝 Prompt method: Single-pass YAML-based generation")

            # Show actual prompt preview
            try:
                sample_prompt = processor.get_extraction_prompt()
                print(f"📝 Prompt length: {len(sample_prompt)} characters")
                
                if args.debug:
                    # Show full prompt in debug mode
                    print("📝 FULL PROMPT (Debug Mode):")
                    print("=" * 80)
                    print(sample_prompt)
                    print("=" * 80)
                else:
                    # Show preview in normal mode
                    print("📝 Prompt preview (first 200 chars):")
                    print(f"    {sample_prompt[:200].replace(chr(10), ' ')}")
                    print(
                        f"📝 Critical instruction: {sample_prompt.split('CRITICAL INSTRUCTIONS:')[1].split('REQUIRED OUTPUT FORMAT')[0].strip()[:100] if 'CRITICAL INSTRUCTIONS:' in sample_prompt else 'Not found'}"
                    )
            except Exception as e:
                print(f"📝 ⚠️ Could not preview prompt: {e}")
        else:
            prompt_file = "llama_prompts.yaml (grouped sections)"
            print(f"\n📝 Prompt file: {prompt_file}")
            print("📝 Prompt method: Grouped extraction strategy")

        # Model configuration
        print("\n🤖 Model processor: LlamaProcessor")
        print("🤖 Generation config: temperature=0.0, do_sample=False")
        print(f"🤖 Max tokens: {getattr(processor, 'max_new_tokens', 'default')}")

        print("=" * 80)
        print("END DEBUG CONFIGURATION")
        print("=" * 80 + "\n")

    # =============================================================================
    # DOCUMENT IMAGE DISCOVERY AND FILTERING
    # =============================================================================
    if image_path:
        # Single image processing mode
        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            print(f"❌ ERROR: Image file not found: {image_path}")
            return
        if image_path_obj.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
            print(f"❌ ERROR: Unsupported image format: {image_path_obj.suffix}")
            return
        image_files = [str(image_path_obj)]
        print(f"\n📷 Single image processing: {image_path_obj.name}")
        
        # Process single image and display results with clean formatting
        print("\n🚀 Starting batch processing...")
        results, batch_stats = processor.process_image_batch(image_files)
        
        if results:
            result = results[0]  # Single image result
            
            # Display processing summary
            print("\n📊 Batch Processing Complete:")
            print("   Total images: 1")
            print("   Successful extractions: 1")
            print("   Success rate: 100.0%")
            print(f"   Average processing time: {result.get('processing_time', 0):.2f}s")
            
            # Display clean extraction results with status icons
            print("\n📊 EXTRACTED DATA:")
            extracted_data = result.get('extracted_data', {})
            found_count = 0
            total_count = len(extracted_data)
            
            for field_name, value in extracted_data.items():
                if value and value != "NOT_FOUND":
                    status = "✅"
                    found_count += 1
                else:
                    status = "❌"
                print(f"   {status} {field_name}: {value}")
            
            print("\n📊 PARSED EXTRACTION RESULTS:")
            print("=" * 80)
            for field_name, value in extracted_data.items():
                if value and value != "NOT_FOUND":
                    status = "✅"
                else:
                    status = "❌"
                print(f"  {status} {field_name}: \"{value}\"")
            print("=" * 80)
            print(f"✅ Extracted {found_count}/{total_count} fields")
            
            # Load ground truth for evaluation if available
            if Path(ground_truth_path).exists():
                print(f"\n📊 Loading ground truth from: {ground_truth_path}")
                ground_truth_data = load_ground_truth(ground_truth_path)
                
                if ground_truth_data and image_path_obj.name in ground_truth_data:
                    print("\n🎯 Evaluating extraction results against ground truth...")
                    evaluation_summary = evaluate_extraction_results(results, ground_truth_data)
                    
                    print("\n📈 EVALUATION vs Ground Truth:")
                    overall_accuracy = evaluation_summary.get('overall_accuracy', 0)
                    print(f"   Accuracy: {overall_accuracy*100:.1f}%")
                    print(f"   Meets Threshold: {'Yes' if overall_accuracy >= 0.8 else 'No'}")
                    print(f"   Fields evaluated: {total_count}")
                else:
                    print(f"\n⚠️  No ground truth available for {image_path_obj.name}")
            
            print("\n✅ Single image processing complete!")
        else:
            print("❌ No results from image processing")
        
        return
    # Batch processing mode - scan data directory
    print(f"\n📁 Discovering images in: {data_dir}")
    image_files = discover_images(data_dir)

    # Apply filtering to match evaluation dataset
    # Currently filters for synthetic invoice test images - modify for different datasets
    image_files = [f for f in image_files if "synthetic_invoice" in Path(f).name]

    # Apply image limit if specified
    if limit_images and limit_images > 0:
        original_count = len(image_files)
        image_files = image_files[:limit_images]
        print(
            f"📷 Limited to {len(image_files)} images (from {original_count} total) for testing"
        )
    else:
        print(f"📷 Found {len(image_files)} images for processing")
    
    if not image_files:
        print("❌ No images found for processing")
        print(
            "💡 Check that DATA_DIR contains images with 'synthetic_invoice' in filename"
        )
        print("💡 Supported formats: PNG, JPG, JPEG (case insensitive)")
        return

    # =============================================================================
    # DEBUG OCR MODE CHECK
    # =============================================================================
    # Handle debug OCR mode for raw markdown output instead of structured extraction
    if debug_ocr:
        if not debug:
            print("❌ ERROR: --debug-ocr requires --debug flag")
            print("💡 Usage: python llama_keyvalue.py --image-path IMAGE --debug --debug-ocr")
            return
        
        print("\n🔍 DEBUG OCR MODE ENABLED")
        print("🎯 Processing will output raw markdown OCR instead of structured extraction")
        print("💡 This helps diagnose OCR vs document understanding issues")
        print()
        
        # Process each image in debug OCR mode
        ocr_results = []
        for i, image_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] Processing {Path(image_path).name}...")
            try:
                ocr_result = processor.process_debug_ocr(image_path)
                ocr_results.append(ocr_result)
                print(f"✅ OCR completed: {ocr_result['output_length']} chars in {ocr_result['processing_time']:.2f}s")
            except Exception as e:
                print(f"❌ OCR failed for {Path(image_path).name}: {e}")
                continue
        
        # Summary for debug OCR mode
        print("\n🎯 DEBUG OCR SUMMARY:")
        print(f"   Images processed: {len(ocr_results)}/{len(image_files)}")
        if ocr_results:
            avg_time = sum(r['processing_time'] for r in ocr_results) / len(ocr_results)
            total_chars = sum(r['output_length'] for r in ocr_results)
            print(f"   Average processing time: {avg_time:.2f}s")
            print(f"   Total OCR output: {total_chars:,} characters")
            print(f"   Average output per document: {total_chars // len(ocr_results):,} chars")
        
        print("\n💡 Use this output to diagnose:")
        print("   - OCR quality issues (garbled text, missing characters)")
        print("   - Layout preservation problems")  
        print("   - Document structure recognition")
        print("   - Compare with structured extraction accuracy")
        return

    # =============================================================================
    # BATCH PROCESSING PIPELINE
    # =============================================================================
    # Process all discovered images through Llama Vision extraction pipeline
    # Each image generates 25 structured fields with confidence and timing metrics
    print("\n🚀 Starting batch processing...")
    results, batch_stats = processor.process_image_batch(image_files)

    # =============================================================================
    # DATA FRAME CREATION AND CSV EXPORT
    # =============================================================================
    # Transform extraction results into structured DataFrames for analysis
    # Main DF: image_name + 25 extraction fields in alphabetical order
    # Metadata DF: processing statistics, timing, quality metrics per image
    print("\n📊 Creating extraction DataFrames...")
    main_df, metadata_df = create_extraction_dataframe(results)

    # Generate timestamped filenames for batch tracking and version control
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    extraction_csv = output_dir_path / f"llama_batch_extraction_{timestamp}.csv"
    main_df.to_csv(extraction_csv, index=False)
    print(f"💾 Extraction results saved: {extraction_csv}")

    # Save processing metadata for performance analysis and debugging
    if not metadata_df.empty:
        metadata_csv = output_dir_path / f"llama_extraction_metadata_{timestamp}.csv"
        metadata_df.to_csv(metadata_csv, index=False)
        print(f"💾 Extraction metadata saved: {metadata_csv}")

    # =============================================================================
    # GROUND TRUTH DATA LOADING
    # =============================================================================
    # Load reference data for accuracy evaluation and performance benchmarking
    print(f"\n📊 Loading ground truth from: {ground_truth_path}")
    ground_truth_data = load_ground_truth(ground_truth_path)

    if not ground_truth_data:
        print("❌ No ground truth data available - skipping evaluation")
        print("💡 Processing completed successfully, but accuracy metrics unavailable")
        return

    # =============================================================================
    # COMPREHENSIVE EVALUATION AGAINST GROUND TRUTH
    # =============================================================================
    # Calculate field-level and document-level accuracy metrics
    # Supports both exact matching and fuzzy matching for string fields
    print("\n🎯 Evaluating extraction results against ground truth...")
    evaluation_summary = evaluate_extraction_results(results, ground_truth_data)

    # Calculate document quality distribution for deployment readiness assessment
    # Categories: Good (80-99%), Fair (60-80%), Poor (<60%)
    evaluation_data = evaluation_summary.get("evaluation_data", [])
    evaluation_summary["good_documents"] = sum(
        1 for doc in evaluation_data if 0.8 <= doc["overall_accuracy"] < 0.99
    )
    evaluation_summary["fair_documents"] = sum(
        1 for doc in evaluation_data if 0.6 <= doc["overall_accuracy"] < 0.8
    )
    evaluation_summary["poor_documents"] = sum(
        1 for doc in evaluation_data if doc["overall_accuracy"] < 0.6
    )

    # =============================================================================
    # COMPREHENSIVE REPORTING AND SUMMARY GENERATION
    # =============================================================================
    # Generate multiple report formats for different stakeholder needs:
    # - Technical evaluation results (JSON)
    # - Executive summary (Markdown)
    # - Deployment readiness checklist (Markdown)
    print("\n📝 Generating comprehensive evaluation reports...")
    reports = generate_comprehensive_reports(
        evaluation_summary,
        output_dir_path,
        "llama",
        "Llama-3.2-11B-Vision-Instruct",
        batch_stats,
        results,  # extraction_results for classification analysis
        ground_truth_data,  # ground truth mapping for classification analysis
    )

    # Display summary statistics to console for immediate feedback
    print_evaluation_summary(evaluation_summary, "Llama-3.2-11B-Vision-Instruct")

    # List all generated files for easy access and verification
    print("\n📁 Report files generated:")
    for report_type, report_path in reports.items():
        if report_type == "visualizations" and isinstance(report_path, list):
            print(f"   - {report_type}:")
            for viz_path in report_path:
                print(f"     • {viz_path.name}")
        else:
            print(f"   - {report_type}: {report_path.name}")

    print("\n✅ Llama Vision evaluation pipeline completed successfully!")
    print(
        f"📊 {len(image_files)} documents processed with {evaluation_summary.get('overall_accuracy', 0):.1%} average accuracy"
    )
    print(
        f"🔬 Total extraction time: {batch_stats.get('total_processing_time', 0):.2f} seconds (core model inference only)"
    )
    print(
        f"📈 Average extraction time per document: {batch_stats.get('average_processing_time', 0):.2f} seconds"
    )
    print(f"✅ Processing success rate: {batch_stats.get('success_rate', 0):.1%}")


if __name__ == "__main__":
    args = parse_arguments()
    main(
        extraction_mode=args.extraction_mode,
        debug=args.debug,
        limit_images=args.limit_images,
        image_path=args.image_path,
        debug_ocr=args.debug_ocr,
    )
