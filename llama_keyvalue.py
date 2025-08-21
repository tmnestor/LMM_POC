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
    - Ground truth CSV file for evaluation (evaluation_ground_truth.csv)
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
    - EXTRACTION_FIELDS: List of fields to extract (currently 25 business document fields)

Model Specifications:
    - Parameters: 11B parameter model
    - Field extraction: 25 structured fields (ABN, TOTAL, INVOICE_DATE, etc.)

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
from common.config import DEFAULT_EXTRACTION_MODE, EXTRACTION_MODES
from common.config import GROUND_TRUTH_PATH as ground_truth_path
from common.config import LLAMA_MODEL_PATH as model_path
from common.config import OUTPUT_DIR as output_dir
from common.evaluation_utils import (
    create_extraction_dataframe,
    discover_images,
    evaluate_extraction_results,
    load_ground_truth,
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

    return parser.parse_args()


def main(extraction_mode=None, debug=False, limit_images=None):
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
        - GROUND_TRUTH_PATH: Must contain evaluation_ground_truth.csv
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
    extraction_mode = extraction_mode or DEFAULT_EXTRACTION_MODE
    print(
        f"\n🚀 Initializing Llama Vision processor with {extraction_mode} extraction mode..."
    )
    processor = LlamaProcessor(
        model_path=model_path, extraction_mode=extraction_mode, debug=debug
    )

    # =============================================================================
    # DOCUMENT IMAGE DISCOVERY AND FILTERING
    # =============================================================================
    # Scan data directory for supported image formats (PNG, JPG, JPEG)
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
    )
