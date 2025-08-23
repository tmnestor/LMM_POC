#!/usr/bin/env python3
"""
InternVL3 Vision Key-Value Extraction with Comprehensive Evaluation Pipeline

This module implements a complete evaluation pipeline for the InternVL3-2B vision-language
model, designed to extract structured key-value data from business documents (invoices,
statements, etc.) and evaluate performance against ground truth data.

Pipeline Overview:
    1. Setup & Validation - Verify paths, create output directories
    2. Model Initialization - Load InternVL3 processor with optimal config
    3. Image Discovery - Find and filter document images for processing
    4. Batch Processing - Extract structured data from all images
    5. Data Generation - Create CSV outputs with extraction results
    6. Evaluation - Compare results against ground truth for accuracy metrics
    7. Reporting - Generate comprehensive evaluation reports and summaries

Input Requirements:
    - Document images in DATA_DIR (PNG, JPG, JPEG formats)
    - Ground truth CSV file for evaluation (evaluation_ground_truth.csv)
    - InternVL3-2B model at INTERNVL3_MODEL_PATH
    - CUDA-compatible GPU recommended (4GB+ VRAM, CPU fallback available)

Output Files:
    - internvl3_batch_extraction_{timestamp}.csv - Main extraction results
    - internvl3_extraction_metadata_{timestamp}.csv - Processing metadata
    - internvl3_evaluation_results_{timestamp}.json - Detailed evaluation metrics
    - internvl3_executive_summary_{timestamp}.md - High-level performance summary
    - internvl3_deployment_checklist_{timestamp}.md - Deployment readiness assessment

Usage:
    python internvl3_keyvalue.py

Configuration:
    Edit common/config.py to adjust:
    - DATA_DIR: Path to document images
    - INTERNVL3_MODEL_PATH: Path to model weights
    - OUTPUT_DIR: Path for generated reports
    - EXTRACTION_FIELDS: List of fields to extract (currently 25 business document fields)

Model Specifications:
    - Parameters: 2B/8B parameter model variants
    - Field extraction: 25 structured fields (ABN, TOTAL, INVOICE_DATE, etc.)

Dependencies:
    - models.internvl3_processor: InternVL3-specific processing logic
    - common.config: Centralized configuration management
    - common.evaluation_utils: Shared evaluation and parsing utilities
    - common.reporting: Report generation and formatting
    - transformers: Hugging Face model loading
    - torch: PyTorch for model inference (CUDA recommended)
    - timm: PyTorch Image Models (required for InternVL3)
    - einops: Tensor operations library (required for InternVL3)

Model Architecture Notes:
    InternVL3-2B uses a vision transformer encoder with language model decoder.
    The model supports dynamic preprocessing with tile-based approach for high-resolution
    images, making it particularly effective for document analysis tasks.

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
from common.config import INTERNVL3_MODEL_PATH as model_path
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
from models.internvl3_processor import InternVL3Processor


def parse_arguments():
    """Parse command line arguments for extraction mode and debugging."""
    parser = argparse.ArgumentParser(
        description="InternVL3 Vision Key-Value Extraction with Comprehensive Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python internvl3_keyvalue.py --extraction-mode grouped --debug
  python internvl3_keyvalue.py --extraction-mode adaptive
  python internvl3_keyvalue.py --extraction-mode single_pass
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
    Execute the complete InternVL3 Vision evaluation pipeline.

    This function orchestrates the entire evaluation workflow from initial setup
    through final report generation. It handles all error conditions gracefully
    and provides detailed progress feedback throughout processing.

    Args:
        extraction_mode (str): Extraction strategy ('single_pass', 'grouped', 'adaptive')
        debug (bool): Enable debug logging for grouped extraction
        limit_images (int): Limit processing to first N images (for testing)

    Pipeline Stages:
        1. Environment validation and setup
        2. Model processor initialization (InternVL3-2B)
        3. Document image discovery and filtering
        4. Batch processing with progress tracking
        5. Data frame creation and CSV export
        6. Ground truth evaluation and metrics calculation
        7. Comprehensive report generation

    Configuration Dependencies:
        - DATA_DIR: Must contain document images (*.png, *.jpg, *.jpeg)
        - INTERNVL3_MODEL_PATH: Must point to valid InternVL3-2B model
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

    InternVL3-Specific Features:
        - Dynamic preprocessing with tile-based approach
        - Optimized for high-resolution document images

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
    print("🔬 INTERNVL3 VISION COMPREHENSIVE EVALUATION PIPELINE")
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

    # Validate critical input paths before proceeding with model loading
    # InternVL3 is faster to load than Llama but still worth validating first
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
    # Load InternVL3 model with optimal configuration for extraction tasks
    extraction_mode = extraction_mode or DEFAULT_EXTRACTION_MODE
    print(
        f"\n🚀 Initializing InternVL3 processor with {extraction_mode} extraction mode..."
    )
    processor = InternVL3Processor(
        model_path=model_path, extraction_mode=extraction_mode, debug=debug
    )

    # =============================================================================
    # DEBUG CONFIGURATION OUTPUT
    # =============================================================================
    if debug:
        print("\n" + "=" * 80)
        print("🔧 COMPLETE INTERNVL3 CONFIGURATION DEBUG")
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
        from common.config import EXTRACTION_FIELDS, FIELD_COUNT
        print(f"\n📋 Total fields: {FIELD_COUNT}")
        print(f"📋 First field: {EXTRACTION_FIELDS[0]}")
        print(f"📋 Last field: {EXTRACTION_FIELDS[-1]}")
        print(f"📋 Field sequence: {' → '.join(EXTRACTION_FIELDS[:3])} ... {' → '.join(EXTRACTION_FIELDS[-3:])}")
        
        # Prompt file information
        prompt_file = "internvl3_prompts.yaml"
        print(f"\n📝 Prompt file: {prompt_file}")
        if extraction_mode == "single_pass":
            print("📝 Prompt method: Single-pass section from YAML")
            print("📝 YAML section: single_pass")
        else:
            print("📝 Prompt method: Grouped extraction sections from YAML")
            print("📝 YAML sections: regulatory_financial, entity_contacts, line_item_transactions, temporal_data, banking_payment, document_balances")
        
        # Show actual prompt preview
        try:
            sample_prompt = processor.get_extraction_prompt()
            print(f"📝 Prompt length: {len(sample_prompt)} characters")
            print("📝 Prompt preview (first 200 chars):")
            print(f"    {sample_prompt[:200].replace(chr(10), ' ')}")
        except Exception as e:
            print(f"📝 ⚠️ Could not preview prompt: {e}")
        
        # Model configuration
        print("\n🤖 Model processor: InternVL3Processor")
        print("🤖 Generation config: temperature=0.0, do_sample=False") 
        print(f"🤖 Max tokens: {getattr(processor, 'max_new_tokens', 'default')}")
        
        print("=" * 80)
        print("END DEBUG CONFIGURATION")
        print("=" * 80 + "\n")

    # Determine model name based on actual model path
    model_display_name = "InternVL3-8B" if "8B" in str(model_path) else "InternVL3-2B"

    # =============================================================================
    # GROUND TRUTH DATA LOADING
    # =============================================================================
    # Load reference data for accuracy evaluation and performance benchmarking
    print(f"\n📊 Loading ground truth from: {ground_truth_path}")
    ground_truth_data = load_ground_truth(ground_truth_path, show_sample=True)

    if ground_truth_data:
        print(
            f"✅ Ground truth loaded successfully for {len(ground_truth_data)} images"
        )
        print("🎯 Evaluation infrastructure ready")
    else:
        print("❌ Failed to load ground truth data - evaluation will be limited")

    # =============================================================================
    # DOCUMENT IMAGE DISCOVERY AND FILTERING
    # =============================================================================
    # Scan data directory for supported image formats (PNG, JPG, JPEG)
    print(f"\n📁 Discovering images in: {data_dir}")
    image_files = discover_images(data_dir)

    if not image_files:
        print("❌ No images found for processing")
        print("💡 Supported formats: PNG, JPG, JPEG (case insensitive)")
        return

    # Apply image limit if specified
    if limit_images and limit_images > 0:
        original_count = len(image_files)
        image_files = image_files[:limit_images]
        print(
            f"📷 Limited to {len(image_files)} images (from {original_count} total) for testing"
        )
    else:
        print(f"📷 Found {len(image_files)} images for processing")

    # Display sample of discovered files for verification
    print("\n📋 Sample of files to be processed:")
    for i, file_path in enumerate(image_files[:5]):
        print(f"   {i + 1}. {Path(file_path).name}")
    if len(image_files) > 5:
        print(f"   ... and {len(image_files) - 5} more files")

    # =============================================================================
    # BATCH PROCESSING PIPELINE
    # =============================================================================
    # Process all discovered images through InternVL3 extraction pipeline
    # Each image generates 25 structured fields with confidence and timing metrics
    print("\n🚀 Starting batch processing...")
    start_time = datetime.now()

    try:
        # Process all images through InternVL3 Vision extraction pipeline
        # InternVL3 uses dynamic preprocessing with tile-based approach for optimal quality
        extraction_results, batch_statistics = processor.process_image_batch(
            image_files
        )

        # =============================================================================
        # DATA FRAME CREATION AND CSV EXPORT
        # =============================================================================
        # Transform extraction results into structured DataFrames for analysis
        # Main DF: image_name + 25 extraction fields in alphabetical order
        # Metadata DF: processing statistics, timing, quality metrics per image
        print("\n📊 Creating extraction DataFrames...")
        df, metadata_df = create_extraction_dataframe(extraction_results)

        print(
            f"✅ Successfully created DataFrame with {len(df)} rows and {len(df.columns)} columns"
        )
        print(
            f"📋 Column structure: image_name + {len(EXTRACTION_FIELDS)} alphabetically ordered fields"
        )

        # Generate timestamped filenames for batch tracking and version control
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extraction_csv = output_dir_path / f"internvl3_batch_extraction_{timestamp}.csv"
        df.to_csv(extraction_csv, index=False)
        print(f"💾 Extraction results saved: {extraction_csv}")

        # Save processing metadata for performance analysis and debugging
        if not metadata_df.empty:
            metadata_csv = (
                output_dir_path / f"internvl3_extraction_metadata_{timestamp}.csv"
            )
            metadata_df.to_csv(metadata_csv, index=False)
            print(f"💾 Extraction metadata saved: {metadata_csv}")

        # =============================================================================
        # COMPREHENSIVE EVALUATION AGAINST GROUND TRUTH
        # =============================================================================
        # Perform accuracy evaluation if ground truth data is available
        # Calculate field-level and document-level accuracy metrics
        # Supports both exact matching and fuzzy matching for string fields
        if ground_truth_data:
            print("\n🎯 Evaluating extraction results against ground truth...")
            evaluation_summary = evaluate_extraction_results(
                extraction_results, ground_truth_data
            )

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
                "internvl3",
                model_display_name,
                batch_statistics,
                extraction_results,  # extraction_results for classification analysis
                ground_truth_data,  # ground truth mapping for classification analysis
            )

            # Display summary statistics to console for immediate feedback
            print_evaluation_summary(evaluation_summary, model_display_name)

            # List all generated files for easy access and verification
            print("\n📁 Report files generated:")
            for report_type, report_path in reports.items():
                if report_type == "visualizations" and isinstance(report_path, list):
                    print(f"   - {report_type}:")
                    for viz_path in report_path:
                        print(f"     • {viz_path.name}")
                else:
                    print(f"   - {report_type}: {report_path.name}")
        else:
            print("❌ No ground truth data available - skipping evaluation")
            print(
                "💡 Processing completed successfully, but accuracy metrics unavailable"
            )

        # =============================================================================
        # FINAL STATISTICS AND PIPELINE COMPLETION
        # =============================================================================
        # Calculate total processing time and performance metrics
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        print("\n✅ InternVL3 Vision evaluation pipeline completed successfully!")
        print(
            f"📊 {len(image_files)} documents processed with "
            f"{evaluation_summary.get('overall_accuracy', 0):.1%} average accuracy"
            if ground_truth_data
            else f"📊 {len(image_files)} documents processed successfully"
        )
        print(
            f"⏱️ Total pipeline time: {total_time:.2f} seconds (includes model loading, evaluation, reporting)"
        )
        print(
            f"📈 Average pipeline time per document: {total_time / len(image_files):.2f} seconds"
        )
        print(
            f"🔬 Average extraction time per document: {batch_statistics['average_processing_time']:.2f} seconds (core model inference only)"
        )
        print(f"✅ Processing success rate: {batch_statistics['success_rate']:.1%}")
        print("🚀 InternVL3 processing completed")

    except Exception as e:
        print(f"\n❌ Error during batch processing: {e}")
        print("💡 Check model path, dependencies, and available GPU memory")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    args = parse_arguments()
    main(
        extraction_mode=args.extraction_mode,
        debug=args.debug,
        limit_images=args.limit_images,
    )
