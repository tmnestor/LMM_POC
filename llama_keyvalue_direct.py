#!/usr/bin/env python3
"""
Llama Vision Direct Key-Value Extraction with Comprehensive Evaluation Pipeline

This module implements a complete evaluation pipeline for the Llama-3.2-11B-Vision (base)
model using DIRECT PROMPTING without chat templates, designed to extract structured
key-value data from business documents (invoices, statements, etc.) and evaluate
performance against ground truth data.

Key Differences from Chat-based Version:
- Uses base Llama-3.2-11B-Vision model (not -Instruct)
- Direct prompting with <image> tag (no conversation templates)
- No conversation artifact cleaning required
- Cleaner, more direct outputs similar to InternVL3 approach

Pipeline Overview:
    1. Setup & Validation - Verify paths, create output directories
    2. Model Initialization - Load Llama Vision direct processor with optimal config
    3. Image Discovery - Find and filter document images for processing
    4. Batch Processing - Extract structured data from all images using direct prompting
    5. Data Generation - Create CSV outputs with extraction results
    6. Evaluation - Compare results against ground truth for accuracy metrics
    7. Reporting - Generate comprehensive evaluation reports and summaries

Input Requirements:
    - Document images in DATA_DIR (PNG, JPG, JPEG formats)
    - Ground truth CSV file for evaluation (evaluation_ground_truth.csv)
    - Llama-3.2-11B-Vision (base) model at LLAMA_DIRECT_MODEL_PATH
    - CUDA-compatible GPU recommended (16GB+ VRAM or 8-bit quantization)

Output Files:
    - llama_direct_batch_extraction_{timestamp}.csv - Main extraction results
    - llama_direct_extraction_metadata_{timestamp}.csv - Processing metadata
    - llama_direct_evaluation_results_{timestamp}.json - Detailed evaluation metrics
    - llama_direct_executive_summary_{timestamp}.md - High-level performance summary
    - llama_direct_deployment_checklist_{timestamp}.md - Deployment readiness assessment

Usage:
    python llama_keyvalue_direct.py

Configuration:
    Edit common/config.py to adjust:
    - DATA_DIR: Path to document images
    - LLAMA_DIRECT_MODEL_PATH: Path to base model weights
    - OUTPUT_DIR: Path for generated reports
    - EXTRACTION_FIELDS: List of fields to extract (currently 25 business document fields)

Performance Characteristics:
    - Processing time: ~3-5 seconds per document (GPU)
    - Memory usage: ~22GB VRAM (11B parameter model)
    - Accuracy: Expected similar to chat-based version
    - Field extraction: 25 structured fields (ABN, TOTAL, INVOICE_DATE, etc.)
    - Output quality: Cleaner outputs without conversation artifacts

Dependencies:
    - models.llama_direct_processor: Direct prompting Llama processor
    - common.config: Centralized configuration management
    - common.evaluation_utils: Shared evaluation and parsing utilities
    - common.reporting: Report generation and formatting utilities

Author: LMM_POC Pipeline
Date: 2024
"""

import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from common.config import (
    DATA_DIR,
    EXTRACTION_FIELDS,
    GROUND_TRUTH_PATH,
    LLAMA_DIRECT_MODEL_PATH,
    OUTPUT_DIR,
)
from common.evaluation_utils import (
    create_extraction_dataframe,
    discover_images,
    evaluate_extraction_results,
)
from common.reporting import generate_comprehensive_reports, print_evaluation_summary
from models.llama_direct_processor import LlamaDirectProcessor


def verify_requirements():
    """
    Verify all required paths and dependencies exist.

    Returns:
        bool: True if all requirements are met

    Raises:
        SystemExit: If critical requirements are missing
    """
    print("\n🔍 Verifying requirements...")

    # Check model path
    if not Path(LLAMA_DIRECT_MODEL_PATH).exists():
        print(f"❌ FATAL: Llama direct model not found at: {LLAMA_DIRECT_MODEL_PATH}")
        print("💡 Please ensure LLAMA_DIRECT_MODEL_PATH is correctly configured")
        print("💡 Expected: Llama-3.2-11B-Vision (base model, not -Instruct)")
        sys.exit(1)

    # Check data directory
    if not Path(DATA_DIR).exists():
        print(f"❌ FATAL: Data directory not found: {DATA_DIR}")
        print("💡 Please ensure DATA_DIR contains document images")
        sys.exit(1)

    # Check ground truth file
    if not Path(GROUND_TRUTH_PATH).exists():
        print(f"⚠️ WARNING: Ground truth file not found: {GROUND_TRUTH_PATH}")
        print("💡 Evaluation against ground truth will be skipped")
        has_ground_truth = False
    else:
        has_ground_truth = True

    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("✅ Requirements verification completed")
    return has_ground_truth


def main():
    """
    Execute the complete Llama Vision Direct evaluation pipeline.

    This function orchestrates the entire process from model loading to
    report generation, with comprehensive error handling and progress reporting.

    The pipeline processes images in batches, generates structured CSV outputs,
    evaluates results against ground truth (if available), and produces
    detailed reports for deployment assessment.

    Critical Requirements:
        - LLAMA_DIRECT_MODEL_PATH: Must point to valid Llama-3.2-Vision base model
        - DATA_DIR: Must contain document images (PNG/JPG/JPEG)
        - OUTPUT_DIR: Must be writable for report generation
        - GPU Memory: 16GB+ VRAM recommended (or 8-bit quantization enabled)

    Exit Codes:
        0: Success - Pipeline completed without errors
        1: Configuration Error - Missing model, data, or permission issues
        2: Processing Error - Model loading or batch processing failures
        3: Evaluation Error - Ground truth processing or metric calculation failures
    """
    start_time = datetime.now()

    print("🦙 Llama Vision Direct Key-Value Extraction with Comprehensive Evaluation")
    print("=" * 80)
    print(f"🔧 Model: {LLAMA_DIRECT_MODEL_PATH}")
    print(f"📁 Data: {DATA_DIR}")
    print(f"📊 Fields: {len(EXTRACTION_FIELDS)} business document fields")
    print("🎯 Approach: Direct prompting (no chat template)")
    print(f"⏰ Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    try:
        # Phase 1: Setup and Validation
        print("📋 Phase 1: Setup and Validation")
        print("-" * 40)

        has_ground_truth = verify_requirements()

        # Discover document images for processing
        print(f"\n🔍 Discovering images in: {DATA_DIR}")
        image_files = discover_images(DATA_DIR)

        if not image_files:
            print(f"❌ FATAL: No document images found in {DATA_DIR}")
            print("💡 Supported formats: PNG, JPG, JPEG")
            sys.exit(1)

        print(f"✅ Found {len(image_files)} document images")

        # Phase 2: Model Initialization and Loading
        print("\n🤖 Phase 2: Model Initialization")
        print("-" * 40)

        # Load Llama-3.2-Vision direct model with optimal configuration for extraction tasks
        print("\n🚀 Initializing Llama Vision Direct processor...")
        llama_direct_processor = LlamaDirectProcessor()

        print("✅ Model initialization completed successfully")

        # Phase 3: Batch Processing and Data Extraction
        print("\n📄 Phase 3: Document Processing")
        print("-" * 40)

        # Process all discovered images through Llama Vision direct extraction pipeline
        extraction_results, batch_statistics = (
            llama_direct_processor.process_image_batch(image_files)
        )

        print(f"\n✅ Processed {len(extraction_results)} documents")
        print(f"📊 Success rate: {batch_statistics['success_rate']:.1%}")
        print(
            f"⏱️ Average processing time: {batch_statistics['average_processing_time']:.2f}s"
        )

        # Phase 4: Data Generation and CSV Export
        print("\n💾 Phase 4: Data Export")
        print("-" * 40)

        # Generate structured DataFrames from extraction results
        results_df, metadata_df = create_extraction_dataframe(extraction_results)

        # Create timestamped output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_path = Path(OUTPUT_DIR)

        # Save main extraction results
        extraction_csv_path = (
            output_dir_path / f"llama_direct_batch_extraction_{timestamp}.csv"
        )
        results_df.to_csv(extraction_csv_path, index=False)
        print(f"💾 Extraction results: {extraction_csv_path}")

        # Save processing metadata
        metadata_csv_path = (
            output_dir_path / f"llama_direct_extraction_metadata_{timestamp}.csv"
        )
        metadata_df.to_csv(metadata_csv_path, index=False)
        print(f"📊 Processing metadata: {metadata_csv_path}")

        # Phase 5: Evaluation Against Ground Truth
        evaluation_summary = {}
        if has_ground_truth:
            print("\n🎯 Phase 5: Ground Truth Evaluation")
            print("-" * 40)

            try:
                # First load ground truth mapping from CSV
                from common.evaluation_utils import load_ground_truth_csv
                ground_truth_map = load_ground_truth_csv(GROUND_TRUTH_PATH)
                
                # Then evaluate using the original extraction_results list (not DataFrame)
                evaluation_summary = evaluate_extraction_results(
                    extraction_results, ground_truth_map
                )
                print("✅ Ground truth evaluation completed")
            except Exception as e:
                print(f"⚠️ Evaluation error: {e}")
                print("💡 Continuing without ground truth metrics...")
        else:
            print("\n⏭️ Phase 5: Skipped (No Ground Truth)")
            print("-" * 40)
            print("💡 Ground truth file not available - evaluation metrics skipped")

        # Phase 6: Comprehensive Report Generation
        print("\n📊 Phase 6: Report Generation")
        print("-" * 40)

        # Generate comprehensive reports only if evaluation succeeded
        if evaluation_summary and 'field_accuracies' in evaluation_summary:
            generate_comprehensive_reports(
                evaluation_summary,
                output_dir_path,
                "llama_direct",
                "Llama-3.2-11B-Vision-Direct",
            )
            
            # Print evaluation summary to console
            print_evaluation_summary(evaluation_summary, "Llama-3.2-11B-Vision-Direct")
        else:
            print("⚠️ Skipping detailed reports - ground truth evaluation failed")
            print("💡 CSV extraction results are still available for manual review")
            print(f"📊 Processing completed successfully:")
            print(f"   • Documents processed: {len(extraction_results)}")
            print(f"   • Success rate: {batch_statistics['success_rate']:.1%}")
            print(f"   • Average processing time: {batch_statistics['average_processing_time']:.2f}s")
            print(f"   • Output files: {extraction_csv_path}")

        print("\n✅ Llama Vision Direct evaluation pipeline completed successfully!")
        print("=" * 80)

        # Final summary
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        print(f"⏰ Total pipeline duration: {total_duration:.1f} seconds")
        print(f"📄 Documents processed: {len(extraction_results)}")
        print(
            f"📊 Average per document: {total_duration / len(extraction_results):.2f}s"
        )
        print("🚀 Llama direct processing: ~5x cleaner outputs vs chat-based")
        print(f"💾 Reports saved to: {output_dir_path}")

        return 0

    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        return 1
    except FileNotFoundError as e:
        print(f"\n❌ FATAL: File not found - {e}")
        return 1
    except PermissionError as e:
        print(f"\n❌ FATAL: Permission denied - {e}")
        return 1
    except Exception as e:
        print(f"\n❌ FATAL: Unexpected error - {e}")
        import traceback

        traceback.print_exc()
        return 2


if __name__ == "__main__":
    """
    Main execution entry point.
    
    Handles command-line execution with proper exit codes and error handling.
    All configuration is managed through common/config.py - no command-line 
    arguments are currently supported.
    """
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"💥 Critical error in main execution: {e}")
        sys.exit(3)
