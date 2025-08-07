#!/usr/bin/env python3
"""
Llama Vision Key-Value Extraction with Comprehensive Evaluation (Refactored)

This refactored version uses shared modules to reduce code duplication
while maintaining all original functionality.
"""

from datetime import datetime
from pathlib import Path

# Import shared modules
from common.config import DATA_DIR as data_dir
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


def main():
    """Main execution pipeline for Llama Vision key-value extraction and evaluation."""
    
    print("\n" + "="*80)
    print("🦙 LLAMA VISION COMPREHENSIVE EVALUATION PIPELINE")
    print("="*80)
    
    print(f"📁 Data directory: {data_dir}")
    print(f"📂 Output directory: {output_dir}")
    print(f"📊 Ground truth: {ground_truth_path}")
    print(f"🔧 Model: {model_path}")
    
    # Ensure output directory exists
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Validate data directory
    if not Path(data_dir).exists():
        print(f"❌ ERROR: Data directory not found: {data_dir}")
        return
    
    # Validate ground truth
    if not Path(ground_truth_path).exists():
        print(f"❌ ERROR: Ground truth file not found: {ground_truth_path}")
        return
    
    # Initialize processor
    print("\n🚀 Initializing Llama Vision processor...")
    processor = LlamaProcessor(model_path=model_path)
    
    # Discover images
    print(f"\n📁 Discovering images in: {data_dir}")
    image_files = discover_images(data_dir)
    
    # Filter for test images (matching original behavior)
    image_files = [f for f in image_files if 'synthetic_invoice' in Path(f).name]
    
    print(f"📷 Found {len(image_files)} images for processing")
    if not image_files:
        print("❌ No images found for processing")
        return
    
    # Process batch
    print("\n🚀 Starting batch processing...")
    results, batch_stats = processor.process_image_batch(image_files)
    
    # Create DataFrames
    print("\n📊 Creating extraction DataFrames...")
    main_df, metadata_df = create_extraction_dataframe(results)
    
    # Save extraction results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    extraction_csv = output_dir_path / f"llama_batch_extraction_{timestamp}.csv"
    main_df.to_csv(extraction_csv, index=False)
    print(f"💾 Extraction results saved: {extraction_csv}")
    
    # Save extraction metadata (to match InternVL3 output structure)
    if not metadata_df.empty:
        metadata_csv = output_dir_path / f"llama_extraction_metadata_{timestamp}.csv"
        metadata_df.to_csv(metadata_csv, index=False)
        print(f"💾 Extraction metadata saved: {metadata_csv}")
    
    # Load ground truth
    print(f"\n📊 Loading ground truth from: {ground_truth_path}")
    ground_truth_data = load_ground_truth(ground_truth_path)
    
    if not ground_truth_data:
        print("❌ No ground truth data available - skipping evaluation")
        return
    
    # Perform evaluation
    print("\n🎯 Evaluating extraction results against ground truth...")
    evaluation_summary = evaluate_extraction_results(results, ground_truth_data)
    
    # Add document quality distribution for reporting
    evaluation_data = evaluation_summary.get('evaluation_data', [])
    evaluation_summary['good_documents'] = sum(1 for doc in evaluation_data if 0.8 <= doc['overall_accuracy'] < 0.99)
    evaluation_summary['fair_documents'] = sum(1 for doc in evaluation_data if 0.6 <= doc['overall_accuracy'] < 0.8)
    evaluation_summary['poor_documents'] = sum(1 for doc in evaluation_data if doc['overall_accuracy'] < 0.6)
    
    # Generate comprehensive reports
    print("\n📝 Generating comprehensive evaluation reports...")
    reports = generate_comprehensive_reports(
        evaluation_summary,
        output_dir_path,
        "llama",
        "Llama-3.2-11B-Vision-Instruct"
    )
    
    # Print final summary
    print_evaluation_summary(evaluation_summary, "Llama-3.2-11B-Vision-Instruct")
    
    # Show report locations
    print("\n📁 Report files generated:")
    for report_type, report_path in reports.items():
        print(f"   - {report_type}: {report_path.name}")
    
    print("\n✅ Llama Vision evaluation pipeline completed successfully!")


if __name__ == "__main__":
    main()