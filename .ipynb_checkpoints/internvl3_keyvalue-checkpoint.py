#!/usr/bin/env python
# coding: utf-8

"""
InternVL3 Key-Value Extraction with Comprehensive Evaluation (Refactored)

This refactored version uses shared modules to reduce code duplication
while maintaining all original functionality.
"""

from datetime import datetime
from pathlib import Path

# Import shared modules
from common.config import DATA_DIR as data_dir
from common.config import EXTRACTION_FIELDS
from common.config import GROUND_TRUTH_PATH as ground_truth_path
from common.config import INTERNVL3_MODEL_PATH as model_path
from common.config import OUTPUT_DIR as output_dir
from common.evaluation_utils import (
    create_extraction_dataframe,
    discover_images,
    evaluate_extraction_results,
    load_ground_truth,
)
from common.reporting import generate_comprehensive_reports, print_evaluation_summary
from models.internvl3_processor import InternVL3Processor


def main():
    """Main execution pipeline for InternVL3 evaluation."""
    
    print("🎯 INTERNVL3 EVALUATION MODE ENABLED")
    print(f"📁 Evaluation data directory: {data_dir}")
    print(f"📊 Ground truth file: {ground_truth_path}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔧 Loading InternVL3-2B model from: {model_path}")
    
    # Validate evaluation setup
    evaluation_data_path = Path(data_dir)
    gt_path = Path(ground_truth_path)
    output_dir_path = Path(output_dir)
    
    # Create output directory if needed
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Check data directory
    if not evaluation_data_path.exists():
        print(f"❌ ERROR: Evaluation data directory not found: {data_dir}")
        print("💡 Please ensure evaluation_data/ directory exists")
        return
    
    # Check ground truth
    if not gt_path.exists():
        print(f"❌ ERROR: Ground truth file not found: {ground_truth_path}")
        print("💡 Please ensure evaluation_ground_truth.csv exists")
        return
    
    # Initialize processor
    print("\n🚀 Initializing InternVL3 processor...")
    processor = InternVL3Processor(model_path=model_path)
    
    # Load ground truth data
    print("\n📊 Loading ground truth data for evaluation...")
    ground_truth_data = load_ground_truth(ground_truth_path, show_sample=True)
    
    if ground_truth_data:
        print(f"✅ Ground truth loaded successfully for {len(ground_truth_data)} images")
        print("🎯 Evaluation infrastructure ready")
    else:
        print("❌ Failed to load ground truth data - evaluation will be limited")
    
    # Discover images
    print("\n🔍 Discovering images in data directory...")
    image_files = discover_images(data_dir)
    
    if not image_files:
        print(f"❌ No image files found in {data_dir}")
        print("💡 Supported formats: PNG, JPG, JPEG (case insensitive)")
        return
    
    print(f"✅ Found {len(image_files)} image files to process")
    
    # Show sample of files
    print("\n📋 Sample of files to be processed:")
    for i, file_path in enumerate(image_files[:5]):
        print(f"   {i + 1}. {Path(file_path).name}")
    if len(image_files) > 5:
        print(f"   ... and {len(image_files) - 5} more files")
    
    # Process images
    print("\n🚀 Starting enhanced batch key-value extraction...")
    start_time = datetime.now()
    
    try:
        # Process all images through extraction pipeline
        extraction_results, batch_statistics = processor.process_image_batch(image_files)
        
        # Create structured DataFrame
        print("\n📊 Creating structured DataFrame with success metrics...")
        df, metadata_df = create_extraction_dataframe(extraction_results)
        
        print(f"✅ Successfully created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        print(f"📋 Column structure: image_name + {len(EXTRACTION_FIELDS)} alphabetically ordered fields")
        
        # Generate CSV output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = "internvl3_batch_extraction.csv"
        csv_filename_timestamped = f"internvl3_batch_extraction_{timestamp}.csv"
        
        csv_path = output_dir_path / csv_filename
        csv_path_timestamped = output_dir_path / csv_filename_timestamped
        
        # Save both versions
        df.to_csv(csv_path, index=False)
        df.to_csv(csv_path_timestamped, index=False)
        
        print("\n💾 CSV files saved:")
        print(f"   - Latest version: {csv_path}")
        print(f"   - Timestamped backup: {csv_path_timestamped}")
        
        # Save metadata if available
        if not metadata_df.empty:
            metadata_csv = output_dir_path / f"internvl3_extraction_metadata_{timestamp}.csv"
            metadata_df.to_csv(metadata_csv, index=False)
            print(f"   - Metadata: {metadata_csv}")
        
        # Perform evaluation if ground truth is available
        if ground_truth_data:
            print("\n🎯 Starting comprehensive evaluation against ground truth...")
            evaluation_summary = evaluate_extraction_results(extraction_results, ground_truth_data)
            
            # Generate comprehensive reports
            reports = generate_comprehensive_reports(
                evaluation_summary,
                output_dir_path,
                "internvl3",
                "InternVL3-2B"
            )
            
            # Print summary
            print_evaluation_summary(evaluation_summary, "InternVL3-2B")
        
        # Calculate and display processing statistics
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        print(f"\n⏱️ Total processing time: {total_time:.2f} seconds")
        print(f"📊 Average time per image: {total_time/len(image_files):.2f} seconds")
        print(f"✅ Success rate: {batch_statistics['success_rate']:.1%}")
        
        print("\n🎉 InternVL3 evaluation pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during batch processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()