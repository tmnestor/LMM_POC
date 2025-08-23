#!/usr/bin/env python3
"""
Compare single_pass vs grouped extraction modes for both Llama and InternVL3.

This script runs both models in both modes on a small test set and summarizes
the performance differences after the scoring fixes.

Usage:
    python compare_extraction_modes.py [--images N]
"""


import argparse
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

from common.config import DATA_DIR
from common.evaluation_metrics import (
    evaluate_extraction_results,
    load_ground_truth,
)
from common.extraction_parser import discover_images


def run_model_evaluation(
    model_name: str,
    extraction_mode: str,
    limit_images: int = 2,
    debug: bool = False
) -> Dict:
    """
    Run evaluation for a specific model and extraction mode.
    
    Args:
        model_name: 'llama' or 'internvl3'
        extraction_mode: 'single_pass' or 'grouped'
        limit_images: Number of images to process
        debug: Enable debug output
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*60}")
    print(f"Testing {model_name.upper()} - {extraction_mode.upper()} mode")
    print(f"{'='*60}")
    
    # Import the appropriate processor
    if model_name == "llama":
        from models.llama_processor import LlamaProcessor
        processor_class = LlamaProcessor
    else:
        from models.internvl3_processor import InternVL3Processor
        processor_class = InternVL3Processor
    
    try:
        # Initialize processor
        print(f"🔧 Initializing {model_name} processor...")
        processor = processor_class(
            extraction_mode=extraction_mode,
            debug=debug
        )
        
        # Discover images
        images = discover_images(DATA_DIR)
        if limit_images:
            images = images[:limit_images]
        
        print(f"📷 Processing {len(images)} images...")
        
        # Process images
        start_time = time.time()
        results, batch_stats = processor.process_image_batch(images)
        processing_time = time.time() - start_time
        
        # Load ground truth
        ground_truth_map = load_ground_truth(Path(DATA_DIR) / "evaluation_ground_truth.csv")
        
        # Evaluate results
        if ground_truth_map:
            try:
                evaluation_summary = evaluate_extraction_results(results, ground_truth_map)
            except Exception as e:
                print(f"❌ Error during evaluation: {e}")
                import traceback
                traceback.print_exc()
                return {
                    "model": model_name,
                    "mode": extraction_mode,
                    "error": f"Evaluation failed: {str(e)}"
                }
            
            # Calculate additional metrics
            try:
                avg_extracted_fields = sum(r.get("extracted_fields_count", 0) for r in results) / len(results) if results else 0
                avg_response_completeness = sum(r.get("response_completeness", 0) for r in results) / len(results) if results else 0
                avg_content_coverage = sum(r.get("content_coverage", 0) for r in results) / len(results) if results else 0
            except (TypeError, KeyError) as e:
                print(f"⚠️ Error calculating averages: {e}")
                avg_extracted_fields = 0
                avg_response_completeness = 0
                avg_content_coverage = 0
            
            return {
                "model": model_name,
                "mode": extraction_mode,
                "images_processed": len(images),
                "overall_accuracy": evaluation_summary["overall_accuracy"],
                "avg_extracted_fields": avg_extracted_fields,
                "avg_response_completeness": avg_response_completeness,
                "avg_content_coverage": avg_content_coverage,
                "processing_time": processing_time,
                "successful_extractions": batch_stats["successful_extractions"],
                "perfect_documents": evaluation_summary.get("perfect_documents", 0),
            }
        else:
            print("⚠️ No ground truth available for evaluation")
            return {
                "model": model_name,
                "mode": extraction_mode,
                "images_processed": len(images),
                "overall_accuracy": 0.0,
                "avg_extracted_fields": 0,
                "avg_response_completeness": 0,
                "avg_content_coverage": 0,
                "processing_time": processing_time,
                "successful_extractions": 0,
                "perfect_documents": 0,
            }
            
    except Exception as e:
        print(f"❌ Error running {model_name} in {extraction_mode} mode: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        return {
            "model": model_name,
            "mode": extraction_mode,
            "error": str(e)
        }


def compare_results(results: List[Dict]) -> None:
    """
    Compare and summarize results across models and modes.
    
    Args:
        results: List of evaluation result dictionaries
    """
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    # Create DataFrame for easy comparison
    results_df = pd.DataFrame(results)
    
    if "error" in results_df.columns:
        # Handle any errors
        errors = results_df[results_df["error"].notna()]
        if not errors.empty:
            print("\n❌ ERRORS:")
            for _, row in errors.iterrows():
                print(f"  {row['model']} ({row['mode']}): {row['error']}")
        
        # Filter to successful runs
        results_df = results_df[results_df["error"].isna()].drop(columns=["error"])
    
    if results_df.empty:
        print("No successful evaluations to compare.")
        return
    
    # Print comparison table
    print("\n📊 PERFORMANCE METRICS:")
    print("-" * 80)
    
    # Format the DataFrame for display
    display_df = results_df[["model", "mode", "overall_accuracy", "avg_extracted_fields", 
                     "avg_content_coverage", "processing_time"]].copy()
    
    # Format percentages
    display_df["overall_accuracy"] = display_df["overall_accuracy"].apply(lambda x: f"{x:.1%}")
    display_df["avg_content_coverage"] = display_df["avg_content_coverage"].apply(lambda x: f"{x:.1%}")
    display_df["avg_extracted_fields"] = display_df["avg_extracted_fields"].apply(lambda x: f"{x:.1f}/25")
    display_df["processing_time"] = display_df["processing_time"].apply(lambda x: f"{x:.1f}s")
    
    # Rename columns for display
    display_df.columns = ["Model", "Mode", "Accuracy", "Avg Fields", "Coverage", "Time"]
    
    print(display_df.to_string(index=False))
    
    # Calculate improvements
    print("\n📈 MODE COMPARISON:")
    print("-" * 80)
    
    for model in results_df["model"].unique():
        model_df = results_df[results_df["model"] == model]
        
        single_pass = model_df[model_df["mode"] == "single_pass"].iloc[0] if not model_df[model_df["mode"] == "single_pass"].empty else None
        grouped = model_df[model_df["mode"] == "grouped"].iloc[0] if not model_df[model_df["mode"] == "grouped"].empty else None
        
        if single_pass is not None and grouped is not None:
            print(f"\n{model.upper()}:")
            
            acc_diff = grouped["overall_accuracy"] - single_pass["overall_accuracy"]
            sign = "+" if acc_diff >= 0 else ""
            print(f"  Accuracy difference: {sign}{acc_diff:.1%} (grouped vs single_pass)")
            
            cov_diff = grouped["avg_content_coverage"] - single_pass["avg_content_coverage"]
            sign = "+" if cov_diff >= 0 else ""
            print(f"  Coverage difference: {sign}{cov_diff:.1%}")
            
            time_diff = grouped["processing_time"] - single_pass["processing_time"]
            print(f"  Time difference: +{time_diff:.1f}s (grouped takes longer)")
            
            # Insights
            if acc_diff > 0:
                print(f"  ✅ Grouped mode performs BETTER for {model}")
            elif acc_diff < -0.05:
                print(f"  ⚠️ Grouped mode performs worse for {model}")
            else:
                print(f"  ≈ Similar performance between modes for {model}")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS AFTER SCORING FIX:")
    print("="*80)
    print("""
1. Content Coverage should now be similar between modes (both count N/A fields)
2. Accuracy differences show the true extraction quality gap
3. Grouped mode takes longer due to multiple extraction passes
4. The scoring fix should have narrowed the gap between modes
""")


def main():
    """Main function to run comparisons."""
    parser = argparse.ArgumentParser(
        description="Compare single_pass vs grouped extraction for both models"
    )
    parser.add_argument(
        "--images",
        type=int,
        default=2,
        help="Number of images to process (default: 2)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("EXTRACTION MODE COMPARISON TEST")
    print("After N/A scoring bias fix")
    print("="*80)
    print("\nConfiguration:")
    print(f"  Images to process: {args.images}")
    print(f"  Debug mode: {args.debug}")
    
    # Run all combinations
    results = []
    
    # Test configurations
    tests = [
        ("llama", "single_pass"),
        ("llama", "grouped"),
        ("internvl3", "single_pass"),
        ("internvl3", "grouped"),
    ]
    
    for model, mode in tests:
        result = run_model_evaluation(
            model_name=model,
            extraction_mode=mode,
            limit_images=args.images,
            debug=args.debug
        )
        results.append(result)
    
    # Compare results
    compare_results(results)
    
    print("\n✅ Comparison complete!")
    print("\nNote: Run with more images for more reliable comparisons:")
    print("  python compare_extraction_modes.py --images 10")


if __name__ == "__main__":
    main()