#!/usr/bin/env python3
"""
Compare 8-group vs 6-group extraction strategies for both Llama and InternVL3.

This script tests the new research-backed 6-group strategy against the proven 
8-group approach to validate cognitive load and performance improvements.

Usage:
    python compare_grouping_strategies.py [--images N] [--model MODEL]
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

from common.config import DATA_DIR
from common.evaluation_utils import (
    discover_images,
    evaluate_extraction_results,
    load_ground_truth,
)


def run_grouping_strategy_evaluation(
    model_name: str,
    grouping_strategy: str,
    limit_images: int = 2,
    debug: bool = False
) -> Dict:
    """
    Run evaluation for a specific model and grouping strategy.
    
    Args:
        model_name: 'llama' or 'internvl3'
        grouping_strategy: '8_groups' or '6_groups'
        limit_images: Number of images to process
        debug: Enable debug output
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*60}")
    print(f"Testing {model_name.upper()} - {grouping_strategy.upper()} strategy")
    print(f"{'='*60}")
    
    # Import the appropriate processor
    if model_name == "llama":
        from models.llama_processor import LlamaProcessor
        processor_class = LlamaProcessor
    else:
        from models.internvl3_processor import InternVL3Processor
        processor_class = InternVL3Processor
    
    try:
        # Initialize processor with specific grouping strategy
        print(f"🔧 Initializing {model_name} processor with {grouping_strategy}...")
        processor = processor_class(
            extraction_mode="grouped",
            debug=debug,
            grouping_strategy=grouping_strategy  # New parameter
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
                    "strategy": grouping_strategy,
                    "error": f"Evaluation failed: {str(e)}"
                }
            
            # Calculate additional metrics
            try:
                avg_extracted_fields = sum(r.get("extracted_fields_count", 0) for r in results) / len(results) if results else 0
                avg_response_completeness = sum(r.get("response_completeness", 0) for r in results) / len(results) if results else 0
                avg_content_coverage = sum(r.get("content_coverage", 0) for r in results) / len(results) if results else 0
                groups_processed = batch_stats.get("total_groups_processed", 0)
            except (TypeError, KeyError) as e:
                print(f"⚠️ Error calculating averages: {e}")
                avg_extracted_fields = 0
                avg_response_completeness = 0
                avg_content_coverage = 0
                groups_processed = 0
            
            return {
                "model": model_name,
                "strategy": grouping_strategy,
                "images_processed": len(images),
                "overall_accuracy": evaluation_summary["overall_accuracy"],
                "avg_extracted_fields": avg_extracted_fields,
                "avg_response_completeness": avg_response_completeness,
                "avg_content_coverage": avg_content_coverage,
                "processing_time": processing_time,
                "successful_extractions": batch_stats["successful_extractions"],
                "perfect_documents": evaluation_summary.get("perfect_documents", 0),
                "groups_processed": groups_processed,
            }
        else:
            print("⚠️ No ground truth available for evaluation")
            return {
                "model": model_name,
                "strategy": grouping_strategy,
                "images_processed": len(images),
                "overall_accuracy": 0.0,
                "avg_extracted_fields": 0,
                "avg_response_completeness": 0,
                "avg_content_coverage": 0,
                "processing_time": processing_time,
                "successful_extractions": 0,
                "perfect_documents": 0,
                "groups_processed": 0,
            }
            
    except Exception as e:
        print(f"❌ Error running {model_name} with {grouping_strategy}: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        return {
            "model": model_name,
            "strategy": grouping_strategy,
            "error": str(e)
        }


def compare_strategies(results: List[Dict]) -> None:
    """
    Compare and summarize results across grouping strategies.
    
    Args:
        results: List of evaluation result dictionaries
    """
    print("\n" + "="*80)
    print("GROUPING STRATEGY COMPARISON")
    print("="*80)
    
    # Create DataFrame for easy comparison
    results_df = pd.DataFrame(results)
    
    if "error" in results_df.columns:
        # Handle any errors
        errors = results_df[results_df["error"].notna()]
        if not errors.empty:
            print("\n❌ ERRORS:")
            for _, row in errors.iterrows():
                print(f"  {row['model']} ({row['strategy']}): {row['error']}")
        
        # Filter to successful runs
        results_df = results_df[results_df["error"].isna()].drop(columns=["error"])
    
    if results_df.empty:
        print("No successful evaluations to compare.")
        return
    
    # Print comparison table
    print("\n📊 PERFORMANCE METRICS:")
    print("-" * 80)
    
    # Format the DataFrame for display
    display_df = results_df[["model", "strategy", "overall_accuracy", "avg_extracted_fields", 
                     "avg_content_coverage", "processing_time", "groups_processed"]].copy()
    
    # Format percentages and values
    display_df["overall_accuracy"] = display_df["overall_accuracy"].apply(lambda x: f"{x:.1%}")
    display_df["avg_content_coverage"] = display_df["avg_content_coverage"].apply(lambda x: f"{x:.1%}")
    display_df["avg_extracted_fields"] = display_df["avg_extracted_fields"].apply(lambda x: f"{x:.1f}/25")
    display_df["processing_time"] = display_df["processing_time"].apply(lambda x: f"{x:.1f}s")
    
    # Rename columns for display
    display_df.columns = ["Model", "Strategy", "Accuracy", "Avg Fields", "Coverage", "Time", "Groups"]
    
    print(display_df.to_string(index=False))
    
    # Calculate improvements
    print("\n📈 STRATEGY COMPARISON:")
    print("-" * 80)
    
    for model in results_df["model"].unique():
        model_df = results_df[results_df["model"] == model]
        
        eight_groups = model_df[model_df["strategy"] == "8_groups"].iloc[0] if not model_df[model_df["strategy"] == "8_groups"].empty else None
        six_groups = model_df[model_df["strategy"] == "6_groups"].iloc[0] if not model_df[model_df["strategy"] == "6_groups"].empty else None
        
        if eight_groups is not None and six_groups is not None:
            print(f"\n{model.upper()}:")
            
            acc_diff = six_groups["overall_accuracy"] - eight_groups["overall_accuracy"]
            sign = "+" if acc_diff >= 0 else ""
            print(f"  Accuracy difference: {sign}{acc_diff:.1%} (6-groups vs 8-groups)")
            
            cov_diff = six_groups["avg_content_coverage"] - eight_groups["avg_content_coverage"]
            sign = "+" if cov_diff >= 0 else ""
            print(f"  Coverage difference: {sign}{cov_diff:.1%}")
            
            time_diff = six_groups["processing_time"] - eight_groups["processing_time"]
            sign = "+" if time_diff > 0 else ""
            print(f"  Time difference: {sign}{time_diff:.1f}s")
            
            group_reduction = eight_groups["groups_processed"] - six_groups["groups_processed"]
            print(f"  Group reduction: -{group_reduction} groups ({eight_groups['groups_processed']} → {six_groups['groups_processed']})")
            
            # Performance insights
            if acc_diff > 0.02:
                print(f"  ✅ 6-group strategy performs BETTER for {model}")
            elif acc_diff < -0.02:
                print(f"  ⚠️ 6-group strategy performs worse for {model}")
            else:
                print(f"  ≈ Similar performance between strategies for {model}")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS - COGNITIVE LOAD OPTIMIZATION:")
    print("="*80)
    print("""
1. 6-group strategy reduces cognitive load through better field relationships
2. Fewer groups = fewer model calls = faster processing
3. Research-backed entity grouping should improve field correlation
4. Azure v4.0 inspired taxonomy balances logical and mental models
""")


def main():
    """Main function to run strategy comparisons."""
    parser = argparse.ArgumentParser(
        description="Compare 8-group vs 6-group extraction strategies"
    )
    parser.add_argument(
        "--images",
        type=int,
        default=2,
        help="Number of images to process (default: 2)"
    )
    parser.add_argument(
        "--model",
        choices=["llama", "internvl3", "both"],
        default="both",
        help="Model to test (default: both)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("GROUPING STRATEGY COMPARISON TEST")
    print("8-groups (proven) vs 6-groups (research-optimized)")
    print("="*80)
    print("\nConfiguration:")
    print(f"  Images to process: {args.images}")
    print(f"  Model(s): {args.model}")
    print(f"  Debug mode: {args.debug}")
    
    # Run all combinations
    results = []
    
    # Determine models to test
    models = ["llama", "internvl3"] if args.model == "both" else [args.model]
    strategies = ["8_groups", "6_groups"]
    
    for model in models:
        for strategy in strategies:
            result = run_grouping_strategy_evaluation(
                model_name=model,
                grouping_strategy=strategy,
                limit_images=args.images,
                debug=args.debug
            )
            results.append(result)
    
    # Compare results
    compare_strategies(results)
    
    print("\n✅ Strategy comparison complete!")
    print("\nNote: Run with more images for more reliable comparisons:")
    print("  python compare_grouping_strategies.py --images 10")


if __name__ == "__main__":
    main()