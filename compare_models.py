#!/usr/bin/env python3
"""
Model Comparison Dashboard Generator

This standalone script reads evaluation JSON files from Llama and InternVL3 models
and generates comparative visualizations to analyze their relative performance.

Usage:
    python compare_models.py --llama path/to/llama_results.json --internvl3 path/to/internvl3_results.json
    python compare_models.py --auto  # Auto-discover latest JSON files in output directory

Features:
- Side-by-side model performance comparison
- Field-level accuracy heatmap
- Relative strengths and weaknesses analysis
- Performance delta visualization
- Executive comparison summary

Output:
- Comparative dashboard images (PNG)
- HTML comparison report
- Summary statistics

Dependencies:
- matplotlib, seaborn, pandas, numpy
- common.config (for styling and field definitions)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Import configuration for styling and field definitions
from common.config import (
    CHART_DPI,
    OUTPUT_DIR,
    VIZ_COLORS,
)


class ModelComparator:
    """Comparative analysis and visualization generator for LMM model evaluation."""

    def __init__(self, output_dir: str = None):
        """Initialize with output directory and styling."""
        self.output_dir = Path(output_dir or OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True)
        self._setup_plotting_style()

    def _setup_plotting_style(self) -> None:
        """Configure matplotlib with professional styling for comparisons."""
        plt.style.use("default")
        plt.rcParams.update(
            {
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "axes.edgecolor": "#CCCCCC",
                "axes.linewidth": 1,
                "axes.labelcolor": VIZ_COLORS["text"],
                "text.color": VIZ_COLORS["text"],
                "xtick.color": VIZ_COLORS["text"],
                "ytick.color": VIZ_COLORS["text"],
                "font.size": 11,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "figure.titlesize": 16,
            }
        )

    def load_evaluation_results(self, json_path: str) -> Optional[Dict]:
        """Load evaluation results from JSON file."""
        try:
            with Path(json_path).open("r") as f:
                data = json.load(f)
            print(f"✅ Loaded results for {data.get('model_name', 'Unknown Model')}")
            return data
        except Exception as e:
            print(f"❌ Failed to load {json_path}: {e}")
            return None

    def _prepare_comparison_data(
        self, llama_data: Dict, internvl3_data: Dict
    ) -> Dict[str, pd.DataFrame]:
        """Prepare comparison data in pandas-friendly format."""

        # Overall metrics comparison
        overall_metrics = pd.DataFrame(
            [
                {
                    "Model": "Llama-3.2-11B",
                    "Overall_Accuracy": llama_data.get("overall_accuracy", 0) * 100,
                    "Total_Images": llama_data.get("total_images", 0),
                    "Perfect_Documents": llama_data.get("perfect_documents", 0),
                    "Best_Accuracy": llama_data.get("best_performance_accuracy", 0)
                    * 100,
                    "Worst_Accuracy": llama_data.get("worst_performance_accuracy", 0)
                    * 100,
                },
                {
                    "Model": "InternVL3-2B",
                    "Overall_Accuracy": internvl3_data.get("overall_accuracy", 0) * 100,
                    "Total_Images": internvl3_data.get("total_images", 0),
                    "Perfect_Documents": internvl3_data.get("perfect_documents", 0),
                    "Best_Accuracy": internvl3_data.get("best_performance_accuracy", 0)
                    * 100,
                    "Worst_Accuracy": internvl3_data.get(
                        "worst_performance_accuracy", 0
                    )
                    * 100,
                },
            ]
        )

        # Field-level comparison
        llama_fields = llama_data.get("field_accuracies", {})
        internvl3_fields = internvl3_data.get("field_accuracies", {})
        all_fields = sorted(set(llama_fields.keys()) | set(internvl3_fields.keys()))

        field_data = []
        for field in all_fields:
            llama_acc = llama_fields.get(field, 0) * 100
            internvl3_acc = internvl3_fields.get(field, 0) * 100

            field_data.extend(
                [
                    {"Field": field, "Model": "Llama-3.2-11B", "Accuracy": llama_acc},
                    {
                        "Field": field,
                        "Model": "InternVL3-2B",
                        "Accuracy": internvl3_acc,
                    },
                ]
            )

        field_comparison_df = pd.DataFrame(field_data)

        # Quality distribution data
        def count_quality_levels(field_accs):
            excellent = sum(1 for acc in field_accs.values() if acc >= 0.9)
            good = sum(1 for acc in field_accs.values() if 0.8 <= acc < 0.9)
            poor = sum(1 for acc in field_accs.values() if acc < 0.8)
            return excellent, good, poor

        llama_counts = count_quality_levels(llama_fields)
        internvl3_counts = count_quality_levels(internvl3_fields)

        quality_data = []
        categories = ["Excellent (≥90%)", "Good (80-90%)", "Poor (<80%)"]
        for i, category in enumerate(categories):
            quality_data.extend(
                [
                    {
                        "Category": category,
                        "Model": "Llama-3.2-11B",
                        "Count": llama_counts[i],
                    },
                    {
                        "Category": category,
                        "Model": "InternVL3-2B",
                        "Count": internvl3_counts[i],
                    },
                ]
            )

        quality_df = pd.DataFrame(quality_data)

        return {
            "overall_metrics": overall_metrics,
            "field_comparison": field_comparison_df,
            "quality_distribution": quality_df,
        }

    def create_side_by_side_comparison(
        self, llama_data: Dict, internvl3_data: Dict, save_path: Optional[str] = None
    ) -> str:
        """Create side-by-side performance comparison dashboard using seaborn."""
        print("🎨 Creating side-by-side model comparison...")

        # Prepare data in pandas-friendly format
        comparison_data = self._prepare_comparison_data(llama_data, internvl3_data)

        # Create figure with 2x2 layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Llama vs InternVL3 Performance Comparison",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # Chart 1: Overall Accuracy Comparison
        accuracy_df = comparison_data["overall_metrics"]
        sns.barplot(
            data=accuracy_df,
            x="Model",
            y="Overall_Accuracy",
            hue="Model",
            palette=[VIZ_COLORS["primary"], VIZ_COLORS["secondary"]],
            alpha=0.8,
            ax=ax1,
            legend=False,
        )
        ax1.set_title("Overall Accuracy", fontweight="bold", fontsize=12)
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_ylim(0, 100)

        # Add threshold lines
        ax1.axhline(
            y=90, color="green", linestyle="--", alpha=0.7, label="Excellent (90%)"
        )
        ax1.axhline(y=80, color="orange", linestyle="--", alpha=0.7, label="Good (80%)")
        ax1.legend()

        # Add value labels
        for i, (_, row) in enumerate(accuracy_df.iterrows()):
            ax1.text(
                i,
                row["Overall_Accuracy"] + 1,
                f"{row['Overall_Accuracy']:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

        # Chart 2: Document Processing Stats
        processing_data = []
        for _, row in accuracy_df.iterrows():
            processing_data.extend(
                [
                    {
                        "Model": row["Model"],
                        "Metric": "Total\nDocuments",
                        "Count": row["Total_Images"],
                    },
                    {
                        "Model": row["Model"],
                        "Metric": "Perfect\nDocuments",
                        "Count": row["Perfect_Documents"],
                    },
                ]
            )

        processing_df = pd.DataFrame(processing_data)

        sns.barplot(
            data=processing_df,
            x="Metric",
            y="Count",
            hue="Model",
            palette=[VIZ_COLORS["primary"], VIZ_COLORS["secondary"]],
            alpha=0.8,
            ax=ax2,
        )
        ax2.set_title("Processing Stats", fontweight="bold", fontsize=12)
        ax2.set_ylabel("Count")
        ax2.legend()

        # Add value labels
        for container in ax2.containers:
            ax2.bar_label(container, fmt="%d", fontweight="bold", fontsize=10)

        # Chart 3: Performance Quality Breakdown
        quality_df = comparison_data["quality_distribution"]

        sns.barplot(
            data=quality_df,
            x="Category",
            y="Count",
            hue="Model",
            palette=[VIZ_COLORS["primary"], VIZ_COLORS["secondary"]],
            alpha=0.8,
            ax=ax3,
        )
        ax3.set_title("Field Quality", fontweight="bold", fontsize=12)
        ax3.set_ylabel("Number of Fields")
        ax3.legend()

        # Add value labels
        for container in ax3.containers:
            ax3.bar_label(container, fmt="%d", fontweight="bold", fontsize=10)

        # Chart 4: Best vs Worst Performance
        performance_range_data = []
        for _, row in accuracy_df.iterrows():
            performance_range_data.extend(
                [
                    {
                        "Model": row["Model"],
                        "Metric": "Best Document\nAccuracy",
                        "Accuracy": row["Best_Accuracy"],
                    },
                    {
                        "Model": row["Model"],
                        "Metric": "Worst Document\nAccuracy",
                        "Accuracy": row["Worst_Accuracy"],
                    },
                ]
            )

        performance_range_df = pd.DataFrame(performance_range_data)

        sns.barplot(
            data=performance_range_df,
            x="Metric",
            y="Accuracy",
            hue="Model",
            palette=[VIZ_COLORS["primary"], VIZ_COLORS["secondary"]],
            alpha=0.8,
            ax=ax4,
        )
        ax4.set_title("Performance Range", fontweight="bold", fontsize=12)
        ax4.set_ylabel("Accuracy (%)")
        ax4.set_ylim(0, 100)
        ax4.legend()

        # Add value labels
        for container in ax4.containers:
            ax4.bar_label(container, fmt="%.1f%%", fontweight="bold", fontsize=10)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save comparison dashboard
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.output_dir / f"model_comparison_dashboard_{timestamp}.png"

        plt.savefig(save_path, dpi=CHART_DPI, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"✅ Comparison dashboard saved: {save_path}")
        return str(save_path)

    def create_field_accuracy_heatmap(
        self, llama_data: Dict, internvl3_data: Dict, save_path: Optional[str] = None
    ) -> str:
        """Create field-by-field accuracy heatmap comparison using seaborn."""
        print("🎨 Creating field accuracy heatmap...")

        # Prepare data using the same helper method
        comparison_data = self._prepare_comparison_data(llama_data, internvl3_data)
        field_df = comparison_data["field_comparison"]

        # Create pivot tables for heatmaps
        accuracy_pivot = field_df.pivot_table(
            index="Field", columns="Model", values="Accuracy"
        )

        # Calculate deltas
        delta_data = []
        for field in accuracy_pivot.index:
            llama_acc = accuracy_pivot.loc[field, "Llama-3.2-11B"]
            internvl3_acc = accuracy_pivot.loc[field, "InternVL3-2B"]
            delta = llama_acc - internvl3_acc
            delta_data.append({"Field": field, "Delta": delta})

        delta_df = pd.DataFrame(delta_data).set_index("Field")

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
        fig.suptitle(
            "Field-Level Performance Comparison", fontsize=16, fontweight="bold"
        )

        # Left plot: Side-by-side accuracy heatmap
        sns.heatmap(
            accuracy_pivot,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            center=50,
            vmin=0,
            vmax=100,
            cbar_kws={"label": "Accuracy (%)"},
            ax=ax1,
            annot_kws={"fontsize": 8, "fontweight": "bold"},
        )
        ax1.set_title("Accuracy by Field (%)", fontweight="bold")
        ax1.set_xlabel("")
        ax1.set_ylabel("Field")

        # Right plot: Performance delta heatmap
        max_delta = max(abs(delta_df["Delta"]))
        sns.heatmap(
            delta_df,
            annot=True,
            fmt="+.1f",
            cmap="RdBu_r",
            center=0,
            vmin=-max_delta,
            vmax=max_delta,
            cbar_kws={"label": "Accuracy Difference (%)"},
            ax=ax2,
            annot_kws={"fontsize": 8, "fontweight": "bold"},
        )
        ax2.set_title("Performance Delta\n(Llama - InternVL3)", fontweight="bold")
        ax2.set_xlabel("")
        ax2.set_ylabel("")
        ax2.tick_params(left=False, labelleft=False)  # Remove duplicate y-labels

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save heatmap
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.output_dir / f"field_accuracy_heatmap_{timestamp}.png"

        plt.savefig(save_path, dpi=CHART_DPI, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"✅ Field accuracy heatmap saved: {save_path}")
        return str(save_path)

    def generate_comparison_summary(
        self, llama_data: Dict, internvl3_data: Dict, save_path: Optional[str] = None
    ) -> str:
        """Generate executive comparison summary in Markdown format."""
        print("📝 Generating comparison summary...")

        # Extract key metrics
        llama_acc = llama_data.get("overall_accuracy", 0)
        internvl3_acc = internvl3_data.get("overall_accuracy", 0)
        accuracy_delta = llama_acc - internvl3_acc

        # Field analysis
        llama_fields = llama_data.get("field_accuracies", {})
        internvl3_fields = internvl3_data.get("field_accuracies", {})

        # Find best performing model per field
        llama_wins = 0
        internvl3_wins = 0
        ties = 0

        for field in set(llama_fields.keys()) | set(internvl3_fields.keys()):
            l_acc = llama_fields.get(field, 0)
            i_acc = internvl3_fields.get(field, 0)

            if abs(l_acc - i_acc) < 0.01:  # Within 1%
                ties += 1
            elif l_acc > i_acc:
                llama_wins += 1
            else:
                internvl3_wins += 1

        # Generate summary
        summary = f"""# Llama vs InternVL3 Model Comparison Report

## Executive Summary
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

### Overall Performance
- **Llama-3.2-11B-Vision:** {llama_acc:.1%} accuracy
- **InternVL3-2B:** {internvl3_acc:.1%} accuracy
- **Performance Delta:** {accuracy_delta:+.1%} ({"Llama leads" if accuracy_delta > 0 else "InternVL3 leads" if accuracy_delta < 0 else "Tied"})

### Field-Level Analysis
- **Total Fields Compared:** {len(set(llama_fields.keys()) | set(internvl3_fields.keys()))}
- **Llama Wins:** {llama_wins} fields
- **InternVL3 Wins:** {internvl3_wins} fields  
- **Ties (≤1% difference):** {ties} fields

### Document Quality
- **Llama Perfect Documents:** {llama_data.get("perfect_documents", 0)}/{llama_data.get("total_images", 0)}
- **InternVL3 Perfect Documents:** {internvl3_data.get("perfect_documents", 0)}/{internvl3_data.get("total_images", 0)}

### Performance Range
- **Llama Best/Worst:** {llama_data.get("best_performance_accuracy", 0):.1%} / {llama_data.get("worst_performance_accuracy", 0):.1%}
- **InternVL3 Best/Worst:** {internvl3_data.get("best_performance_accuracy", 0):.1%} / {internvl3_data.get("worst_performance_accuracy", 0):.1%}

## Deployment Recommendations

### Model Selection Criteria
"""

        # Add deployment recommendations based on performance
        if accuracy_delta > 0.05:  # Llama significantly better
            summary += """
✅ **Recommend Llama-3.2-11B-Vision** for production deployment
- Superior overall accuracy performance
- Better consistency across field types
- Suitable for high-accuracy requirements

⚠️ **Consider InternVL3-2B** for:
- Resource-constrained environments (4GB vs 22GB VRAM)
- High-throughput scenarios requiring speed over accuracy
- Cost-sensitive deployments
"""
        elif accuracy_delta < -0.05:  # InternVL3 significantly better
            summary += """
✅ **Recommend InternVL3-2B** for production deployment
- Superior overall accuracy performance
- More memory efficient (4GB vs 22GB VRAM)
- Better cost-performance ratio

⚠️ **Consider Llama-3.2-11B-Vision** for:
- Maximum accuracy requirements
- Complex document analysis tasks
- When computational resources are abundant
"""
        else:  # Performance is close
            summary += """
🤔 **Models show comparable accuracy** - selection based on operational factors:

✅ **Choose InternVL3-2B** for:
- Memory-constrained environments  
- High-throughput processing
- Cost optimization

✅ **Choose Llama-3.2-11B-Vision** for:
- Maximum accuracy requirements
- Resource-abundant environments
- Complex document types
"""

        summary += f"""
## Technical Specifications Comparison

| Metric | Llama-3.2-11B | InternVL3-2B |
|--------|---------------|--------------|
| Parameters | 11B | 2B |
| Memory (VRAM) | ~22GB | ~4GB |
| Processing Speed | 3-5s/doc | 1-3s/doc |
| Overall Accuracy | {llama_acc:.1%} | {internvl3_acc:.1%} |
| Perfect Documents | {llama_data.get("perfect_documents", 0)}/{llama_data.get("total_images", 0)} | {internvl3_data.get("perfect_documents", 0)}/{internvl3_data.get("total_images", 0)} |

---
*Generated by LMM POC Model Comparison Pipeline*
"""

        # Save summary
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.output_dir / f"model_comparison_summary_{timestamp}.md"

        with Path(save_path).open("w") as f:
            f.write(summary)

        print(f"✅ Comparison summary saved: {save_path}")
        return str(save_path)


def find_latest_results(output_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """Auto-discover the latest JSON evaluation files."""
    output_path = Path(output_dir)

    # Find latest Llama results
    llama_files = list(output_path.glob("llama_evaluation_results_*.json"))
    llama_file = (
        max(llama_files, key=lambda x: x.stat().st_mtime) if llama_files else None
    )

    # Find latest InternVL3 results
    internvl3_files = list(output_path.glob("internvl3_evaluation_results_*.json"))
    internvl3_file = (
        max(internvl3_files, key=lambda x: x.stat().st_mtime)
        if internvl3_files
        else None
    )

    return (
        str(llama_file) if llama_file else None,
        str(internvl3_file) if internvl3_file else None,
    )


def main():
    """Main execution function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Compare Llama and InternVL3 model evaluation results"
    )
    parser.add_argument("--llama", type=str, help="Path to Llama evaluation JSON file")
    parser.add_argument(
        "--internvl3", type=str, help="Path to InternVL3 evaluation JSON file"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-discover latest JSON files in OUTPUT_DIR",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory for comparison results (default: {OUTPUT_DIR})",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("📊 MODEL COMPARISON DASHBOARD GENERATOR")
    print("=" * 80)

    # Determine input files
    if args.auto:
        print("🔍 Auto-discovering latest evaluation results...")
        llama_file, internvl3_file = find_latest_results(args.output_dir)
    else:
        llama_file = args.llama
        internvl3_file = args.internvl3

    # Validate inputs
    if not llama_file or not internvl3_file:
        print("❌ ERROR: Could not find evaluation files")
        if args.auto:
            print("💡 No JSON files found in output directory")
            print(f"💡 Searched in: {args.output_dir}")
        else:
            print("💡 Please specify both --llama and --internvl3 file paths")
        sys.exit(1)

    if not Path(llama_file).exists():
        print(f"❌ ERROR: Llama results file not found: {llama_file}")
        sys.exit(1)

    if not Path(internvl3_file).exists():
        print(f"❌ ERROR: InternVL3 results file not found: {internvl3_file}")
        sys.exit(1)

    print(f"📁 Llama results: {llama_file}")
    print(f"📁 InternVL3 results: {internvl3_file}")

    # Initialize comparator
    comparator = ModelComparator(args.output_dir)

    # Load evaluation data
    print("\n📊 Loading evaluation results...")
    llama_data = comparator.load_evaluation_results(llama_file)
    internvl3_data = comparator.load_evaluation_results(internvl3_file)

    if not llama_data or not internvl3_data:
        print("❌ Failed to load evaluation data")
        sys.exit(1)

    # Generate comparisons
    print("\n🎨 Generating comparison visualizations...")
    generated_files = []

    try:
        # 1. Side-by-side dashboard
        dashboard_path = comparator.create_side_by_side_comparison(
            llama_data, internvl3_data
        )
        generated_files.append(dashboard_path)

        # 2. Field accuracy heatmap
        heatmap_path = comparator.create_field_accuracy_heatmap(
            llama_data, internvl3_data
        )
        generated_files.append(heatmap_path)

        # 3. Executive summary
        summary_path = comparator.generate_comparison_summary(
            llama_data, internvl3_data
        )
        generated_files.append(summary_path)

        print("\n✅ MODEL COMPARISON COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("📁 Generated files:")
        for file_path in generated_files:
            print(f"   • {Path(file_path).name}")

        # Show key insights
        llama_acc = llama_data.get("overall_accuracy", 0)
        internvl3_acc = internvl3_data.get("overall_accuracy", 0)
        delta = llama_acc - internvl3_acc

        print("\n📈 Key Insights:")
        print(f"   • Llama accuracy: {llama_acc:.1%}")
        print(f"   • InternVL3 accuracy: {internvl3_acc:.1%}")
        print(f"   • Performance delta: {delta:+.1%}")

        if abs(delta) < 0.01:
            print("   🤔 Models show comparable performance")
        elif delta > 0:
            print("   🦙 Llama leads in overall accuracy")
        else:
            print("   🔬 InternVL3 leads in overall accuracy")

    except Exception as e:
        print(f"❌ Error during comparison generation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
