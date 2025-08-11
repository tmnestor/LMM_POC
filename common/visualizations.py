"""
Advanced Visualization Module for LMM POC Model Evaluation

This module provides comprehensive visualization capabilities for vision model
evaluation results, creating professional-grade charts and dashboards suitable
for business presentation and technical analysis.

Features:
- Field accuracy visualizations with business intelligence focus
- Performance dashboards with key metrics
- Document quality distribution analysis
- Field category analysis based on business importance
- Multi-model comparison charts
- Professional styling for stakeholder presentations
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .config import (
    CHART_DPI,
    DEPLOYMENT_READY_THRESHOLD,
    EXCELLENT_FIELD_THRESHOLD,
    EXTRACTION_FIELDS,
    FIELD_DEFINITIONS,
    GOOD_FIELD_THRESHOLD,
    PILOT_READY_THRESHOLD,
    POOR_FIELD_THRESHOLD,
    VISUALIZATION_ENABLED,
    VIZ_COLORS,
    VIZ_OUTPUT_PATTERNS,
    VIZ_QUALITY_THRESHOLDS,
)


class LMMVisualizer:
    """Professional visualization generator for LMM model evaluation results."""

    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize the visualizer with output directory and styling.

        Args:
            output_dir: Directory to save generated charts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Set up professional plotting style
        self._setup_plotting_style()

        # Field categorization for business intelligence
        self.field_categories = self._categorize_fields_by_importance()

    def _setup_plotting_style(self) -> None:
        """Configure matplotlib with professional styling."""
        # Use clean, professional style
        plt.style.use("default")

        # Configure matplotlib defaults for business presentations
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

    def _categorize_fields_by_importance(self) -> Dict[str, List[str]]:
        """Categorize fields based on business importance from FIELD_DEFINITIONS."""
        categories = {
            "High Priority": [],  # Required fields
            "Standard": [],  # Important but optional
            "Supporting": [],  # Nice to have
        }

        for field_name, definition in FIELD_DEFINITIONS.items():
            if definition.get("required", False):
                categories["High Priority"].append(field_name)
            elif definition["type"] in ["monetary", "date"]:
                categories["Standard"].append(field_name)
            else:
                categories["Supporting"].append(field_name)

        return categories

    def create_field_accuracy_chart(
        self,
        evaluation_summary: Dict[str, Any],
        model_name: str,
        save_path: Optional[str] = None,
    ) -> str:
        """
        Create field-wise accuracy bar chart with color coding.

        Args:
            evaluation_summary: Evaluation results from evaluate_extraction_results()
            model_name: Model name for file naming
            save_path: Optional custom save path

        Returns:
            Path to saved chart file
        """
        if not VISUALIZATION_ENABLED:
            return ""

        print(f"🎨 Creating field accuracy chart for {model_name}...")

        # Extract field accuracies and sort by performance
        field_accuracies = evaluation_summary.get("field_accuracies", {})
        if not field_accuracies:
            print("❌ No field accuracy data available")
            return ""

        # Sort fields by accuracy for better visualization
        sorted_fields = sorted(
            field_accuracies.items(), key=lambda x: x[1], reverse=True
        )

        # Create figure with appropriate size for 25 fields
        fig, ax = plt.subplots(figsize=(16, 10))

        # Prepare data
        fields = [field for field, _ in sorted_fields]
        accuracies = [acc * 100 for _, acc in sorted_fields]  # Convert to percentage

        # Color code bars based on performance thresholds
        colors = []
        for acc in accuracies:
            if acc >= VIZ_QUALITY_THRESHOLDS["excellent"] * 100:
                colors.append(VIZ_COLORS["success"])
            elif acc >= VIZ_QUALITY_THRESHOLDS["good"] * 100:
                colors.append(VIZ_COLORS["info"])
            else:
                colors.append(VIZ_COLORS["warning"])

        # Create horizontal bar chart for better field name readability
        bars = ax.barh(range(len(fields)), accuracies, color=colors, alpha=0.8)

        # Customize chart
        ax.set_yticks(range(len(fields)))
        ax.set_yticklabels(fields)
        ax.set_xlabel("Accuracy (%)", fontweight="bold")
        ax.set_title(
            f"Field-wise Accuracy Performance\n{model_name.upper()} - {len(fields)} Fields",
            fontweight="bold",
            pad=20,
        )

        # Add accuracy value labels on bars
        for bar, acc in zip(bars, accuracies, strict=False):
            ax.text(
                bar.get_width() + 1,
                bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f}%",
                ha="left",
                va="center",
                fontweight="bold",
                fontsize=9,
            )

        # Add threshold reference lines
        ax.axvline(
            x=VIZ_QUALITY_THRESHOLDS["excellent"] * 100,
            color="green",
            linestyle="--",
            alpha=0.7,
            label=f"Excellent ({VIZ_QUALITY_THRESHOLDS['excellent']:.0%}+)",
        )
        ax.axvline(
            x=VIZ_QUALITY_THRESHOLDS["good"] * 100,
            color="orange",
            linestyle="--",
            alpha=0.7,
            label=f"Good ({VIZ_QUALITY_THRESHOLDS['good']:.0%}+)",
        )

        ax.set_xlim(0, 105)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()

        # Save chart
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = VIZ_OUTPUT_PATTERNS["field_accuracy"].format(
                model=model_name, timestamp=timestamp
            )
            save_path = self.output_dir / filename
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=CHART_DPI, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"✅ Field accuracy chart saved: {save_path}")
        return str(save_path)

    def create_performance_dashboard(
        self,
        evaluation_summary: Dict[str, Any],
        batch_statistics: Dict[str, Any],
        model_name: str,
        save_path: Optional[str] = None,
    ) -> str:
        """
        Create comprehensive 2x2 performance dashboard.

        Args:
            evaluation_summary: Evaluation results
            batch_statistics: Processing statistics
            model_name: Model name for file naming
            save_path: Optional custom save path

        Returns:
            Path to saved dashboard file
        """
        if not VISUALIZATION_ENABLED:
            return ""

        print(f"🎨 Creating performance dashboard for {model_name}...")

        # Create 2x2 subplot dashboard
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f"{model_name.upper()} Performance Dashboard",
            fontsize=18,
            fontweight="bold",
            y=0.95,
        )

        # Chart 1: Overall Accuracy vs Thresholds
        overall_acc = evaluation_summary.get("overall_accuracy", 0) * 100

        bars1 = ax1.bar(
            ["Model Performance"], [overall_acc], color=VIZ_COLORS["primary"], alpha=0.8
        )

        # Add threshold lines
        ax1.axhline(
            y=DEPLOYMENT_READY_THRESHOLD * 100,
            color="green",
            linestyle="--",
            alpha=0.7,
            label=f"Production Ready ({DEPLOYMENT_READY_THRESHOLD:.0%})",
        )
        ax1.axhline(
            y=PILOT_READY_THRESHOLD * 100,
            color="orange",
            linestyle="--",
            alpha=0.7,
            label=f"Pilot Ready ({PILOT_READY_THRESHOLD:.0%})",
        )

        ax1.set_title("Overall Accuracy", fontweight="bold")
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_ylim(0, 100)
        ax1.legend()

        # Add value label
        ax1.text(
            0,
            overall_acc + 2,
            f"{overall_acc:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=14,
        )

        # Chart 2: Processing Speed
        avg_time = batch_statistics.get("average_processing_time", 0)
        throughput = 60.0 / avg_time if avg_time > 0 else 0

        bars2 = ax2.bar(
            ["Avg Time/Image", "Throughput"],
            [avg_time, throughput],
            color=[VIZ_COLORS["info"], VIZ_COLORS["success"]],
            alpha=0.8,
        )

        ax2.set_title("Processing Performance", fontweight="bold")
        ax2.set_ylabel("Time (s) | Images/min")

        # Add value labels
        ax2.text(
            0,
            avg_time + 0.5,
            f"{avg_time:.1f}s",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
        ax2.text(
            1,
            throughput + 0.5,
            f"{throughput:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

        # Chart 3: Document Quality Distribution
        evaluation_data = evaluation_summary.get("evaluation_data", [])
        if evaluation_data:
            perfect_docs = sum(
                1 for doc in evaluation_data if doc["overall_accuracy"] >= 0.99
            )
            good_docs = sum(
                1 for doc in evaluation_data if 0.8 <= doc["overall_accuracy"] < 0.99
            )
            fair_docs = sum(
                1 for doc in evaluation_data if 0.6 <= doc["overall_accuracy"] < 0.8
            )
            poor_docs = sum(
                1 for doc in evaluation_data if doc["overall_accuracy"] < 0.6
            )

            categories = [
                "Perfect\n(99%+)",
                "Good\n(80-99%)",
                "Fair\n(60-80%)",
                "Poor\n(<60%)",
            ]
            counts = [perfect_docs, good_docs, fair_docs, poor_docs]
            colors = [
                VIZ_COLORS["success"],
                VIZ_COLORS["info"],
                VIZ_COLORS["secondary"],
                VIZ_COLORS["warning"],
            ]

            bars3 = ax3.bar(categories, counts, color=colors, alpha=0.8)
            ax3.set_title("Document Quality Distribution", fontweight="bold")
            ax3.set_ylabel("Number of Documents")

            # Add count labels
            for bar, count in zip(bars3, counts, strict=False):
                if count > 0:
                    ax3.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.1,
                        str(count),
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )

        # Chart 4: Field Performance Summary
        field_accuracies = evaluation_summary.get("field_accuracies", {})
        if field_accuracies:
            excellent_count = sum(
                1
                for acc in field_accuracies.values()
                if acc >= EXCELLENT_FIELD_THRESHOLD
            )
            good_count = sum(
                1
                for acc in field_accuracies.values()
                if GOOD_FIELD_THRESHOLD <= acc < EXCELLENT_FIELD_THRESHOLD
            )
            poor_count = sum(
                1 for acc in field_accuracies.values() if acc < POOR_FIELD_THRESHOLD
            )

            field_categories = ["Excellent\nFields", "Good\nFields", "Poor\nFields"]
            field_counts = [excellent_count, good_count, poor_count]

            bars4 = ax4.bar(
                field_categories,
                field_counts,
                color=[
                    VIZ_COLORS["success"],
                    VIZ_COLORS["info"],
                    VIZ_COLORS["warning"],
                ],
                alpha=0.8,
            )

            ax4.set_title(
                f"Field Performance Summary\n(of {len(field_accuracies)} total fields)",
                fontweight="bold",
            )
            ax4.set_ylabel("Number of Fields")

            # Add count labels
            for bar, count in zip(bars4, field_counts, strict=False):
                if count > 0:
                    ax4.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.1,
                        str(count),
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )

        plt.tight_layout()

        # Save dashboard
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = VIZ_OUTPUT_PATTERNS["performance_dashboard"].format(
                model=model_name, timestamp=timestamp
            )
            save_path = self.output_dir / filename
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=CHART_DPI, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"✅ Performance dashboard saved: {save_path}")
        return str(save_path)

    def create_field_category_analysis(
        self,
        evaluation_summary: Dict[str, Any],
        model_name: str,
        save_path: Optional[str] = None,
    ) -> str:
        """
        Create field category performance analysis based on business importance.

        Args:
            evaluation_summary: Evaluation results
            model_name: Model name for file naming
            save_path: Optional custom save path

        Returns:
            Path to saved chart file
        """
        if not VISUALIZATION_ENABLED:
            return ""

        print(f"🎨 Creating field category analysis for {model_name}...")

        field_accuracies = evaluation_summary.get("field_accuracies", {})
        if not field_accuracies:
            print("❌ No field accuracy data available for category analysis")
            return ""

        # Calculate average accuracy per category
        category_performance = {}
        for category, fields in self.field_categories.items():
            if not fields:
                continue

            category_accs = []
            for field in fields:
                if field in field_accuracies:
                    category_accs.append(field_accuracies[field])

            if category_accs:
                category_performance[category] = np.mean(category_accs) * 100

        if not category_performance:
            print("❌ No category performance data available")
            return ""

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(
            f"Field Category Performance Analysis\n{model_name.upper()}",
            fontsize=16,
            fontweight="bold",
            y=0.95,
        )

        # Left chart: Category performance bars
        categories = list(category_performance.keys())
        performances = list(category_performance.values())

        bars = ax1.bar(
            categories,
            performances,
            color=[VIZ_COLORS["primary"], VIZ_COLORS["secondary"], VIZ_COLORS["info"]],
            alpha=0.8,
        )

        ax1.set_title("Average Accuracy by Category", fontweight="bold")
        ax1.set_ylabel("Average Accuracy (%)")
        ax1.set_ylim(0, 100)

        # Add value labels
        for bar, perf in zip(bars, performances, strict=False):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{perf:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Add threshold lines
        ax1.axhline(
            y=VIZ_QUALITY_THRESHOLDS["excellent"] * 100,
            color="green",
            linestyle="--",
            alpha=0.7,
            label="Excellent",
        )
        ax1.axhline(
            y=VIZ_QUALITY_THRESHOLDS["good"] * 100,
            color="orange",
            linestyle="--",
            alpha=0.7,
            label="Good",
        )
        ax1.legend()

        # Right chart: Field count per category
        category_counts = [
            len(fields) for fields in self.field_categories.values() if fields
        ]

        pie_colors = [
            VIZ_COLORS["primary"],
            VIZ_COLORS["secondary"],
            VIZ_COLORS["info"],
        ]
        wedges, texts, autotexts = ax2.pie(
            category_counts,
            labels=categories,
            colors=pie_colors,
            autopct="%1.0f",
            startangle=90,
        )

        ax2.set_title("Field Distribution by Category", fontweight="bold")

        # Enhance pie chart text
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
            autotext.set_fontsize(12)

        plt.tight_layout()

        # Save chart
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = VIZ_OUTPUT_PATTERNS["field_category"].format(timestamp=timestamp)
            save_path = self.output_dir / filename
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=CHART_DPI, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"✅ Field category analysis saved: {save_path}")
        return str(save_path)

    def generate_model_visualizations(
        self,
        evaluation_summary: Dict[str, Any],
        batch_statistics: Dict[str, Any],
        model_name: str,
    ) -> List[str]:
        """
        Generate complete visualization suite for a single model.

        Args:
            evaluation_summary: Evaluation results from evaluate_extraction_results()
            batch_statistics: Processing statistics from model evaluation
            model_name: Model name for file naming

        Returns:
            List of paths to generated visualization files
        """
        if not VISUALIZATION_ENABLED:
            print("📊 Visualizations disabled in configuration")
            return []

        print(f"🎨 Generating complete visualization suite for {model_name}...")

        generated_files = []

        try:
            # 1. Field accuracy chart
            field_chart = self.create_field_accuracy_chart(
                evaluation_summary, model_name
            )
            if field_chart:
                generated_files.append(field_chart)

            # 2. Performance dashboard
            dashboard = self.create_performance_dashboard(
                evaluation_summary, batch_statistics, model_name
            )
            if dashboard:
                generated_files.append(dashboard)

            # 3. Field category analysis
            category_chart = self.create_field_category_analysis(
                evaluation_summary, model_name
            )
            if category_chart:
                generated_files.append(category_chart)

            print(
                f"✅ Generated {len(generated_files)} visualizations for {model_name}"
            )

        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")

        return generated_files

    def create_html_summary(
        self,
        evaluation_summary: Dict[str, Any],
        model_name: str,
        visualization_paths: List[str],
        save_path: Optional[str] = None,
    ) -> str:
        """
        Create HTML summary report with embedded visualizations.

        Args:
            evaluation_summary: Evaluation results
            model_name: Model name for display
            visualization_paths: List of paths to visualization files
            save_path: Optional custom save path

        Returns:
            Path to saved HTML file
        """
        if not VISUALIZATION_ENABLED or not visualization_paths:
            return ""

        print(f"📄 Creating HTML summary report for {model_name}...")

        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{model_name.upper()} Evaluation Report</title>
            <style>
                body {{
                    font-family: 'Arial', sans-serif;
                    margin: 40px;
                    background-color: #f8f9fa;
                    color: #2c3e50;
                }}
                .header {{
                    background: linear-gradient(135deg, {VIZ_COLORS["primary"]}, {VIZ_COLORS["secondary"]});
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                }}
                h1 {{ margin: 0; font-size: 2.5em; }}
                h2 {{ color: {VIZ_COLORS["primary"]}; border-bottom: 2px solid {VIZ_COLORS["primary"]}; padding-bottom: 10px; }}
                .metric-card {{
                    background: white;
                    padding: 20px;
                    margin: 15px 0;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    border-left: 5px solid {VIZ_COLORS["primary"]};
                }}
                .metric-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: {VIZ_COLORS["primary"]};
                }}
                .visualization {{
                    text-align: center;
                    margin: 40px 0;
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .visualization img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 5px;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 50px;
                    color: #7f8c8d;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{model_name.upper()} Evaluation Report</h1>
                <p>Generated on {datetime.now().strftime("%Y-%m-%d at %H:%M:%S")}</p>
            </div>
            
            <h2>📊 Key Performance Metrics</h2>
            
            <div class="metric-card">
                <h3>Overall Accuracy</h3>
                <div class="metric-value">{evaluation_summary.get("overall_accuracy", 0):.1%}</div>
                <p>Average accuracy across all {evaluation_summary.get("total_images", 0)} documents processed</p>
            </div>
            
            <div class="metric-card">
                <h3>Document Processing</h3>
                <div class="metric-value">{evaluation_summary.get("total_images", 0)}</div>
                <p>Total documents evaluated with {evaluation_summary.get("perfect_documents", 0)} achieving perfect scores</p>
            </div>
            
            <div class="metric-card">
                <h3>Field Extraction</h3>
                <div class="metric-value">{len(evaluation_summary.get("field_accuracies", {}))}</div>
                <p>Fields analyzed for structured data extraction performance</p>
            </div>
        """

        # Add visualizations
        if visualization_paths:
            html_content += "<h2>📈 Detailed Analysis</h2>\n"

            for viz_path in visualization_paths:
                viz_name = Path(viz_path).stem.replace("_", " ").title()
                html_content += f"""
                <div class="visualization">
                    <h3>{viz_name}</h3>
                    <img src="{Path(viz_path).name}" alt="{viz_name}">
                </div>
                """

        # Add footer
        html_content += f"""
            <div class="footer">
                <p>Report generated by LMM POC Evaluation Pipeline</p>
                <p>Configuration-driven analysis with {len(EXTRACTION_FIELDS)} business document fields</p>
            </div>
        </body>
        </html>
        """

        # Save HTML report
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = VIZ_OUTPUT_PATTERNS["html_summary"].format(timestamp=timestamp)
            save_path = self.output_dir / filename
        else:
            save_path = Path(save_path)

        with save_path.open("w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"✅ HTML summary report saved: {save_path}")
        return str(save_path)
