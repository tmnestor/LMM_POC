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
import pandas as pd
import seaborn as sns

from .config import (
    CHART_DPI,
    CHART_SIZES,
    DEPLOYMENT_READY_THRESHOLD,
    EXCELLENT_FIELD_THRESHOLD,
    EXTRACTION_FIELDS,
    FIELD_TYPES,
    GOOD_FIELD_THRESHOLD,
    PILOT_READY_THRESHOLD,
    POOR_FIELD_THRESHOLD,
    VISUALIZATION_ENABLED,
    VIZ_COLORS,
    VIZ_OUTPUT_PATTERNS,
    VIZ_QUALITY_THRESHOLDS,
)
from .evaluation_metrics import generate_overall_classification_summary


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
        """Categorize fields based on data type for visualization grouping."""
        categories = {
            "Financial": [],  # Monetary fields
            "Identification": [],  # IDs and business info
            "Supporting": [],  # Everything else
        }

        for field_name in EXTRACTION_FIELDS:
            field_type = FIELD_TYPES.get(field_name, "text")
            if field_type == "monetary":
                categories["Financial"].append(field_name)
            elif field_type in ["numeric_id", "text"] and any(
                term in field_name.lower()
                for term in ["abn", "business", "supplier", "name"]
            ):
                categories["Identification"].append(field_name)
            else:
                categories["Supporting"].append(field_name)

        return categories

    def _prepare_dashboard_data(
        self, evaluation_summary: Dict[str, Any], batch_statistics: Dict[str, Any]
    ) -> Dict[str, pd.DataFrame]:
        """Prepare dashboard data in pandas format."""

        # Overall performance metrics
        overall_acc = evaluation_summary.get("overall_accuracy", 0) * 100
        performance_data = pd.DataFrame(
            [
                {
                    "Metric": "Model Performance",
                    "Value": overall_acc,
                    "Category": "Accuracy",
                }
            ]
        )

        # Processing performance
        avg_time = batch_statistics.get("average_processing_time", 0)
        throughput = 60.0 / avg_time if avg_time > 0 else 0
        processing_data = pd.DataFrame(
            [
                {"Metric": "Avg Time/Image", "Value": avg_time, "Unit": "seconds"},
                {"Metric": "Throughput", "Value": throughput, "Unit": "images/min"},
            ]
        )

        # Document quality distribution
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

            quality_data = pd.DataFrame(
                [
                    {
                        "Quality": "Perfect\n(99%+)",
                        "Count": perfect_docs,
                        "Color": VIZ_COLORS["success"],
                    },
                    {
                        "Quality": "Good\n(80-99%)",
                        "Count": good_docs,
                        "Color": VIZ_COLORS["info"],
                    },
                    {
                        "Quality": "Fair\n(60-80%)",
                        "Count": fair_docs,
                        "Color": VIZ_COLORS["secondary"],
                    },
                    {
                        "Quality": "Poor\n(<60%)",
                        "Count": poor_docs,
                        "Color": VIZ_COLORS["warning"],
                    },
                ]
            )
        else:
            quality_data = pd.DataFrame()

        # Field performance summary
        field_accuracies = evaluation_summary.get("field_accuracies", {})
        if field_accuracies:
            excellent_count = sum(
                1
                for acc in field_accuracies.values()
                if acc["accuracy"] >= EXCELLENT_FIELD_THRESHOLD
            )
            good_count = sum(
                1
                for acc in field_accuracies.values()
                if GOOD_FIELD_THRESHOLD <= acc["accuracy"] < EXCELLENT_FIELD_THRESHOLD
            )
            poor_count = sum(
                1
                for acc in field_accuracies.values()
                if acc["accuracy"] < POOR_FIELD_THRESHOLD
            )

            field_summary_data = pd.DataFrame(
                [
                    {
                        "Category": "Excellent\nFields",
                        "Count": excellent_count,
                        "Color": VIZ_COLORS["success"],
                    },
                    {
                        "Category": "Good\nFields",
                        "Count": good_count,
                        "Color": VIZ_COLORS["info"],
                    },
                    {
                        "Category": "Poor\nFields",
                        "Count": poor_count,
                        "Color": VIZ_COLORS["warning"],
                    },
                ]
            )
        else:
            field_summary_data = pd.DataFrame()

        return {
            "performance": performance_data,
            "processing": processing_data,
            "quality": quality_data,
            "field_summary": field_summary_data,
        }

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

        # Extract field accuracies and convert to DataFrame
        field_accuracies = evaluation_summary.get("field_accuracies", {})
        if not field_accuracies:
            print("❌ No field accuracy data available")
            return ""

        # Create DataFrame with field data
        field_data = []
        for field, accuracy in field_accuracies.items():
            acc_pct = accuracy["accuracy"] * 100

            # Determine quality category for color coding
            if acc_pct >= VIZ_QUALITY_THRESHOLDS["excellent"] * 100:
                quality = "Excellent"
                color = VIZ_COLORS["success"]
            elif acc_pct >= VIZ_QUALITY_THRESHOLDS["good"] * 100:
                quality = "Good"
                color = VIZ_COLORS["info"]
            else:
                quality = "Poor"
                color = VIZ_COLORS["warning"]

            field_data.append(
                {
                    "Field": field,
                    "Accuracy": acc_pct,
                    "Quality": quality,
                    "Color": color,
                }
            )

        field_df = pd.DataFrame(field_data)
        field_df = field_df.sort_values(
            "Accuracy", ascending=True
        )  # Ascending for horizontal bars

        # Create figure
        fig, ax = plt.subplots(figsize=CHART_SIZES["field_accuracy"])

        # Create horizontal bar plot with seaborn
        sns.barplot(
            data=field_df,
            y="Field",
            x="Accuracy",
            hue="Field",
            palette=field_df["Color"].tolist(),
            alpha=0.8,
            ax=ax,
            legend=False,
        )

        # Customize chart
        ax.set_xlabel("Accuracy (%)", fontweight="bold")
        ax.set_ylabel("")
        ax.set_title(
            f"Field-wise Accuracy Performance\n{model_name.upper()} - {len(field_df)} Fields",
            fontweight="bold",
            pad=20,
        )

        # Add accuracy value labels on bars
        for i, (_, row) in enumerate(field_df.iterrows()):
            ax.text(
                row["Accuracy"] + 1,
                i,
                f"{row['Accuracy']:.1f}%",
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

        # Prepare data using pandas
        dashboard_data = self._prepare_dashboard_data(
            evaluation_summary, batch_statistics
        )

        # Create 2x2 subplot dashboard
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, figsize=CHART_SIZES["performance_dashboard"]
        )
        fig.suptitle(
            f"{model_name.upper()} Performance Dashboard",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # Chart 1: Overall Accuracy vs Thresholds
        performance_df = dashboard_data["performance"]
        if not performance_df.empty:
            overall_acc = performance_df.iloc[0]["Value"]

            sns.barplot(
                data=performance_df,
                x="Metric",
                y="Value",
                color=VIZ_COLORS["primary"],
                alpha=0.8,
                ax=ax1,
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

            ax1.set_title("Overall Accuracy", fontweight="bold", fontsize=12)
            ax1.set_xlabel("")
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
                fontsize=12,
            )

        # Chart 2: Processing Speed
        processing_df = dashboard_data["processing"]
        if not processing_df.empty:
            # Create custom palette for different metrics
            processing_colors = [VIZ_COLORS["info"], VIZ_COLORS["success"]]

            sns.barplot(
                data=processing_df,
                x="Metric",
                y="Value",
                hue="Metric",
                palette=processing_colors,
                alpha=0.8,
                ax=ax2,
                legend=False,
            )

            ax2.set_title("Processing Performance", fontweight="bold", fontsize=12)
            ax2.set_xlabel("")
            ax2.set_ylabel("Time (s) | Images/min")

            # Add value labels with units
            for i, (_, row) in enumerate(processing_df.iterrows()):
                unit_suffix = "s" if "Time" in row["Metric"] else ""
                ax2.text(
                    i,
                    row["Value"] + 0.5,
                    f"{row['Value']:.1f}{unit_suffix}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=10,
                )

        # Chart 3: Document Quality Distribution
        quality_df = dashboard_data["quality"]
        if not quality_df.empty:
            sns.barplot(
                data=quality_df,
                x="Quality",
                y="Count",
                hue="Quality",
                palette=quality_df["Color"].tolist(),
                alpha=0.8,
                ax=ax3,
                legend=False,
            )

            ax3.set_title(
                "Document Quality Distribution", fontweight="bold", fontsize=12
            )
            ax3.set_xlabel("")
            ax3.set_ylabel("Number of Documents")

            # Add count labels
            for i, (_, row) in enumerate(quality_df.iterrows()):
                if row["Count"] > 0:
                    ax3.text(
                        i,
                        row["Count"] + 0.1,
                        str(int(row["Count"])),
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                        fontsize=10,
                    )

        # Chart 4: Field Performance Summary
        field_summary_df = dashboard_data["field_summary"]
        if not field_summary_df.empty:
            total_fields = len(evaluation_summary.get("field_accuracies", {}))

            sns.barplot(
                data=field_summary_df,
                x="Category",
                y="Count",
                hue="Category",
                palette=field_summary_df["Color"].tolist(),
                alpha=0.8,
                ax=ax4,
                legend=False,
            )

            ax4.set_title(
                f"Field Performance Summary\n(of {total_fields} total fields)",
                fontweight="bold",
                fontsize=12,
            )
            ax4.set_xlabel("")
            ax4.set_ylabel("Number of Fields")

            # Add count labels
            for i, (_, row) in enumerate(field_summary_df.iterrows()):
                if row["Count"] > 0:
                    ax4.text(
                        i,
                        row["Count"] + 0.1,
                        str(int(row["Count"])),
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                        fontsize=10,
                    )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

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

        # Prepare data in pandas format
        category_data = []
        field_count_data = []

        for category, fields in self.field_categories.items():
            if not fields:
                continue

            category_accs = []
            for field in fields:
                if field in field_accuracies:
                    category_accs.append(field_accuracies[field]["accuracy"])

            if category_accs:
                avg_accuracy = np.mean(category_accs) * 100
                category_data.append(
                    {
                        "Category": category,
                        "Average_Accuracy": avg_accuracy,
                        "Field_Count": len(fields),
                    }
                )

        if not category_data:
            print("❌ No category performance data available")
            return ""

        category_df = pd.DataFrame(category_data)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=CHART_SIZES["field_category"])
        fig.suptitle(
            f"Field Category Performance Analysis\n{model_name.upper()}",
            fontsize=16,
            fontweight="bold",
            y=0.95,
        )

        # Left chart: Category performance bars
        category_colors = [
            VIZ_COLORS["primary"],
            VIZ_COLORS["secondary"],
            VIZ_COLORS["info"],
        ]

        sns.barplot(
            data=category_df,
            x="Category",
            y="Average_Accuracy",
            hue="Category",
            palette=category_colors,
            alpha=0.8,
            ax=ax1,
            legend=False,
        )

        ax1.set_title("Average Accuracy by Category", fontweight="bold")
        ax1.set_xlabel("")
        ax1.set_ylabel("Average Accuracy (%)")
        ax1.set_ylim(0, 100)

        # Add value labels
        for i, (_, row) in enumerate(category_df.iterrows()):
            ax1.text(
                i,
                row["Average_Accuracy"] + 1,
                f"{row['Average_Accuracy']:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
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
        pie_colors = [
            VIZ_COLORS["primary"],
            VIZ_COLORS["secondary"],
            VIZ_COLORS["info"],
        ]

        wedges, texts, autotexts = ax2.pie(
            category_df["Field_Count"].values,
            labels=category_df["Category"].values,
            colors=pie_colors[: len(category_df)],
            autopct="%1.0f",
            startangle=90,
        )

        ax2.set_title("Field Distribution by Category", fontweight="bold")

        # Enhance pie chart text
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
            autotext.set_fontsize(12)

        plt.tight_layout(rect=[0, 0.03, 1, 0.92])

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

    def create_classification_metrics_dashboard(
        self,
        extraction_results: List[Dict[str, Any]],
        ground_truth_data: Dict[str, Any],
        model_name: str,
        evaluation_summary: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> str:
        """
        Create classification metrics dashboard with precision, recall, and F1 scores.

        Args:
            extraction_results: Raw extraction results for classification analysis
            ground_truth_data: Ground truth mapping for classification analysis
            model_name: Model name for file naming
            save_path: Optional custom save path

        Returns:
            Path to saved dashboard file
        """
        if not VISUALIZATION_ENABLED:
            return ""

        print(f"🎨 Creating classification metrics dashboard for {model_name}...")

        # Generate classification summary
        try:
            classification_summary = generate_overall_classification_summary(
                evaluation_summary
            )
        except Exception as e:
            print(f"❌ Error generating classification data: {e}")
            return ""

        # Extract field-level metrics
        field_metrics = classification_summary.get("field_metrics", {})
        if not field_metrics:
            print("❌ No field classification metrics available")
            return ""

        # Prepare data in pandas format
        metrics_data = []
        for field, metrics in field_metrics.items():
            if "error" not in metrics:
                metrics_data.append(
                    {
                        "Field": field,
                        "Precision": metrics.get("precision", 0),
                        "Recall": metrics.get("recall", 0),
                        "F1_Score": metrics.get("f1_score", 0),
                        "Support": metrics.get("support", 0),
                    }
                )

        if not metrics_data:
            print("❌ No valid classification metrics to visualize")
            return ""

        metrics_df = pd.DataFrame(metrics_data)
        metrics_df = metrics_df.sort_values("F1_Score", ascending=False)

        # Create 2x2 dashboard
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, figsize=CHART_SIZES["classification_metrics"]
        )
        fig.suptitle(
            f"{model_name.upper()} Classification Metrics Dashboard",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # Chart 1: F1-Score by Field (Top 10)
        top_f1_df = metrics_df.head(10)
        sns.barplot(
            data=top_f1_df,
            y="Field",
            x="F1_Score",
            hue="Field",
            palette="viridis",
            alpha=0.8,
            ax=ax1,
            legend=False,
        )
        ax1.set_title("Top 10 Fields by F1-Score", fontweight="bold", fontsize=12)
        ax1.set_xlabel("F1-Score")
        ax1.set_ylabel("")
        ax1.set_xlim(0, 1.0)

        # Add F1-score labels
        for i, (_, row) in enumerate(top_f1_df.iterrows()):
            ax1.text(
                row["F1_Score"] + 0.01,
                i,
                f"{row['F1_Score']:.3f}",
                ha="left",
                va="center",
                fontweight="bold",
                fontsize=9,
            )

        # Chart 2: Precision vs Recall Scatter
        sns.scatterplot(
            data=metrics_df,
            x="Recall",
            y="Precision",
            size="Support",
            sizes=(50, 200),
            alpha=0.7,
            color=VIZ_COLORS["primary"],
            ax=ax2,
        )
        ax2.set_title("Precision vs Recall", fontweight="bold", fontsize=12)
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_xlim(0, 1.0)
        ax2.set_ylim(0, 1.0)

        # Add diagonal reference line (perfect balance)
        ax2.plot([0, 1], [0, 1], "r--", alpha=0.5, label="Perfect Balance")
        ax2.legend()

        # Chart 3: Overall Performance Metrics
        overall_metrics = classification_summary.get("overall_metrics", {})
        if overall_metrics and "error" not in overall_metrics:
            overall_data = []
            for avg_type in ["macro_avg", "micro_avg", "weighted_avg"]:
                if avg_type in overall_metrics:
                    metrics = overall_metrics[avg_type]
                    avg_name = avg_type.replace("_", " ").title()
                    overall_data.extend(
                        [
                            {
                                "Average": avg_name,
                                "Metric": "Precision",
                                "Value": metrics.get("precision", 0),
                            },
                            {
                                "Average": avg_name,
                                "Metric": "Recall",
                                "Value": metrics.get("recall", 0),
                            },
                            {
                                "Average": avg_name,
                                "Metric": "F1-Score",
                                "Value": metrics.get("f1_score", 0),
                            },
                        ]
                    )

            if overall_data:
                overall_df = pd.DataFrame(overall_data)
                sns.barplot(
                    data=overall_df,
                    x="Average",
                    y="Value",
                    hue="Metric",
                    palette=[
                        VIZ_COLORS["primary"],
                        VIZ_COLORS["secondary"],
                        VIZ_COLORS["success"],
                    ],
                    alpha=0.8,
                    ax=ax3,
                )
                ax3.set_title(
                    "Overall Performance Averages", fontweight="bold", fontsize=12
                )
                ax3.set_xlabel("")
                ax3.set_ylabel("Score")
                ax3.set_ylim(0, 1.0)
                ax3.legend()

                # Add value labels
                for container in ax3.containers:
                    ax3.bar_label(container, fmt="%.3f", fontweight="bold", fontsize=8)

        # Chart 4: Performance Distribution
        performance_categories = {
            "Excellent\n(F1 ≥ 0.9)": len(metrics_df[metrics_df["F1_Score"] >= 0.9]),
            "Good\n(F1 0.7-0.9)": len(
                metrics_df[
                    (metrics_df["F1_Score"] >= 0.7) & (metrics_df["F1_Score"] < 0.9)
                ]
            ),
            "Fair\n(F1 0.5-0.7)": len(
                metrics_df[
                    (metrics_df["F1_Score"] >= 0.5) & (metrics_df["F1_Score"] < 0.7)
                ]
            ),
            "Poor\n(F1 < 0.5)": len(metrics_df[metrics_df["F1_Score"] < 0.5]),
        }

        distribution_df = pd.DataFrame(
            [
                {"Category": cat, "Count": count, "Color": color}
                for cat, count, color in zip(
                    performance_categories.keys(),
                    performance_categories.values(),
                    [
                        VIZ_COLORS["success"],
                        VIZ_COLORS["info"],
                        VIZ_COLORS["secondary"],
                        VIZ_COLORS["warning"],
                    ],
                    strict=False,
                )
            ]
        )

        sns.barplot(
            data=distribution_df,
            x="Category",
            y="Count",
            hue="Category",
            palette=distribution_df["Color"].tolist(),
            alpha=0.8,
            ax=ax4,
            legend=False,
        )
        ax4.set_title("F1-Score Distribution", fontweight="bold", fontsize=12)
        ax4.set_xlabel("")
        ax4.set_ylabel("Number of Fields")

        # Add count labels
        for i, (_, row) in enumerate(distribution_df.iterrows()):
            if row["Count"] > 0:
                ax4.text(
                    i,
                    row["Count"] + 0.1,
                    str(int(row["Count"])),
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=10,
                )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save dashboard
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = VIZ_OUTPUT_PATTERNS["classification_metrics"].format(
                model=model_name, timestamp=timestamp
            )
            save_path = self.output_dir / filename
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=CHART_DPI, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"✅ Classification metrics dashboard saved: {save_path}")
        return str(save_path)

    def generate_model_visualizations(
        self,
        evaluation_summary: Dict[str, Any],
        batch_statistics: Dict[str, Any],
        model_name: str,
        extraction_results: Optional[List[Dict[str, Any]]] = None,
        ground_truth_data: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Generate complete visualization suite for a single model.

        Args:
            evaluation_summary: Evaluation results from evaluate_extraction_results()
            batch_statistics: Processing statistics from model evaluation
            model_name: Model name for file naming
            extraction_results: Raw extraction results for classification analysis
            ground_truth_data: Ground truth mapping for classification analysis

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

            # 4. Classification metrics dashboard (if data available)
            if extraction_results is not None and ground_truth_data is not None:
                classification_chart = self.create_classification_metrics_dashboard(
                    extraction_results,
                    ground_truth_data,
                    model_name,
                    evaluation_summary,
                )
                if classification_chart:
                    generated_files.append(classification_chart)

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
