"""
Batch Visualization Module for Document Extraction Results

Creates comprehensive visualizations including dashboards and heatmaps.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rich import print as rprint


class BatchVisualizer:
    """Generate visualizations from batch processing results."""

    def __init__(self, style: str = "default", palette: str = "husl"):
        """
        Initialize visualizer with style settings.

        Args:
            style: Matplotlib style
            palette: Seaborn color palette
        """
        plt.style.use(style)
        sns.set_palette(palette)

    def create_dashboard(
        self,
        df_results: pd.DataFrame,
        df_doctype_stats: pd.DataFrame,
        timestamp: str,
        save_path: Path | None = None,
        show: bool = True,
    ) -> Path | None:
        """
        Create 2x2 performance dashboard.

        Args:
            df_results: Results DataFrame
            df_doctype_stats: Document type statistics DataFrame
            timestamp: Timestamp for title
            save_path: Path to save figure
            show: Whether to display the figure

        Returns:
            Path to saved figure or None
        """
        if len(df_results) == 0:
            rprint("[yellow]⚠️ No data available for dashboard[/yellow]")
            return None

        # Create 2x2 dashboard
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Extract model name from timestamp for dynamic title
        model_name = "Vision Model"  # Default fallback
        if timestamp.startswith("internvl3_"):
            model_name = "InternVL3"
        elif timestamp.startswith("llama_") or "llama" in timestamp.lower():
            model_name = "Llama Vision"

        fig.suptitle(
            f"{model_name} Batch Processing Dashboard - {timestamp}",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Accuracy Distribution Histogram with Document Type Counts
        sns.histplot(
            data=df_results,
            x="overall_accuracy",
            bins=20,
            kde=True,
            ax=ax1,
            color="skyblue",
        )
        ax1.axvline(x=80, color="orange", linestyle="--", label="Pilot Ready (80%)")
        ax1.axvline(x=95, color="green", linestyle="--", label="Production Ready (95%)")

        # Add document type counts to title
        if "document_type" in df_results.columns:
            doc_counts = df_results["document_type"].value_counts()
            count_text = " | ".join(
                [f"{doc}: {count}" for doc, count in doc_counts.items()]
            )
            ax1.set_title(
                f"Accuracy Distribution\n({count_text})", fontweight="bold", fontsize=10
            )
        else:
            ax1.set_title("Accuracy Distribution", fontweight="bold")

        ax1.set_xlabel("Overall Accuracy (%)")
        ax1.set_ylabel("Count")
        ax1.legend()

        # 2. Processing Time Distribution or by Document Type
        if (
            "document_type" in df_results.columns
            and len(df_results["document_type"].unique()) > 1
        ):
            sns.boxplot(data=df_results, x="document_type", y="processing_time", ax=ax2)
            ax2.set_title("Processing Time by Document Type", fontweight="bold")
            ax2.set_xlabel("Document Type")
            ax2.set_ylabel("Processing Time (seconds)")
            ax2.tick_params(axis="x", rotation=45)
        else:
            sns.histplot(
                data=df_results,
                x="processing_time",
                bins=15,
                kde=True,
                ax=ax2,
                color="coral",
            )
            ax2.set_title("Processing Time Distribution", fontweight="bold")
            ax2.set_xlabel("Processing Time (seconds)")
            ax2.set_ylabel("Count")

        # 3. Accuracy vs Processing Time Scatter
        scatter = ax3.scatter(
            df_results["processing_time"],
            df_results["overall_accuracy"],
            c=df_results["overall_accuracy"],
            cmap="RdYlGn",
            alpha=0.6,
            s=100,
        )
        ax3.set_title("Accuracy vs Processing Time", fontweight="bold")
        ax3.set_xlabel("Processing Time (seconds)")
        ax3.set_ylabel("Overall Accuracy (%)")
        plt.colorbar(scatter, ax=ax3, label="Accuracy (%)")

        # 4. Document Type Performance or Field Success Rate
        if not df_doctype_stats.empty:
            doc_perf = (
                df_results.groupby("document_type")["overall_accuracy"]
                .mean()
                .sort_values(ascending=True)
            )
            colors = [
                "red" if acc < 60 else "orange" if acc < 80 else "green"
                for acc in doc_perf.to_numpy()
            ]
            doc_perf.plot(kind="barh", ax=ax4, color=colors)
            ax4.set_title("Average Accuracy by Document Type", fontweight="bold")
            ax4.set_xlabel("Average Accuracy (%)")
            ax4.set_ylabel("Document Type")
            ax4.axvline(x=80, color="orange", linestyle="--", alpha=0.5)
            ax4.axvline(x=95, color="green", linestyle="--", alpha=0.5)

            # Add value labels
            for i, (_idx, val) in enumerate(doc_perf.items()):
                ax4.text(val + 1, i, f"{val:.1f}%", va="center")
        else:
            # Field extraction success rate
            if "fields_matched" in df_results.columns:
                field_success = (
                    df_results["fields_matched"] / df_results["total_fields"] * 100
                ).mean()
                ax4.bar(
                    ["Field Extraction\nSuccess Rate"],
                    [field_success],
                    color="steelblue",
                )
                ax4.set_ylabel("Success Rate (%)")
                ax4.set_title("Overall Field Extraction Performance", fontweight="bold")
                ax4.set_ylim(0, 100)
                ax4.text(0, field_success + 2, f"{field_success:.1f}%", ha="center")

        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            rprint(f"[green]✅ Dashboard saved to {save_path}[/green]")

        if show:
            plt.show()
        else:
            plt.close()

        return save_path

    def create_field_heatmap(
        self,
        df_results: pd.DataFrame,
        timestamp: str,
        save_path: Path | None = None,
        show: bool = True,
        max_fields: int = 30,
    ) -> tuple[Path | None, pd.DataFrame | None]:
        """
        Create field-level accuracy heatmap.

        Args:
            df_results: Results DataFrame
            timestamp: Timestamp for title
            save_path: Path to save figure
            show: Whether to display the figure
            max_fields: Maximum number of fields to display

        Returns:
            Tuple of (saved path, field statistics DataFrame)
        """
        if len(df_results) == 0:
            rprint("[yellow]⚠️ No data available for heatmap[/yellow]")
            return None, None

        # Extract field accuracy columns
        field_cols = [col for col in df_results.columns if col.startswith("accuracy_")]

        if not field_cols:
            rprint("[yellow]⚠️ No field-level accuracy data available[/yellow]")
            return None, None

        # Create field accuracy matrix
        field_accuracy_data = df_results[["image_name"] + field_cols].set_index(
            "image_name"
        )
        field_accuracy_data.columns = [
            col.replace("accuracy_", "") for col in field_accuracy_data.columns
        ]

        # Calculate average accuracy per field and sort
        field_avg = field_accuracy_data.mean().sort_values(ascending=False)

        # Limit to top fields if too many
        if len(field_avg) > max_fields:
            field_avg = field_avg.head(max_fields)
            field_accuracy_data = field_accuracy_data[field_avg.index]

        # Create heatmap
        fig_height = max(8, len(df_results) * 0.3)
        plt.figure(figsize=(14, fig_height))

        # Reorder columns by average accuracy
        field_accuracy_ordered = field_accuracy_data[field_avg.index]

        # Create heatmap
        sns.heatmap(
            field_accuracy_ordered.T,
            cmap="RdYlGn",
            vmin=0,
            vmax=100,
            annot=False,
            fmt=".0f",
            cbar_kws={"label": "Accuracy (%)"},
            linewidths=0.5,
        )

        plt.title(
            f"Field-Level Accuracy Heatmap - {timestamp}",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Document")
        plt.ylabel("Field")
        plt.xticks(rotation=90)
        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            rprint(f"[green]✅ Heatmap saved to {save_path}[/green]")

        if show:
            plt.show()
        else:
            plt.close()

        # Create field statistics
        field_stats = (
            pd.DataFrame(
                {
                    "Average Accuracy (%)": field_accuracy_data.mean(),
                    "Min Accuracy (%)": field_accuracy_data.min(),
                    "Max Accuracy (%)": field_accuracy_data.max(),
                    "Std Dev (%)": field_accuracy_data.std(),
                }
            )
            .round(2)
            .sort_values("Average Accuracy (%)", ascending=False)
        )

        return save_path, field_stats

    def create_all_visualizations(
        self,
        df_results: pd.DataFrame,
        df_doctype_stats: pd.DataFrame,
        output_dir: Path,
        timestamp: str,
        show: bool = False,
    ) -> dict[str, Path]:
        """
        Create all visualizations and save them.

        Args:
            df_results: Results DataFrame
            df_doctype_stats: Document type statistics
            output_dir: Directory to save visualizations
            timestamp: Timestamp for filenames
            show: Whether to display figures

        Returns:
            Dictionary mapping visualization names to saved paths
        """
        saved_files = {}

        # Create dashboard
        dashboard_path = output_dir / f"dashboard_{timestamp}.png"
        self.create_dashboard(
            df_results, df_doctype_stats, timestamp, dashboard_path, show
        )
        saved_files["dashboard"] = dashboard_path

        # Create heatmap
        heatmap_path = output_dir / f"field_accuracy_heatmap_{timestamp}.png"
        heatmap_result, field_stats = self.create_field_heatmap(
            df_results, timestamp, heatmap_path, show
        )
        if heatmap_result:
            saved_files["heatmap"] = heatmap_path

            # Save field statistics if available
            if field_stats is not None:
                field_stats_path = (
                    output_dir.parent / "csv" / f"batch_{timestamp}_field_stats.csv"
                )
                field_stats.to_csv(field_stats_path)
                saved_files["field_stats"] = field_stats_path

        return saved_files
