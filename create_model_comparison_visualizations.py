#!/usr/bin/env python3
"""
Model Comparison Visualizations

Generates comparison charts for vision-language model evaluation results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print as rprint
from rich.console import Console

console = Console()

# Model configurations
MODELS = {
    "Llama-3.2-11B": {
        "csv": "output/csv/llama_batch_results_20251115_042949.csv",
        "size": "11B",
        "color": "#FF6B6B",  # Red
    },
    "Llama-4-Scout": {
        "csv": "output/csv/llama_batch_results_20251115_041934.csv",
        "size": "17B active (109B total)",
        "color": "#FF8E53",  # Orange
    },
    "InternVL3-2B": {
        "csv": "output/csv/internvl3_non_quantized_batch_results_20251115_035828.csv",
        "size": "2B",
        "color": "#4ECDC4",  # Teal
    },
    "InternVL3-8B (Q)": {
        "csv": "output/csv/internvl3_batch_results_20251115_040353.csv",
        "size": "8B",
        "color": "#95E1D3",  # Light teal
    },
    "InternVL3.5-8B": {
        "csv": "output/csv/internvl3_5_8b_batch_results_20251115_041347.csv",
        "size": "8.5B",
        "color": "#45B7D1",  # Blue
    },
}

# Memory usage data (from MODEL_COMPARISON_REPORT.md)
MEMORY_USAGE = {
    "Llama-3.2-11B": 21.3,  # GB (21.3GB allocated across 2x H200)
    "Llama-4-Scout": 217.28,  # GB (MoE multi-GPU)
    "InternVL3-2B": 2.29,
    "InternVL3-8B (Q)": 2.38,
    "InternVL3.5-8B": 17.06,  # GB (17.06GB allocated across 2x H200)
}


def load_model_data(csv_path):
    """Load and parse model results CSV."""
    if csv_path is None or not Path(csv_path).exists():
        return None
    return pd.read_csv(csv_path)


def calculate_metrics(df):
    """Calculate key metrics from results DataFrame."""
    if df is None or len(df) == 0:
        return None

    metrics = {
        "total_images": len(df),
        "avg_processing_time": df["processing_time"].mean(),
        "overall_accuracy": df["overall_accuracy"].mean() if "overall_accuracy" in df.columns else None,
    }

    # Per-document-type accuracy
    if "document_type" in df.columns and "overall_accuracy" in df.columns:
        doc_type_acc = df.groupby("document_type")["overall_accuracy"].agg(["mean", "count"])

        for doc_type in ["receipt", "invoice", "bank_statement"]:
            if doc_type in doc_type_acc.index:
                metrics[f"{doc_type}_accuracy"] = doc_type_acc.loc[doc_type, "mean"]
                metrics[f"{doc_type}_count"] = int(doc_type_acc.loc[doc_type, "count"])
            else:
                metrics[f"{doc_type}_accuracy"] = None
                metrics[f"{doc_type}_count"] = 0

    return metrics


def create_accuracy_comparison(all_metrics, output_dir):
    """Create overall accuracy comparison bar chart."""
    rprint("[cyan]Creating overall accuracy comparison chart...[/cyan]")

    # Filter models with accuracy data
    models_with_data = {
        name: data
        for name, data in all_metrics.items()
        if data["metrics"] is not None
        and data["metrics"].get("overall_accuracy") is not None
    }

    if not models_with_data:
        rprint("[yellow]⚠️ No accuracy data available[/yellow]")
        return None

    # Sort by accuracy (descending)
    sorted_models = sorted(
        models_with_data.items(),
        key=lambda x: x[1]["metrics"]["overall_accuracy"],
        reverse=True,
    )

    model_names = [name for name, _ in sorted_models]
    accuracies = [data["metrics"]["overall_accuracy"] for _, data in sorted_models]
    colors = [MODELS[name]["color"] for name in model_names]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(model_names, accuracies, color=colors, alpha=0.8, edgecolor="black")

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies, strict=False):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_ylabel("Overall Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Vision-Language Model Accuracy Comparison", fontsize=14, fontweight="bold", pad=20)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    # Save
    output_path = output_dir / "model_accuracy_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    rprint(f"[green]✅ Saved: {output_path}[/green]")
    return output_path


def create_speed_comparison(all_metrics, output_dir):
    """Create speed comparison bar chart."""
    rprint("[cyan]Creating speed comparison chart...[/cyan]")

    # Filter models with speed data
    models_with_data = {
        name: data
        for name, data in all_metrics.items()
        if data["metrics"] is not None
    }

    if not models_with_data:
        rprint("[yellow]⚠️ No speed data available[/yellow]")
        return None

    # Sort by speed (ascending - faster is better)
    sorted_models = sorted(
        models_with_data.items(),
        key=lambda x: x[1]["metrics"]["avg_processing_time"],
    )

    model_names = [name for name, _ in sorted_models]
    speeds = [data["metrics"]["avg_processing_time"] for _, data in sorted_models]
    colors = [MODELS[name]["color"] for name in model_names]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(model_names, speeds, color=colors, alpha=0.8, edgecolor="black")

    # Add value labels on bars
    for bar, speed in zip(bars, speeds, strict=False):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{speed:.1f}s",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_ylabel("Processing Time (seconds/image)", fontsize=12, fontweight="bold")
    ax.set_title("Vision-Language Model Speed Comparison", fontsize=14, fontweight="bold", pad=20)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    # Save
    output_path = output_dir / "model_speed_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    rprint(f"[green]✅ Saved: {output_path}[/green]")
    return output_path


def create_memory_comparison(output_dir):
    """Create memory usage comparison bar chart."""
    rprint("[cyan]Creating memory usage comparison chart...[/cyan]")

    # Filter models with memory data
    models_with_data = {
        name: mem for name, mem in MEMORY_USAGE.items() if mem is not None
    }

    if not models_with_data:
        rprint("[yellow]⚠️ No memory data available[/yellow]")
        return None

    # Sort by memory usage (ascending - less is better)
    sorted_models = sorted(models_with_data.items(), key=lambda x: x[1])

    model_names = [name for name, _ in sorted_models]
    memory_usage = [mem for _, mem in sorted_models]
    colors = [MODELS[name]["color"] for name in model_names]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(model_names, memory_usage, color=colors, alpha=0.8, edgecolor="black")

    # Add value labels on bars
    for bar, mem in zip(bars, memory_usage, strict=False):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{mem:.1f}GB",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_ylabel("GPU Memory Usage (GB)", fontsize=12, fontweight="bold")
    ax.set_title("Vision-Language Model Memory Efficiency", fontsize=14, fontweight="bold", pad=20)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    # Save
    output_path = output_dir / "model_memory_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    rprint(f"[green]✅ Saved: {output_path}[/green]")
    return output_path


def create_document_type_comparison(all_metrics, output_dir):
    """Create grouped bar chart for document type accuracy."""
    rprint("[cyan]Creating document type accuracy comparison chart...[/cyan]")

    # Prepare data
    doc_types = ["receipt", "invoice", "bank_statement"]
    doc_type_labels = ["Receipt", "Invoice", "Bank Statement"]

    # Filter models with document type data
    models_with_data = {}
    for name, data in all_metrics.items():
        if data["metrics"] is None:
            continue

        has_doc_data = any(
            data["metrics"].get(f"{dt}_accuracy") is not None for dt in doc_types
        )
        if has_doc_data:
            models_with_data[name] = data

    if not models_with_data:
        rprint("[yellow]⚠️ No document type data available[/yellow]")
        return None

    # Create data arrays
    model_names = list(models_with_data.keys())
    n_models = len(model_names)
    n_doc_types = len(doc_types)

    # Build accuracy matrix
    accuracy_matrix = np.zeros((n_models, n_doc_types))
    for i, name in enumerate(model_names):
        for j, doc_type in enumerate(doc_types):
            acc = models_with_data[name]["metrics"].get(f"{doc_type}_accuracy")
            accuracy_matrix[i, j] = acc if acc is not None else 0

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(n_doc_types)
    width = 0.15
    offset = np.linspace(-(n_models - 1) * width / 2, (n_models - 1) * width / 2, n_models)

    for i, (name, model_offset) in enumerate(zip(model_names, offset, strict=False)):
        color = MODELS[name]["color"]
        bars = ax.bar(
            x + model_offset,
            accuracy_matrix[i],
            width,
            label=name,
            color=color,
            alpha=0.8,
            edgecolor="black",
        )

        # Add value labels on bars
        for bar, acc in zip(bars, accuracy_matrix[i], strict=False):
            height = bar.get_height()
            if height > 0:  # Only show label if data exists
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{acc:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Model Accuracy by Document Type", fontsize=14, fontweight="bold", pad=20
    )
    ax.set_xticks(x)
    ax.set_xticklabels(doc_type_labels)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()

    # Save
    output_path = output_dir / "model_doctype_accuracy_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    rprint(f"[green]✅ Saved: {output_path}[/green]")
    return output_path


def create_speed_vs_accuracy_scatter(all_metrics, output_dir):
    """Create scatter plot comparing speed vs accuracy."""
    rprint("[cyan]Creating speed vs accuracy scatter plot...[/cyan]")

    # Filter models with both speed and accuracy data
    models_with_data = {
        name: data
        for name, data in all_metrics.items()
        if data["metrics"] is not None
        and data["metrics"].get("overall_accuracy") is not None
    }

    if not models_with_data:
        rprint("[yellow]⚠️ No speed/accuracy data available[/yellow]")
        return None

    # Prepare data
    model_names = list(models_with_data.keys())
    speeds = [data["metrics"]["avg_processing_time"] for data in models_with_data.values()]
    accuracies = [data["metrics"]["overall_accuracy"] for data in models_with_data.values()]
    colors = [MODELS[name]["color"] for name in model_names]
    sizes = [MODELS[name].get("size", "?") for name in model_names]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    for name, speed, acc, color in zip(model_names, speeds, accuracies, colors, strict=False):
        ax.scatter(speed, acc, s=300, alpha=0.6, color=color, edgecolor="black", linewidth=2)
        ax.annotate(
            name,
            (speed, acc),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("Processing Time (seconds/image)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Overall Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Model Performance: Speed vs Accuracy Trade-off",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add ideal region annotation
    ax.axhline(y=85, color="green", linestyle="--", alpha=0.3, label="85% accuracy target")
    ax.axvline(x=15, color="blue", linestyle="--", alpha=0.3, label="15s/image target")

    ax.legend(loc="lower right")
    plt.tight_layout()

    # Save
    output_path = output_dir / "model_speed_accuracy_tradeoff.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    rprint(f"[green]✅ Saved: {output_path}[/green]")
    return output_path


def main():
    """Main visualization generation function."""
    console.rule("[bold blue]Model Comparison Visualization Generator[/bold blue]")
    rprint()

    # Setup output directory
    output_dir = Path("output/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all model data
    rprint("[bold green]Loading model data...[/bold green]")
    all_metrics = {}
    for model_name, config in MODELS.items():
        results_df = load_model_data(config["csv"])
        metrics = calculate_metrics(results_df)
        all_metrics[model_name] = {
            "config": config,
            "metrics": metrics,
        }

        if metrics:
            rprint(f"[green]✅ {model_name}: {metrics['total_images']} images[/green]")
        else:
            rprint(f"[yellow]⚠️ {model_name}: No data[/yellow]")

    rprint()

    # Generate visualizations
    console.rule("[bold blue]Generating Visualizations[/bold blue]")
    rprint()

    viz_files = {}

    # 1. Overall accuracy comparison
    viz_files["accuracy"] = create_accuracy_comparison(all_metrics, output_dir)

    # 2. Speed comparison
    viz_files["speed"] = create_speed_comparison(all_metrics, output_dir)

    # 3. Memory comparison
    viz_files["memory"] = create_memory_comparison(output_dir)

    # 4. Document type accuracy comparison
    viz_files["doctype"] = create_document_type_comparison(all_metrics, output_dir)

    # 5. Speed vs accuracy scatter
    viz_files["scatter"] = create_speed_vs_accuracy_scatter(all_metrics, output_dir)

    # Summary
    rprint()
    console.rule("[bold green]Visualization Generation Complete[/bold green]")
    rprint()
    rprint(f"[cyan]Output directory: {output_dir}[/cyan]")
    rprint(f"[green]✅ Generated {len([v for v in viz_files.values() if v is not None])} visualizations[/green]")
    rprint()

    # List generated files
    for viz_type, path in viz_files.items():
        if path:
            rprint(f"  [cyan]• {viz_type}: {path.name}[/cyan]")


if __name__ == "__main__":
    main()
