#!/usr/bin/env python3
"""
Model Comparison Script

Compares memory usage, inference speed, and accuracy across multiple vision-language models.
"""

from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

# Model configurations
MODELS = {
    "Llama-3.2-11B": {
        "csv": "output/csv/llama_batch_results_20251115_042949.csv",
        "size": "11B",
        "type": "Vision-Language",
    },
    "Llama-4-Scout": {
        "csv": "output/csv/llama_batch_results_20251115_041934.csv",
        "size": "17B active (109B total)",
        "type": "MoE Vision-Language",
    },
    "InternVL3-2B": {
        "csv": "output/csv/internvl3_non_quantized_batch_results_20251115_035828.csv",
        "size": "2B",
        "type": "Vision-Language",
    },
    "InternVL3-8B (Q)": {
        "csv": "output/csv/internvl3_batch_results_20251115_040353.csv",
        "size": "8B",
        "type": "Vision-Language (Quantized)",
    },
    "InternVL3.5-8B": {
        "csv": "output/csv/internvl3_5_8b_batch_results_20251115_041347.csv",
        "size": "8.5B",
        "type": "Vision-Language (Cascade RL)",
    },
}


def load_model_data(csv_path):
    """Load and parse model results CSV."""
    if csv_path is None:
        return None

    csv_file = Path(csv_path)
    if not csv_file.exists():
        return None

    results_df = pd.read_csv(csv_file)
    return results_df


def calculate_metrics(results_df):
    """Calculate key metrics from results DataFrame."""
    if results_df is None or len(results_df) == 0:
        return None

    metrics = {
        "total_images": len(results_df),
        "avg_processing_time": results_df["processing_time"].mean(),
        "total_processing_time": results_df["processing_time"].sum(),
        "throughput_per_min": 60 / results_df["processing_time"].mean() if results_df["processing_time"].mean() > 0 else 0,
    }

    # Overall accuracy
    if "overall_accuracy" in results_df.columns:
        metrics["overall_accuracy"] = results_df["overall_accuracy"].mean()

    # Per-document-type accuracy
    if "document_type" in results_df.columns and "overall_accuracy" in results_df.columns:
        doc_type_acc = results_df.groupby("document_type")["overall_accuracy"].agg(["mean", "count"])

        for doc_type in ["receipt", "invoice", "bank_statement"]:
            if doc_type in doc_type_acc.index:
                metrics[f"{doc_type}_accuracy"] = doc_type_acc.loc[doc_type, "mean"]
                metrics[f"{doc_type}_count"] = int(doc_type_acc.loc[doc_type, "count"])
            else:
                metrics[f"{doc_type}_accuracy"] = None
                metrics[f"{doc_type}_count"] = 0

    return metrics


def create_comparison_table(all_metrics):
    """Create Rich table comparing all models."""

    # Main comparison table
    table = Table(title="🔬 Vision-Language Model Comparison", show_header=True, header_style="bold magenta")

    table.add_column("Model", style="cyan", width=18)
    table.add_column("Size", style="dim")
    table.add_column("Avg Time (s)", justify="right")
    table.add_column("Throughput\n(img/min)", justify="right")
    table.add_column("Overall\nAccuracy", justify="right")
    table.add_column("Receipt\nAccuracy", justify="right")
    table.add_column("Invoice\nAccuracy", justify="right")
    table.add_column("Bank Stmt\nAccuracy", justify="right")

    for model_name, data in all_metrics.items():
        metrics = data["metrics"]
        config = data["config"]

        if metrics is None:
            table.add_row(
                model_name,
                config["size"],
                "[dim]Not run[/dim]",
                "[dim]N/A[/dim]",
                "[dim]N/A[/dim]",
                "[dim]N/A[/dim]",
                "[dim]N/A[/dim]",
                "[dim]N/A[/dim]",
            )
            continue

        # Format values with colors
        overall_acc = metrics.get("overall_accuracy")
        overall_str = f"{overall_acc:.1f}%" if overall_acc is not None else "N/A"

        receipt_acc = metrics.get("receipt_accuracy")
        receipt_str = f"{receipt_acc:.1f}%" if receipt_acc is not None else "N/A"

        invoice_acc = metrics.get("invoice_accuracy")
        invoice_str = f"{invoice_acc:.1f}%" if invoice_acc is not None else "N/A"

        bank_acc = metrics.get("bank_statement_accuracy")
        bank_str = f"{bank_acc:.1f}%" if bank_acc is not None else "N/A"

        # Color code based on accuracy
        def color_code(acc_str, acc_val):
            if acc_val is None:
                return f"[dim]{acc_str}[/dim]"
            elif acc_val >= 90:
                return f"[green]{acc_str}[/green]"
            elif acc_val >= 80:
                return f"[yellow]{acc_str}[/yellow]"
            else:
                return f"[red]{acc_str}[/red]"

        table.add_row(
            model_name,
            config["size"],
            f"{metrics['avg_processing_time']:.2f}",
            f"{metrics['throughput_per_min']:.1f}",
            color_code(overall_str, overall_acc),
            color_code(receipt_str, receipt_acc),
            color_code(invoice_str, invoice_acc),
            color_code(bank_str, bank_acc),
        )

    return table


def create_performance_summary(all_metrics):
    """Create performance summary table."""
    table = Table(title="⚡ Performance Summary", show_header=True, header_style="bold blue")

    table.add_column("Model", style="cyan")
    table.add_column("Images/Min", justify="right")
    table.add_column("Avg Time (s)", justify="right")
    table.add_column("Total Time (s)", justify="right")
    table.add_column("Speed Rank", justify="center")

    # Calculate speed rankings
    valid_models = {k: v for k, v in all_metrics.items() if v["metrics"] is not None}
    sorted_by_speed = sorted(
        valid_models.items(),
        key=lambda x: x[1]["metrics"]["avg_processing_time"]
    )

    for rank, (model_name, data) in enumerate(sorted_by_speed, 1):
        metrics = data["metrics"]

        # Speed rank emoji
        if rank == 1:
            rank_str = "🥇 1st"
        elif rank == 2:
            rank_str = "🥈 2nd"
        elif rank == 3:
            rank_str = "🥉 3rd"
        else:
            rank_str = f"{rank}th"

        table.add_row(
            model_name,
            f"{metrics['throughput_per_min']:.1f}",
            f"{metrics['avg_processing_time']:.2f}",
            f"{metrics['total_processing_time']:.1f}",
            rank_str,
        )

    return table


def create_accuracy_summary(all_metrics):
    """Create accuracy summary table."""
    table = Table(title="🎯 Accuracy Summary", show_header=True, header_style="bold green")

    table.add_column("Model", style="cyan")
    table.add_column("Overall Acc", justify="right")
    table.add_column("Receipt Acc", justify="right")
    table.add_column("Invoice Acc", justify="right")
    table.add_column("Bank Stmt Acc", justify="right")
    table.add_column("Accuracy Rank", justify="center")

    # Calculate accuracy rankings
    valid_models = {k: v for k, v in all_metrics.items() if v["metrics"] is not None and v["metrics"].get("overall_accuracy") is not None}
    sorted_by_accuracy = sorted(
        valid_models.items(),
        key=lambda x: x[1]["metrics"]["overall_accuracy"],
        reverse=True
    )

    for rank, (model_name, data) in enumerate(sorted_by_accuracy, 1):
        metrics = data["metrics"]

        # Accuracy rank emoji
        if rank == 1:
            rank_str = "🥇 1st"
        elif rank == 2:
            rank_str = "🥈 2nd"
        elif rank == 3:
            rank_str = "🥉 3rd"
        else:
            rank_str = f"{rank}th"

        overall = metrics.get("overall_accuracy")
        receipt = metrics.get("receipt_accuracy")
        invoice = metrics.get("invoice_accuracy")
        bank = metrics.get("bank_statement_accuracy")

        table.add_row(
            model_name,
            f"{overall:.1f}%" if overall is not None else "N/A",
            f"{receipt:.1f}%" if receipt is not None else "N/A",
            f"{invoice:.1f}%" if invoice is not None else "N/A",
            f"{bank:.1f}%" if bank is not None else "N/A",
            rank_str,
        )

    return table


def main():
    """Main comparison function."""
    console.rule("[bold blue]Vision-Language Model Comparison[/bold blue]")
    console.print()

    # Load all model data
    all_metrics = {}
    for model_name, config in MODELS.items():
        results_df = load_model_data(config["csv"])
        metrics = calculate_metrics(results_df)
        all_metrics[model_name] = {
            "config": config,
            "metrics": metrics,
        }

    # Display comparison table
    comparison_table = create_comparison_table(all_metrics)
    console.print(comparison_table)
    console.print()

    # Display performance summary
    performance_table = create_performance_summary(all_metrics)
    console.print(performance_table)
    console.print()

    # Display accuracy summary
    accuracy_table = create_accuracy_summary(all_metrics)
    console.print(accuracy_table)
    console.print()

    # Key findings
    console.rule("[bold green]Key Findings[/bold green]")

    valid_models = {k: v for k, v in all_metrics.items() if v["metrics"] is not None}

    if len(valid_models) > 0:
        # Fastest model
        fastest = min(valid_models.items(), key=lambda x: x[1]["metrics"]["avg_processing_time"])
        console.print(f"[green]⚡ Fastest:[/green] {fastest[0]} ({fastest[1]['metrics']['avg_processing_time']:.2f}s/image)")

        # Most accurate model
        accurate_models = {k: v for k, v in valid_models.items() if v["metrics"].get("overall_accuracy") is not None}
        if len(accurate_models) > 0:
            most_accurate = max(accurate_models.items(), key=lambda x: x[1]["metrics"]["overall_accuracy"])
            console.print(f"[green]🎯 Most Accurate:[/green] {most_accurate[0]} ({most_accurate[1]['metrics']['overall_accuracy']:.1f}%)")

        # Best for receipts
        receipt_models = {k: v for k, v in valid_models.items() if v["metrics"].get("receipt_accuracy") is not None}
        if len(receipt_models) > 0:
            best_receipt = max(receipt_models.items(), key=lambda x: x[1]["metrics"]["receipt_accuracy"])
            console.print(f"[green]🧾 Best for Receipts:[/green] {best_receipt[0]} ({best_receipt[1]['metrics']['receipt_accuracy']:.1f}%)")

        # Best for invoices
        invoice_models = {k: v for k, v in valid_models.items() if v["metrics"].get("invoice_accuracy") is not None}
        if len(invoice_models) > 0:
            best_invoice = max(invoice_models.items(), key=lambda x: x[1]["metrics"]["invoice_accuracy"])
            console.print(f"[green]📄 Best for Invoices:[/green] {best_invoice[0]} ({best_invoice[1]['metrics']['invoice_accuracy']:.1f}%)")

        # Best for bank statements
        bank_models = {k: v for k, v in valid_models.items() if v["metrics"].get("bank_statement_accuracy") is not None}
        if len(bank_models) > 0:
            best_bank = max(bank_models.items(), key=lambda x: x[1]["metrics"]["bank_statement_accuracy"])
            console.print(f"[green]🏦 Best for Bank Statements:[/green] {best_bank[0]} ({best_bank[1]['metrics']['bank_statement_accuracy']:.1f}%)")

    console.print()
    console.rule("[bold blue]Comparison Complete[/bold blue]")


if __name__ == "__main__":
    main()
