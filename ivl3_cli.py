#!/usr/bin/env python3
# ruff: noqa: B008 - typer.Option in defaults is the standard Typer pattern
"""
InternVL3.5-8B Document Extraction CLI

A production-ready CLI for document field extraction using InternVL3.5-8B.
Supports evaluation mode (with ground truth) and inference-only mode.

Usage:
    python ivl3_cli.py --data-dir ./images --output-dir ./output
    python ivl3_cli.py --config run_config.yaml
    python ivl3_cli.py --help
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import typer
import yaml
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text
from transformers import AutoModel, AutoTokenizer

# Local imports
from common.bank_statement_adapter import BankStatementAdapter
from common.batch_analytics import BatchAnalytics
from common.batch_processor import (
    BatchDocumentProcessor,
    load_document_field_definitions,
)
from common.batch_reporting import BatchReporter
from common.batch_visualizations import BatchVisualizer
from common.pipeline_config import (
    PipelineConfig,
    discover_images,
    load_env_config,
    load_yaml_config,
    merge_configs,
    validate_config,
)
from models.document_aware_internvl3_processor import (
    DocumentAwareInternVL3HybridProcessor,
)

# ============================================================================
# Constants
# ============================================================================

APP_NAME = "ivl3-cli"
VERSION = "1.0.0"

# Exit codes
EXIT_SUCCESS = 0
EXIT_CONFIG_ERROR = 1
EXIT_MODEL_ERROR = 2
EXIT_PROCESSING_ERROR = 3

console = Console()
app = typer.Typer(
    name=APP_NAME,
    help="InternVL3.5-8B Document Extraction CLI",
    add_completion=False,
)


# ============================================================================
# Helper Functions
# ============================================================================


def _util_color(pct: float) -> str:
    """Return Rich color name for a GPU utilization percentage."""
    if pct < 50:
        return "green"
    return "yellow" if pct < 80 else "red"


def create_gpu_status_table() -> Table | None:
    """Create a Rich table showing GPU memory status."""
    if not torch.cuda.is_available():
        return None

    gpu_table = Table(
        title="GPU Status",
        show_header=True,
        header_style="bold cyan",
    )
    gpu_table.add_column("GPU", style="white")
    gpu_table.add_column("Total", justify="right", style="dim")
    gpu_table.add_column("Allocated", justify="right")
    gpu_table.add_column("Reserved", justify="right")
    gpu_table.add_column("Utilization", justify="right")

    device_count = torch.cuda.device_count()
    total_vram = 0.0
    total_allocated = 0.0
    total_reserved = 0.0

    for gpu_id in range(device_count):
        props = torch.cuda.get_device_properties(gpu_id)
        gpu_name = props.name
        vram_gb = props.total_memory / (1024**3)
        allocated_gb = torch.cuda.memory_allocated(gpu_id) / (1024**3)
        reserved_gb = torch.cuda.memory_reserved(gpu_id) / (1024**3)
        utilization = (reserved_gb / vram_gb) * 100 if vram_gb > 0 else 0

        total_vram += vram_gb
        total_allocated += allocated_gb
        total_reserved += reserved_gb

        color = _util_color(utilization)
        gpu_table.add_row(
            f"{gpu_id}: {gpu_name}",
            f"{vram_gb:.1f} GB",
            f"{allocated_gb:.2f} GB",
            f"{reserved_gb:.2f} GB",
            f"[{color}]{utilization:.1f}%[/{color}]",
        )

    # Add total row for multi-GPU
    if device_count > 1:
        total_util = (total_reserved / total_vram) * 100 if total_vram > 0 else 0
        color = _util_color(total_util)
        gpu_table.add_row(
            "[bold]Total[/bold]",
            f"[bold]{total_vram:.1f} GB[/bold]",
            f"[bold]{total_allocated:.2f} GB[/bold]",
            f"[bold]{total_reserved:.2f} GB[/bold]",
            f"[bold][{color}]{total_util:.1f}%[/{color}][/bold]",
        )

    return gpu_table


# ============================================================================
# Pipeline Components
# ============================================================================


def setup_output_directories(config: PipelineConfig) -> dict[str, Path]:
    """Create output directory structure."""
    output_dirs = {
        "base": config.output_dir,
        "batch": config.output_dir / "batch_results",
        "csv": config.output_dir / "csv",
        "visualizations": config.output_dir / "visualizations",
        "reports": config.output_dir / "reports",
    }

    for _name, path in output_dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        if config.verbose:
            console.print(f"  [dim]Created: {path}[/dim]")

    return output_dirs


def load_prompt_config() -> dict[str, Any]:
    """Load prompt configuration from YAML."""
    prompt_path = Path(__file__).parent / "prompts" / "internvl3_prompts.yaml"
    if not prompt_path.exists():
        console.print(
            f"[yellow]Warning: Prompt config not found at {prompt_path}[/yellow]"
        )
        return {}

    with prompt_path.open() as f:
        config = yaml.safe_load(f)

    # Add detection config keys expected by DocumentAwareInternVL3HybridProcessor
    detection_path = Path(__file__).parent / "prompts" / "document_type_detection.yaml"
    config["detection_file"] = str(detection_path)
    config["detection_key"] = "detection"

    return config


def load_pipeline_configs() -> tuple[dict[str, Any], list[str], dict[str, list[str]]]:
    """Load prompt configuration and build universal field list.

    Returns:
        Tuple of (prompt_config, sorted universal_fields, field_definitions).
    """
    prompt_config = load_prompt_config()
    field_definitions = load_document_field_definitions()

    all_fields: set[str] = set()
    for fields in field_definitions.values():
        all_fields.update(fields)
    universal_fields = sorted(all_fields)

    if not universal_fields:
        console.print(
            "[red]FATAL: No field definitions found in config/field_definitions.yaml[/red]"
        )
        console.print(
            "[yellow]Expected: document_fields section with field lists[/yellow]"
        )
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    return prompt_config, universal_fields, field_definitions


@contextmanager
def load_model(config: PipelineConfig):
    """Context manager for loading and cleaning up model resources."""
    model = None
    tokenizer = None

    try:
        # Clear GPU memory before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        console.print(f"\n[bold]Loading model from: {config.model_path}[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading tokenizer...", total=None)

            tokenizer = AutoTokenizer.from_pretrained(
                str(config.model_path),
                trust_remote_code=True,
                use_fast=False,
            )

            progress.update(task, description="Loading model weights...")

            model = AutoModel.from_pretrained(
                str(config.model_path),
                dtype=config.torch_dtype,
                low_cpu_mem_usage=True,
                use_flash_attn=config.flash_attn,
                trust_remote_code=True,
                device_map="auto",
            ).eval()

            progress.update(task, description="Model loaded!")

        # Display GPU memory status table
        gpu_table = create_gpu_status_table()
        if gpu_table:
            console.print(gpu_table)

        yield model, tokenizer

    finally:
        # Cleanup
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_processor(
    model,
    tokenizer,
    config: PipelineConfig,
    prompt_config: dict[str, Any],
    universal_fields: list[str],
    field_definitions: dict[str, list[str]],
) -> DocumentAwareInternVL3HybridProcessor:
    """Create document extraction processor from loaded components."""
    return DocumentAwareInternVL3HybridProcessor(
        field_list=universal_fields,
        model_path=str(config.model_path),
        debug=config.verbose,
        pre_loaded_model=model,
        pre_loaded_tokenizer=tokenizer,
        prompt_config=prompt_config,
        max_tiles=config.max_tiles,
        field_definitions=field_definitions,
    )


def run_batch_processing(
    config: PipelineConfig,
    processor: DocumentAwareInternVL3HybridProcessor,
    prompt_config: dict[str, Any],
    images: list[Path],
    field_definitions: dict[str, list[str]],
) -> tuple[list[dict], list[float], dict[str, int], dict[str, float]]:
    """Run batch document processing with optional bank statement adapter."""
    # Create bank adapter when V2 bank extraction is enabled
    bank_adapter = None
    if config.bank_v2:
        console.print(
            "[bold cyan]Setting up sophisticated bank statement extraction...[/bold cyan]"
        )
        bank_adapter = BankStatementAdapter(
            model=processor,
            verbose=config.verbose,
            use_balance_correction=config.balance_correction,
        )
        console.print(
            "[green]V2: Sophisticated bank statement extraction enabled[/green]"
        )
        console.print(
            f"[dim]  Balance correction: {'Enabled' if config.balance_correction else 'Disabled'}[/dim]"
        )

    # Create batch processor with bank adapter (routing is handled internally)
    batch_processor = BatchDocumentProcessor(
        model=processor,
        prompt_config=prompt_config,
        ground_truth_csv=str(config.ground_truth) if config.ground_truth else None,
        console=console,
        enable_math_enhancement=False,
        bank_adapter=bank_adapter,
        field_definitions=field_definitions,
        batch_size=config.batch_size,
    )

    # Process batch
    console.print(f"\n[bold]Processing {len(images)} images...[/bold]")

    batch_results, processing_times, document_types_found = (
        batch_processor.process_batch(
            images,
            verbose=config.verbose,
        )
    )

    return (
        batch_results,
        processing_times,
        document_types_found,
        batch_processor.batch_stats,
    )


def generate_analytics(
    config: PipelineConfig,
    output_dirs: dict[str, Path],
    batch_results: list[dict],
    processing_times: list[float],
) -> tuple[Any, ...]:
    """Generate analytics and save CSV files."""
    console.print("\n[bold]Generating analytics...[/bold]")

    analytics = BatchAnalytics(batch_results, processing_times)

    saved_files, df_results, df_summary, df_doctype_stats, df_field_stats = (
        analytics.save_all_dataframes(
            output_dirs["csv"],
            config.timestamp,
        )
    )

    for f in saved_files:
        console.print(f"  [green]Saved: {f}[/green]")

    return analytics, df_results, df_summary, df_doctype_stats, df_field_stats


def generate_visualizations(
    config: PipelineConfig,
    output_dirs: dict[str, Path],
    df_results,
    df_doctype_stats,
) -> list[str]:
    """Generate visualization files."""
    if config.skip_visualizations:
        console.print("[dim]Skipping visualizations (--no-viz)[/dim]")
        return []

    console.print("\n[bold]Generating visualizations...[/bold]")

    visualizer = BatchVisualizer()
    viz_files = visualizer.create_all_visualizations(
        df_results,
        df_doctype_stats,
        output_dirs["visualizations"],
        config.timestamp,
        show=False,
    )

    for f in viz_files:
        console.print(f"  [green]Saved: {f}[/green]")

    return viz_files


def generate_reports(
    config: PipelineConfig,
    output_dirs: dict[str, Path],
    batch_results: list[dict],
    processing_times: list[float],
    document_types_found: dict[str, int],
    df_results,
    df_summary,
    df_doctype_stats,
) -> list[str]:
    """Generate report files."""
    if config.skip_reports:
        console.print("[dim]Skipping reports (--no-reports)[/dim]")
        return []

    console.print("\n[bold]Generating reports...[/bold]")

    reporter = BatchReporter(
        batch_results,
        processing_times,
        document_types_found,
        config.timestamp,
    )

    # Build batch config for report
    batch_config = {
        "model_path": str(config.model_path),
        "batch_size": config.batch_size or "auto",
        "max_tiles": config.max_tiles,
        "flash_attn": config.flash_attn,
        "dtype": config.dtype,
        "bank_v2": config.bank_v2,
        "balance_correction": config.balance_correction,
    }

    v100_config = {
        "dtype": "float32" if config.dtype == "float32" else None,
        "no_flash_attn": not config.flash_attn,
    }

    report_files = reporter.save_all_reports(
        output_dirs,
        df_results,
        df_summary,
        df_doctype_stats,
        str(config.model_path),
        batch_config,
        v100_config,
        verbose=config.verbose,
    )

    for f in report_files:
        console.print(f"  [green]Saved: {f}[/green]")

    return report_files


def print_summary(
    config: PipelineConfig,
    batch_results: list[dict],
    processing_times: list[float],
    document_types_found: dict[str, int],
    batch_stats: dict[str, float] | None = None,
) -> None:
    """Print execution summary."""
    # Calculate metrics
    total_time = sum(processing_times)
    avg_time = total_time / len(processing_times) if processing_times else 0

    # Extract accuracy from evaluation dict (uses overall_accuracy = mean F1)
    accuracies = []
    for r in batch_results:
        eval_data = r.get("evaluation", {})
        if eval_data:
            # Use overall_accuracy (mean F1), not median_f1
            acc = eval_data.get("overall_accuracy", 0)
            accuracies.append(acc)
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0

    # Create summary table
    table = Table(title="Execution Summary", show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Images Processed", str(len(batch_results)))
    table.add_row("Total Time", f"{total_time:.1f}s")
    table.add_row("Avg Time/Image", f"{avg_time:.2f}s")

    # Batch stats
    if batch_stats:
        configured = batch_stats.get("configured_batch_size", 1)
        avg_extract = batch_stats.get("avg_extraction_batch", 1.0)
        table.add_row("Batch Size (configured)", str(configured))
        table.add_row("Avg Batch Size (actual)", f"{avg_extract:.1f}")

    if config.ground_truth:
        table.add_row("Avg Accuracy", f"{avg_accuracy:.1%}")

    table.add_row("Output Directory", str(config.output_dir))

    console.print()
    console.print(table)

    # Document type breakdown
    if document_types_found:
        console.print("\n[bold]Document Types:[/bold]")
        for doc_type, count in sorted(document_types_found.items()):
            console.print(f"  {doc_type}: {count}")


# ============================================================================
# Main CLI Command
# ============================================================================


@app.command()
def main(
    data_dir: Path = typer.Option(
        None,
        "--data-dir",
        "-d",
        help="Directory containing images to process",
    ),
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for results",
    ),
    config_file: Path = typer.Option(
        None,
        "--config",
        "-c",
        help="YAML configuration file",
    ),
    model_path: Path = typer.Option(
        None,
        "--model-path",
        "-m",
        help="Path to InternVL3.5-8B model",
    ),
    ground_truth: Path = typer.Option(
        None,
        "--ground-truth",
        "-g",
        help="Ground truth CSV file (omit for inference-only)",
    ),
    max_images: int = typer.Option(
        None,
        "--max-images",
        help="Maximum number of images to process",
    ),
    document_types: str = typer.Option(
        None,
        "--document-types",
        help="Filter by document types (comma-separated)",
    ),
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        "-b",
        help="Images per batch (auto-detect from VRAM if omitted)",
    ),
    bank_v2: bool | None = typer.Option(
        None,
        "--bank-v2/--no-bank-v2",
        help="Use V2 bank statement extraction. Default from config or True.",
    ),
    balance_correction: bool | None = typer.Option(
        None,
        "--balance-correction/--no-balance-correction",
        help="Enable balance validation. Default from config or True.",
    ),
    max_tiles: int | None = typer.Option(
        None,
        "--max-tiles",
        help="Max image tiles (H200: 11, V100: 14). Default from config or 11.",
    ),
    flash_attn: bool | None = typer.Option(
        None,
        "--flash-attn/--no-flash-attn",
        help="Use Flash Attention 2. Default from config or True.",
    ),
    dtype: str | None = typer.Option(
        None,
        "--dtype",
        help="Torch dtype (bfloat16, float16, float32). Default from config or bfloat16.",
    ),
    no_viz: bool | None = typer.Option(
        None,
        "--no-viz/--viz",
        help="Skip visualization generation. Default from config or False.",
    ),
    no_reports: bool | None = typer.Option(
        None,
        "--no-reports/--reports",
        help="Skip report generation. Default from config or False.",
    ),
    verbose: bool | None = typer.Option(
        None,
        "--verbose/--quiet",
        "-v/-q",
        help="Verbose output. Default from config or True.",
    ),
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version and exit",
    ),
) -> None:
    """
    InternVL3.5-8B Document Extraction CLI

    Process document images and extract structured fields using InternVL3.5-8B.
    Supports evaluation mode (with ground truth) and inference-only mode.

    Examples:

        # Evaluation mode with ground truth
        python ivl3_cli.py -d ./data -o ./output -g ./ground_truth.csv

        # Inference-only mode
        python ivl3_cli.py -d ./images -o ./results

        # Using config file
        python ivl3_cli.py --config run_config.yaml

        # V100 configuration
        python ivl3_cli.py -d ./data -o ./output --max-tiles 14 --no-flash-attn --dtype float32
    """
    # Handle version flag
    if version:
        console.print(f"{APP_NAME} v{VERSION}")
        raise typer.Exit(EXIT_SUCCESS)

    # Header will be printed after config is loaded (to show actual values)

    # Build CLI args dict (only non-None values)
    arg_mapping = {
        "data_dir": data_dir,
        "output_dir": output_dir,
        "model_path": model_path,
        "ground_truth": ground_truth,
        "max_images": max_images,
        "batch_size": batch_size,
        "max_tiles": max_tiles,
        "flash_attn": flash_attn,
        "dtype": dtype,
        "bank_v2": bank_v2,
        "balance_correction": balance_correction,
        "verbose": verbose,
        "skip_visualizations": no_viz,
        "skip_reports": no_reports,
    }
    cli_args = {k: v for k, v in arg_mapping.items() if v is not None}

    # Special handling: comma-split
    if document_types is not None:
        cli_args["document_types"] = [t.strip() for t in document_types.split(",")]

    # Load configs from different sources
    yaml_config: dict[str, Any] = {}
    if config_file:
        try:
            yaml_config = load_yaml_config(config_file)
        except FileNotFoundError:
            console.print(f"[red]FATAL: Config file not found: {config_file}[/red]")
            console.print("[yellow]Expected: YAML configuration file[/yellow]")
            raise typer.Exit(EXIT_CONFIG_ERROR) from None
    env_config = load_env_config()

    # Check required fields
    merged_check = {**env_config, **yaml_config, **cli_args}
    if not merged_check.get("data_dir"):
        console.print("[red]FATAL: --data-dir is required[/red]")
        console.print(
            "[yellow]Specify via CLI, config file, or IVL_DATA_DIR environment variable[/yellow]"
        )
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    if not merged_check.get("output_dir"):
        console.print("[red]FATAL: --output-dir is required[/red]")
        console.print(
            "[yellow]Specify via CLI, config file, or IVL_OUTPUT_DIR environment variable[/yellow]"
        )
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    # Merge configurations
    config = merge_configs(cli_args, yaml_config, env_config)

    # Validate configuration
    errors = validate_config(config)
    if errors:
        for error in errors:
            console.print(f"[red]FATAL: {error}[/red]")
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    # Print header with configuration table (always shown for audit purposes)
    config_table = Table(show_header=False, box=None, padding=(0, 1))
    config_table.add_column("Option", style="cyan")
    config_table.add_column("Value", style="white")

    # Data paths
    config_table.add_row("Data directory", str(config.data_dir))
    config_table.add_row("Output directory", str(config.output_dir))
    config_table.add_row("Model path", str(config.model_path))
    config_table.add_row(
        "Ground truth",
        str(config.ground_truth)
        if config.ground_truth
        else "[dim]None (inference-only)[/dim]",
    )

    # Model settings
    config_table.add_row("Max tiles", str(config.max_tiles))
    config_table.add_row("Flash attention", str(config.flash_attn))
    config_table.add_row("Dtype", config.dtype)
    config_table.add_row("Max new tokens", str(config.max_new_tokens))

    # Processing options
    config_table.add_row(
        "Batch size",
        str(config.batch_size) if config.batch_size else "[dim]auto[/dim]",
    )
    config_table.add_row("Bank V2", str(config.bank_v2))
    config_table.add_row("Balance correction", str(config.balance_correction))
    config_table.add_row("Verbose", str(config.verbose))

    # Output options
    config_table.add_row("Skip visualizations", str(config.skip_visualizations))
    config_table.add_row("Skip reports", str(config.skip_reports))

    # Optional filters
    if config.max_images:
        config_table.add_row("Max images", str(config.max_images))
    if config.document_types:
        config_table.add_row("Document types", ", ".join(config.document_types))

    header_content = Group(
        Text.from_markup(f"[bold blue]{APP_NAME}[/bold blue] v{VERSION}"),
        Text.from_markup("[dim]InternVL3.5-8B Document Extraction[/dim]"),
        Text(""),  # Blank line
        config_table,
    )

    console.print(
        Panel(
            header_content,
            border_style="blue",
            title="Configuration",
            title_align="left",
        )
    )

    # Setup output directories
    console.print("\n[bold]Setting up output directories...[/bold]")
    output_dirs = setup_output_directories(config)

    # Discover images
    images = list(discover_images(config.data_dir, config.document_types))
    if config.max_images:
        images = images[: config.max_images]

    if not images:
        from common.pipeline_config import IMAGE_EXTENSIONS

        exts = ", ".join(IMAGE_EXTENSIONS)
        console.print(
            f"[red]FATAL: No images found in: {config.data_dir}. Supported formats: {exts}[/red]"
        )
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    console.print(f"\n[bold]Found {len(images)} images to process[/bold]")

    # Load configs (no GPU needed)
    prompt_config, universal_fields, field_definitions = load_pipeline_configs()

    # Run pipeline
    try:
        # Model context: load model, run extraction, then free GPU memory
        with load_model(config) as (model, tokenizer):
            processor = create_processor(
                model,
                tokenizer,
                config,
                prompt_config,
                universal_fields,
                field_definitions,
            )

            batch_results, processing_times, document_types_found, batch_stats = (
                run_batch_processing(
                    config,
                    processor,
                    prompt_config,
                    images,
                    field_definitions,
                )
            )
        # Model freed here — post-processing is CPU-only

        # Generate analytics
        analytics, df_results, df_summary, df_doctype_stats, df_field_stats = (
            generate_analytics(
                config,
                output_dirs,
                batch_results,
                processing_times,
            )
        )

        # Generate visualizations
        generate_visualizations(config, output_dirs, df_results, df_doctype_stats)

        # Generate reports
        generate_reports(
            config,
            output_dirs,
            batch_results,
            processing_times,
            document_types_found,
            df_results,
            df_summary,
            df_doctype_stats,
        )

        # Print summary
        print_summary(
            config, batch_results, processing_times, document_types_found, batch_stats
        )

    except KeyboardInterrupt:
        console.print("\n[yellow]Processing interrupted by user[/yellow]")
        raise typer.Exit(EXIT_PROCESSING_ERROR) from None
    except Exception as e:
        console.print(f"\n[red]FATAL: Processing error: {e}[/red]")
        if config.verbose:
            console.print_exception()
        raise typer.Exit(EXIT_PROCESSING_ERROR) from None

    console.print("\n[bold green]Pipeline completed successfully![/bold green]")
    raise typer.Exit(EXIT_SUCCESS)


if __name__ == "__main__":
    app()
