#!/usr/bin/env python3
# ruff: noqa: B008 - typer.Option in defaults is the standard Typer pattern
"""
Document Extraction CLI

A production-ready CLI for document field extraction using vision-language models.
Supports evaluation mode (with ground truth) and inference-only mode.

Usage:
    python cli.py --data-dir ./images --output-dir ./output
    python cli.py --config run_config.yaml
    python cli.py --help
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

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
    load_structure_suffixes,
    load_yaml_config,
    merge_configs,
    strip_structure_suffixes,
    validate_config,
)
from models.registry import get_model, list_models

# ============================================================================
# Constants
# ============================================================================

APP_NAME = "doc-extract"
VERSION = "2.0.0"

# Exit codes
EXIT_SUCCESS = 0
EXIT_CONFIG_ERROR = 1
EXIT_PROCESSING_ERROR = 3

console = Console()
app = typer.Typer(
    name=APP_NAME,
    help="Document Extraction CLI",
    add_completion=False,
)


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


def load_prompt_config(model_type: str = "internvl3") -> dict[str, Any]:
    """Build prompt routing config from YAML files (single source of truth).

    Derives supported document types from the extraction prompt YAML keys
    rather than maintaining a hardcoded list.
    """
    import yaml

    registration = get_model(model_type)
    base = Path(__file__).parent / "prompts"
    detection_file = base / "document_type_detection.yaml"
    extraction_path = base / registration.prompt_file

    if not detection_file.exists():
        console.print(f"[red]FATAL: Detection prompt not found: {detection_file}[/red]")
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    if not extraction_path.exists():
        console.print(
            f"[red]FATAL: Extraction prompts not found: {extraction_path}[/red]"
        )
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    # Derive supported document types from extraction YAML prompt keys
    with extraction_path.open() as f:
        extraction_yaml = yaml.safe_load(f)

    prompt_keys = set(extraction_yaml.get("prompts", {}).keys())

    # Derive non-doc keys by cross-referencing with field_definitions.yaml
    # A prompt key is a document type if its suffix-stripped form appears
    # in supported_document_types; everything else is a non-doc key.
    field_defs_path = Path(__file__).parent / "config" / "field_definitions.yaml"
    supported_types: set[str] = set()
    if field_defs_path.exists():
        with field_defs_path.open() as f:
            field_defs = yaml.safe_load(f)
        supported_types = set(field_defs.get("supported_document_types", []))

    suffixes = load_structure_suffixes(extraction_path)
    doc_keys: set[str] = set()
    for key in prompt_keys:
        base = strip_structure_suffixes(key, suffixes)
        if base in supported_types:
            doc_keys.add(key)

    # Map prompt keys to uppercase canonical types, stripping structure suffixes
    # e.g. bank_statement_flat → BANK_STATEMENT, invoice → INVOICE
    extraction_file = str(extraction_path)
    extraction_files: dict[str, str] = {}
    for key in doc_keys:
        canonical = strip_structure_suffixes(key, suffixes).upper()
        extraction_files[canonical] = extraction_file

    if not extraction_files:
        console.print(
            "[red]FATAL: No document type prompts found in "
            f"{extraction_path.name}[/red]"
        )
        console.print(
            "[yellow]Expected: prompts section with keys like 'invoice', "
            "'receipt', etc.[/yellow]"
        )
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    return {
        "detection_file": str(detection_file),
        "detection_key": "detection",
        "extraction_files": extraction_files,
    }


def load_pipeline_configs(
    model_type: str = "internvl3",
) -> tuple[dict[str, Any], list[str], dict[str, list[str]]]:
    """Load prompt configuration and build universal field list.

    Returns:
        Tuple of (prompt_config, sorted universal_fields, field_definitions).
    """
    prompt_config = load_prompt_config(model_type)
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


def load_model(config: PipelineConfig):
    """Context manager for loading and cleaning up model resources.

    Delegates to the registered loader for config.model_type.
    """
    registration = get_model(config.model_type)
    return registration.loader(config)


def create_processor(
    model,
    tokenizer,
    config: PipelineConfig,
    prompt_config: dict[str, Any],
    universal_fields: list[str],
    field_definitions: dict[str, list[str]],
) -> Any:
    """Create document extraction processor from loaded components.

    Delegates to the registered processor_creator for config.model_type.
    """
    registration = get_model(config.model_type)
    return registration.processor_creator(
        model, tokenizer, config, prompt_config, universal_fields, field_definitions
    )


def run_batch_processing(
    config: PipelineConfig,
    processor: Any,
    prompt_config: dict[str, Any],
    images: list[Path],
    field_definitions: dict[str, list[str]],
) -> tuple[list[dict], list[float], dict[str, int], dict[str, float]]:
    """Run batch document processing with optional bank statement adapter."""
    # Create bank adapter when V2 bank extraction is enabled AND model supports
    # multi-turn .chat() API (required by UnifiedBankExtractor).
    # Models without .chat() (e.g. Llama) use standard single-pass extraction.
    bank_adapter = None
    if config.bank_v2:
        if hasattr(processor.model, "chat"):
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
        else:
            console.print(
                "[yellow]Bank V2 skipped: model does not support multi-turn .chat() API[/yellow]"
            )
            console.print(
                "[dim]  Bank statements will use standard single-pass extraction[/dim]"
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

    report_files = reporter.save_all_reports(
        output_dirs,
        df_results,
        df_summary,
        df_doctype_stats,
        config,
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


def print_config(config: PipelineConfig) -> None:
    """Print configuration panel with all settings."""
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
    config_table.add_row("Model type", config.model_type)
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
        Text.from_markup(f"[dim]{config.model_type} Document Extraction[/dim]"),
        Text(""),
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


# ============================================================================
# Pipeline Orchestration
# ============================================================================


def run_pipeline(config: PipelineConfig) -> None:
    """Run the full extraction pipeline from a validated config.

    This function is independently callable from tests, notebooks, or the CLI.
    It handles: output setup, image discovery, model loading, batch processing,
    analytics, visualizations, reports, and summary.
    """
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
    prompt_config, universal_fields, field_definitions = load_pipeline_configs(
        config.model_type
    )

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
        _analytics, df_results, df_summary, df_doctype_stats, _df_field_stats = (
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
    model_type: str | None = typer.Option(
        None,
        "--model",
        help=f"Model type ({', '.join(list_models())}). Default from config or internvl3.",
    ),
    model_path: Path = typer.Option(
        None,
        "--model-path",
        "-m",
        help="Path to model weights directory",
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
    Document Extraction CLI

    Process document images and extract structured fields using vision-language models.
    Supports evaluation mode (with ground truth) and inference-only mode.

    Examples:

        # Evaluation mode with ground truth
        python cli.py -d ./data -o ./output -g ./ground_truth.csv

        # Inference-only mode
        python cli.py -d ./images -o ./results

        # Using config file
        python cli.py --config run_config.yaml

        # Specify model type
        python cli.py --model internvl3 -d ./data -o ./output
    """
    # Handle version flag
    if version:
        console.print(f"{APP_NAME} v{VERSION}")
        raise typer.Exit(EXIT_SUCCESS)

    # Build CLI args dict (only non-None values)
    arg_mapping = {
        "data_dir": data_dir,
        "output_dir": output_dir,
        "model_type": model_type,
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
    # Always load the default run_config.yml; --config overrides the path
    default_config = Path(__file__).parent / "config" / "run_config.yml"
    resolved_config = config_file or (
        default_config if default_config.exists() else None
    )

    yaml_config: dict[str, Any] = {}
    raw_config: dict[str, Any] = {}
    if resolved_config:
        try:
            yaml_config, raw_config = load_yaml_config(resolved_config)
        except FileNotFoundError:
            # Only fatal when the user explicitly passed --config
            if config_file:
                console.print(f"[red]FATAL: Config file not found: {config_file}[/red]")
                console.print("[yellow]Expected: YAML configuration file[/yellow]")
                raise typer.Exit(EXIT_CONFIG_ERROR) from None

    # Apply YAML overrides to module-level constants (batch, generation, gpu)
    if raw_config:
        from common.config import apply_yaml_overrides

        apply_yaml_overrides(raw_config)

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

    # Validate model_type early (fail fast with available models)
    resolved_model_type = merged_check.get("model_type", "internvl3")
    try:
        get_model(resolved_model_type)
    except ValueError:
        available = ", ".join(list_models()) or "(none)"
        console.print(f"[red]FATAL: Unknown model type: '{resolved_model_type}'[/red]")
        console.print(f"[yellow]Available models: {available}[/yellow]")
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    # Merge and validate configuration
    config = merge_configs(cli_args, yaml_config, env_config, raw_config)

    errors = validate_config(config)
    if errors:
        for error in errors:
            console.print(f"[red]FATAL: {error}[/red]")
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    # Display configuration and run pipeline
    print_config(config)
    run_pipeline(config)
    raise typer.Exit(EXIT_SUCCESS)


if __name__ == "__main__":
    app()
