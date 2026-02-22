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

import logging
import os
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
EXIT_MODEL_LOAD_ERROR = 2
EXIT_PROCESSING_ERROR = 3
EXIT_PARTIAL_SUCCESS = 4

console = Console()
app = typer.Typer(
    name=APP_NAME,
    help="Document Extraction CLI",
    add_completion=False,
    invoke_without_command=True,
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
    # Create bank adapter when V2 bank extraction is enabled.
    bank_adapter = None
    if config.bank_v2 and getattr(processor, "supports_multi_turn", True):
        console.print(
            "[bold cyan]Setting up sophisticated bank statement extraction...[/bold cyan]"
        )

        bank_adapter = BankStatementAdapter(
            generate_fn=processor.generate,
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
    wall_clock_time: float | None = None,
    inference_time: float | None = None,
) -> None:
    """Print execution summary."""
    # Wall clock is the true elapsed time; fall back to sum for backwards compat
    total_time = (
        wall_clock_time if wall_clock_time is not None else sum(processing_times)
    )
    num_images = len(batch_results)
    throughput = (num_images / total_time * 60) if total_time > 0 else 0
    # Inference rate excludes model loading overhead
    inference_rate = (
        (num_images / inference_time * 60)
        if inference_time and inference_time > 0
        else None
    )

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

    table.add_row("Images Processed", str(num_images))
    table.add_row("Wall Clock Time", f"{total_time:.1f}s")
    table.add_row("Throughput", f"{throughput:.2f} images/min")
    if inference_rate is not None:
        table.add_row("Inference Rate", f"{inference_rate:.2f} images/min")

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
    config_table.add_row(
        "GPUs",
        "auto-detect"
        if config.num_gpus == 0
        else f"{config.num_gpus} (parallel)"
        if config.num_gpus > 1
        else "1 (single)",
    )
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


def _resolve_gpu_count(config: PipelineConfig) -> int:
    """Resolve num_gpus: 0 = auto-detect all, N = explicit. Returns resolved count."""
    try:
        import torch

        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        available_gpus = 0

    if config.num_gpus == 0:
        resolved = max(available_gpus, 1)
        if resolved > 1:
            console.print(
                f"\n[bold cyan]Auto-detected {resolved} GPUs — "
                f"enabling parallel processing[/bold cyan]"
            )
        return resolved

    if config.num_gpus > available_gpus > 0:
        console.print(
            f"[red]FATAL: Requested {config.num_gpus} GPUs but only "
            f"{available_gpus} available[/red]"
        )
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    return config.num_gpus


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

    resolved_gpus = _resolve_gpu_count(config)

    import time as _time

    _wall_start = _time.time()

    try:
        if resolved_gpus > 1:
            # Multi-GPU parallel processing (model loading happens per-worker)
            from common.multi_gpu import MultiGPUOrchestrator

            _inference_start = _time.time()
            orchestrator = MultiGPUOrchestrator(config, resolved_gpus)
            batch_results, processing_times, document_types_found, batch_stats = (
                orchestrator.run(
                    images, prompt_config, universal_fields, field_definitions
                )
            )
            inference_time = _time.time() - _inference_start
        else:
            # Single-GPU path (default)
            try:
                model_cm = load_model(config)
                model, tokenizer = model_cm.__enter__()
            except typer.Exit:
                raise
            except Exception as e:
                console.print(f"\n[red]FATAL: Model loading failed: {e}[/red]")
                if config.verbose:
                    console.print_exception()
                raise typer.Exit(EXIT_MODEL_LOAD_ERROR) from None

            try:
                processor = create_processor(
                    model,
                    tokenizer,
                    config,
                    prompt_config,
                    universal_fields,
                    field_definitions,
                )

                _inference_start = _time.time()
                batch_results, processing_times, document_types_found, batch_stats = (
                    run_batch_processing(
                        config,
                        processor,
                        prompt_config,
                        images,
                        field_definitions,
                    )
                )
                inference_time = _time.time() - _inference_start
            finally:
                model_cm.__exit__(None, None, None)
            # Model freed here — post-processing is CPU-only

    except KeyboardInterrupt:
        console.print("\n[yellow]Processing interrupted by user[/yellow]")
        raise typer.Exit(EXIT_PROCESSING_ERROR) from None
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"\n[red]FATAL: Processing error: {e}[/red]")
        if config.verbose:
            console.print_exception()
        raise typer.Exit(EXIT_PROCESSING_ERROR) from None

    wall_clock_time = _time.time() - _wall_start

    # Check for partial success: some images succeeded, some failed
    failed_count = sum(1 for r in batch_results if "error" in r)
    total_count = len(batch_results)

    # Generate analytics, visualizations, and reports regardless of partial failures
    _analytics, df_results, df_summary, df_doctype_stats, _df_field_stats = (
        generate_analytics(
            config,
            output_dirs,
            batch_results,
            processing_times,
        )
    )

    generate_visualizations(config, output_dirs, df_results, df_doctype_stats)

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

    print_summary(
        config,
        batch_results,
        processing_times,
        document_types_found,
        batch_stats,
        wall_clock_time,
        inference_time,
    )

    if failed_count == total_count:
        console.print("\n[red]All images failed processing[/red]")
        raise typer.Exit(EXIT_PROCESSING_ERROR) from None
    if failed_count > 0:
        console.print(
            f"\n[yellow]Pipeline completed with errors: "
            f"{failed_count}/{total_count} images failed[/yellow]"
        )
        raise typer.Exit(EXIT_PARTIAL_SUCCESS) from None

    console.print("\n[bold green]Pipeline completed successfully![/bold green]")


# ============================================================================
# Config Building (shared by callback + subcommands)
# ============================================================================


def _build_config(
    cli_args: dict[str, Any],
    config_file: Path | None = None,
    document_types: str | None = None,
    *,
    require_data_dir: bool = True,
    require_output_dir: bool = True,
) -> PipelineConfig:
    """Build PipelineConfig from config cascade (CLI > YAML > ENV > defaults).

    Shared by the backward-compat callback and all subcommands so that
    YAML / ENV / CLI merging logic is defined once.
    """
    # Special handling: comma-split document_types
    if document_types is not None:
        cli_args["document_types"] = [t.strip() for t in document_types.split(",")]

    # Load configs from different sources
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
            if config_file:
                console.print(f"[red]FATAL: Config file not found: {config_file}[/red]")
                console.print("[yellow]Expected: YAML configuration file[/yellow]")
                raise typer.Exit(EXIT_CONFIG_ERROR) from None

    # Apply YAML overrides to module-level constants (batch, generation, gpu)
    if raw_config:
        from common.model_config import apply_yaml_overrides

        apply_yaml_overrides(raw_config)

    # Check required fields
    merged_check = {**yaml_config, **cli_args}

    if require_data_dir and not merged_check.get("data_dir"):
        console.print("[red]FATAL: --data-dir is required[/red]")
        console.print(
            "[yellow]Specify via CLI flag or data.dir in config file[/yellow]"
        )
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    if require_output_dir and not merged_check.get("output_dir"):
        console.print("[red]FATAL: --output-dir is required[/red]")
        console.print(
            "[yellow]Specify via CLI flag or output.dir in config file[/yellow]"
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
    # Extract run_id before merge (not a PipelineConfig field in cli_args)
    run_id = cli_args.pop("run_id", None)

    config = merge_configs(cli_args, yaml_config, raw_config)

    # Override timestamp with run_id for predictable inter-stage file paths
    if run_id:
        config.timestamp = run_id

    errors = validate_config(config)
    if errors:
        for error in errors:
            console.print(f"[red]FATAL: {error}[/red]")
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    # Configure logging level from --verbose/--quiet flag
    logging.basicConfig(
        level=logging.DEBUG if config.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    return config


def _collect_run_args(
    *,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    model_type: str | None = None,
    model_path: Path | None = None,
    ground_truth: Path | None = None,
    max_images: int | None = None,
    batch_size: int | None = None,
    max_tiles: int | None = None,
    flash_attn: bool | None = None,
    dtype: str | None = None,
    bank_v2: bool | None = None,
    balance_correction: bool | None = None,
    num_gpus: int | None = None,
    verbose: bool | None = None,
    skip_visualizations: bool | None = None,
    skip_reports: bool | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Collect non-None CLI args into a dict for _build_config()."""
    mapping = {
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
        "num_gpus": num_gpus,
        "verbose": verbose,
        "skip_visualizations": skip_visualizations,
        "skip_reports": skip_reports,
        "run_id": run_id,
    }
    return {k: v for k, v in mapping.items() if v is not None}


# ============================================================================
# Main CLI Callback (backward-compat: bare `python cli.py --flags`)
# ============================================================================


@app.callback()
def main(
    ctx: typer.Context,
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
    num_gpus: int | None = typer.Option(
        None,
        "--num-gpus",
        help="Number of GPUs (0 = auto-detect all, 1 = single GPU, N = use N GPUs)",
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
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Shared run identifier for predictable inter-stage file paths. "
        "Overrides the auto-generated timestamp.",
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

    Subcommands:

        classify   Run classification stage only (produces CSV)
        extract    Run extraction stage (reads classification CSV, produces JSON)
        evaluate   Run evaluation stage (CPU-only, reads extraction JSON)
        run        Run full pipeline (classify + extract + evaluate)

    Without a subcommand, all flags are passed to the full pipeline (backward compat).
    """
    # Handle version flag
    if version:
        console.print(f"{APP_NAME} v{VERSION}")
        raise typer.Exit(EXIT_SUCCESS)

    # When a subcommand is invoked, let it handle everything
    if ctx.invoked_subcommand is not None:
        return

    # No subcommand → backward-compat: run full pipeline
    cli_args = _collect_run_args(
        data_dir=data_dir,
        output_dir=output_dir,
        model_type=model_type,
        model_path=model_path,
        ground_truth=ground_truth,
        max_images=max_images,
        batch_size=batch_size,
        max_tiles=max_tiles,
        flash_attn=flash_attn,
        dtype=dtype,
        bank_v2=bank_v2,
        balance_correction=balance_correction,
        num_gpus=num_gpus,
        verbose=verbose,
        skip_visualizations=no_viz,
        skip_reports=no_reports,
        run_id=run_id,
    )

    config = _build_config(cli_args, config_file, document_types)
    print_config(config)
    run_pipeline(config)
    raise typer.Exit(EXIT_SUCCESS)


# ============================================================================
# Subcommands
# ============================================================================


@app.command()
def run(
    data_dir: Path = typer.Option(
        None, "--data-dir", "-d", help="Directory containing images to process"
    ),
    output_dir: Path = typer.Option(
        None, "--output-dir", "-o", help="Output directory for results"
    ),
    config_file: Path = typer.Option(
        None, "--config", "-c", help="YAML configuration file"
    ),
    model_type: str | None = typer.Option(
        None,
        "--model",
        help=f"Model type ({', '.join(list_models())}). Default from config or internvl3.",
    ),
    model_path: Path = typer.Option(
        None, "--model-path", "-m", help="Path to model weights directory"
    ),
    ground_truth: Path = typer.Option(
        None,
        "--ground-truth",
        "-g",
        help="Ground truth CSV file (omit for inference-only)",
    ),
    max_images: int = typer.Option(
        None, "--max-images", help="Maximum number of images to process"
    ),
    document_types: str = typer.Option(
        None, "--document-types", help="Filter by document types (comma-separated)"
    ),
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        "-b",
        help="Images per batch (auto-detect from VRAM if omitted)",
    ),
    bank_v2: bool | None = typer.Option(
        None, "--bank-v2/--no-bank-v2", help="Use V2 bank statement extraction."
    ),
    balance_correction: bool | None = typer.Option(
        None,
        "--balance-correction/--no-balance-correction",
        help="Enable balance validation.",
    ),
    max_tiles: int | None = typer.Option(None, "--max-tiles", help="Max image tiles."),
    flash_attn: bool | None = typer.Option(
        None, "--flash-attn/--no-flash-attn", help="Use Flash Attention 2."
    ),
    dtype: str | None = typer.Option(
        None, "--dtype", help="Torch dtype (bfloat16, float16, float32)."
    ),
    num_gpus: int | None = typer.Option(
        None, "--num-gpus", help="Number of GPUs (0 = auto-detect)."
    ),
    no_viz: bool | None = typer.Option(
        None, "--no-viz/--viz", help="Skip visualization generation."
    ),
    no_reports: bool | None = typer.Option(
        None, "--no-reports/--reports", help="Skip report generation."
    ),
    verbose: bool | None = typer.Option(
        None, "--verbose/--quiet", "-v/-q", help="Verbose output."
    ),
    run_id: str | None = typer.Option(
        None, "--run-id", help="Shared run identifier for inter-stage file paths."
    ),
) -> None:
    """Run the full extraction pipeline (classify -> extract -> evaluate)."""
    cli_args = _collect_run_args(
        data_dir=data_dir,
        output_dir=output_dir,
        model_type=model_type,
        model_path=model_path,
        ground_truth=ground_truth,
        max_images=max_images,
        batch_size=batch_size,
        max_tiles=max_tiles,
        flash_attn=flash_attn,
        dtype=dtype,
        bank_v2=bank_v2,
        balance_correction=balance_correction,
        num_gpus=num_gpus,
        verbose=verbose,
        skip_visualizations=no_viz,
        skip_reports=no_reports,
        run_id=run_id,
    )

    config = _build_config(cli_args, config_file, document_types)
    print_config(config)
    run_pipeline(config)
    raise typer.Exit(EXIT_SUCCESS)


@app.command()
def classify(
    data_dir: Path = typer.Option(
        None, "--data-dir", "-d", help="Directory containing images to classify"
    ),
    output_dir: Path = typer.Option(
        None, "--output-dir", "-o", help="Output directory for classification CSV"
    ),
    config_file: Path = typer.Option(
        None, "--config", "-c", help="YAML configuration file"
    ),
    model_type: str | None = typer.Option(
        None,
        "--model",
        help=f"Model type ({', '.join(list_models())}). Default from config or internvl3.",
    ),
    model_path: Path = typer.Option(
        None, "--model-path", "-m", help="Path to model weights directory"
    ),
    max_images: int = typer.Option(
        None, "--max-images", help="Maximum number of images to process"
    ),
    document_types: str = typer.Option(
        None, "--document-types", help="Filter by document types (comma-separated)"
    ),
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        "-b",
        help="Images per batch (auto-detect from VRAM if omitted)",
    ),
    num_gpus: int | None = typer.Option(
        None, "--num-gpus", help="Number of GPUs (0 = auto-detect)."
    ),
    max_tiles: int | None = typer.Option(None, "--max-tiles", help="Max image tiles."),
    flash_attn: bool | None = typer.Option(
        None, "--flash-attn/--no-flash-attn", help="Use Flash Attention 2."
    ),
    dtype: str | None = typer.Option(
        None, "--dtype", help="Torch dtype (bfloat16, float16, float32)."
    ),
    verbose: bool | None = typer.Option(
        None, "--verbose/--quiet", "-v/-q", help="Verbose output."
    ),
    run_id: str | None = typer.Option(
        None, "--run-id", help="Shared run identifier for inter-stage file paths."
    ),
) -> None:
    """Run classification stage only. Produces a classification CSV.

    Example:
        python cli.py classify -d ./images -o ./output --model internvl3
    """
    from pipeline.classify import classify_images, write_classification_csv

    cli_args = _collect_run_args(
        data_dir=data_dir,
        output_dir=output_dir,
        model_type=model_type,
        model_path=model_path,
        max_images=max_images,
        batch_size=batch_size,
        max_tiles=max_tiles,
        flash_attn=flash_attn,
        dtype=dtype,
        num_gpus=num_gpus,
        verbose=verbose,
        run_id=run_id,
    )

    config = _build_config(cli_args, config_file, document_types)
    output_dirs = setup_output_directories(config)

    # Discover images
    images = list(discover_images(config.data_dir, config.document_types))
    if config.max_images:
        images = images[: config.max_images]

    if not images:
        from common.pipeline_config import IMAGE_EXTENSIONS

        exts = ", ".join(IMAGE_EXTENSIONS)
        console.print(
            f"[red]FATAL: No images found in: {config.data_dir}. "
            f"Supported formats: {exts}[/red]"
        )
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    console.print(f"\n[bold]Classifying {len(images)} images...[/bold]")

    # Load configs (no GPU needed)
    prompt_config, universal_fields, field_definitions = load_pipeline_configs(
        config.model_type
    )

    resolved_gpus = _resolve_gpu_count(config)

    if resolved_gpus > 1:
        from common.multi_gpu import MultiGPUOrchestrator

        orchestrator = MultiGPUOrchestrator(config, resolved_gpus)
        classification_output = orchestrator.run_classify(
            images, prompt_config, universal_fields, field_definitions
        )
    else:
        # Single-GPU path
        try:
            model_cm = load_model(config)
            model, tokenizer = model_cm.__enter__()
        except typer.Exit:
            raise
        except Exception as e:
            console.print(f"\n[red]FATAL: Model loading failed: {e}[/red]")
            if config.verbose:
                console.print_exception()
            raise typer.Exit(EXIT_MODEL_LOAD_ERROR) from None

        try:
            processor = create_processor(
                model,
                tokenizer,
                config,
                prompt_config,
                universal_fields,
                field_definitions,
            )
            classification_output = classify_images(
                processor=processor,
                image_paths=images,
                batch_size=config.batch_size or 1,
                verbose=config.verbose,
                console=console,
            )
        finally:
            model_cm.__exit__(None, None, None)

    # Write classification CSV
    csv_path = output_dirs["csv"] / f"batch_{config.timestamp}_classifications.csv"
    write_classification_csv(classification_output, csv_path)

    console.print(f"\n[green]Classifications written to: {csv_path}[/green]")
    console.print(f"  {len(classification_output.rows)} images classified")

    type_counts: dict[str, int] = {}
    for row in classification_output.rows:
        type_counts[row.document_type] = type_counts.get(row.document_type, 0) + 1
    for doc_type, count in sorted(type_counts.items()):
        console.print(f"  {doc_type}: {count}")

    raise typer.Exit(EXIT_SUCCESS)


@app.command()
def extract(
    classifications: Path = typer.Option(
        ..., "--classifications", help="Path to classification CSV from classify stage"
    ),
    output_dir: Path = typer.Option(
        None, "--output-dir", "-o", help="Output directory for extraction JSON"
    ),
    config_file: Path = typer.Option(
        None, "--config", "-c", help="YAML configuration file"
    ),
    model_type: str | None = typer.Option(
        None,
        "--model",
        help=f"Model type ({', '.join(list_models())}). Default from config or internvl3.",
    ),
    model_path: Path = typer.Option(
        None, "--model-path", "-m", help="Path to model weights directory"
    ),
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        "-b",
        help="Images per batch (auto-detect from VRAM if omitted)",
    ),
    bank_v2: bool | None = typer.Option(
        None, "--bank-v2/--no-bank-v2", help="Use V2 bank statement extraction."
    ),
    balance_correction: bool | None = typer.Option(
        None,
        "--balance-correction/--no-balance-correction",
        help="Enable balance validation.",
    ),
    num_gpus: int | None = typer.Option(
        None, "--num-gpus", help="Number of GPUs (0 = auto-detect)."
    ),
    max_tiles: int | None = typer.Option(None, "--max-tiles", help="Max image tiles."),
    flash_attn: bool | None = typer.Option(
        None, "--flash-attn/--no-flash-attn", help="Use Flash Attention 2."
    ),
    dtype: str | None = typer.Option(
        None, "--dtype", help="Torch dtype (bfloat16, float16, float32)."
    ),
    verbose: bool | None = typer.Option(
        None, "--verbose/--quiet", "-v/-q", help="Verbose output."
    ),
    run_id: str | None = typer.Option(
        None, "--run-id", help="Shared run identifier for inter-stage file paths."
    ),
) -> None:
    """Run extraction stage. Reads classification CSV, produces extraction JSON.

    Example:
        python cli.py extract --classifications ./output/csv/batch_*_classifications.csv -o ./output
    """
    from pipeline.classify import read_classification_csv
    from pipeline.extract import extract_documents, write_extraction_json

    if not classifications.exists():
        console.print(
            f"[red]FATAL: Classification CSV not found: {classifications}[/red]"
        )
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    cli_args = _collect_run_args(
        output_dir=output_dir,
        model_type=model_type,
        model_path=model_path,
        batch_size=batch_size,
        bank_v2=bank_v2,
        balance_correction=balance_correction,
        max_tiles=max_tiles,
        flash_attn=flash_attn,
        dtype=dtype,
        num_gpus=num_gpus,
        verbose=verbose,
        run_id=run_id,
    )

    config = _build_config(cli_args, config_file, require_data_dir=False)
    output_dirs = setup_output_directories(config)

    classification_output = read_classification_csv(classifications)
    console.print(
        f"\n[bold]Extracting {len(classification_output.rows)} documents...[/bold]"
    )

    # Load configs (no GPU needed)
    prompt_config, universal_fields, field_definitions = load_pipeline_configs(
        config.model_type
    )

    resolved_gpus = _resolve_gpu_count(config)

    if resolved_gpus > 1:
        from common.multi_gpu import MultiGPUOrchestrator

        orchestrator = MultiGPUOrchestrator(config, resolved_gpus)
        extraction_output = orchestrator.run_extract(
            classification_output, prompt_config, universal_fields, field_definitions
        )
    else:
        # Single-GPU path
        try:
            model_cm = load_model(config)
            model, tokenizer = model_cm.__enter__()
        except typer.Exit:
            raise
        except Exception as e:
            console.print(f"\n[red]FATAL: Model loading failed: {e}[/red]")
            if config.verbose:
                console.print_exception()
            raise typer.Exit(EXIT_MODEL_LOAD_ERROR) from None

        try:
            processor = create_processor(
                model,
                tokenizer,
                config,
                prompt_config,
                universal_fields,
                field_definitions,
            )

            # Create bank adapter when V2 bank extraction is enabled
            bank_adapter = None
            if config.bank_v2 and getattr(processor, "supports_multi_turn", True):
                bank_adapter = BankStatementAdapter(
                    generate_fn=processor.generate,
                    verbose=config.verbose,
                    use_balance_correction=config.balance_correction,
                )

            extraction_output = extract_documents(
                processor=processor,
                classifications=classification_output,
                bank_adapter=bank_adapter,
                batch_size=config.batch_size or 1,
                verbose=config.verbose,
                console=console,
            )
        finally:
            model_cm.__exit__(None, None, None)

    # Write extraction JSON
    json_path = output_dirs["batch"] / f"batch_{config.timestamp}_extractions.json"
    write_extraction_json(extraction_output, json_path)

    console.print(f"\n[green]Extractions written to: {json_path}[/green]")
    console.print(f"  {len(extraction_output.records)} documents extracted")

    raise typer.Exit(EXIT_SUCCESS)


@app.command()
def evaluate(
    extractions: Path = typer.Option(
        ..., "--extractions", help="Path to extraction JSON from extract stage"
    ),
    ground_truth: Path = typer.Option(
        ..., "--ground-truth", "-g", help="Ground truth CSV file (required)"
    ),
    output_dir: Path = typer.Option(
        None, "--output-dir", "-o", help="Output directory for evaluation results"
    ),
    config_file: Path = typer.Option(
        None, "--config", "-c", help="YAML configuration file"
    ),
    no_viz: bool | None = typer.Option(
        None, "--no-viz/--viz", help="Skip visualization generation."
    ),
    no_reports: bool | None = typer.Option(
        None, "--no-reports/--reports", help="Skip report generation."
    ),
    verbose: bool | None = typer.Option(
        None, "--verbose/--quiet", "-v/-q", help="Verbose output."
    ),
    run_id: str | None = typer.Option(
        None, "--run-id", help="Shared run identifier for inter-stage file paths."
    ),
) -> None:
    """Run evaluation stage. CPU-only, no model needed.

    Reads extraction JSON and ground truth CSV, produces evaluation summary CSV
    and field-level detail JSON.

    Example:
        python cli.py evaluate --extractions ./output/batch_results/batch_*_extractions.json \\
            -g ./ground_truth.csv -o ./output
    """
    from pipeline.evaluate import (
        evaluate_extractions,
        write_evaluation_csv,
        write_evaluation_json,
    )
    from pipeline.extract import read_extraction_json

    if not extractions.exists():
        console.print(f"[red]FATAL: Extraction JSON not found: {extractions}[/red]")
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    if not ground_truth.exists():
        console.print(f"[red]FATAL: Ground truth CSV not found: {ground_truth}[/red]")
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    cli_args: dict[str, Any] = {"output_dir": output_dir} if output_dir else {}
    if run_id is not None:
        cli_args["run_id"] = run_id
    if verbose is not None:
        cli_args["verbose"] = verbose
    if no_viz is not None:
        cli_args["skip_visualizations"] = no_viz
    if no_reports is not None:
        cli_args["skip_reports"] = no_reports

    config = _build_config(cli_args, config_file, require_data_dir=False)
    output_dirs = setup_output_directories(config)

    extraction_output = read_extraction_json(extractions)
    console.print(
        f"\n[bold]Evaluating {len(extraction_output.records)} extractions...[/bold]"
    )

    field_definitions = load_document_field_definitions()
    evaluation_method = os.environ.get("EVALUATION_METHOD", "order_aware_f1")

    eval_output = evaluate_extractions(
        extractions=extraction_output,
        ground_truth_csv=ground_truth,
        field_definitions=field_definitions,
        evaluation_method=evaluation_method,
        enable_math_enhancement=False,
        verbose=config.verbose,
    )

    # Write evaluation outputs
    csv_path = output_dirs["csv"] / f"batch_{config.timestamp}_eval_summary.csv"
    json_path = output_dirs["batch"] / f"batch_{config.timestamp}_eval_detail.json"
    write_evaluation_csv(eval_output, csv_path)
    write_evaluation_json(eval_output, json_path)

    console.print(f"\n[green]Evaluation summary: {csv_path}[/green]")
    console.print(f"[green]Evaluation detail:  {json_path}[/green]")

    # Print summary stats
    if eval_output.image_evaluations:
        f1_scores = [ie.median_f1 for ie in eval_output.image_evaluations]
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
        console.print("\n[bold]Results:[/bold]")
        console.print(f"  Documents evaluated: {len(eval_output.image_evaluations)}")
        console.print(f"  Average median F1:   {avg_f1:.1%}")

    raise typer.Exit(EXIT_SUCCESS)


if __name__ == "__main__":
    app()
