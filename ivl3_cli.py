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

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from transformers import AutoModel, AutoTokenizer

# Local imports
from common.batch_analytics import BatchAnalytics
from common.batch_processor import BatchDocumentProcessor
from common.batch_reporting import BatchReporter
from common.batch_visualizations import BatchVisualizer
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

# Default model paths to search
DEFAULT_MODEL_PATHS = [
    "/home/jovyan/nfs_share/models/InternVL3_5-8B",
    "/models/InternVL3_5-8B",
    "./models/InternVL3_5-8B",
]

# Supported image extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}

# Environment variable prefix
ENV_PREFIX = "IVL_"

console = Console()
app = typer.Typer(
    name=APP_NAME,
    help="InternVL3.5-8B Document Extraction CLI",
    add_completion=False,
)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class PipelineConfig:
    """Configuration for the extraction pipeline."""

    # Data paths
    data_dir: Path
    output_dir: Path
    model_path: Path | None = None
    ground_truth: Path | None = None

    # Processing options
    max_images: int | None = None
    document_types: list[str] | None = None
    bank_v2: bool = True
    balance_correction: bool = True

    # Model options
    max_tiles: int = 11
    flash_attn: bool = True
    dtype: str = "bfloat16"
    max_new_tokens: int = 2000

    # Output options
    skip_visualizations: bool = False
    skip_reports: bool = False
    verbose: bool = True

    # Runtime state (set during execution)
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    def __post_init__(self) -> None:
        """Convert string paths to Path objects."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.model_path, str):
            self.model_path = Path(self.model_path)
        if isinstance(self.ground_truth, str):
            self.ground_truth = Path(self.ground_truth)

    @property
    def torch_dtype(self) -> torch.dtype:
        """Convert dtype string to torch.dtype."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.dtype, torch.bfloat16)


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load configuration from YAML file."""
    if not config_path.exists():
        console.print(f"[red]FATAL: Config file not found: {config_path}[/red]")
        console.print("[yellow]Expected: YAML configuration file[/yellow]")
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    with config_path.open() as f:
        config = yaml.safe_load(f)

    # Flatten nested structure
    flat_config = {}
    if "model" in config:
        flat_config["model_path"] = config["model"].get("path")
        flat_config["max_tiles"] = config["model"].get("max_tiles")
        flat_config["flash_attn"] = config["model"].get("flash_attn")
        flat_config["dtype"] = config["model"].get("dtype")

    if "data" in config:
        flat_config["data_dir"] = config["data"].get("dir")
        flat_config["ground_truth"] = config["data"].get("ground_truth")
        flat_config["max_images"] = config["data"].get("max_images")
        flat_config["document_types"] = config["data"].get("document_types")

    if "output" in config:
        flat_config["output_dir"] = config["output"].get("dir")
        flat_config["skip_visualizations"] = config["output"].get("skip_visualizations")
        flat_config["skip_reports"] = config["output"].get("skip_reports")

    if "processing" in config:
        flat_config["bank_v2"] = config["processing"].get("bank_v2")
        flat_config["balance_correction"] = config["processing"].get(
            "balance_correction"
        )
        flat_config["verbose"] = config["processing"].get("verbose")

    # Remove None values
    return {k: v for k, v in flat_config.items() if v is not None}


def load_env_config() -> dict[str, Any]:
    """Load configuration from environment variables."""
    env_config = {}

    env_mappings = {
        f"{ENV_PREFIX}DATA_DIR": ("data_dir", str),
        f"{ENV_PREFIX}OUTPUT_DIR": ("output_dir", str),
        f"{ENV_PREFIX}MODEL_PATH": ("model_path", str),
        f"{ENV_PREFIX}GROUND_TRUTH": ("ground_truth", str),
        f"{ENV_PREFIX}MAX_IMAGES": ("max_images", int),
        f"{ENV_PREFIX}MAX_TILES": ("max_tiles", int),
        f"{ENV_PREFIX}FLASH_ATTN": ("flash_attn", lambda x: x.lower() == "true"),
        f"{ENV_PREFIX}DTYPE": ("dtype", str),
        f"{ENV_PREFIX}BANK_V2": ("bank_v2", lambda x: x.lower() == "true"),
        f"{ENV_PREFIX}VERBOSE": ("verbose", lambda x: x.lower() == "true"),
    }

    for env_var, (config_key, converter) in env_mappings.items():
        value = os.environ.get(env_var)
        if value is not None:
            env_config[config_key] = converter(value)

    return env_config


def auto_detect_model_path() -> Path | None:
    """Auto-detect model path from common locations."""
    for path_str in DEFAULT_MODEL_PATHS:
        path = Path(path_str)
        if path.exists() and (path / "config.json").exists():
            return path
    return None


def merge_configs(
    cli_args: dict[str, Any],
    yaml_config: dict[str, Any],
    env_config: dict[str, Any],
) -> PipelineConfig:
    """Merge configs with CLI > YAML > ENV > defaults priority."""
    # Start with defaults (handled by dataclass)
    merged = {}

    # Layer in env config (lowest priority)
    merged.update({k: v for k, v in env_config.items() if v is not None})

    # Layer in YAML config
    merged.update({k: v for k, v in yaml_config.items() if v is not None})

    # Layer in CLI args (highest priority)
    merged.update({k: v for k, v in cli_args.items() if v is not None})

    # Auto-detect model path if not specified
    if not merged.get("model_path"):
        detected = auto_detect_model_path()
        if detected:
            merged["model_path"] = detected

    return PipelineConfig(**merged)


def validate_config(config: PipelineConfig) -> None:
    """Validate configuration with fail-fast diagnostics."""
    errors = []

    # Validate data directory
    if not config.data_dir.exists():
        console.print(f"[red]FATAL: Data directory not found: {config.data_dir}[/red]")
        console.print(
            f"[yellow]Expected: Directory containing {', '.join(IMAGE_EXTENSIONS)} images[/yellow]"
        )
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    # Validate model path
    if not config.model_path:
        console.print(
            "[red]FATAL: Model path not specified and could not be auto-detected[/red]"
        )
        console.print("[yellow]Searched locations:[/yellow]")
        for p in DEFAULT_MODEL_PATHS:
            console.print(f"  - {p}")
        console.print(
            "[yellow]Specify with --model-path or IVL_MODEL_PATH environment variable[/yellow]"
        )
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    if not config.model_path.exists():
        console.print(f"[red]FATAL: Model path not found: {config.model_path}[/red]")
        console.print(
            "[yellow]Expected: Directory containing InternVL3.5-8B model files[/yellow]"
        )
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    # Validate ground truth if specified
    if config.ground_truth and not config.ground_truth.exists():
        console.print(
            f"[red]FATAL: Ground truth file not found: {config.ground_truth}[/red]"
        )
        console.print(
            "[yellow]Expected: CSV file with columns: file, field_name, ground_truth_value[/yellow]"
        )
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    # Validate dtype
    valid_dtypes = {"bfloat16", "float16", "float32"}
    if config.dtype not in valid_dtypes:
        console.print(f"[red]FATAL: Invalid dtype: {config.dtype}[/red]")
        console.print(f"[yellow]Valid options: {', '.join(valid_dtypes)}[/yellow]")
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    # Check for images in data directory
    images = list(discover_images(config.data_dir, config.document_types))
    if not images:
        console.print(f"[red]FATAL: No images found in: {config.data_dir}[/red]")
        console.print(
            f"[yellow]Supported formats: {', '.join(IMAGE_EXTENSIONS)}[/yellow]"
        )
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    if errors:
        for error in errors:
            console.print(f"[red]ERROR: {error}[/red]")
        raise typer.Exit(EXIT_CONFIG_ERROR) from None


# ============================================================================
# Pipeline Components
# ============================================================================


def setup_output_directories(config: PipelineConfig) -> dict[str, Path]:
    """Create output directory structure."""
    output_dirs = {
        "base": config.output_dir,
        "batch_results": config.output_dir / "batch_results",
        "csv": config.output_dir / "csv",
        "visualizations": config.output_dir / "visualizations",
        "reports": config.output_dir / "reports",
    }

    for _name, path in output_dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        if config.verbose:
            console.print(f"  [dim]Created: {path}[/dim]")

    return output_dirs


def discover_images(
    data_dir: Path,
    document_types: list[str] | None = None,
) -> list[Path]:
    """Discover images in data directory."""
    images = []

    for ext in IMAGE_EXTENSIONS:
        images.extend(data_dir.glob(f"*{ext}"))
        images.extend(data_dir.glob(f"*{ext.upper()}"))

    # Sort by filename for reproducibility
    images = sorted(images, key=lambda p: p.name.lower())

    # Filter by document type if specified (based on filename patterns)
    if document_types:
        filtered = []
        type_patterns = [t.lower() for t in document_types]
        for img in images:
            name_lower = img.name.lower()
            if any(pattern in name_lower for pattern in type_patterns):
                filtered.append(img)
        images = filtered

    return images


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


def load_field_definitions() -> dict[str, list[str]]:
    """Load field definitions from YAML."""
    config_path = Path(__file__).parent / "config" / "field_definitions.yaml"
    if not config_path.exists():
        console.print(
            f"[yellow]Warning: Field definitions not found at {config_path}[/yellow]"
        )
        return {}

    with config_path.open() as f:
        config = yaml.safe_load(f)

    # Extract fields from nested structure: document_fields.{type}.fields
    document_fields = config.get("document_fields", {})
    result = {}
    for doc_type, doc_config in document_fields.items():
        if isinstance(doc_config, dict) and "fields" in doc_config:
            result[doc_type] = doc_config["fields"]
        elif isinstance(doc_config, list):
            result[doc_type] = doc_config

    return result


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
                torch_dtype=config.torch_dtype,
                low_cpu_mem_usage=True,
                use_flash_attn=config.flash_attn,
                trust_remote_code=True,
                device_map="auto",
            ).eval()

            progress.update(task, description="Model loaded!")

        # Log GPU memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            console.print(f"[dim]GPU memory allocated: {allocated:.2f} GB[/dim]")

        # Load configs
        prompt_config = load_prompt_config()
        field_definitions = load_field_definitions()

        # Create universal field list
        all_fields = set()
        for fields in field_definitions.values():
            all_fields.update(fields)
        universal_fields = sorted(list(all_fields))

        if not universal_fields:
            console.print(
                "[red]FATAL: No field definitions found in config/field_definitions.yaml[/red]"
            )
            console.print(
                "[yellow]Expected: document_fields section with field lists[/yellow]"
            )
            raise typer.Exit(EXIT_CONFIG_ERROR) from None

        # Create processor
        processor = DocumentAwareInternVL3HybridProcessor(
            field_list=universal_fields,
            model_path=str(config.model_path),
            debug=config.verbose,
            pre_loaded_model=model,
            pre_loaded_tokenizer=tokenizer,
            prompt_config=prompt_config,
            max_tiles=config.max_tiles,
        )

        yield processor, prompt_config, field_definitions

    finally:
        # Cleanup
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_batch_processing(
    config: PipelineConfig,
    processor: DocumentAwareInternVL3HybridProcessor,
    prompt_config: dict[str, Any],
    images: list[Path],
) -> tuple[list[dict], list[float], dict[str, int]]:
    """Run batch document processing."""
    # Create batch processor
    batch_processor = BatchDocumentProcessor(
        model=processor,
        processor=None,
        prompt_config=prompt_config,
        ground_truth_csv=str(config.ground_truth) if config.ground_truth else None,
        console=console,
        enable_math_enhancement=False,
    )

    # Apply bank statement V2 settings
    if hasattr(processor, "use_bank_v2"):
        processor.use_bank_v2 = config.bank_v2

    # Process batch
    console.print(f"\n[bold]Processing {len(images)} images...[/bold]")

    batch_results, processing_times, document_types_found = (
        batch_processor.process_batch(
            images,
            verbose=config.verbose,
        )
    )

    return batch_results, processing_times, document_types_found


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
        {k: str(v) for k, v in output_dirs.items()},
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
) -> None:
    """Print execution summary."""
    # Calculate metrics
    total_time = sum(processing_times)
    avg_time = total_time / len(processing_times) if processing_times else 0

    accuracies = [
        r.get("accuracy", 0) for r in batch_results if r.get("accuracy") is not None
    ]
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0

    # Create summary table
    table = Table(title="Execution Summary", show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Images Processed", str(len(batch_results)))
    table.add_row("Total Time", f"{total_time:.1f}s")
    table.add_row("Avg Time/Image", f"{avg_time:.2f}s")

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
    bank_v2: bool = typer.Option(
        True,
        "--bank-v2/--no-bank-v2",
        help="Use V2 bank statement extraction",
    ),
    balance_correction: bool = typer.Option(
        True,
        "--balance-correction/--no-balance-correction",
        help="Enable balance validation",
    ),
    max_tiles: int = typer.Option(
        11,
        "--max-tiles",
        help="Max image tiles (H200: 11, V100: 14)",
    ),
    flash_attn: bool = typer.Option(
        True,
        "--flash-attn/--no-flash-attn",
        help="Use Flash Attention 2",
    ),
    dtype: str = typer.Option(
        "bfloat16",
        "--dtype",
        help="Torch dtype (bfloat16, float16, float32)",
    ),
    no_viz: bool = typer.Option(
        False,
        "--no-viz",
        help="Skip visualization generation",
    ),
    no_reports: bool = typer.Option(
        False,
        "--no-reports",
        help="Skip report generation",
    ),
    verbose: bool = typer.Option(
        True,
        "--verbose/--quiet",
        "-v/-q",
        help="Verbose output",
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

    # Print header
    console.print(
        Panel.fit(
            f"[bold blue]{APP_NAME}[/bold blue] v{VERSION}\n"
            "[dim]InternVL3.5-8B Document Extraction[/dim]",
            border_style="blue",
        )
    )

    # Build CLI args dict (only non-None values)
    cli_args = {}
    if data_dir is not None:
        cli_args["data_dir"] = data_dir
    if output_dir is not None:
        cli_args["output_dir"] = output_dir
    if model_path is not None:
        cli_args["model_path"] = model_path
    if ground_truth is not None:
        cli_args["ground_truth"] = ground_truth
    if max_images is not None:
        cli_args["max_images"] = max_images
    if document_types is not None:
        cli_args["document_types"] = [t.strip() for t in document_types.split(",")]

    cli_args["bank_v2"] = bank_v2
    cli_args["balance_correction"] = balance_correction
    cli_args["max_tiles"] = max_tiles
    cli_args["flash_attn"] = flash_attn
    cli_args["dtype"] = dtype
    cli_args["skip_visualizations"] = no_viz
    cli_args["skip_reports"] = no_reports
    cli_args["verbose"] = verbose

    # Load configs from different sources
    yaml_config = load_yaml_config(config_file) if config_file else {}
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
    validate_config(config)

    # Print configuration
    if config.verbose:
        console.print("\n[bold]Configuration:[/bold]")
        console.print(f"  Data directory:  {config.data_dir}")
        console.print(f"  Output directory: {config.output_dir}")
        console.print(f"  Model path:      {config.model_path}")
        console.print(
            f"  Ground truth:    {config.ground_truth or 'None (inference-only)'}"
        )
        console.print(f"  Max tiles:       {config.max_tiles}")
        console.print(f"  Flash attention: {config.flash_attn}")
        console.print(f"  Dtype:           {config.dtype}")

    # Setup output directories
    console.print("\n[bold]Setting up output directories...[/bold]")
    output_dirs = setup_output_directories(config)

    # Discover images
    images = list(discover_images(config.data_dir, config.document_types))
    if config.max_images:
        images = images[: config.max_images]

    console.print(f"\n[bold]Found {len(images)} images to process[/bold]")

    # Run pipeline
    try:
        with load_model(config) as (processor, prompt_config, field_definitions):
            # Process batch
            batch_results, processing_times, document_types_found = (
                run_batch_processing(
                    config,
                    processor,
                    prompt_config,
                    images,
                )
            )

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
            print_summary(config, batch_results, processing_times, document_types_found)

    except KeyboardInterrupt:
        console.print("\n[yellow]Processing interrupted by user[/yellow]")
        raise typer.Exit(EXIT_PROCESSING_ERROR) from None
    except Exception as e:
        console.print(f"\n[red]FATAL: Processing error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(EXIT_PROCESSING_ERROR) from None

    console.print("\n[bold green]Pipeline completed successfully![/bold green]")
    raise typer.Exit(EXIT_SUCCESS)


if __name__ == "__main__":
    app()
