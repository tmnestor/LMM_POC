"""Pipeline operations: model loading, processor creation, batch execution.

Extracted from cli.py to break the circular dependency between cli.py and
multi_gpu.py.  Both modules import from here; neither imports from the other.
"""

from pathlib import Path
from typing import Any

from rich.console import Console

from common.pipeline_config import PipelineConfig

type BatchOutput = tuple[list[dict], list[float], dict[str, int], dict[str, float]]

console = Console()


def load_model(config: PipelineConfig):
    """Context manager for loading and cleaning up model resources.

    Delegates to the registered loader for config.model_type.
    """
    from models.registry import get_model

    registration = get_model(config.model_type)
    return registration.loader(config)


def create_processor(
    model,
    tokenizer,
    config: PipelineConfig,
    prompt_config: dict[str, Any],
    universal_fields: list[str],
    field_definitions: dict[str, list[str]],
    *,
    app_config: Any | None = None,
) -> Any:
    """Create document extraction processor from loaded components.

    Delegates to the registered processor_creator for config.model_type.
    """
    from models.registry import get_model

    registration = get_model(config.model_type)
    return registration.processor_creator(
        model,
        tokenizer,
        config,
        prompt_config,
        universal_fields,
        field_definitions,
        app_config=app_config,
    )


def run_batch(
    config: PipelineConfig,
    processor: Any,
    images: list[Path],
    field_definitions: dict[str, list[str]],
) -> BatchOutput:
    """Run batch document processing with optional bank statement adapter."""
    from common.document_pipeline import create_document_pipeline
    from common.unified_bank_extractor import UnifiedBankExtractor

    bank_adapter = None
    if config.bank_v2 and getattr(processor, "supports_multi_turn", True):
        console.print(
            "[bold cyan]Setting up sophisticated bank statement extraction...[/bold cyan]"
        )

        bank_adapter = UnifiedBankExtractor(
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

    pipeline = create_document_pipeline(
        processor,
        ground_truth_csv=str(config.ground_truth) if config.ground_truth else None,
        bank_adapter=bank_adapter,
        field_definitions=field_definitions,
        batch_size=config.batch_size,
        enable_math_enhancement=False,
        console=console,
    )

    console.print(f"\n[bold]Processing {len(images)} images...[/bold]")

    batch_results, processing_times, document_types_found = pipeline.process_batch(
        [str(img) for img in images],
        verbose=config.verbose,
    )

    return (
        batch_results,
        processing_times,
        document_types_found,
        pipeline.batch_stats,
    )
