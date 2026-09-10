"""Pipeline operations: model loading, processor creation, batch execution.

Extracted from cli.py so the stages can share the load/create/run seams
without importing the CLI.
"""

from typing import Any

from rich.console import Console

from common.pipeline_config import PipelineConfig

type BatchOutput = tuple[list[dict], list[float], dict[str, int], dict[str, float]]

console = Console()


def load_model(config: PipelineConfig, *, app_config: Any | None = None):
    """Context manager for loading and cleaning up model resources.

    Delegates to the registered loader for config.model_type, forwarding
    ``app_config`` so vLLM loaders can resolve per-model engine tuning from
    ``run_config.yml`` (mirrors ``create_processor``). HF loaders ignore it.
    """
    from models.registry import get_model

    registration = get_model(config.model_type)
    return registration.loader(config, app_config=app_config)


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
