"""Model loading and orchestrator construction, shared by both entry points.

The screen's single-GPU path and its per-GPU data-parallel worker build the
model the same way, so the seam lives here rather than in either of them.
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
    *,
    app_config: Any | None = None,
) -> Any:
    """Build the orchestrator for a loaded model.

    Delegates to the registered processor_creator for config.model_type.

    It used to also take a prompt-routing config, a universal field list and
    per-type field definitions. The screen supplied all three and used none of
    them -- they existed only because the orchestrator's extraction half
    demanded them, and demanding them kept the extraction prompt file, the
    field schema and the prompt catalogue alive behind a constructor argument.
    """
    from models.registry import get_model

    registration = get_model(config.model_type)
    return registration.processor_creator(model, tokenizer, config, app_config=app_config)
