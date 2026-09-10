"""Prompt routing and field lists needed to construct the orchestrator.

Lifted out of the old extraction CLI when that was deleted. The image-quality
screen needs none of this for its own work -- it sends one prompt and reads
seven answers -- but `create_processor` still requires a prompt config and a
field list to build a DocumentOrchestrator, so the screen supplies them.

That coupling is incidental rather than intended, and worth removing: the
orchestrator's extraction and detection paths are dead on this branch, and with
them these arguments could go too.
"""

from pathlib import Path
from typing import Any

from common.field_schema import get_field_schema
from common.prompt_catalog import PromptCatalog
from models.registry import get_model


class PipelinePromptError(RuntimeError):
    """Raised when prompt routing or field definitions cannot be built."""


def load_prompt_config(model_type: str = "internvl3-vllm") -> dict[str, Any]:
    """Build prompt routing config from PromptCatalog.

    Args:
        model_type: Registered model type, e.g. "internvl3-vllm".

    Returns:
        Routing config in the shape the orchestrator expects.

    Raises:
        PipelinePromptError: The catalog cannot build routing for this model.
    """
    catalog = PromptCatalog()

    try:
        routing = catalog.build_extraction_routing(model_type)
    except (FileNotFoundError, ValueError) as err:
        raise PipelinePromptError(
            f"Cannot build prompt routing.\n"
            f"  What:        {err}\n"
            f"  Where:       prompts/, for model type {model_type!r}\n"
            f"  Expected:    the prompt file named by the model registration to exist.\n"
            f"  How to fix:  check the model type is registered and its prompt file is present."
        ) from None

    registration = get_model(model_type)
    root = Path(__file__).resolve().parent.parent
    extraction_path = str(root / "prompts" / registration.prompt_file)
    detection_path = str(root / "prompts" / "document_type_detection.yaml")

    return {
        "detection_file": detection_path,
        "detection_key": "detection",
        "extraction_files": {doc_type: extraction_path for doc_type in routing},
    }


def load_pipeline_configs(
    model_type: str = "internvl3-vllm",
) -> tuple[dict[str, Any], list[str], dict[str, list[str]]]:
    """Load prompt configuration and build the universal field list.

    Args:
        model_type: Registered model type.

    Returns:
        `(prompt_config, sorted universal_fields, field_definitions)`.

    Raises:
        PipelinePromptError: No field definitions could be loaded.
    """
    prompt_config = load_prompt_config(model_type)
    field_definitions = get_field_schema().get_all_doc_type_fields()

    all_fields: set[str] = set()
    for fields in field_definitions.values():
        all_fields.update(fields)
    universal_fields = sorted(all_fields)

    if not universal_fields:
        raise PipelinePromptError(
            "No field definitions found.\n"
            "  What:        config/field_definitions.yaml yielded no fields.\n"
            "  Where:       config/field_definitions.yaml -> document_fields\n"
            "  Expected:    a document_fields section with per-type field lists.\n"
            "  How to fix:  restore the document_fields section."
        )

    return prompt_config, universal_fields, field_definitions
