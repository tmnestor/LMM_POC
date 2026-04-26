"""vLLM data-parallel worker functions.

Each function is a top-level callable invoked in a subprocess by
``vllm_dp.run_dp``. Workers build their own vLLM engine (TP=1)
inside a process pinned to a single GPU via CUDA_VISIBLE_DEVICES.

All arguments must be picklable (strings, dicts, ints -- no model
objects, no Path objects across the process boundary).
"""

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def classify_worker(
    gpu_id: int,
    image_paths: list[str],
    *,
    config_path: str | None,
    cli_overrides: dict[str, Any],
) -> list[dict[str, Any]]:
    """Worker: build vLLM engine + processor, classify each image.

    Args:
        gpu_id: GPU rank (for logging).
        image_paths: Absolute paths to images (strings).
        config_path: Path to run_config.yml (or None).
        cli_overrides: CLI args dict for AppConfig.load().

    Returns:
        List of classification record dicts.
    """
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(levelname)s [GPU {gpu_id}] %(name)s: %(message)s",
    )

    from cli import load_pipeline_configs
    from common.app_config import AppConfig
    from common.pipeline_ops import create_processor, load_model

    # Force single-GPU in this worker
    overrides = dict(cli_overrides)
    overrides["num_gpus"] = 1

    cfg_path = Path(config_path) if config_path else None
    app_cfg = AppConfig.load(overrides, config_path=cfg_path)
    config = app_cfg.pipeline

    prompt_config, universal_fields, field_definitions = load_pipeline_configs(
        config.model_type
    )

    logger.info("Loading model: %s (GPU %d)", config.model_type, gpu_id)
    model_cm = load_model(config)
    model, tokenizer = model_cm.__enter__()

    try:
        processor = create_processor(
            model,
            tokenizer,
            config,
            prompt_config,
            universal_fields,
            field_definitions,
            app_config=app_cfg,
        )

        records: list[dict[str, Any]] = []
        for idx, image_path in enumerate(image_paths):
            result = processor.detect_and_classify_document(
                image_path, verbose=config.verbose
            )
            image_name = Path(image_path).name
            records.append(
                {
                    "image_path": image_path,
                    "image_name": image_name,
                    "document_type": result["document_type"],
                    "confidence": result.get("confidence", 1.0),
                    "raw_response": result.get("raw_response", ""),
                    "prompt_used": result.get("prompt_used", "detection"),
                }
            )
            logger.info(
                "[%d/%d] %s -> %s",
                idx + 1,
                len(image_paths),
                image_name,
                result["document_type"],
            )

        return records
    finally:
        model_cm.__exit__(None, None, None)


def extract_worker(
    gpu_id: int,
    image_paths: list[str],
    *,
    config_path: str | None,
    cli_overrides: dict[str, Any],
    workflow_name: str,
    label: str,
) -> list[dict[str, Any]]:
    """Worker: build vLLM engine + GraphExecutor, extract per image.

    Args:
        gpu_id: GPU rank (for logging).
        image_paths: Absolute paths to images (strings).
        config_path: Path to run_config.yml (or None).
        cli_overrides: CLI args dict for AppConfig.load().
        workflow_name: YAML file under prompts/workflows/.
        label: Label for prompt_used prefix.

    Returns:
        List of extraction record dicts.
    """
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(levelname)s [GPU {gpu_id}] %(name)s: %(message)s",
    )

    import yaml

    from common.app_config import AppConfig
    from common.graph_executor import GraphExecutor
    from common.graph_generate import make_vllm_generate_fn
    from common.pipeline_ops import load_model
    from common.turn_parsers import build_parser_registry

    # Force single-GPU in this worker
    overrides = dict(cli_overrides)
    overrides["num_gpus"] = 1

    cfg_path = Path(config_path) if config_path else None
    app_cfg = AppConfig.load(overrides, config_path=cfg_path)
    config = app_cfg.pipeline

    logger.info("Loading model: %s (GPU %d)", config.model_type, gpu_id)
    model_cm = load_model(config)
    engine, _ = model_cm.__enter__()

    try:
        generate_fn = make_vllm_generate_fn(engine)

        workflow_path = (
            Path(__file__).resolve().parent.parent
            / "prompts"
            / "workflows"
            / workflow_name
        )
        with workflow_path.open() as f:
            definition = yaml.safe_load(f)

        executor = GraphExecutor(generate_fn, build_parser_registry())

        records: list[dict[str, Any]] = []
        total = len(image_paths)

        for idx, image_path in enumerate(image_paths):
            image_name = Path(image_path).name
            img_start = time.time()
            doc_type = "UNKNOWN"

            try:
                session = executor.run(
                    document_type="UNKNOWN",
                    definition=definition,
                    image_path=image_path,
                    image_name=image_name,
                )
                record = session.to_record()
                record["prompt_used"] = f"graph_{label}_{session.strategy}"
                records.append(record)
                doc_type = session.document_type
            except Exception as e:
                logger.error("Error processing %s: %s", image_name, e)
                img_time = time.time() - img_start
                records.append(
                    {
                        "image_name": image_name,
                        "image_path": image_path,
                        "document_type": "UNKNOWN",
                        "raw_response": "",
                        "processing_time": img_time,
                        "prompt_used": "error",
                        "error": str(e),
                    }
                )

            img_time = time.time() - img_start
            logger.info(
                "[%d/%d] %s: %s (%.1fs)",
                idx + 1,
                total,
                image_name,
                doc_type,
                img_time,
            )

        return records
    finally:
        model_cm.__exit__(None, None, None)
