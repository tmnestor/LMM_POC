"""HF data-parallel worker functions.

Each function is a top-level callable invoked in a subprocess by
``vllm_dp.run_dp``. Workers build their own HF model (single-GPU)
inside a process pinned to a single GPU via CUDA_VISIBLE_DEVICES.

All arguments must be picklable (strings, dicts, ints -- no model
objects, no Path objects across the process boundary).
"""

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def extract_worker(
    gpu_id: int,
    image_paths: list[str],
    *,
    config_path: str | None,
    cli_overrides: dict[str, Any],
    classifications: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Worker: build HF model + processor, extract fields per image.

    Args:
        gpu_id: GPU rank (for logging).
        image_paths: Absolute paths to images (strings) for this chunk.
        config_path: Path to run_config.yml (or None).
        cli_overrides: CLI args dict for AppConfig.load().
        classifications: Full list of classification dicts from Stage 1.
            Each worker filters to its own chunk via image_path lookup.

    Returns:
        List of extraction record dicts.
    """
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(levelname)s [GPU {gpu_id}] %(name)s: %(message)s",
    )

    from cli import load_pipeline_configs
    from common.app_config import AppConfig
    from common.pipeline_ops import create_processor, load_model
    from common.unified_bank_extractor import UnifiedBankExtractor

    # Force single-GPU in this worker
    overrides = dict(cli_overrides)
    overrides["num_gpus"] = 1

    cfg_path = Path(config_path) if config_path else None
    app_cfg = AppConfig.load(overrides, config_path=cfg_path)
    config = app_cfg.pipeline
    effective_verbose = config.verbose

    prompt_config, universal_fields, field_definitions = load_pipeline_configs(
        config.model_type
    )

    # Build lookup: image_path -> classification dict
    cls_lookup: dict[str, dict[str, Any]] = {
        c["image_path"]: c for c in classifications
    }

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

        # Set up bank adapter if any bank statements in this chunk
        bank_adapter = None
        has_bank = any(
            cls_lookup.get(p, {}).get("document_type", "").upper() == "BANK_STATEMENT"
            for p in image_paths
        )
        if (
            has_bank
            and config.bank_v2
            and getattr(processor, "supports_multi_turn", True)
        ):
            bank_adapter = UnifiedBankExtractor(
                generate_fn=processor.generate,
                verbose=effective_verbose,
                use_balance_correction=config.balance_correction,
            )
            logger.info("Bank adapter enabled (GPU %d)", gpu_id)

        records: list[dict[str, Any]] = []
        total = len(image_paths)

        for idx, image_path in enumerate(image_paths):
            image_name = Path(image_path).name
            classification = cls_lookup.get(image_path, {})
            doc_type = classification.get("document_type", "UNKNOWN")
            img_start = time.time()

            try:
                if doc_type.upper() == "BANK_STATEMENT" and bank_adapter is not None:
                    schema_fields, metadata = bank_adapter.extract_bank_statement(
                        image_path
                    )
                    raw_response_str = "\n".join(
                        f"{field}: {value}" for field, value in schema_fields.items()
                    )
                    strategy = metadata.get("strategy_used", "unknown")
                    img_time = time.time() - img_start
                    records.append(
                        {
                            "image_name": image_name,
                            "image_path": image_path,
                            "document_type": doc_type,
                            "raw_response": raw_response_str,
                            "processing_time": img_time,
                            "prompt_used": f"unified_bank_{strategy}",
                            "error": None,
                        }
                    )
                else:
                    result = processor.process_document_aware(
                        image_path, classification, verbose=effective_verbose
                    )
                    img_time = time.time() - img_start
                    records.append(
                        {
                            "image_name": image_name,
                            "image_path": image_path,
                            "document_type": doc_type,
                            "raw_response": result.get("raw_response", ""),
                            "processing_time": img_time,
                            "prompt_used": doc_type.lower(),
                            "error": None,
                        }
                    )
            except Exception as e:
                logger.error("Error extracting %s: %s", image_name, e)
                img_time = time.time() - img_start
                records.append(
                    {
                        "image_name": image_name,
                        "image_path": image_path,
                        "document_type": doc_type,
                        "raw_response": "",
                        "processing_time": img_time,
                        "prompt_used": "error",
                        "error": str(e),
                    }
                )

            logger.info(
                "[%d/%d] %s: %s (%.1fs)",
                idx + 1,
                total,
                image_name,
                doc_type,
                time.time() - img_start,
            )

        return records
    finally:
        model_cm.__exit__(None, None, None)
