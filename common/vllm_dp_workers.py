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

from common import prompt_trace

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

    prompt_config, universal_fields, field_definitions = load_pipeline_configs(config.model_type)

    logger.info("Loading model: %s (GPU %d)", config.model_type, gpu_id)
    model_cm = load_model(config, app_config=app_cfg)
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
            img_start = time.time()
            # image_name is resolved BEFORE the call so the raw-prompt trace can
            # attribute each line to its image; without this context every trace
            # row carries image_name/pipeline/label = null.
            image_name = Path(image_path).name
            with prompt_trace.trace_context(
                image_name=image_name,
                pipeline="information_extraction",
                label="classify",
            ):
                result = processor.detect_and_classify_document(image_path, verbose=config.verbose)
            records.append(
                {
                    "image_path": image_path,
                    "image_name": image_name,
                    "document_type": result["document_type"],
                    "confidence": result.get("confidence", 1.0),
                    "raw_response": result.get("raw_response", ""),
                    "prompt_used": result.get("prompt_used", "detection"),
                    # Timed so the stage can report inference separately
                    # from wall clock, which includes engine startup.
                    "processing_time": time.time() - img_start,
                }
            )
            logger.info(
                "[%d/%d] %s -> %s",
                idx + 1,
                len(image_paths),
                image_name,
                result["document_type"],
            )

        # Tag every record with the rank that produced it. Workers run
        # concurrently, so the parent needs this to report the SLOWEST
        # worker's inference time — summing across workers would give total
        # compute and understate throughput by roughly the GPU count.
        for record in records:
            record["gpu_id"] = gpu_id
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
    from common.pipeline_ops import load_model
    from common.prompt_trace import effective_trace_path
    from common.turn_parsers import build_parser_registry
    from models.backends.vllm_backend import VllmBackend

    # Force single-GPU in this worker
    overrides = dict(cli_overrides)
    overrides["num_gpus"] = 1

    cfg_path = Path(config_path) if config_path else None
    app_cfg = AppConfig.load(overrides, config_path=cfg_path)
    config = app_cfg.pipeline

    logger.info("Loading model: %s (GPU %d)", config.model_type, gpu_id)
    model_cm = load_model(config, app_config=app_cfg)
    engine, _ = model_cm.__enter__()

    try:
        backend = VllmBackend(
            engine,
            model_type_key=config.model_type,
            chat_template=config.chat_template,
            trace_path=effective_trace_path(config),
            pre_tiling_enabled=config.pre_tiling_enabled,
            tile_image_size=config.pre_tiling_image_size,
            tile_use_thumbnail=config.pre_tiling_use_thumbnail,
        )
        generate_fn = backend.generate_for_graph

        workflow_path = Path(__file__).resolve().parent.parent / "prompts" / "workflows" / workflow_name
        with workflow_path.open() as f:
            definition = yaml.safe_load(f)

        parsers = build_parser_registry(fallback_type=app_cfg.classification_fallback_type)
        executor = GraphExecutor(generate_fn, parsers, budget_resolver=app_cfg.get_token_budget)

        records: list[dict[str, Any]] = []
        total = len(image_paths)

        for idx, image_path in enumerate(image_paths):
            image_name = Path(image_path).name
            img_start = time.time()
            doc_type = "UNKNOWN"

            try:
                # A graph run makes SEVERAL VLM calls per image, so without this
                # context the trace is an unattributable stream of prompts.
                with prompt_trace.trace_context(
                    image_name=image_name,
                    pipeline="information_extraction",
                    label=f"graph_{label}",
                ):
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

        # Tag every record with the rank that produced it. Workers run
        # concurrently, so the parent needs this to report the SLOWEST
        # worker's inference time — summing across workers would give total
        # compute and understate throughput by roughly the GPU count.
        for record in records:
            record["gpu_id"] = gpu_id
        return records
    finally:
        model_cm.__exit__(None, None, None)


def _extract_batch_records(
    processor: Any,
    batch: list[dict[str, Any]],
    *,
    verbose: bool,
) -> list[dict[str, Any]]:
    """Extract a batch of standard documents in one engine call.

    Returns the same record shape the per-image path writes — RAW
    responses, which the clean stage parses. The two paths must stay
    indistinguishable in raw_extractions.jsonl, or resume and re-cleaning
    break.

    A batch that raises is retried one document at a time, so an
    out-of-memory costs only the document that genuinely fails.
    """
    started = time.time()
    image_paths = [c["image_path"] for c in batch]

    try:
        results = processor.extract_batch(image_paths, batch, verbose=verbose)
        if len(results) != len(batch):
            raise ValueError(f"backend returned {len(results)} results for {len(batch)} images")
    except Exception as err:  # noqa: BLE001 - one batch must not end the run
        logger.warning(
            "Batch of %d %s failed (%s) — retrying one at a time",
            len(batch),
            batch[0]["document_type"],
            err,
        )
        records = []
        for classification in batch:
            img_start = time.time()
            try:
                result = processor.process_document_aware(
                    classification["image_path"], classification, verbose=verbose
                )
                raw, error = result.get("raw_response", ""), None
            except Exception as inner:  # noqa: BLE001
                logger.error("Error extracting %s: %s", classification["image_name"], inner)
                raw, error = "", str(inner)
            records.append(
                {
                    "image_name": classification["image_name"],
                    "image_path": classification["image_path"],
                    "document_type": classification["document_type"],
                    "raw_response": raw,
                    "processing_time": time.time() - img_start,
                    "prompt_used": classification["document_type"].lower() if error is None else "error",
                    "error": error,
                }
            )
        return records

    # Inside a batch the requests overlap, so per-image timing is not
    # separable; charge each an equal share of the call's wall clock.
    share = (time.time() - started) / len(batch)
    return [
        {
            "image_name": classification["image_name"],
            "image_path": classification["image_path"],
            "document_type": classification["document_type"],
            "raw_response": result.get("raw_response", ""),
            "processing_time": share,
            "prompt_used": classification["document_type"].lower(),
            "error": None,
        }
        for classification, result in zip(batch, results, strict=True)
    ]


def classified_extract_worker(
    gpu_id: int,
    image_paths: list[str],
    *,
    config_path: str | None,
    cli_overrides: dict[str, Any],
    classifications: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Worker: build vLLM engine + processor, run classified extraction.

    Handles standard documents via ``process_document_aware`` and bank
    statements via ``UnifiedBankExtractor``.  Each worker receives the
    full classifications list and filters to its own image chunk.

    Args:
        gpu_id: GPU rank (for logging).
        image_paths: Absolute paths to images assigned to this GPU.
        config_path: Path to run_config.yml (or None).
        cli_overrides: CLI args dict for AppConfig.load().
        classifications: Full list of classification records (filtered
            inside the worker to match ``image_paths``).

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

    prompt_config, universal_fields, field_definitions = load_pipeline_configs(config.model_type)

    # Filter classifications to this worker's image chunk
    chunk_set = set(image_paths)
    my_classifications = [c for c in classifications if c["image_path"] in chunk_set]

    logger.info("Loading model: %s (GPU %d)", config.model_type, gpu_id)
    model_cm = load_model(config, app_config=app_cfg)
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

        # Set up bank adapter if needed
        bank_adapter = None
        has_bank = any(c["document_type"].upper() == "BANK_STATEMENT" for c in my_classifications)
        if has_bank and config.bank_v2 and getattr(processor, "supports_multi_turn", True):
            bank_budget = app_cfg.get_image_budget("bank_statement")
            bank_adapter = UnifiedBankExtractor(
                generate_fn=processor.generate,
                verbose=effective_verbose,
                use_balance_correction=config.balance_correction,
                max_tiles=(bank_budget["max_tiles"] if config.pre_tiling_enabled else None),
                min_tiles=(bank_budget["min_tiles"] if config.pre_tiling_enabled else None),
            )
            logger.info("Bank adapter enabled (GPU %d)", gpu_id)

        records: list[dict[str, Any]] = []
        total = len(my_classifications)

        # Group into engine calls. Batching happens INSIDE the worker, after
        # run_dp has partitioned across GPUs, so each rank batches only its
        # own share and the two levels of parallelism compose.
        from common.extraction_batching import plan_extraction_batches, read_batch_sizes
        from common.pipeline_config import load_yaml_config
        from stages.extract import UNBATCHABLE_TYPES

        _, raw_config = load_yaml_config(
            cfg_path or Path(__file__).resolve().parent.parent / "config" / "run_config.yml"
        )
        batch_sizes = read_batch_sizes(
            raw_config,
            config_path=cfg_path or Path("config/run_config.yml"),
        )
        batches = plan_extraction_batches(
            my_classifications,
            batch_sizes=batch_sizes,
            unbatchable=UNBATCHABLE_TYPES,
        )
        logger.info(
            "GPU %d: %d documents in %d engine calls", gpu_id, len(my_classifications), len(batches)
        )

        idx = 0
        for batch in batches:
            # A multi-document batch is one engine call; singletons fall
            # through to the per-image path below unchanged.
            if len(batch) > 1:
                records.extend(_extract_batch_records(processor, batch, verbose=effective_verbose))
                idx += len(batch)
                logger.info("[%d/%d] batch of %d %s", idx, total, len(batch), batch[0]["document_type"])
                continue

            classification = batch[0]
            idx += 1
            image_path = classification["image_path"]
            image_name = classification["image_name"]
            doc_type = classification["document_type"]

            # Progress up front: a dense bank statement can take a while, so log
            # the start (not just the end) to track which of N is in flight.
            logger.info("[%d/%d] extracting %s (%s)...", idx, total, image_name, doc_type)

            img_start = time.time()

            # Attribution for the raw-prompt trace. Shared by both branches
            # below; a bank statement makes several VLM calls per image, so
            # without it the trace cannot be tied back to a document.
            trace_ctx = {
                "image_name": image_name,
                "pipeline": "information_extraction",
                "label": f"extract_{doc_type.lower()}",
            }

            try:
                if doc_type.upper() == "BANK_STATEMENT" and bank_adapter is not None:
                    with prompt_trace.trace_context(**trace_ctx):
                        schema_fields, metadata = bank_adapter.extract_bank_statement(image_path)
                    img_time = time.time() - img_start
                    raw_response_str = "\n".join(
                        f"{field}: {value}" for field, value in schema_fields.items()
                    )
                    strategy = metadata.get("strategy_used", "unknown")
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
                    with prompt_trace.trace_context(**trace_ctx):
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
                idx,
                total,
                image_name,
                doc_type,
                time.time() - img_start,
            )

        # Tag every record with the rank that produced it. Workers run
        # concurrently, so the parent needs this to report the SLOWEST
        # worker's inference time — summing across workers would give total
        # compute and understate throughput by roughly the GPU count.
        for record in records:
            record["gpu_id"] = gpu_id
        return records
    finally:
        model_cm.__exit__(None, None, None)
