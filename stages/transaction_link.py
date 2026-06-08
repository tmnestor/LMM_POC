# ruff: noqa: B008 - typer.Option in defaults is the standard Typer pattern
"""Stage: Receipt-to-bank-statement transaction linking (matcher-first, hybrid).

Restores the original transaction-linking purpose as a STANDALONE task that
coexists with the trust-distribution linking in ``stages/link.py`` — this
module never touches the trust pipeline.

This is an ADDITIVE step that runs after extract+clean. It does NOT re-extract:
the extract stage already pulled every bank-statement transaction row, so this
stage works from the cleaned records.

Control flow (matcher-first, VLM-fallback — the inverse of the original
VLM-first design):

    1. Group cleaned records by case ID (regex on filename).
    2. Per case, build a flat transaction index from the bank-statement rows
       and a list of receipt summaries.
    3. Run the pure algorithmic matcher (amount/date/description gates) with a
       one-to-one constraint. Receipts that match at or above
       ``hybrid_min_confidence`` are accepted immediately — no model call.
    4. Receipts that DON'T confidently match fall through to a targeted VLM
       lookup: the bank-statement image + the receipt's details as text
       ("find the debit matching amount=X, date=Y, supplier=Z"). These
       unmatched receipts skew toward the unreliable tail of long transaction
       tables, which is exactly where the model beats the extracted rows.
    5. Write transaction_links.jsonl.

Usage:
    python -m stages.transaction_link \
        --extractions /artifacts/cleaned_extractions.jsonl \
        --output      /artifacts/transaction_links.jsonl \
        --data-dir    /path/to/images \
        --config      config/run_config.yml \
        --model       internvl3-vllm
"""

import functools
import logging
import re
import time
from pathlib import Path
from typing import Any

import typer

from common.transaction_matcher import (
    BankTransaction,
    LinkResult,
    ReceiptSummary,
    build_receipt_summaries,
    build_transaction_index,
    group_by_case,
    match_all_receipts,
)
from common.vlm_linker import LinkPrompt, call_vlm_linker, load_link_prompt

from .io import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)
app = typer.Typer()

_RECEIPT_TYPES = frozenset({"RECEIPT", "INVOICE"})
_BANK_TYPES = frozenset({"BANK_STATEMENT"})

# Higher = better. Used to threshold matcher confidence against
# linking.hybrid_min_confidence.
_CONFIDENCE_ORDER = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}
_ALLOWED_MIN_CONFIDENCE = ("HIGH", "MEDIUM", "LOW")


# ---------------------------------------------------------------------------
# Configuration (fail-fast, 4-element diagnostics)
# ---------------------------------------------------------------------------


def _diagnostic(what: str, where: str, example: str, fix: str) -> str:
    """Assemble a 4-element diagnostic error message."""
    return f"What: {what}\nWhere: {where}\nExpected: {example}\nHow to fix: {fix}"


def _require(linking: dict[str, Any], key: str, config_path: Path, example: str) -> Any:
    """Return ``linking[key]`` or raise a 4-element diagnostic ValueError."""
    if key not in linking:
        raise ValueError(
            _diagnostic(
                what=f"required key 'linking.{key}' is missing.",
                where=f"{config_path} -> linking.{key}",
                example=example,
                fix=f"add '{key}' under the 'linking:' section in {config_path}.",
            )
        )
    return linking[key]


def _load_linking_config(config_path: Path | None) -> dict[str, Any]:
    """Load and validate the ``linking:`` block from run_config.yml.

    Every key the orchestration reads is REQUIRED; a missing or invalid value
    fails fast with a 4-element diagnostic (what / where / valid example /
    remediation) before any work begins.

    Args:
        config_path: Path to run_config.yml. Defaults to the repo config.

    Returns:
        The validated ``linking`` mapping (with the compiled regex available
        under the original ``case_key_pattern`` string).
    """
    import yaml

    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "run_config.yml"

    if not config_path.exists():
        raise FileNotFoundError(
            _diagnostic(
                what="the configuration file is missing.",
                where=str(config_path),
                example="a YAML file with a 'linking:' section.",
                fix=f"create {config_path} or pass --config with a valid path.",
            )
        )

    with config_path.open() as f:
        raw = yaml.safe_load(f) or {}

    if "linking" not in raw:
        raise ValueError(
            _diagnostic(
                what="the top-level key 'linking' is absent.",
                where=f"{config_path} -> linking",
                example=(
                    "linking:\n"
                    '    case_key_pattern: "^(?P<case>[^_]+)_"\n'
                    "    vlm_prompt: single_receipt_link\n"
                    "    vlm_max_tokens: 4096\n"
                    "    vlm_temperature: 0.0\n"
                    "    hybrid_amount_tolerance: 0.01\n"
                    "    hybrid_date_window_days: 5\n"
                    "    hybrid_description_threshold: 0.3\n"
                    "    hybrid_min_confidence: LOW"
                ),
                fix=f"add a 'linking:' section to {config_path}.",
            )
        )

    linking = raw["linking"]

    # case_key_pattern — must compile and expose a (?P<case>...) group.
    pattern_str = _require(
        linking,
        "case_key_pattern",
        config_path,
        'a regex with a (?P<case>...) named group, e.g.: "^(?P<case>[^_]+)_"',
    )
    try:
        compiled = re.compile(pattern_str)
    except re.error as exc:
        raise ValueError(
            _diagnostic(
                what=f"'linking.case_key_pattern' is not a valid regex: {exc}.",
                where=f"{config_path} -> linking.case_key_pattern (pattern={pattern_str!r})",
                example='a valid regex with a (?P<case>...) group, e.g.: "^(?P<case>[^_]+)_"',
                fix="correct the regex in 'linking.case_key_pattern'.",
            )
        ) from None
    if "case" not in compiled.groupindex:
        raise ValueError(
            _diagnostic(
                what=f"'linking.case_key_pattern' has no named group 'case': {pattern_str!r}.",
                where=f"{config_path} -> linking.case_key_pattern",
                example='regex with a (?P<case>...) group, e.g.: "^(?P<case>[^_]+)_"',
                fix="add a (?P<case>...) named group to the regex.",
            )
        )

    _require(linking, "vlm_prompt", config_path, "a prompt file name, e.g.: single_receipt_link")
    _require(linking, "vlm_max_tokens", config_path, "an integer, e.g.: 4096")

    # vlm_temperature — the shared extract seam (processor.generate) decodes
    # deterministically (temperature is fixed at 0 in the backend), so 0.0 is
    # the only honourable value. Reject anything else rather than silently
    # ignore it.
    vlm_temperature = _require(linking, "vlm_temperature", config_path, "0.0 (deterministic decoding)")
    if float(vlm_temperature) != 0.0:
        raise ValueError(
            _diagnostic(
                what=(
                    f"'linking.vlm_temperature' is {vlm_temperature!r}, but the linking VLM seam "
                    "(processor.generate, shared with extract) decodes deterministically at "
                    "temperature 0 and cannot honour another value."
                ),
                where=f"{config_path} -> linking.vlm_temperature",
                example="0.0",
                fix="set 'linking.vlm_temperature: 0.0' in run_config.yml.",
            )
        )

    _require(linking, "hybrid_amount_tolerance", config_path, "a dollar tolerance, e.g.: 0.01")
    _require(linking, "hybrid_date_window_days", config_path, "an integer day window, e.g.: 5")
    _require(linking, "hybrid_description_threshold", config_path, "a 0..1 fraction, e.g.: 0.3")

    min_confidence = _require(
        linking,
        "hybrid_min_confidence",
        config_path,
        "one of HIGH | MEDIUM | LOW, e.g.: LOW",
    )
    if str(min_confidence).upper() not in _ALLOWED_MIN_CONFIDENCE:
        raise ValueError(
            _diagnostic(
                what=f"'linking.hybrid_min_confidence' is {min_confidence!r}, not an allowed value.",
                where=f"{config_path} -> linking.hybrid_min_confidence",
                example=f"one of {', '.join(_ALLOWED_MIN_CONFIDENCE)}, e.g.: LOW",
                fix="set 'linking.hybrid_min_confidence' to HIGH, MEDIUM, or LOW.",
            )
        )

    return dict(linking)


# ---------------------------------------------------------------------------
# Record building
# ---------------------------------------------------------------------------


def _confidence_level(confidence: str) -> int:
    """Map a confidence string to its ranked level (higher = stronger)."""
    return _CONFIDENCE_ORDER.get(confidence.upper(), 0)


def _empty_match_fields() -> dict[str, Any]:
    """The null bank-side fields shared by every output record."""
    return {
        "bank_statement_file": None,
        "bank_transaction_date": None,
        "bank_transaction_description": None,
        "bank_transaction_amount": None,
    }


def _receipt_date_str(receipt: ReceiptSummary) -> str:
    """Render the receipt date as DD/MM/YYYY, or '' when absent."""
    return receipt.date.strftime("%d/%m/%Y") if receipt.date else ""


def _base_record(receipt: ReceiptSummary, case_id: str) -> dict[str, Any]:
    """Build the common output-record skeleton for one receipt."""
    record: dict[str, Any] = {
        "image_name": receipt.image_name,
        "case_id": case_id,
        "document_type": receipt.document_type or "RECEIPT",
        "supplier_name": receipt.supplier_name,
        "receipt_date": _receipt_date_str(receipt),
        "receipt_total": receipt.total,
        "matched": False,
        "confidence": "NONE",
        "match_scores": {},
        "reasoning": "",
    }
    record.update(_empty_match_fields())
    return record


def _apply_transaction(record: dict[str, Any], txn: BankTransaction) -> None:
    """Populate the bank-side fields of ``record`` from a matched transaction."""
    record["bank_statement_file"] = txn.source_image
    record["bank_transaction_date"] = txn.date.strftime("%d/%m/%Y") if txn.date else None
    record["bank_transaction_description"] = txn.description
    record["bank_transaction_amount"] = txn.amount


def _link_result_to_record(result: LinkResult, case_id: str) -> dict[str, Any]:
    """Convert an algorithmic LinkResult into an output record."""
    record = _base_record(result.receipt, case_id)
    record["matched"] = result.matched
    record["confidence"] = result.confidence
    record["match_scores"] = result.match_scores
    record["reasoning"] = result.reasoning
    if result.matched and result.transaction is not None:
        _apply_transaction(record, result.transaction)
    return record


# ---------------------------------------------------------------------------
# VLM fallback helpers
# ---------------------------------------------------------------------------


def _parse_amount_str(s: str) -> float | None:
    """Parse a currency/amount string to an absolute float, None on failure."""
    if not s or s.strip().upper() == "NOT_FOUND":
        return None
    cleaned = re.sub(r"[$,\s]", "", s.strip())
    try:
        return abs(float(cleaned))
    except ValueError:
        return None


def _is_bank_row_echo(match: dict[str, str]) -> bool:
    """Detect a hallucinated block that just echoes a bank-statement row.

    A legitimate match never has RECEIPT_STORE exactly equal to
    TRANSACTION_DESCRIPTION (receipts carry full names; banks abbreviate).
    """
    store = (match.get("RECEIPT_STORE") or "").strip().upper()
    desc = (match.get("TRANSACTION_DESCRIPTION") or "").strip().upper()
    if not store or not desc or desc == "NOT_FOUND":
        return False
    return store == desc


def _bank_images_for_case(bank_records: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any] | None]]:
    """List (image_name, bank_columns) pairs for a case's bank statements."""
    seen: dict[str, dict[str, Any] | None] = {}
    for record in bank_records:
        name = record.get("image_name", "")
        if not name:
            continue
        if name not in seen:
            seen[name] = record.get("bank_columns")
    return sorted(seen.items())


def _apply_vlm_match(record: dict[str, Any], match: dict[str, str], bank_image_name: str) -> None:
    """Update ``record`` in place from a FOUND VLM match block."""
    record["matched"] = True
    record["confidence"] = match.get("CONFIDENCE", "MEDIUM") or "MEDIUM"
    record["reasoning"] = f"VLM fallback: {match.get('REASONING', '')}".strip()
    record["bank_statement_file"] = bank_image_name
    record["bank_transaction_date"] = match.get("TRANSACTION_DATE") or None
    record["bank_transaction_description"] = match.get("TRANSACTION_DESCRIPTION") or None
    record["bank_transaction_amount"] = _parse_amount_str(match.get("TRANSACTION_AMOUNT", ""))


def _attempt_on_image(
    record: dict[str, Any],
    receipt: ReceiptSummary,
    bank_image_name: str,
    bank_columns: dict[str, Any] | None,
    *,
    generate_fn: Any,
    data_dir: Path,
    prompt: LinkPrompt,
    max_tokens: int,
) -> bool:
    """Query ONE statement image for ONE receipt; apply + return True on FOUND.

    A missing image, a generate error, a bank-row echo, or a NOT_FOUND all
    return False without aborting — the caller moves on to the next statement.
    """
    bank_path = data_dir / bank_image_name
    if not bank_path.exists():
        logger.warning("Bank image not found, skipping: %s", bank_path)
        return False

    try:
        matches = call_vlm_linker(
            generate_fn,
            bank_path,
            receipt,
            max_tokens=max_tokens,
            prompt=prompt,
            bank_columns=bank_columns,
        )
    except Exception:
        logger.exception(
            "VLM fallback failed for %s x %s — leaving NOT_FOUND",
            receipt.image_name,
            bank_image_name,
        )
        return False

    for match in matches:
        if _is_bank_row_echo(match):
            continue
        if match.get("MATCHED_TRANSACTION", "NOT_FOUND") == "FOUND":
            _apply_vlm_match(record, match, bank_image_name)
            logger.info(
                "VLM fallback matched %s (case %s) on %s",
                receipt.image_name,
                record["case_id"],
                bank_image_name,
            )
            return True
    return False


def _run_case_fallback(
    case_items: list[tuple[int, ReceiptSummary]],
    bank_images: list[tuple[str, dict[str, Any] | None]],
    all_results: list[dict[str, Any]],
    *,
    attempt_fn: Any,
) -> int:
    """Per-statement fallback for one case.

    Iterates STATEMENTS in the outer loop and the case's still-unmatched
    receipts in the inner loop, so every query against a given statement is
    issued consecutively — keeping vLLM's shared-prefix cache for that statement
    warm. A receipt matched on one statement is dropped from ``remaining`` so it
    is never re-queried against a later statement (first-FOUND-wins, preserving
    the original semantics). Returns the number of receipts matched by the VLM.
    """
    remaining = list(case_items)
    matched_count = 0
    for bank_image_name, bank_columns in bank_images:
        if not remaining:
            break
        still: list[tuple[int, ReceiptSummary]] = []
        for record_index, receipt in remaining:
            record = all_results[record_index]
            if attempt_fn(record, receipt, bank_image_name, bank_columns):
                matched_count += 1
            else:
                still.append((record_index, receipt))
        remaining = still

    for record_index, _receipt in remaining:
        record = all_results[record_index]
        record["reasoning"] = (
            record["reasoning"] or "No amount match in extracted rows; VLM lookup found no debit."
        )
    return matched_count


def _log_cache_summary(processor: Any, elapsed_s: float) -> None:
    """Log the fallback pass's wall-clock timing + prefix-cache hit rate."""
    summary = processor.cache_hit_summary()
    calls = int(summary.get("calls", 0))
    per_call_ms = (elapsed_s / calls * 1000.0) if calls else 0.0
    logger.info(
        "fallback timing: %d VLM calls in %.1fs (%.0f ms/call)",
        calls,
        elapsed_s,
        per_call_ms,
    )
    if summary.get("available"):
        logger.info(
            "prefix-cache: %d/%d fallback prompt tokens served from cache (%.0f%%)",
            summary["cached_prompt_tokens"],
            summary["total_prompt_tokens"],
            summary["hit_ratio"] * 100,
        )
    else:
        logger.info("prefix-cache: unavailable on this vLLM build")


def _build_processor(model_type: str | None, data_dir: Path, config_path: Path | None) -> Any:
    """Load the model and wrap it in a processor exposing ``.generate``.

    Mirrors the single-GPU path of ``stages.extract`` so the linking VLM call
    routes through the same ``processor.generate(image, prompt, max_tokens=...)``
    seam — NOT a freshly-built engine.
    """
    from cli import load_pipeline_configs
    from common.app_config import AppConfig
    from common.pipeline_ops import create_processor, load_model

    cli_args: dict[str, Any] = {"data_dir": str(data_dir), "output_dir": str(data_dir)}
    if model_type:
        cli_args["model_type"] = model_type

    app_cfg = AppConfig.load(cli_args, config_path=config_path)
    config = app_cfg.pipeline

    prompt_config, universal_fields, field_definitions = load_pipeline_configs(config.model_type)

    logger.info("Loading model for VLM linking fallback (model=%s)...", config.model_type)
    model_cm = load_model(config, app_config=app_cfg)
    model, tokenizer = model_cm.__enter__()
    processor = create_processor(
        model,
        tokenizer,
        config,
        prompt_config,
        universal_fields,
        field_definitions,
        app_config=app_cfg,
    )
    logger.info("Model loaded for VLM linking fallback.")
    return processor, model_cm


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(
    extractions_path: Path,
    output_path: Path,
    *,
    data_dir: Path,
    config_path: Path | None = None,
    model_type: str | None = None,
) -> Path:
    """Link receipts to bank-statement debits: matcher-first, VLM fallback.

    Args:
        extractions_path: Path to cleaned_extractions.jsonl (clean stage output).
        output_path: Path to write transaction_links.jsonl.
        data_dir: Image directory (source images, required for the VLM fallback).
        config_path: Optional explicit config file path.
        model_type: Model type string (e.g. "internvl3-vllm"). If None, the
            VLM fallback falls back to YAML model.type.

    Returns:
        Path to the written transaction_links.jsonl.
    """
    linking_cfg = _load_linking_config(config_path)
    case_pattern = re.compile(linking_cfg["case_key_pattern"])
    amount_tolerance = float(linking_cfg["hybrid_amount_tolerance"])
    date_window = int(linking_cfg["hybrid_date_window_days"])
    desc_threshold = float(linking_cfg["hybrid_description_threshold"])
    min_conf_level = _confidence_level(str(linking_cfg["hybrid_min_confidence"]))
    vlm_max_tokens = int(linking_cfg["vlm_max_tokens"])
    vlm_prompt_name = linking_cfg["vlm_prompt"]

    records = read_jsonl(extractions_path)
    if not records:
        raise FileNotFoundError(
            _diagnostic(
                what=f"no records found in {extractions_path}.",
                where=str(extractions_path),
                example="a non-empty cleaned_extractions.jsonl from the clean stage.",
                fix="run the classify -> extract -> clean stages before transaction_link.",
            )
        )
    logger.info("Read %d cleaned extraction records from %s", len(records), extractions_path)

    case_groups = group_by_case(records, case_pattern)
    logger.info("Found %d case groups", len(case_groups))

    all_results: list[dict[str, Any]] = []
    # Receipts the matcher could not confidently link: (record_index, case_id, receipt).
    fallback_queue: list[tuple[int, str, ReceiptSummary]] = []
    # Bank images per case for the fallback pass.
    bank_images_by_case: dict[str, list[tuple[str, dict[str, Any] | None]]] = {}

    total_receipts = 0
    matcher_matched = 0

    # -- Pass 1: algorithmic matcher (no model) --------------------------------
    for case_id, case_records in sorted(case_groups.items()):
        receipt_records = [r for r in case_records if r.get("document_type", "").upper() in _RECEIPT_TYPES]
        bank_records = [r for r in case_records if r.get("document_type", "").upper() in _BANK_TYPES]

        if not receipt_records:
            logger.debug("Case %s: no receipts/invoices, skipping", case_id)
            continue

        receipts: list[ReceiptSummary] = []
        for record in receipt_records:
            receipts.extend(build_receipt_summaries(record))
        total_receipts += len(receipts)

        if not bank_records:
            logger.warning(
                "Case %s: no bank statements -- all %d receipts unmatched", case_id, len(receipts)
            )
            for receipt in receipts:
                record = _base_record(receipt, case_id)
                record["reasoning"] = "No bank statements available in this case"
                all_results.append(record)
            continue

        bank_images_by_case[case_id] = _bank_images_for_case(bank_records)

        index = build_transaction_index(bank_records)
        results = match_all_receipts(
            receipts,
            index,
            amount_tolerance=amount_tolerance,
            date_window_days=date_window,
            description_threshold=desc_threshold,
        )

        for result in results:
            record = _link_result_to_record(result, case_id)
            confident = result.matched and _confidence_level(result.confidence) >= min_conf_level
            if confident:
                matcher_matched += 1
            else:
                # Not confidently matched -> queue for VLM fallback.
                record["matched"] = False
                if record["confidence"] != "NONE":
                    record["confidence"] = "NONE"
                fallback_queue.append((len(all_results), case_id, result.receipt))
            all_results.append(record)

    logger.info(
        "Matcher pass: %d/%d receipts matched at >= %s; %d queued for VLM fallback",
        matcher_matched,
        total_receipts,
        str(linking_cfg["hybrid_min_confidence"]).upper(),
        len(fallback_queue),
    )

    # -- Pass 2: targeted VLM fallback (lazy model load) -----------------------
    actionable = [item for item in fallback_queue if bank_images_by_case.get(item[1])]
    vlm_matched = 0
    if actionable:
        processor, model_cm = _build_processor(model_type, data_dir, config_path)
        prompt = load_link_prompt(vlm_prompt_name)
        # Group by case so a case's receipts are queried one statement at a time
        # (consecutive same-image calls keep the shared-prefix cache warm).
        by_case: dict[str, list[tuple[int, ReceiptSummary]]] = {}
        for record_index, case_id, receipt in actionable:
            by_case.setdefault(case_id, []).append((record_index, receipt))
        try:
            attempt_fn = functools.partial(
                _attempt_on_image,
                generate_fn=processor.generate,
                data_dir=data_dir,
                prompt=prompt,
                max_tokens=vlm_max_tokens,
            )
            fallback_start = time.perf_counter()
            for case_id, items in by_case.items():
                vlm_matched += _run_case_fallback(
                    items, bank_images_by_case[case_id], all_results, attempt_fn=attempt_fn
                )
            _log_cache_summary(processor, time.perf_counter() - fallback_start)
        finally:
            model_cm.__exit__(None, None, None)
        logger.info("VLM fallback: %d/%d queued receipts recovered", vlm_matched, len(actionable))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = write_jsonl(output_path, all_results)
    total_matched = matcher_matched + vlm_matched
    logger.info("Wrote %d linking results to %s", count, output_path)
    logger.info(
        "Summary: %d/%d receipts matched (%.1f%%) — %d matcher, %d VLM fallback",
        total_matched,
        total_receipts,
        (total_matched / total_receipts * 100) if total_receipts else 0.0,
        matcher_matched,
        vlm_matched,
    )

    return output_path


@app.command()
def main(
    extractions: Path = typer.Option(
        ...,
        "--extractions",
        "-i",
        help="Path to cleaned_extractions.jsonl from the clean stage",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Path to write transaction_links.jsonl",
    ),
    data_dir: Path = typer.Option(..., "--data-dir", help="Image directory (required for VLM fallback)"),
    config: Path | None = typer.Option(None, "--config", help="YAML configuration file"),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Model type (e.g. internvl3-vllm). Default: YAML model.type",
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    """Stage: Link receipts to bank-statement debits (matcher-first, VLM fallback)."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    run(
        extractions,
        output,
        data_dir=data_dir,
        config_path=config,
        model_type=model,
    )


if __name__ == "__main__":
    app()
