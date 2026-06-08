# ruff: noqa: B008 - typer.Option in defaults is the standard Typer pattern
"""Trust document type classification stage (GPU).

Scans a flat directory for CASEXXX_* documents, uses VLM inference to classify
each into one of 4 trust document types, then assembles a quads CSV so
trust_extract works unchanged.

Pipeline position:
    trust_classify (GPU) -> trust_extract (GPU) -> trust_clean (CPU) -> trust_evaluate (CPU)

Usage:
    python -m stages.trust_classify \\
        --data-dir /flat_docs \\
        --output-dir /artifacts \\
        [--model internvl3-vllm]
"""

import logging
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from common.trust_classify_parser import parse_trust_classification
from stages.io import StreamingJsonlWriter, read_completed_images

logger = logging.getLogger(__name__)
app = typer.Typer()

# Supported document extensions (images + PDF)
TRUST_DOC_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
    ".bmp",
    ".webp",
    ".pdf",
}

# Case ID regex — extracts CASEXXX from filenames
_CASE_ID_RE = re.compile(r"(CASE\d+)")

# Quads CSV columns (must match link.py._read_quads expectations)
_QUADS_COLUMNS = ["case_id", "trust_return", "distribution_stmt", "income_schedule", "beneficiary_itr"]

# Document type -> quads column name
_TYPE_TO_COLUMN = {
    "TRUST_RETURN": "trust_return",
    "DISTRIBUTION_STMT": "distribution_stmt",
    "INCOME_SCHEDULE": "income_schedule",
    "BENEFICIARY_ITR": "beneficiary_itr",
}


def extract_case_id(filename: str) -> str:
    """Extract case ID from a filename (e.g. 'CASE201_001.pdf' -> 'CASE201').

    Args:
        filename: Filename or full path string.

    Returns:
        Case ID string, or empty string if no match.
    """
    basename = Path(filename).name
    match = _CASE_ID_RE.match(basename)
    return match.group(1) if match else ""


def discover_trust_documents(data_dir: Path) -> list[Path]:
    """Discover trust documents in a directory.

    Finds all files matching CASE\\d+_* prefix with supported extensions
    (images + PDF). Returns sorted list.

    Args:
        data_dir: Directory to scan.

    Returns:
        Sorted list of document paths.
    """
    docs: list[Path] = []
    for path in data_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in TRUST_DOC_EXTENSIONS:
            continue
        if not _CASE_ID_RE.match(path.name):
            continue
        docs.append(path)
    return sorted(docs, key=lambda p: p.name)


def group_by_case(classifications: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group classification records by case_id.

    Args:
        classifications: List of classification dicts with 'case_id' key.

    Returns:
        Dict mapping case_id -> list of classification records.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in classifications:
        case_id = record.get("case_id", "")
        if case_id:
            groups.setdefault(case_id, []).append(record)
    return groups


def assemble_quads(
    groups: dict[str, list[dict[str, Any]]],
    quads_path: Path,
    incomplete_path: Path,
) -> tuple[int, int]:
    """Assemble quads CSV from grouped classification records.

    Complete cases (all 4 types present) go to quads_path.
    Incomplete cases go to incomplete_path.

    For duplicate types within a case, the highest-confidence record wins.

    Args:
        groups: Dict mapping case_id -> list of classification records.
        quads_path: Output path for complete quads CSV.
        incomplete_path: Output path for incomplete quads CSV.

    Returns:
        Tuple of (complete_count, incomplete_count).
    """
    slot_cols = list(_TYPE_TO_COLUMN.values())
    all_records = [rec for recs in groups.values() for rec in recs]

    if not all_records:
        empty = pd.DataFrame(columns=_QUADS_COLUMNS)
        quads_path.parent.mkdir(parents=True, exist_ok=True)
        empty.to_csv(quads_path, index=False)
        incomplete_path.parent.mkdir(parents=True, exist_ok=True)
        empty.to_csv(incomplete_path, index=False)
        logger.info("Quads assembly: 0 complete, 0 incomplete")
        return 0, 0

    df = pd.DataFrame(all_records)

    # Filter to known document types
    known = df[df["document_type"].isin(_TYPE_TO_COLUMN)].copy()

    # Log duplicate warnings before dedup
    dupes = known.duplicated(subset=["case_id", "document_type"], keep=False)
    if dupes.any():
        for (case_id, doc_type), grp in known[dupes].groupby(["case_id", "document_type"]):
            sorted_grp = grp.sort_values("confidence", ascending=False)
            kept = sorted_grp.iloc[0]
            for _, row in sorted_grp.iloc[1:].iterrows():
                logger.warning(
                    "Case %s: duplicate %s — keeping %s (confidence %.2f) over %s (%.2f)",
                    case_id,
                    doc_type,
                    kept["image_name"],
                    kept["confidence"],
                    row["image_name"],
                    row["confidence"],
                )

    # Keep highest confidence per (case_id, document_type)
    best = known.sort_values("confidence", ascending=False).drop_duplicates(
        subset=["case_id", "document_type"]
    )

    if best.empty:
        pivoted = pd.DataFrame(columns=_QUADS_COLUMNS)
    else:
        # Pivot: case_id rows × document_type columns → image_name values
        pivoted = best.pivot(index="case_id", columns="document_type", values="image_name")
        pivoted = pivoted.rename(columns=_TYPE_TO_COLUMN)
        for col in slot_cols:
            if col not in pivoted.columns:
                pivoted[col] = ""
        pivoted = pivoted.fillna("").sort_index().reset_index()
        pivoted = pivoted[_QUADS_COLUMNS]

    # Split complete (all 4 slots filled) vs incomplete
    complete_mask = (pivoted[slot_cols] != "").all(axis=1)
    complete = pivoted[complete_mask]
    incomplete = pivoted[~complete_mask]

    # Log incomplete cases
    for _, row in incomplete.iterrows():
        missing = [col for col in slot_cols if not row[col]]
        logger.warning("Case %s incomplete — missing: %s", row["case_id"], ", ".join(missing))

    # Write CSVs
    quads_path.parent.mkdir(parents=True, exist_ok=True)
    complete.to_csv(quads_path, index=False)
    incomplete_path.parent.mkdir(parents=True, exist_ok=True)
    incomplete.to_csv(incomplete_path, index=False)

    logger.info("Quads assembly: %d complete, %d incomplete", len(complete), len(incomplete))
    return len(complete), len(incomplete)


def _load_trust_prompt() -> str:
    """Load the trust detection prompt from YAML."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "trust_document_type_detection.yaml"
    import yaml

    with prompt_path.open() as f:
        data = yaml.safe_load(f)
    return data["prompts"]["trust_detection"]["prompt"]


def run(
    data_dir: Path,
    output_dir: Path,
    *,
    model_type: str | None = None,
    config_path: Path | None = None,
    classifications_path: Path | None = None,
    quads_path: Path | None = None,
    quads_incomplete_path: Path | None = None,
) -> Path:
    """Classify trust documents in a flat directory.

    Args:
        data_dir: Directory containing CASEXXX_* document files.
        output_dir: Directory for output files (used as parent dir for mkdir).
        model_type: Model type override (default: from config).
        config_path: Path to run_config.yml.
        classifications_path: Explicit output path for classifications JSONL.
        quads_path: Explicit output path for complete quads CSV.
        quads_incomplete_path: Explicit output path for incomplete quads CSV.

    Returns:
        Path to the written trust_classifications.jsonl.
    """
    from common.app_config import AppConfig
    from common.pipeline_ops import load_model
    from models.backend import GenerationParams
    from common.prompt_trace import effective_trace_path
    from models.backends.vllm_backend import VllmBackend

    # Discover documents
    documents = discover_trust_documents(data_dir)
    if not documents:
        msg = f"No trust documents found in {data_dir}"
        raise FileNotFoundError(msg)
    logger.info("Discovered %d trust documents in %s", len(documents), data_dir)

    # Output paths — explicit overrides fall back to output_dir-derived defaults
    output_dir.mkdir(parents=True, exist_ok=True)
    classifications_path = classifications_path or output_dir / "trust_classifications.jsonl"
    quads_path = quads_path or output_dir / "trust_quads.csv"
    incomplete_path = quads_incomplete_path or output_dir / "trust_quads_incomplete.csv"

    # Check for resumption
    completed = read_completed_images(classifications_path)
    if completed:
        original_count = len(documents)
        documents = [d for d in documents if d.name not in completed]
        logger.info(
            "Resuming: %d already done, %d remaining (of %d total)",
            len(completed),
            len(documents),
            original_count,
        )

    if not documents:
        logger.info("All documents already classified — assembling quads from existing results")
        # Still need to assemble quads from existing classifications
        from stages.io import read_jsonl

        all_classifications = read_jsonl(classifications_path)
        groups = group_by_case(all_classifications)
        assemble_quads(groups, quads_path, incomplete_path)
        return classifications_path

    # Load config
    cli_args: dict[str, Any] = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
    }
    if model_type:
        cli_args["model_type"] = model_type
    app_cfg = AppConfig.load(cli_args, config_path=config_path)
    config = app_cfg.pipeline

    # Load prompt
    prompt_text = _load_trust_prompt()

    # Load model
    logger.info("Loading model: %s", config.model_type)
    model_cm = load_model(config, app_config=app_cfg)
    model, tokenizer = model_cm.__enter__()

    try:
        # Use VllmBackend — same path as stages/classify.py on other branches.
        # VllmBackend uses text-first ordering and handles thinking mode
        # suppression internally via model_type_key.
        backend = VllmBackend(
            model,
            model_type_key=config.model_type,
            chat_template=config.chat_template,
            trace_path=effective_trace_path(config),
            pre_tiling_enabled=config.pre_tiling_enabled,
            tile_image_size=config.pre_tiling_image_size,
            tile_use_thumbnail=config.pre_tiling_use_thumbnail,
        )

        # Read token budget from config
        import yaml

        cfg_path = config_path or Path("config/run_config.yml")
        token_budget = 300  # default
        if cfg_path.exists():
            with cfg_path.open() as f:
                raw_cfg = yaml.safe_load(f) or {}
            budgets = raw_cfg.get("pipeline", {}).get("token_budgets", {})
            token_budget = budgets.get("trust_classify", 300)

        gen_params = GenerationParams(max_tokens=token_budget)

        # Classify each document
        from PIL import Image

        with StreamingJsonlWriter(classifications_path) as writer:
            for doc_path in documents:
                case_id = extract_case_id(doc_path.name)
                start = time.time()

                try:
                    image = Image.open(doc_path).convert("RGB")
                    raw_text = backend.generate(image, prompt_text, gen_params)
                    parsed = parse_trust_classification(raw_text)

                    record: dict[str, Any] = {
                        "image_name": doc_path.name,
                        "case_id": case_id,
                        "document_type": parsed["document_type"],
                        "confidence": parsed["confidence"],
                        "evidence": parsed["evidence"],
                        "raw_response": parsed["raw_response"],
                        "processing_time": time.time() - start,
                    }
                except Exception as exc:
                    logger.exception("Failed to classify %s", doc_path.name)
                    record = {
                        "image_name": doc_path.name,
                        "case_id": case_id,
                        "document_type": "UNKNOWN_TRUST_DOC",
                        "confidence": 0.0,
                        "evidence": {},
                        "raw_response": "",
                        "error": str(exc),
                        "processing_time": time.time() - start,
                    }

                writer.write(record)
                logger.info(
                    "Classified %s -> %s (%.2f confidence, %.1fs)",
                    doc_path.name,
                    record["document_type"],
                    record["confidence"],
                    record["processing_time"],
                )

    finally:
        model_cm.__exit__(None, None, None)

    # Assemble quads from ALL classifications (including previously completed)
    from stages.io import read_jsonl

    all_classifications = read_jsonl(classifications_path)
    groups = group_by_case(all_classifications)
    assemble_quads(groups, quads_path, incomplete_path)

    logger.info("Trust classification complete: %s", classifications_path)
    return classifications_path


@app.command()
def main(
    data_dir: Path = typer.Option(
        ..., "--data-dir", "-d", help="Directory containing CASEXXX_* document files"
    ),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Directory for classification output"),
    model: str | None = typer.Option(None, "--model", help="Model type override"),
    config: Path | None = typer.Option(None, "--config", help="Path to run_config.yml"),
    classifications: Path | None = typer.Option(
        None, "--classifications", help="Explicit output path for classifications JSONL"
    ),
    quads: Path | None = typer.Option(None, "--quads", help="Explicit output path for complete quads CSV"),
    quads_incomplete: Path | None = typer.Option(
        None, "--quads-incomplete", help="Explicit output path for incomplete quads CSV"
    ),
) -> None:
    """Classify trust documents by type using VLM inference (GPU)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    run(
        data_dir,
        output_dir,
        model_type=model,
        config_path=config,
        classifications_path=classifications,
        quads_path=quads,
        quads_incomplete_path=quads_incomplete,
    )


if __name__ == "__main__":
    app()
