#!/usr/bin/env python3
# ruff: noqa: B008 - typer.Option in defaults is the standard Typer pattern
"""WildReceipt Benchmark Runner — fine-grained receipt entity extraction.

Standalone benchmark script for WildReceipt (472 test images, 12 value
classes).  Evaluates entity-level micro-average F1 with normalised text
matching.

Usage:
    python benchmark_wildreceipt.py --model internvl3 --data-dir data/wildreceipt
    python benchmark_wildreceipt.py --model internvl3-vllm --data-dir data/wildreceipt
    python benchmark_wildreceipt.py --model internvl3 internvl3-vllm --data-dir data/wildreceipt
    python benchmark_wildreceipt.py --model internvl3-vllm -n 10 --data-dir data/wildreceipt
"""

import csv
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer
from PIL import Image
from rich.console import Console
from rich.table import Table

from benchmark_sroie import run_inference
from common.pipeline_config import (
    PipelineConfig,
    auto_detect_model_path,
    load_yaml_config,
)
from models.registry import get_model

# Suppress "Setting pad_token_id to eos_token_id" spam from transformers
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

console = Console()
app = typer.Typer(
    name="benchmark-wildreceipt",
    help="WildReceipt benchmark for vision-language model comparison.",
    add_completion=False,
)


# ============================================================================
# Entity class definitions
# ============================================================================

# WildReceipt label ID -> (class_name, json_field_path)
# Only value classes — key classes, Ignore (0), Others (25) excluded.
VALUE_CLASSES: dict[int, tuple[str, str]] = {
    1: ("store_name", "store_name"),
    3: ("store_address", "store_address"),
    5: ("telephone", "telephone"),
    7: ("date", "date"),
    9: ("time", "time"),
    11: ("prod_item", "items[].name"),
    13: ("prod_quantity", "items[].quantity"),
    15: ("prod_price", "items[].price"),
    17: ("subtotal", "subtotal"),
    19: ("tax", "tax"),
    21: ("tips", "tips"),
    23: ("total", "total"),
}

# Class names in display order (sorted by label ID)
CLASS_NAMES = [name for _, (name, _) in sorted(VALUE_CLASSES.items())]

# Scalar: model outputs one value, GT may have multiple annotations to concatenate
SCALAR_CLASSES = {
    "store_name",
    "store_address",
    "telephone",
    "date",
    "time",
    "subtotal",
    "tax",
    "tips",
    "total",
}

# List: model outputs items array, GT has per-entity annotations
LIST_CLASSES = {"prod_item", "prod_quantity", "prod_price"}


# ============================================================================
# Extraction prompt
# ============================================================================

WILDRECEIPT_PROMPT = """\
Extract all information from this receipt image. Return a JSON object with these fields:
{
  "store_name": "...",
  "store_address": "...",
  "telephone": "...",
  "date": "...",
  "time": "...",
  "items": [{"name": "...", "quantity": "...", "price": "..."}],
  "subtotal": "...",
  "tax": "...",
  "tips": "...",
  "total": "..."
}
Omit fields not present on the receipt. For items, list each product as a separate entry.\
"""


# ============================================================================
# Data loading
# ============================================================================


@dataclass
class WildReceiptGT:
    """Ground truth for one WildReceipt image."""

    file_name: str
    # class_name -> list of text values (one per annotation)
    entities: dict[str, list[str]] = field(default_factory=dict)


def _bbox_sort_key(box: list[int | float]) -> tuple[float, float]:
    """Compute (centroid_y, centroid_x) from a WildReceipt quad bbox.

    Box format: [x1,y1, x2,y2, x3,y3, x4,y4] — 4 corners.
    Returns (cy, cx) for top-to-bottom, left-to-right reading order.
    """
    cx = (box[0] + box[2] + box[4] + box[6]) / 4
    cy = (box[1] + box[3] + box[5] + box[7]) / 4
    return (cy, cx)


def load_wildreceipt_ground_truth(test_txt: Path) -> dict[str, WildReceiptGT]:
    """Parse test.txt JSON-lines into ground truth dict keyed by file_name.

    Annotations are sorted into reading order (top-to-bottom, left-to-right)
    using bbox centroid positions.  This ensures list fields (prod_item, etc.)
    are ordered correctly for position-aware evaluation.
    """
    ground_truth: dict[str, WildReceiptGT] = {}

    with test_txt.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            file_name = data["file_name"]

            # Collect (text, sort_key) per class, then sort by reading order
            class_annotations: dict[str, list[tuple[str, tuple[float, float]]]] = {}

            for ann in data.get("annotations", []):
                label_id = ann["label"]
                text = ann["text"].strip()

                if label_id not in VALUE_CLASSES or not text:
                    continue

                class_name = VALUE_CLASSES[label_id][0]
                sort_key = _bbox_sort_key(ann["box"])
                class_annotations.setdefault(class_name, []).append((text, sort_key))

            # Sort each class by reading order and keep only text
            gt = WildReceiptGT(file_name=file_name)
            for class_name, ann_list in class_annotations.items():
                ann_list.sort(key=lambda x: x[1])
                gt.entities[class_name] = [text for text, _ in ann_list]

            ground_truth[file_name] = gt

    return ground_truth


# ============================================================================
# Normalization
# ============================================================================

_WHITESPACE_RE = re.compile(r"\s+")
_CURRENCY_RE = re.compile(r"[$\u00a3\u20ac\u00a5]|RM")
_COMMA_IN_NUMBER_RE = re.compile(r"(\d),(\d)")
_LEADING_TRAILING_PUNCT_RE = re.compile(r"^[^\w]+|[^\w]+$")


def _normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip leading/trailing punctuation."""
    text = text.lower().strip()
    text = _WHITESPACE_RE.sub(" ", text)
    text = _LEADING_TRAILING_PUNCT_RE.sub("", text)
    return text.strip()


def _normalize_currency(text: str) -> str:
    """Strip currency symbols, commas; format to 2 decimal places."""
    text = text.strip()
    text = _CURRENCY_RE.sub("", text)
    text = _COMMA_IN_NUMBER_RE.sub(r"\1\2", text)
    text = text.strip()
    try:
        return f"{float(text):.2f}"
    except ValueError:
        return _normalize_text(text)


def _normalize_phone(text: str) -> str:
    """Keep only digits."""
    return re.sub(r"\D", "", text)


def _normalize_date(text: str) -> str:
    """Normalize date to DD/MM/YYYY if possible."""
    text = text.strip()

    # ISO: YYYY-MM-DD
    m = re.match(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", text)
    if m:
        return f"{int(m.group(3)):02d}/{int(m.group(2)):02d}/{m.group(1)}"

    # DD/MM/YYYY or DD-MM-YYYY
    m = re.match(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", text)
    if m:
        return f"{int(m.group(1)):02d}/{int(m.group(2)):02d}/{m.group(3)}"

    return _normalize_text(text)


def normalize_entity(class_name: str, text: str) -> str:
    """Apply class-specific normalization."""
    if not text:
        return ""
    if class_name in ("total", "subtotal", "tax", "tips", "prod_price"):
        return _normalize_currency(text)
    if class_name == "telephone":
        return _normalize_phone(text)
    if class_name == "date":
        return _normalize_date(text)
    if class_name == "prod_quantity":
        digits = re.sub(r"\D", "", text)
        return digits if digits else _normalize_text(text)
    return _normalize_text(text)


# ============================================================================
# Response parsing
# ============================================================================

# Regex to find a top-level JSON object (supports one level of nesting)
_JSON_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def parse_wildreceipt_response(raw: str) -> dict:
    """Extract JSON from model response, handling markdown fences."""
    text = raw.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_RE.search(text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}


# ============================================================================
# Evaluation
# ============================================================================


# ============================================================================
# Matching primitives
# ============================================================================

# Monetary classes use exact numeric comparison after normalization to 2dp.
MONETARY_CLASSES = {"total", "subtotal", "tax", "tips", "prod_price"}

# ANLS threshold — below this similarity score counts as mismatch.
ANLS_THRESHOLD = 0.5

# Word-overlap Jaccard threshold for fuzzy list-item matching.
JACCARD_THRESHOLD = 0.75


def _levenshtein(s: str, t: str) -> int:
    """Levenshtein edit distance (Wagner-Fischer, O(mn) space)."""
    m, n = len(s), len(t)
    if m == 0:
        return n
    if n == 0:
        return m

    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev

    return prev[n]


def _anls_score(predicted: str, ground_truth: str) -> float:
    """ANLS (Average Normalised Levenshtein Similarity).

    Returns continuous score 0.0–1.0.  Below ANLS_THRESHOLD → 0.0.
    """
    if not predicted and not ground_truth:
        return 1.0
    if not predicted or not ground_truth:
        return 0.0

    max_len = max(len(predicted), len(ground_truth))
    if max_len == 0:
        return 1.0

    dist = _levenshtein(predicted, ground_truth)
    similarity = 1.0 - dist / max_len
    return similarity if similarity >= ANLS_THRESHOLD else 0.0


def _monetary_match(predicted: str, ground_truth: str) -> bool:
    """Exact match after normalization to 2 decimal places."""
    return predicted == ground_truth


def _jaccard_word_overlap(a: str, b: str) -> float:
    """Word-level Jaccard similarity: |intersection| / |union|."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def _entity_matches(class_name: str, pred: str, gt: str) -> bool:
    """Check whether a predicted entity matches a GT entity.

    - Monetary classes: numeric tolerance (1%)
    - List text classes (prod_item): word-overlap Jaccard >= 0.75
    - Other text classes: ANLS >= 0.5
    """
    if class_name in MONETARY_CLASSES:
        return _monetary_match(pred, gt)
    if class_name in LIST_CLASSES:
        return _jaccard_word_overlap(pred, gt) >= JACCARD_THRESHOLD
    return _anls_score(pred, gt) >= ANLS_THRESHOLD


# ============================================================================
# Result dataclasses
# ============================================================================


@dataclass
class ClassResult:
    """TP/FP/FN counts for one entity class."""

    class_name: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@dataclass
class ImageResult:
    """Per-image evaluation result."""

    image_id: str
    per_class: dict[str, ClassResult] = field(default_factory=dict)
    predicted_raw: dict = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Full benchmark result for one model run."""

    model_name: str
    image_results: list[ImageResult]
    class_results: dict[str, ClassResult]
    overall_precision: float
    overall_recall: float
    overall_f1: float
    total_images: int
    elapsed_seconds: float = 0.0


# ============================================================================
# Entity extraction helpers
# ============================================================================


def _extract_predicted_entities(predicted: dict) -> dict[str, list[str]]:
    """Extract per-class entity lists from parsed model response."""
    entities: dict[str, list[str]] = {}

    # Scalar fields
    scalar_map = {
        "store_name": "store_name",
        "store_address": "store_address",
        "telephone": "telephone",
        "date": "date",
        "time": "time",
        "subtotal": "subtotal",
        "tax": "tax",
        "tips": "tips",
        "total": "total",
    }

    for json_key, class_name in scalar_map.items():
        val = predicted.get(json_key)
        if val and str(val).strip():
            entities[class_name] = [str(val).strip()]

    # List fields from items array
    items = predicted.get("items", [])
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            for item_key, class_name in [
                ("name", "prod_item"),
                ("quantity", "prod_quantity"),
                ("price", "prod_price"),
            ]:
                val = item.get(item_key)
                if val and str(val).strip():
                    entities.setdefault(class_name, []).append(str(val).strip())

    return entities


def _prepare_gt_entities(gt: WildReceiptGT) -> dict[str, list[str]]:
    """Prepare GT entities for matching.

    Scalar classes: concatenate all annotations into one entity (e.g. multi-word
    store names that are split across annotations).
    List classes: keep each annotation as a separate entity (already bbox-sorted).
    """
    prepared: dict[str, list[str]] = {}

    for class_name, texts in gt.entities.items():
        if class_name in SCALAR_CLASSES:
            prepared[class_name] = [" ".join(texts)]
        else:
            prepared[class_name] = list(texts)

    return prepared


# ============================================================================
# Position-aware evaluation
# ============================================================================


def _evaluate_scalar(
    class_name: str, pred_vals: list[str], gt_vals: list[str]
) -> ClassResult:
    """Evaluate a scalar class (single GT value vs single predicted value).

    Uses ANLS for text, monetary tolerance for currency, exact for phone/date.
    """
    cr = ClassResult(class_name=class_name)

    gt_norm = normalize_entity(class_name, gt_vals[0]) if gt_vals else ""
    pred_norm = normalize_entity(class_name, pred_vals[0]) if pred_vals else ""

    if not gt_norm and not pred_norm:
        return cr

    if gt_norm and pred_norm:
        if _entity_matches(class_name, pred_norm, gt_norm):
            cr.true_positives = 1
        else:
            cr.false_positives = 1
            cr.false_negatives = 1
    elif gt_norm:
        cr.false_negatives = 1
    else:
        cr.false_positives = 1

    return cr


def _evaluate_list_position_aware(
    class_name: str, pred_vals: list[str], gt_vals: list[str]
) -> ClassResult:
    """Position-aware F1 for list classes.

    Compares index-by-index: pred[0] vs gt[0], pred[1] vs gt[1], etc.
    - Both present and match  → TP
    - Both present, no match  → FN (wrong item at this position)
    - Only GT present (model under-extracted) → FN
    - Only pred present (model over-extracted) → FP
    """
    cr = ClassResult(class_name=class_name)

    gt_norm = [normalize_entity(class_name, v) for v in gt_vals]
    pred_norm = [normalize_entity(class_name, v) for v in pred_vals]

    max_len = max(len(gt_norm), len(pred_norm))

    for i in range(max_len):
        g = gt_norm[i] if i < len(gt_norm) else ""
        p = pred_norm[i] if i < len(pred_norm) else ""

        if g and p:
            if _entity_matches(class_name, p, g):
                cr.true_positives += 1
            else:
                cr.false_negatives += 1
        elif g:
            cr.false_negatives += 1
        elif p:
            cr.false_positives += 1

    return cr


def evaluate_image(predicted: dict, gt: WildReceiptGT) -> ImageResult:
    """Compare predicted fields to GT entities. Return per-class TP/FP/FN.

    Scalar classes use ANLS/monetary matching (single value).
    List classes use position-aware F1 with fuzzy word-overlap matching.
    """
    pred_entities = _extract_predicted_entities(predicted)
    gt_entities = _prepare_gt_entities(gt)

    result = ImageResult(image_id=gt.file_name, predicted_raw=predicted)

    for class_name in CLASS_NAMES:
        gt_vals = gt_entities.get(class_name, [])
        pred_vals = pred_entities.get(class_name, [])

        if not gt_vals and not pred_vals:
            result.per_class[class_name] = ClassResult(class_name=class_name)
        elif class_name in LIST_CLASSES:
            result.per_class[class_name] = _evaluate_list_position_aware(
                class_name, pred_vals, gt_vals
            )
        else:
            result.per_class[class_name] = _evaluate_scalar(
                class_name, pred_vals, gt_vals
            )

    return result


def compute_metrics(
    image_results: list[ImageResult],
    model_name: str = "unknown",
    elapsed_seconds: float = 0.0,
) -> BenchmarkResult:
    """Aggregate per-image results into micro-average F1."""
    class_results = {name: ClassResult(class_name=name) for name in CLASS_NAMES}

    for img in image_results:
        for class_name in CLASS_NAMES:
            cr = img.per_class.get(class_name)
            if cr:
                class_results[class_name].true_positives += cr.true_positives
                class_results[class_name].false_positives += cr.false_positives
                class_results[class_name].false_negatives += cr.false_negatives

    # Micro-average: pool all TP/FP/FN across classes
    total_tp = sum(cr.true_positives for cr in class_results.values())
    total_fp = sum(cr.false_positives for cr in class_results.values())
    total_fn = sum(cr.false_negatives for cr in class_results.values())

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return BenchmarkResult(
        model_name=model_name,
        image_results=image_results,
        class_results=class_results,
        overall_precision=precision,
        overall_recall=recall,
        overall_f1=f1,
        total_images=len(image_results),
        elapsed_seconds=elapsed_seconds,
    )


# ============================================================================
# Benchmark runner
# ============================================================================


def run_benchmark(
    model_type: str,
    data_dir: Path,
    model_path: str | None = None,
    max_images: int | None = None,
    max_tokens: int = 1024,
) -> BenchmarkResult:
    """Run WildReceipt benchmark for a single model."""
    test_txt = data_dir / "test.txt"

    if not test_txt.exists():
        raise FileNotFoundError(f"Test annotations not found: {test_txt}")

    # Load ground truth
    ground_truth = load_wildreceipt_ground_truth(test_txt)
    console.print(f"Loaded {len(ground_truth)} ground truth entries")

    # Filter to images that exist on disk
    all_entries = []
    for file_name, gt in sorted(ground_truth.items()):
        img_path = data_dir / file_name
        if img_path.exists():
            all_entries.append((file_name, gt, img_path))

    console.print(f"Found {len(all_entries)} images on disk")

    if max_images:
        all_entries = all_entries[:max_images]

    console.print(f"Evaluating {len(all_entries)} images with model: {model_type}")

    # Load model via registry
    registration = get_model(model_type)

    # Resolve model path: CLI > run_config.yml > auto-detect
    resolved_model_path: Path | None = Path(model_path) if model_path else None

    if resolved_model_path is None:
        run_config_path = Path(__file__).parent / "config" / "run_config.yml"
        if run_config_path.exists():
            _, raw_config = load_yaml_config(run_config_path)
            default_paths = raw_config.get("model_loading", {}).get("default_paths", {})
            if isinstance(default_paths, dict) and model_type in default_paths:
                candidate = Path(default_paths[model_type])
                if candidate.exists():
                    resolved_model_path = candidate
                    console.print(
                        f"Model path from run_config.yml: {resolved_model_path}"
                    )

    if resolved_model_path is None:
        resolved_model_path = auto_detect_model_path()
        if resolved_model_path:
            console.print(f"Auto-detected model path: {resolved_model_path}")

    if resolved_model_path is None:
        raise FileNotFoundError(
            f"No model path found for '{model_type}'. "
            "Use --model-path or add to config/run_config.yml model_loading.default_paths."
        )

    # Build PipelineConfig
    config_kwargs: dict[str, Any] = {
        "data_dir": data_dir,
        "output_dir": data_dir / "output",
        "model_type": model_type,
        "model_path": resolved_model_path,
    }

    # Model-specific defaults
    if model_type in ("internvl3", "internvl3-14b", "internvl3-38b"):
        config_kwargs.setdefault("flash_attn", True)
    elif model_type in (
        "internvl3-vllm",
        "internvl3-14b-vllm",
        "internvl3-38b-vllm",
        "llama4scout-w4a16",
    ):
        config_kwargs.setdefault("flash_attn", False)
    else:
        config_kwargs.setdefault("flash_attn", False)
    config_kwargs.setdefault("device_map", "auto")

    cfg = PipelineConfig(**config_kwargs)

    # Run inference
    image_results: list[ImageResult] = []
    start_time = time.time()

    with registration.loader(cfg) as (model, tokenizer_or_processor):
        from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

        with Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"[cyan]{model_type}", total=len(all_entries))

            for file_name, gt, img_path in all_entries:
                try:
                    image = Image.open(img_path).convert("RGB")
                    raw_response = run_inference(
                        model_type,
                        model,
                        tokenizer_or_processor,
                        image,
                        WILDRECEIPT_PROMPT,
                        max_tokens,
                    )
                    predicted = parse_wildreceipt_response(raw_response)
                    del image
                except Exception as e:
                    console.print(f"[red]Error on {file_name}: {e}[/red]")
                    predicted = {}

                img_result = evaluate_image(predicted, gt)
                image_results.append(img_result)
                progress.advance(task)

    elapsed = time.time() - start_time
    return compute_metrics(image_results, model_type, elapsed)


# ============================================================================
# Display and export
# ============================================================================


def print_results_table(results: list[BenchmarkResult]) -> None:
    """Print Rich comparison table of benchmark results."""
    # Per-class F1
    table = Table(
        title="WildReceipt Benchmark Results (Per-Class F1)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Class", style="white")
    for r in results:
        table.add_column(r.model_name, justify="right")

    for class_name in CLASS_NAMES:
        row: list[str] = [class_name]
        for r in results:
            cr = r.class_results[class_name]
            f1_pct = cr.f1 * 100
            color = "green" if f1_pct >= 60 else ("yellow" if f1_pct >= 30 else "red")
            row.append(f"[{color}]{f1_pct:.1f}%[/{color}]")
        table.add_row(*row)

    # Overall micro-average row
    row = ["[bold]Overall (micro-avg)[/bold]"]
    for r in results:
        f1_pct = r.overall_f1 * 100
        color = "green" if f1_pct >= 60 else ("yellow" if f1_pct >= 30 else "red")
        row.append(f"[bold {color}]{f1_pct:.1f}%[/bold {color}]")
    table.add_row(*row)

    console.print(table)

    # Summary
    summary = Table(
        title="Summary",
        show_header=True,
        header_style="bold cyan",
    )
    summary.add_column("Metric", style="white")
    for r in results:
        summary.add_column(r.model_name, justify="right")

    summary.add_row("Images", *[str(r.total_images) for r in results])
    summary.add_row(
        "Precision",
        *[f"{r.overall_precision * 100:.1f}%" for r in results],
    )
    summary.add_row(
        "Recall",
        *[f"{r.overall_recall * 100:.1f}%" for r in results],
    )
    summary.add_row(
        "F1",
        *[f"{r.overall_f1 * 100:.1f}%" for r in results],
    )
    summary.add_row("Time (s)", *[f"{r.elapsed_seconds:.1f}" for r in results])
    summary.add_row(
        "Img/min",
        *[
            f"{r.total_images / r.elapsed_seconds * 60:.1f}"
            if r.elapsed_seconds > 0
            else "N/A"
            for r in results
        ],
    )

    console.print(summary)


def save_results_json(results: list[BenchmarkResult], output_dir: Path) -> Path:
    """Save benchmark results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    output: dict[str, Any] = {}
    for r in results:
        per_class: dict[str, Any] = {}
        model_data: dict[str, Any] = {
            "overall_f1": round(r.overall_f1, 4),
            "overall_precision": round(r.overall_precision, 4),
            "overall_recall": round(r.overall_recall, 4),
            "total_images": r.total_images,
            "elapsed_seconds": round(r.elapsed_seconds, 2),
            "per_class": per_class,
        }
        for class_name in CLASS_NAMES:
            cr = r.class_results[class_name]
            per_class[class_name] = {
                "precision": round(cr.precision, 4),
                "recall": round(cr.recall, 4),
                "f1": round(cr.f1, 4),
                "tp": cr.true_positives,
                "fp": cr.false_positives,
                "fn": cr.false_negatives,
            }
        output[r.model_name] = model_data

    out_path = output_dir / "wildreceipt_results.json"
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)

    console.print(f"Results saved to {out_path}")
    return out_path


def save_results_csv(results: list[BenchmarkResult], output_dir: Path) -> Path:
    """Save benchmark summary and per-image results to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Summary CSV (one row per model) ---
    summary_path = output_dir / "wildreceipt_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "model",
            "overall_f1",
            "overall_precision",
            "overall_recall",
            "total_images",
            "elapsed_seconds",
            "images_per_min",
        ]
        for class_name in CLASS_NAMES:
            header.extend(
                [
                    f"{class_name}_precision",
                    f"{class_name}_recall",
                    f"{class_name}_f1",
                ]
            )
        writer.writerow(header)

        for r in results:
            img_per_min = (
                r.total_images / r.elapsed_seconds * 60
                if r.elapsed_seconds > 0
                else 0.0
            )
            row: list[Any] = [
                r.model_name,
                round(r.overall_f1, 4),
                round(r.overall_precision, 4),
                round(r.overall_recall, 4),
                r.total_images,
                round(r.elapsed_seconds, 2),
                round(img_per_min, 2),
            ]
            for class_name in CLASS_NAMES:
                cr = r.class_results[class_name]
                row.extend(
                    [round(cr.precision, 4), round(cr.recall, 4), round(cr.f1, 4)]
                )
            writer.writerow(row)

    console.print(f"Summary CSV saved to {summary_path}")

    # --- Per-image CSV (one row per model+image) ---
    detail_path = output_dir / "wildreceipt_per_image.csv"
    with detail_path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["model", "image_id"]
        for class_name in CLASS_NAMES:
            header.extend([f"{class_name}_tp", f"{class_name}_fp", f"{class_name}_fn"])
        writer.writerow(header)

        for r in results:
            for img in r.image_results:
                row: list[Any] = [r.model_name, img.image_id]
                for class_name in CLASS_NAMES:
                    cr = img.per_class.get(
                        class_name, ClassResult(class_name=class_name)
                    )
                    row.extend(
                        [cr.true_positives, cr.false_positives, cr.false_negatives]
                    )
                writer.writerow(row)

    console.print(f"Per-image CSV saved to {detail_path}")
    return summary_path


# ============================================================================
# CLI
# ============================================================================


@app.command()
def main(
    model: list[str] = typer.Option(
        ...,
        "--model",
        "-m",
        help="Model type(s) to benchmark.",
    ),
    data_dir: Path = typer.Option(
        Path("data/wildreceipt"),
        "--data-dir",
        "-d",
        help="Path to WildReceipt data directory.",
    ),
    model_path: str | None = typer.Option(
        None,
        "--model-path",
        "-p",
        help="Override model path (auto-detected if omitted).",
    ),
    max_images: int | None = typer.Option(
        None,
        "--max-images",
        "-n",
        help="Maximum images to evaluate (all if omitted).",
    ),
    output_dir: Path = typer.Option(
        Path("output/wildreceipt"),
        "--output-dir",
        "-o",
        help="Directory for results output.",
    ),
    max_tokens: int = typer.Option(
        1024,
        "--max-tokens",
        help="Maximum generation tokens per image.",
    ),
) -> None:
    """Run WildReceipt benchmark on one or more vision-language models."""
    console.print(
        f"[bold]WildReceipt Benchmark[/bold] | "
        f"Models: {', '.join(model)} | Data: {data_dir}"
    )

    if not data_dir.exists():
        console.print(f"[red]Data directory not found: {data_dir}[/red]")
        raise typer.Exit(1) from None

    all_results: list[BenchmarkResult] = []

    for model_type in model:
        console.print(f"\n{'=' * 60}")
        console.print(f"[bold cyan]Benchmarking: {model_type}[/bold cyan]")
        console.print(f"{'=' * 60}")

        result = run_benchmark(
            model_type=model_type,
            data_dir=data_dir,
            model_path=model_path,
            max_images=max_images,
            max_tokens=max_tokens,
        )
        all_results.append(result)

    # Print comparison table
    console.print()
    print_results_table(all_results)

    # Save results
    save_results_json(all_results, output_dir)
    save_results_csv(all_results, output_dir)


if __name__ == "__main__":
    app()
