"""SROIE benchmark evaluation — entity-level exact match after normalization.

Implements the official ICDAR 2019 SROIE Task 3 metrics:
- Per-field: precision, recall, F1
- Overall: mean of per-field F1
- Normalization: lowercase, collapse whitespace, strip currency, normalize dates

Reference: https://rrc.cvc.uab.es/?ch=13&com=tasks
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# ============================================================================
# Result dataclasses
# ============================================================================

SROIE_FIELDS = ("company", "date", "address", "total")


@dataclass
class SROIEFieldResult:
    """Aggregate metrics for one SROIE field across all images."""

    field: str
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
class SROIEImageResult:
    """Per-image extraction result."""

    image_id: str
    ground_truth: dict[str, str]
    predicted: dict[str, str]
    matches: dict[str, bool] = field(default_factory=dict)


@dataclass
class SROIEBenchmarkResult:
    """Full benchmark result for one model run."""

    model_name: str
    image_results: list[SROIEImageResult]
    field_results: dict[str, SROIEFieldResult]
    overall_f1: float
    total_images: int
    elapsed_seconds: float = 0.0


# ============================================================================
# Ground truth loading
# ============================================================================


def load_sroie_ground_truth(key_dir: Path) -> dict[str, dict[str, str]]:
    """Load per-image JSON ground truth from SROIE key/ directory.

    Each .txt file contains a JSON object with keys: company, date, address, total.

    Returns:
        Dict mapping image_id (stem) to {field: value}.
    """
    ground_truth: dict[str, dict[str, str]] = {}

    for txt_path in sorted([*key_dir.glob("*.json"), *key_dir.glob("*.txt")]):
        image_id = txt_path.stem
        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Some SROIE files have line-separated values instead of JSON
            lines = text.splitlines()
            if len(lines) >= 4:
                data = {
                    "company": lines[0].strip(),
                    "date": lines[1].strip(),
                    "address": lines[2].strip(),
                    "total": lines[3].strip(),
                }
            else:
                continue

        # Ensure all 4 keys exist
        gt_entry = {}
        for f in SROIE_FIELDS:
            gt_entry[f] = data.get(f, "")
        ground_truth[image_id] = gt_entry

    return ground_truth


# ============================================================================
# Normalization functions
# ============================================================================

# Regex patterns compiled once
_WHITESPACE_RE = re.compile(r"\s+")
_CURRENCY_RE = re.compile(r"[$\u00a3\u20ac\u00a5RM]")
_COMMA_IN_NUMBER_RE = re.compile(r"(\d),(\d)")


def normalize_text(text: str) -> str:
    """General text normalization: lowercase, collapse whitespace, strip."""
    text = text.lower().strip()
    text = _WHITESPACE_RE.sub(" ", text)
    return text


def normalize_total(text: str) -> str:
    """Normalize monetary total: strip currency symbols, commas, whitespace."""
    text = text.strip()
    text = _CURRENCY_RE.sub("", text)
    text = _COMMA_IN_NUMBER_RE.sub(r"\1\2", text)
    text = text.strip()
    # Try to normalize to consistent decimal format
    try:
        val = float(text)
        return f"{val:.2f}"
    except ValueError:
        return normalize_text(text)


def normalize_date(text: str) -> str:
    """Normalize date to DD/MM/YYYY format if possible.

    Handles common SROIE date formats:
    - DD/MM/YYYY (already correct)
    - DD-MM-YYYY
    - DD.MM.YYYY
    - YYYY-MM-DD (ISO)
    - DD Mon YYYY
    - Various separator inconsistencies
    """
    text = text.strip()

    # Try ISO format: YYYY-MM-DD
    m = re.match(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", text)
    if m:
        return f"{int(m.group(3)):02d}/{int(m.group(2)):02d}/{m.group(1)}"

    # Try DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
    m = re.match(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", text)
    if m:
        return f"{int(m.group(1)):02d}/{int(m.group(2)):02d}/{m.group(3)}"

    # Try DD Mon YYYY
    months = {
        "jan": "01",
        "feb": "02",
        "mar": "03",
        "apr": "04",
        "may": "05",
        "jun": "06",
        "jul": "07",
        "aug": "08",
        "sep": "09",
        "oct": "10",
        "nov": "11",
        "dec": "12",
    }
    m = re.match(r"(\d{1,2})\s+(\w{3,})\s+(\d{4})", text, re.IGNORECASE)
    if m:
        month_str = m.group(2)[:3].lower()
        if month_str in months:
            return f"{int(m.group(1)):02d}/{months[month_str]}/{m.group(3)}"

    # Fallback: return lowercased
    return normalize_text(text)


# ============================================================================
# Matching
# ============================================================================


def sroie_field_match(field_name: str, predicted: str, ground_truth: str) -> bool:
    """Check exact match after field-specific normalization.

    Args:
        field_name: One of "company", "date", "address", "total".
        predicted: Model's predicted value.
        ground_truth: Ground truth value.

    Returns:
        True if normalized values match exactly.
    """
    if not predicted or not ground_truth:
        return False

    if field_name == "total":
        return normalize_total(predicted) == normalize_total(ground_truth)
    if field_name == "date":
        return normalize_date(predicted) == normalize_date(ground_truth)
    # company and address: general text normalization
    return normalize_text(predicted) == normalize_text(ground_truth)


# ============================================================================
# Metrics computation
# ============================================================================


def compute_sroie_metrics(
    image_results: list[SROIEImageResult],
    model_name: str = "unknown",
    elapsed_seconds: float = 0.0,
) -> SROIEBenchmarkResult:
    """Aggregate per-image results into per-field and overall metrics.

    Each image contributes one TP or one FN+FP per field.
    """
    field_results = {f: SROIEFieldResult(field=f) for f in SROIE_FIELDS}

    for img_result in image_results:
        for f in SROIE_FIELDS:
            gt_val = img_result.ground_truth.get(f, "")
            pred_val = img_result.predicted.get(f, "")

            match = sroie_field_match(f, pred_val, gt_val)
            img_result.matches[f] = match

            if match:
                field_results[f].true_positives += 1
            else:
                if pred_val:
                    field_results[f].false_positives += 1
                if gt_val:
                    field_results[f].false_negatives += 1

    # Overall F1 = mean of per-field F1
    f1_scores = [fr.f1 for fr in field_results.values()]
    overall_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return SROIEBenchmarkResult(
        model_name=model_name,
        image_results=image_results,
        field_results=field_results,
        overall_f1=overall_f1,
        total_images=len(image_results),
        elapsed_seconds=elapsed_seconds,
    )
