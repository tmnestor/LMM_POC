"""Tests for the additive throughput + summary-JSON path in stages.evaluate_linking.

Covers the §6 change from the multi-pod linking KFP plan:
  - run(..., inference_seconds=N) computes receipts_per_min == n/N*60 and
    avg_seconds_per_receipt == N/n, with inference_seconds == N;
  - the no-timing path (inference_seconds=None, no per-record processing_time)
    reports throughput as 0.0 (rendered N/A) — never a fake number;
  - per-record processing_time is the fallback when no GPU timing is supplied;
  - linking_evaluation_summary.json is written with the throughput block.
"""

import json

import yaml

from stages import evaluate_linking as el

# Two scored FOUND receipts that both match correctly → n (scored) == 2.
_LINKS = [
    {
        "image_name": "CASE001_receipt.png",
        "matched": True,
        "confidence": "HIGH",
        "receipt_total": 83.48,
        "bank_transaction_amount": 83.48,
        "bank_transaction_date": "15/01/2024",
        "bank_transaction_description": "OFFICE PLUS",
    },
    {
        "image_name": "CASE002_receipt.png",
        "matched": True,
        "confidence": "HIGH",
        "receipt_total": 39.70,
        "bank_transaction_amount": 39.70,
        "bank_transaction_date": "16/01/2024",
        "bank_transaction_description": "CAFE ROMA",
    },
]

# Raw YAML ground-truth format (as written to disk + loaded by run()).
_GT = {
    "CASE001_receipt.png": [
        {
            "receipt_total": 83.48,
            "match_status": "FOUND",
            "bank_date": "15/01/2024",
            "bank_description": "OFFICE PLUS",
            "bank_amount": 83.48,
        }
    ],
    "CASE002_receipt.png": [
        {
            "receipt_total": 39.70,
            "match_status": "FOUND",
            "bank_date": "16/01/2024",
            "bank_description": "CAFE ROMA",
            "bank_amount": 39.70,
        }
    ],
}

# Converted internal format (what evaluate_linking() consumes directly).
_GT_INTERNAL = {
    "CASE001_receipt.png": [
        {
            "expected_match": "FOUND",
            "receipt_total": 83.48,
            "bank_date": "15/01/2024",
            "bank_description": "OFFICE PLUS",
            "bank_amount": 83.48,
            "mismatch_type": "NONE",
        }
    ],
    "CASE002_receipt.png": [
        {
            "expected_match": "FOUND",
            "receipt_total": 39.70,
            "bank_date": "16/01/2024",
            "bank_description": "CAFE ROMA",
            "bank_amount": 39.70,
            "mismatch_type": "NONE",
        }
    ],
}


def _write_inputs(tmp_path, links):
    links_path = tmp_path / "transaction_links.jsonl"
    with links_path.open("w") as f:
        for rec in links:
            f.write(json.dumps(rec) + "\n")
    gt_path = tmp_path / "gt.yml"
    gt_path.write_text(yaml.safe_dump(_GT))
    return links_path, gt_path


def _read_summary(out_dir):
    summary_path = out_dir / "linking_evaluation_summary.json"
    assert summary_path.exists(), "linking_evaluation_summary.json was not written"
    return json.loads(summary_path.read_text())


# ---------------------------------------------------------------------------
# _compute_summary_metrics — direct unit coverage of the throughput math
# ---------------------------------------------------------------------------


def test_compute_metrics_with_inference_seconds():
    results = el.evaluate_linking(_LINKS, _GT_INTERNAL, amount_tolerance=0.01)
    summary = el._compute_summary_metrics(results, inference_seconds=120.0)
    n = summary["total_receipts_evaluated"]
    assert n == 2
    tp = summary["throughput"]
    assert tp["inference_seconds"] == 120.0
    assert tp["receipts_per_min"] == n / 120.0 * 60.0
    assert tp["avg_seconds_per_receipt"] == 120.0 / n


def test_compute_metrics_fallback_to_processing_time():
    # No GPU timing → fall back to summed per-record processing_time.
    results = el.evaluate_linking(_LINKS, _GT_INTERNAL, amount_tolerance=0.01)
    summary = el._compute_summary_metrics(results, inference_seconds=None, fallback_seconds=60.0)
    n = summary["total_receipts_evaluated"]
    tp = summary["throughput"]
    assert tp["inference_seconds"] == 60.0
    assert tp["receipts_per_min"] == n / 60.0 * 60.0


def test_compute_metrics_no_timing_reports_na_not_fake():
    results = el.evaluate_linking(_LINKS, _GT_INTERNAL, amount_tolerance=0.01)
    summary = el._compute_summary_metrics(results, inference_seconds=None, fallback_seconds=0.0)
    tp = summary["throughput"]
    # No GPU timing and no processing_time → 0.0 (rendered N/A), never a fake rate.
    assert tp["inference_seconds"] == 0.0
    assert tp["receipts_per_min"] == 0.0
    assert tp["avg_seconds_per_receipt"] == 0.0


# ---------------------------------------------------------------------------
# run() end-to-end — summary JSON written with the throughput block
# ---------------------------------------------------------------------------


def test_run_with_inference_seconds_writes_summary(tmp_path):
    links_path, gt_path = _write_inputs(tmp_path, _LINKS)
    out_dir = tmp_path / "eval"

    el.run(links_path, gt_path, out_dir, inference_seconds=120.0)

    summary = _read_summary(out_dir)
    n = summary["total_receipts_evaluated"]
    assert n == 2
    tp = summary["throughput"]
    assert tp["inference_seconds"] == 120.0
    assert tp["receipts_per_min"] == n / 120.0 * 60.0
    assert tp["avg_seconds_per_receipt"] == 120.0 / n


def test_run_fallback_uses_processing_time(tmp_path):
    links = [dict(rec, processing_time=30.0) for rec in _LINKS]  # 60s total
    links_path, gt_path = _write_inputs(tmp_path, links)
    out_dir = tmp_path / "eval"

    el.run(links_path, gt_path, out_dir, inference_seconds=None)

    tp = _read_summary(out_dir)["throughput"]
    assert tp["inference_seconds"] == 60.0
    assert tp["receipts_per_min"] == 2 / 60.0 * 60.0


def test_run_no_timing_reports_na(tmp_path):
    links_path, gt_path = _write_inputs(tmp_path, _LINKS)  # no processing_time
    out_dir = tmp_path / "eval"

    el.run(links_path, gt_path, out_dir, inference_seconds=None)

    tp = _read_summary(out_dir)["throughput"]
    assert tp["inference_seconds"] == 0.0
    assert tp["receipts_per_min"] == 0.0
