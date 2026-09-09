"""Tests for stages.evaluate_linking metric computation + ground-truth loaders."""

import json

import pytest
import yaml

from stages import evaluate_linking as el


# ---------------------------------------------------------------------------
# Ground-truth loaders
# ---------------------------------------------------------------------------


def test_load_ground_truth_yaml(tmp_path):
    gt = {
        "CASE001_receipt.png": [
            {
                "supplier": "Office Plus",
                "receipt_total": 83.48,
                "match_status": "FOUND",
                "bank_date": "15/01/2024",
                "bank_description": "OFFICE PLUS",
                "bank_amount": 83.48,
            }
        ]
    }
    path = tmp_path / "gt.yml"
    path.write_text(yaml.safe_dump(gt))
    loaded = el.load_linking_ground_truth(path)
    assert loaded["CASE001_receipt.png"][0]["expected_match"] == "FOUND"
    assert loaded["CASE001_receipt.png"][0]["receipt_total"] == 83.48


def test_load_ground_truth_unsupported_extension(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text("nope")
    with pytest.raises(ValueError):
        el.load_linking_ground_truth(path)


def test_load_ground_truth_csv(tmp_path):
    path = tmp_path / "gt.csv"
    path.write_text(
        "image_file,EXPECTED_MATCH_STATUS,RECEIPT_TOTAL,BANK_TRANSACTION_DATE,"
        "BANK_TRANSACTION_DESCRIPTION,BANK_TRANSACTION_DEBIT,EXPECTED_MISMATCH_TYPE\n"
        "CASE001_a.png,FOUND|FOUND,83.48|39.70,15/01/2024|16/01/2024,"
        "OFFICE PLUS|CAFE,83.48|39.70,NONE|NONE\n"
    )
    loaded = el.load_linking_ground_truth(path)
    assert len(loaded["CASE001_a.png"]) == 2
    assert loaded["CASE001_a.png"][1]["receipt_total"] == 39.70


def test_load_ground_truth_jsonl(tmp_path):
    path = tmp_path / "gt.jsonl"
    path.write_text(
        json.dumps(
            {
                "image_file": "CASE001_a.png",
                "EXPECTED_MATCH_STATUS": "FOUND",
                "RECEIPT_TOTAL": "83.48",
                "BANK_TRANSACTION_DATE": "15/01/2024",
                "BANK_TRANSACTION_DESCRIPTION": "OFFICE PLUS",
                "BANK_TRANSACTION_DEBIT": "83.48",
                "EXPECTED_MISMATCH_TYPE": "NONE",
            }
        )
        + "\n"
    )
    loaded = el.load_linking_ground_truth(path)
    assert loaded["CASE001_a.png"][0]["expected_match"] == "FOUND"


def test_split_pipe_and_parse_float():
    assert el._split_pipe("a | b | c") == ["a", "b", "c"]
    assert el._split_pipe("") == []
    assert el._parse_float("1,234.56") == 1234.56
    assert el._parse_float("") is None
    assert el._parse_float("xx") is None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_dates_match():
    assert el._dates_match("18/03/2024", "18/03/2024")
    assert not el._dates_match("18/03/2024", "19/03/2024")
    assert el._dates_match("18/03/2024", "19/03/2024", tolerance_days=2)


def test_descriptions_match():
    assert el._descriptions_match("WOOLWORTHS 2847", "WOOLWORTHS")
    assert el._descriptions_match("OFFICE SUPPLIES PLUS", "OFFICE SUPPLIES")
    assert not el._descriptions_match("ALPHA", "BETA")


def test_pair_by_amount():
    links = [{"receipt_total": 83.48}, {"receipt_total": 39.70}]
    gts = [{"receipt_total": 39.70}, {"receipt_total": 83.48}]
    pairs = el._pair_by_amount(links, gts, tolerance=0.01)
    # each link paired with the gt of equal amount
    for link, gt in pairs:
        assert link["receipt_total"] == gt["receipt_total"]


# ---------------------------------------------------------------------------
# evaluate_linking metrics
# ---------------------------------------------------------------------------


def test_evaluate_linking_correct_match():
    links = [
        {
            "image_name": "CASE001_receipt.png",
            "matched": True,
            "confidence": "HIGH",
            "receipt_total": 83.48,
            "bank_transaction_amount": 83.48,
            "bank_transaction_date": "15/01/2024",
            "bank_transaction_description": "OFFICE PLUS",
        }
    ]
    gt = {
        "CASE001_receipt.png": [
            {
                "expected_match": "FOUND",
                "receipt_total": 83.48,
                "bank_date": "15/01/2024",
                "bank_description": "OFFICE PLUS",
                "bank_amount": 83.48,
                "mismatch_type": "NONE",
            }
        ]
    }
    results = el.evaluate_linking(links, gt, amount_tolerance=0.01)
    assert results[0]["correct"] is True
    assert results[0]["amount_correct"] is True
    assert results[0]["date_correct"] is True


def test_evaluate_linking_missing_link_output():
    gt = {
        "CASE001_receipt.png": [
            {
                "expected_match": "FOUND",
                "receipt_total": 83.48,
                "bank_date": "",
                "bank_description": "",
                "bank_amount": 83.48,
                "mismatch_type": "NONE",
            }
        ]
    }
    results = el.evaluate_linking([], gt, amount_tolerance=0.01)
    assert results[0]["correct"] is False
    assert results[0]["error"] == "missing_link_output"


def test_evaluate_linking_no_ground_truth_for_image():
    links = [{"image_name": "CASE099_x.png", "matched": True, "confidence": "LOW"}]
    results = el.evaluate_linking(links, {}, amount_tolerance=0.01)
    assert results[0]["error"] == "no_ground_truth"
    assert results[0]["correct"] is None


def test_evaluate_linking_false_positive_not_found():
    links = [{"image_name": "CASE001_r.png", "matched": True, "confidence": "LOW"}]
    gt = {
        "CASE001_r.png": [
            {
                "expected_match": "NOT_FOUND",
                "receipt_total": None,
                "bank_date": "",
                "bank_description": "",
                "bank_amount": None,
                "mismatch_type": "NONE",
            }
        ]
    }
    results = el.evaluate_linking(links, gt, amount_tolerance=0.01)
    # expected NOT_FOUND but matched -> incorrect (false positive)
    assert results[0]["correct"] is False


def test_print_summary_runs():
    # Smoke test: print_summary should render without raising for mixed results.
    results = [
        {
            "image_name": "a.png",
            "expected_match": "FOUND",
            "actual_match": True,
            "correct": True,
            "confidence": "HIGH",
            "amount_correct": True,
            "date_correct": True,
            "description_correct": True,
        },
        {
            "image_name": "b.png",
            "expected_match": "NOT_FOUND",
            "actual_match": False,
            "correct": True,
            "confidence": "NONE",
            "amount_correct": False,
            "date_correct": False,
            "description_correct": False,
        },
        {"image_name": "c.png", "correct": None, "error": "no_ground_truth"},
    ]
    summary = el._compute_summary_metrics(results)
    el.print_summary(results, summary)  # no assertion — must not raise


def test_print_summary_no_scored():
    results = [{"image_name": "a.png", "correct": None, "error": "no_ground_truth"}]
    el.print_summary(results, el._compute_summary_metrics(results))


# ---------------------------------------------------------------------------
# run() end-to-end against a fixture
# ---------------------------------------------------------------------------


def test_run_writes_results(tmp_path):
    links_path = tmp_path / "transaction_links.jsonl"
    with links_path.open("w") as f:
        f.write(
            json.dumps(
                {
                    "image_name": "CASE001_receipt.png",
                    "matched": True,
                    "confidence": "HIGH",
                    "receipt_total": 83.48,
                    "bank_transaction_amount": 83.48,
                    "bank_transaction_date": "15/01/2024",
                    "bank_transaction_description": "OFFICE PLUS",
                }
            )
            + "\n"
        )

    gt_path = tmp_path / "gt.yml"
    gt_path.write_text(
        yaml.safe_dump(
            {
                "CASE001_receipt.png": [
                    {
                        "receipt_total": 83.48,
                        "match_status": "FOUND",
                        "bank_date": "15/01/2024",
                        "bank_description": "OFFICE PLUS",
                        "bank_amount": 83.48,
                    }
                ]
            }
        )
    )

    out_dir = tmp_path / "eval"
    result_path = el.run(links_path, gt_path, out_dir)
    assert result_path.exists()
    rows = [json.loads(line) for line in result_path.read_text().splitlines()]
    assert rows[0]["correct"] is True


# ---------------------------------------------------------------------------
# Tolerance comes from YAML (pipeline.linking.hybrid_amount_tolerance)
# ---------------------------------------------------------------------------

_VALID_LINKING = {
    "case_key_pattern": "^(?P<case>[^_]+)_",
    "vlm_prompt": "single_receipt_link",
    "vlm_max_tokens": 4096,
    "vlm_temperature": 0.0,
    "hybrid_amount_tolerance": 0.01,
    "hybrid_date_window_days": 5,
    "hybrid_description_threshold": 0.3,
    "hybrid_min_confidence": "LOW",
    "data_dir": "/tmp/x",
    "output": "/tmp/x/transaction_links.jsonl",
    "ground_truth": "/tmp/x/gt.yml",
    "evaluation_dir": "/tmp/x/eval",
    "log_dir": "/tmp/x/logs",
}


def test_evaluate_linking_requires_amount_tolerance():
    """No silent Python default — callers must pass the YAML-sourced tolerance."""
    with pytest.raises(TypeError):
        el.evaluate_linking([], {})


def _run_with_tolerance(tmp_path, tolerance):
    cfg = {
        "bootstrap": {"model": {"type": "internvl3-vllm"}},
        "pipeline": {"linking": {**_VALID_LINKING, "hybrid_amount_tolerance": tolerance}},
    }
    cfg_path = tmp_path / "run_config.yml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    links_path = tmp_path / "links.jsonl"
    links_path.write_text(
        json.dumps(
            {
                "image_name": "CASE001_receipt.png",
                "matched": True,
                "confidence": "HIGH",
                "receipt_total": 100.0,
                "bank_transaction_amount": 98.0,
                "bank_transaction_date": "15/01/2024",
                "bank_transaction_description": "OFFICE PLUS",
            }
        )
        + "\n"
    )
    gt_path = tmp_path / "gt.yml"
    gt_path.write_text(
        yaml.safe_dump(
            {
                "CASE001_receipt.png": [
                    {
                        "supplier": "Office Plus",
                        "receipt_total": 100.0,
                        "match_status": "FOUND",
                        "bank_date": "15/01/2024",
                        "bank_description": "OFFICE PLUS",
                        "bank_amount": 100.0,
                    }
                ]
            }
        )
    )
    result_path = el.run(links_path, gt_path, tmp_path / "eval", config_path=cfg_path)
    return [json.loads(line) for line in result_path.read_text().splitlines()]


def test_run_amount_tolerance_from_yaml_loose(tmp_path):
    """A $2 amount delta scores correct when the YAML tolerance allows it."""
    rows = _run_with_tolerance(tmp_path, 10.0)
    assert rows[0]["amount_correct"] is True


def test_run_amount_tolerance_from_yaml_strict(tmp_path):
    """The same $2 delta scores incorrect under the strict YAML tolerance."""
    rows = _run_with_tolerance(tmp_path, 0.01)
    assert rows[0]["amount_correct"] is False
