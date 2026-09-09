"""Tests for the F1 computation in notebooks/LMMPOC_comparison.ipynb.

The dashboard in that notebook labels its panels "Mean F1" but historically plotted
the ``overall_accuracy`` column, because the F1 helper discarded its per-document
scores and returned only a scalar. These tests pin the corrected contract: the
helper must expose genuine per-document F1, and that decomposition must aggregate
back to the scalar the summary table reports.
"""

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from common.evaluation_metrics import calculate_field_accuracy_f1

NOTEBOOK = Path(__file__).parents[2] / "notebooks" / "LMMPOC_comparison.ipynb"
HELPER = "compute_model_mean_f1"


def _load_helper():
    """Exec the helper straight out of the notebook so we test the shipped code."""
    nb = json.loads(NOTEBOOK.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if f"def {HELPER}" not in source:
            continue
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == HELPER:
                namespace = {
                    "pd": pd,
                    "Path": Path,
                    "calculate_field_accuracy_f1": calculate_field_accuracy_f1,
                    "FIELD_COLUMNS": FIELD_COLUMNS,
                }
                exec(compile(ast.Module([node], []), str(NOTEBOOK), "exec"), namespace)
                return namespace[HELPER]
    raise AssertionError(f"{HELPER} not found in {NOTEBOOK}")


FIELD_COLUMNS = ["SUPPLIER_NAME", "TOTAL_AMOUNT", "GST_AMOUNT"]


@pytest.fixture
def batch_df() -> pd.DataFrame:
    """Three documents whose per-document F1 deliberately differs."""
    return pd.DataFrame(
        [
            # Perfect document.
            {
                "image_file": "doc_a.png",
                "SUPPLIER_NAME": "Acme Pty Ltd",
                "TOTAL_AMOUNT": "110.00",
                "GST_AMOUNT": "10.00",
            },
            # Two of three fields wrong.
            {
                "image_file": "doc_b.png",
                "SUPPLIER_NAME": "WRONG",
                "TOTAL_AMOUNT": "999.00",
                "GST_AMOUNT": "10.00",
            },
            # One of three fields wrong.
            {
                "image_file": "doc_c.png",
                "SUPPLIER_NAME": "Beta Corp",
                "TOTAL_AMOUNT": "55.00",
                "GST_AMOUNT": "NOT_FOUND",
            },
        ]
    )


@pytest.fixture
def gt_df() -> pd.DataFrame:
    gt = pd.DataFrame(
        [
            {
                "image_file": "doc_a.png",
                "SUPPLIER_NAME": "Acme Pty Ltd",
                "TOTAL_AMOUNT": "110.00",
                "GST_AMOUNT": "10.00",
            },
            {
                "image_file": "doc_b.png",
                "SUPPLIER_NAME": "Acme Pty Ltd",
                "TOTAL_AMOUNT": "110.00",
                "GST_AMOUNT": "10.00",
            },
            {
                "image_file": "doc_c.png",
                "SUPPLIER_NAME": "Beta Corp",
                "TOTAL_AMOUNT": "55.00",
                "GST_AMOUNT": "5.00",
            },
        ]
    )
    gt["image_stem"] = gt["image_file"].apply(lambda x: Path(x).stem)
    return gt


def test_returns_scalar_and_per_document_scores(batch_df, gt_df):
    """The helper must hand back per-document F1, not just a grand mean."""
    mean_f1, per_doc = _load_helper()(batch_df, gt_df, "test-model")

    assert isinstance(mean_f1, float)
    assert isinstance(per_doc, pd.Series)
    assert sorted(per_doc.index) == ["doc_a", "doc_b", "doc_c"]


def test_per_document_scores_vary(batch_df, gt_df):
    """Guards the original bug: a constant column cannot be a real distribution."""
    _, per_doc = _load_helper()(batch_df, gt_df, "test-model")

    assert per_doc.nunique() > 1, "per-document F1 collapsed to a constant"
    assert per_doc["doc_a"] > per_doc["doc_c"] > per_doc["doc_b"]


def test_per_document_mean_matches_scalar(batch_df, gt_df):
    """The decomposition must aggregate back to the reported scalar."""
    mean_f1, per_doc = _load_helper()(batch_df, gt_df, "test-model")

    assert per_doc.mean() == pytest.approx(mean_f1, abs=1e-12)


def test_scores_are_bounded(batch_df, gt_df):
    mean_f1, per_doc = _load_helper()(batch_df, gt_df, "test-model")

    assert 0.0 <= mean_f1 <= 1.0
    assert per_doc.between(0.0, 1.0).all()


def test_empty_inputs_return_empty_series(gt_df):
    mean_f1, per_doc = _load_helper()(pd.DataFrame(), gt_df, "test-model")

    assert mean_f1 == 0.0
    assert per_doc.empty
