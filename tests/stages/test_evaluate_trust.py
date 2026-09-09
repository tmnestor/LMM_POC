"""Tests for stages.evaluate_trust config wire-up + behavior preservation.

tests/ is gitignored — local-only. Covers the fail-fast loader for the
pipeline.trust eval settings and proves the config-driven field-accuracy path
is behavior-identical to the old hardcoded path.
"""

import pytest
import yaml

from stages import evaluate_trust as et
from stages.io import write_jsonl

_VALID_TRUST = {
    "amount_tolerance": 0.01,
    "linking_fields": {
        "id_fields": ["trust_abn", "beneficiary_tfn"],
        "amount_fields": ["share_of_net_income", "franking_credit", "capital_gain_component"],
    },
}


def _write_config(tmp_path, trust=None, omit=None):
    block = dict(_VALID_TRUST) if trust is None else trust
    if omit:
        # omit supports dotted "linking_fields.id_fields"
        if "." in omit:
            top, sub = omit.split(".", 1)
            block = {**block, top: {k: v for k, v in block[top].items() if k != sub}}
        else:
            block = {k: v for k, v in block.items() if k != omit}
    cfg = {"pipeline": {"trust": block}}
    path = tmp_path / "run_config.yml"
    path.write_text(yaml.safe_dump(cfg))
    return path


# ---------------------------------------------------------------------------
# Loader: valid config
# ---------------------------------------------------------------------------


def test_load_trust_eval_config_valid(tmp_path):
    path = _write_config(tmp_path)
    cfg = et._load_trust_eval_config(path)
    assert cfg == {
        "amount_tolerance": 0.01,
        "id_fields": {"trust_abn", "beneficiary_tfn"},
        "amount_fields": {"share_of_net_income", "franking_credit", "capital_gain_component"},
    }


def test_load_trust_eval_config_default_path_loads_repo_config():
    # config_path=None must resolve to the repo run_config.yml and validate.
    cfg = et._load_trust_eval_config(None)
    assert cfg["amount_tolerance"] == 0.01
    assert cfg["id_fields"] == {"trust_abn", "beneficiary_tfn"}
    assert cfg["amount_fields"] == {
        "share_of_net_income",
        "franking_credit",
        "capital_gain_component",
    }


# ---------------------------------------------------------------------------
# Loader: fail-fast diagnostics
# ---------------------------------------------------------------------------


def test_load_missing_file_is_diagnostic(tmp_path, assert_diagnostic_error):
    with pytest.raises(FileNotFoundError) as exc:
        et._load_trust_eval_config(tmp_path / "does_not_exist.yml")
    assert_diagnostic_error(str(exc.value))


def test_load_missing_trust_section_is_diagnostic(tmp_path, assert_diagnostic_error):
    path = tmp_path / "run_config.yml"
    path.write_text(yaml.safe_dump({"pipeline": {}}))
    with pytest.raises(ValueError) as exc:
        et._load_trust_eval_config(path)
    assert_diagnostic_error(str(exc.value))


@pytest.mark.parametrize(
    "omit",
    ["amount_tolerance", "linking_fields.id_fields", "linking_fields.amount_fields"],
)
def test_load_missing_key_is_diagnostic(tmp_path, assert_diagnostic_error, omit):
    path = _write_config(tmp_path, omit=omit)
    with pytest.raises(ValueError) as exc:
        et._load_trust_eval_config(path)
    assert_diagnostic_error(str(exc.value))


def test_load_bad_tolerance_is_diagnostic(tmp_path, assert_diagnostic_error):
    bad = {**_VALID_TRUST, "amount_tolerance": "not_a_number"}
    path = _write_config(tmp_path, trust=bad)
    with pytest.raises(ValueError) as exc:
        et._load_trust_eval_config(path)
    assert_diagnostic_error(str(exc.value))


def test_load_empty_id_fields_is_diagnostic(tmp_path, assert_diagnostic_error):
    bad = {**_VALID_TRUST, "linking_fields": {"id_fields": [], "amount_fields": ["x"]}}
    path = _write_config(tmp_path, trust=bad)
    with pytest.raises(ValueError) as exc:
        et._load_trust_eval_config(path)
    assert_diagnostic_error(str(exc.value))


# ---------------------------------------------------------------------------
# Behavior preservation: config-driven path == old hardcoded path
# ---------------------------------------------------------------------------


def _run_with(tmp_path, extracted_data, gt_fields, config_path):
    """Run evaluate_trust.run for a single CASE001 record and return field_accuracy."""
    extractions = tmp_path / "raw_extractions.jsonl"
    write_jsonl(
        extractions,
        [{"image_name": "CASE001", "extracted_data": extracted_data, "processing_time": 1.0}],
    )
    gt = tmp_path / "gt.yml"
    gt.write_text(
        yaml.safe_dump(
            {
                "CASE001_dist.png": {
                    "linking_fields": gt_fields,
                    "compliance_status": "compliant",
                }
            }
        )
    )
    out_dir = tmp_path / "out"
    et.run(extractions, gt, out_dir, config_path=config_path)
    results = [
        __import__("json").loads(line)
        for line in (out_dir / "trust_evaluation_results.jsonl").read_text().splitlines()
    ]
    return next(r["field_accuracy"] for r in results if r.get("case_id") == "CASE001")


def test_run_field_accuracy_matches_old_hardcoded_behavior(tmp_path):
    """The new config-driven path reproduces the old hardcoded results.

    Exact ID match + an amount within 0.01 relative -> correct; an amount off by
    >1% -> incorrect. This is identical to the old hardcoded tolerance=0.01 and
    field split.
    """
    cfg = _write_config(tmp_path)
    extracted = {
        "TRUST_ABN": "12 345 678 901",
        "BENEFICIARY_TFN": "123456782",
        "SHARE_OF_NET_INCOME": "1000.00",  # exact -> correct
        "FRANKING_CREDIT": "100.50",  # 0.5% off -> within tolerance -> correct
        "CAPITAL_GAIN_COMPONENT": "200.00",  # 100% off -> incorrect
    }
    gt_fields = {
        "trust_abn": "12345678901",  # space-normalised exact match
        "beneficiary_tfn": "123 456 782",
        "share_of_net_income": "1000.00",
        "franking_credit": "100.00",
        "capital_gain_component": "100.00",
    }
    acc = _run_with(tmp_path, extracted, gt_fields, cfg)
    assert acc == {
        "trust_abn": True,
        "beneficiary_tfn": True,
        "share_of_net_income": True,
        "franking_credit": True,
        "capital_gain_component": False,
    }


def test_parse_amount_strips_currency_and_whitespace():
    """Regression: r"[$,\\s]" matched backslash + literal 's', not whitespace."""
    assert et._parse_amount("1 234.56") == 1234.56  # internal space must be stripped
    assert et._parse_amount("$1,234.56") == 1234.56
    assert et._parse_amount(" 83.48 ") == 83.48
    # The old class silently stripped literal 's' chars -> "12s34" parsed as 1234.0
    assert et._parse_amount("12s34") is None
    assert et._parse_amount("NOT_FOUND") is None
    assert et._parse_amount("") is None
