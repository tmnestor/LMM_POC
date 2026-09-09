"""Tests for the trust_clean amount-tolerance config wire-up (no GPU/model)."""

import json

import pytest
import yaml

from stages import trust_clean as tc
from stages.trust_clean import _load_trust_amount_tolerance


def _write_cfg(tmp_path, tol=0.01, include=True):
    trust = {"amount_tolerance": tol} if include else {}
    cfg = {"pipeline": {"trust": trust}}
    p = tmp_path / "run_config.yml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_load_tolerance_valid(tmp_path):
    assert _load_trust_amount_tolerance(_write_cfg(tmp_path, 0.02)) == 0.02


def test_load_tolerance_missing_key_is_diagnostic(tmp_path, assert_diagnostic_error):
    with pytest.raises(ValueError) as exc:
        _load_trust_amount_tolerance(_write_cfg(tmp_path, include=False))
    assert_diagnostic_error(str(exc.value))


def test_load_tolerance_missing_file_is_diagnostic(tmp_path, assert_diagnostic_error):
    with pytest.raises(FileNotFoundError) as exc:
        _load_trust_amount_tolerance(tmp_path / "nope.yml")
    assert_diagnostic_error(str(exc.value))


def test_run_passes_config_tolerance_to_compliance(tmp_path, monkeypatch):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps({"image_name": "CASE001", "nodes": []}) + "\n")
    cfg = _write_cfg(tmp_path, 0.05)

    captured = {}

    def fake_compliance(state, tolerance=0.01):
        captured["tolerance"] = tolerance
        return True, {"COMPLIANCE_STATUS": "compliant"}

    monkeypatch.setattr(tc, "run_trust_compliance", fake_compliance)
    tc.run(raw, tmp_path / "out.jsonl", config_path=cfg)
    assert captured["tolerance"] == 0.05  # config value reached the compliance validator
