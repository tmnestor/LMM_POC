"""Tests for the extract stage's CLI-arg construction (no GPU/model needed).

Locks the YAML-single-source contract for ``bank_v2`` / ``balance_correction``:
they must NOT enter the config cascade unless the caller passed an explicit
bool — None means "let YAML win" (run_config.yml pipeline.processing.*).

Regression for the bug where typer defaults (True) were injected
unconditionally, silently inverting the committed ``balance_correction: false``
on every entrypoint run.
"""

from pathlib import Path

import yaml
from typer.testing import CliRunner

import stages.extract as extract_mod
from stages.extract import _build_cli_args

_REPO_CONFIG = Path(__file__).parents[2] / "config" / "run_config.yml"


def test_bank_flags_omitted_when_none(tmp_path):
    args = _build_cli_args(tmp_path, tmp_path / "out.jsonl")
    assert "bank_v2" not in args
    assert "balance_correction" not in args


def test_bank_flags_included_when_explicit(tmp_path):
    args = _build_cli_args(
        tmp_path,
        tmp_path / "out.jsonl",
        bank_v2=False,
        balance_correction=True,
    )
    assert args["bank_v2"] is False
    assert args["balance_correction"] is True


def test_optional_args_follow_same_pattern(tmp_path):
    args = _build_cli_args(tmp_path, tmp_path / "out.jsonl")
    assert args == {
        "data_dir": str(tmp_path),
        "output_dir": str(tmp_path),
    }
    full = _build_cli_args(
        tmp_path,
        tmp_path / "out.jsonl",
        model_type="internvl3-vllm",
        batch_size=4,
        max_num_seqs=8,
        verbose=True,
        debug=False,
    )
    assert full["model_type"] == "internvl3-vllm"
    assert full["batch_size"] == 4
    assert full["max_num_seqs"] == 8
    assert full["verbose"] is True
    assert full["debug"] is False


def test_yaml_wins_when_bank_flags_not_passed(tmp_path):
    """The effective config must equal whatever run_config.yml says."""
    from common.app_config import AppConfig

    raw = yaml.safe_load(_REPO_CONFIG.read_text())
    yaml_bank_v2 = raw["pipeline"]["processing"]["bank_v2"]
    yaml_balance = raw["pipeline"]["processing"]["balance_correction"]

    cli = _build_cli_args(tmp_path, tmp_path / "out.jsonl")
    cli["model_path"] = str(tmp_path)  # satisfy path-exists validation locally
    app_cfg = AppConfig.load(cli, config_path=_REPO_CONFIG)

    assert app_cfg.pipeline.bank_v2 is yaml_bank_v2
    assert app_cfg.pipeline.balance_correction is yaml_balance


def test_explicit_flag_overrides_yaml(tmp_path):
    from common.app_config import AppConfig

    raw = yaml.safe_load(_REPO_CONFIG.read_text())
    yaml_balance = raw["pipeline"]["processing"]["balance_correction"]

    cli = _build_cli_args(
        tmp_path,
        tmp_path / "out.jsonl",
        balance_correction=not yaml_balance,
    )
    cli["model_path"] = str(tmp_path)  # satisfy path-exists validation locally
    app_cfg = AppConfig.load(cli, config_path=_REPO_CONFIG)

    assert app_cfg.pipeline.balance_correction is (not yaml_balance)


def test_cli_defaults_bank_flags_to_none(monkeypatch, tmp_path):
    """`stages.extract` invoked without bank flags must pass None to run()."""
    captured: dict = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return tmp_path / "out.jsonl"

    monkeypatch.setattr(extract_mod, "run", fake_run)
    result = CliRunner().invoke(
        extract_mod.app,
        ["-d", str(tmp_path), "-o", str(tmp_path / "out.jsonl")],
    )
    assert result.exit_code == 0
    assert captured["bank_v2"] is None
    assert captured["balance_correction"] is None


def test_cli_explicit_bank_flags_pass_through(monkeypatch, tmp_path):
    captured: dict = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return tmp_path / "out.jsonl"

    monkeypatch.setattr(extract_mod, "run", fake_run)
    result = CliRunner().invoke(
        extract_mod.app,
        [
            "-d",
            str(tmp_path),
            "-o",
            str(tmp_path / "out.jsonl"),
            "--no-balance-correction",
            "--bank-v2",
        ],
    )
    assert result.exit_code == 0
    assert captured["balance_correction"] is False
    assert captured["bank_v2"] is True
