"""Regression tests: the IVL_* environment-variable config layer is gone.

The env layer sat below YAML in the cascade, nothing in the repo set it, and a
manifest-set IVL_* var silently lost to the YAML — so it was removed entirely.
Config has exactly two surfaces: CLI flags > YAML > dataclass defaults.
"""

import common
from common import pipeline_config
from common.pipeline_config import merge_configs


class TestEnvLayerRemoved:
    def test_load_env_config_is_gone(self) -> None:
        assert not hasattr(pipeline_config, "load_env_config")
        assert not hasattr(pipeline_config, "ENV_PREFIX")

    def test_not_exported_from_common(self) -> None:
        assert "load_env_config" not in common.__all__
        assert not hasattr(common, "load_env_config")

    def test_ivl_env_vars_have_no_effect(self, monkeypatch) -> None:
        monkeypatch.setenv("IVL_MAX_TILES", "99")
        monkeypatch.setenv("IVL_MODEL_TYPE", "internvl3-38b-vllm")
        monkeypatch.setenv("IVL_BANK_V2", "false")
        config = merge_configs(
            cli_args={},
            yaml_config={
                "data_dir": "/tmp/images",
                "output_dir": "/tmp/out",
                "max_tiles": 18,
                "model_type": "internvl3-vllm",
                "bank_v2": True,
            },
        )
        assert config.max_tiles == 18
        assert config.model_type == "internvl3-vllm"
        assert config.bank_v2 is True

    def test_cli_still_beats_yaml(self) -> None:
        config = merge_configs(
            cli_args={"max_tiles": 6},
            yaml_config={
                "data_dir": "/tmp/images",
                "output_dir": "/tmp/out",
                "max_tiles": 18,
            },
        )
        assert config.max_tiles == 6
