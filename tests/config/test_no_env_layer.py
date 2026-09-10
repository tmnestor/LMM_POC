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
        # `common` has no __init__.py any more -- it is an implicit namespace
        # package, so there is no __all__ to check. What still matters is that
        # the name is not reachable through the package by any route.
        assert not hasattr(common, "load_env_config")

    def test_ivl_env_vars_have_no_effect(self, monkeypatch) -> None:
        monkeypatch.setenv("IVL_MAX_TILES", "99")
        monkeypatch.setenv("IVL_MODEL_TYPE", "internvl3-38b-vllm")
        monkeypatch.setenv("IVL_VERBOSE", "true")
        config = merge_configs(
            cli_args={},
            yaml_config={
                "data_dir": "/tmp/images",
                "output_dir": "/tmp/out",
                "max_tiles": 18,
                "model_type": "internvl3-vllm",
                "verbose": False,
            },
        )
        assert config.max_tiles == 18
        assert config.model_type == "internvl3-vllm"
        assert config.verbose is False

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
