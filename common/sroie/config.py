"""The ``pipeline.sroie`` configuration block.

Every key is required. Nothing here carries a Python-side default, so
reading the YAML alone answers what a run is configured to do — and a
config that is missing a key stops the run instead of quietly benchmarking
something nobody chose.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SECTION = "pipeline.sroie"

_EXAMPLE = """\
pipeline:
  sroie:
    data_dir: /home/jovyan/nfs_share/tod_2026/data/sroie/test
    output_dir: /home/jovyan/nfs_share/tod_2026/data/sroie/output_internvl3
    max_tiles: 12
    max_new_tokens: 256"""


class SroieConfigError(ValueError):
    """The SROIE configuration block is missing or invalid."""


def _fail(what: str, config_path: Path, key: str, expected: str) -> None:
    """Raise a config error carrying all four diagnostic elements."""
    raise SroieConfigError(
        f"What: {what}\n"
        f"Where: {config_path}, under the {key} section.\n"
        f"Expected: {expected}\n"
        f"How to fix: add or correct that key in {config_path}. A complete "
        f"block looks like:\n\n{_EXAMPLE}"
    )


@dataclass(frozen=True)
class SroieSettings:
    """Everything a SROIE benchmark run needs from the YAML."""

    data_dir: Path
    output_dir: Path
    max_tiles: int
    max_new_tokens: int

    @classmethod
    def from_raw(cls, raw_config: dict[str, Any], *, config_path: Path) -> "SroieSettings":
        """Build settings from the raw run_config mapping.

        Args:
            raw_config: The parsed run_config.yml contents.
            config_path: Path to that file, quoted in diagnostics.

        Returns:
            The validated settings.

        Raises:
            SroieConfigError: If the block or any key is missing or invalid.
        """
        block = raw_config.get("pipeline", {}).get("sroie")
        if not isinstance(block, dict):
            _fail(
                what=f"the {_SECTION} configuration block is missing.",
                config_path=config_path,
                key=_SECTION,
                expected="a mapping holding data_dir, output_dir, max_tiles and max_new_tokens.",
            )

        paths = {}
        for key in ("data_dir", "output_dir"):
            value = block.get(key)
            if not isinstance(value, str) or not value.strip():
                _fail(
                    what=f"{_SECTION}.{key} is missing or is not a path string.",
                    config_path=config_path,
                    key=_SECTION,
                    expected=f"{key}: an absolute path, e.g. /home/jovyan/nfs_share/tod_2026/data/sroie/test",
                )
            paths[key] = Path(value)

        integers = {}
        for key in ("max_tiles", "max_new_tokens"):
            value = block.get(key)
            # bool is an int subclass; a YAML `true` here is a config error.
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                _fail(
                    what=f"{_SECTION}.{key} is missing or is not a positive integer.",
                    config_path=config_path,
                    key=_SECTION,
                    expected=f"{key}: a positive integer, e.g. {12 if key == 'max_tiles' else 256}",
                )
            integers[key] = value

        return cls(
            data_dir=paths["data_dir"],
            output_dir=paths["output_dir"],
            max_tiles=integers["max_tiles"],
            max_new_tokens=integers["max_new_tokens"],
        )
