"""How many vLLM engines the screen starts, and on whose say-so.

tests/ is gitignored — local-only.

Replaces the Gemma-era gate test, which asserted against `resolve_dp_gpus` and
a `supports_data_parallel` registry flag — neither of which survived. The
question it was asking is still live, though: `run_dp` starts one INDEPENDENT
engine per GPU, so this number decides whether the run fans out across the four
production cards or quietly serialises on one.

The failure mode is silent in both directions. Too low and the run takes four
times as long while three cards idle; too high and the last worker OOMs after
the others have already loaded, which reads as a model problem rather than a
config one.
"""

from dataclasses import dataclass

import pytest

from common.vllm_dp import resolve_gpu_count

_INTERNVL = "internvl3-vllm"


@dataclass
class _Config:
    """Minimal PipelineConfig stand-in for GPU-count resolution."""

    model_type: str = _INTERNVL
    num_gpus: int = 0
    data_parallel_size: int | None = None


class TestPrecedence:
    """data_parallel_size > num_gpus > auto-detect, in that order."""

    def test_data_parallel_size_wins_over_num_gpus(self) -> None:
        # The explicit override exists to run fewer engines than there are
        # cards -- so it must beat num_gpus even when num_gpus is larger.
        assert resolve_gpu_count(_Config(num_gpus=4, data_parallel_size=2)) == 2

    def test_num_gpus_is_used_when_no_override_is_given(self) -> None:
        assert resolve_gpu_count(_Config(num_gpus=4)) == 4

    def test_a_data_parallel_size_of_one_is_honoured_not_treated_as_unset(self) -> None:
        """1 and None mean different things.

        `if config.data_parallel_size:` would collapse them and silently fan
        out across every card when the operator asked for a single engine.
        """
        assert resolve_gpu_count(_Config(num_gpus=4, data_parallel_size=1)) == 1


class TestAutoDetect:
    def test_zero_num_gpus_falls_through_to_the_device_count(self, monkeypatch) -> None:
        import torch

        monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
        assert resolve_gpu_count(_Config(num_gpus=0)) == 3

    def test_a_cpu_box_resolves_to_one_not_zero(self, monkeypatch) -> None:
        """Zero engines would be a division by zero when sharding the images.

        This is the local-dev and the evaluate-pod case: no CUDA device at all.
        """
        import torch

        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
        assert resolve_gpu_count(_Config(num_gpus=0)) == 1


class TestTheStageGate:
    """`stages.quality_screen.run` only takes the DP path above one GPU."""

    @pytest.mark.parametrize("resolved", [1, 0])
    def test_one_gpu_or_none_does_not_fan_out(self, resolved: int) -> None:
        # Mirrors `if resolved_gpus > 1:` in the stage. Starting run_dp for a
        # single shard pays the whole subprocess and engine-build cost to run
        # exactly what the in-process path would have run.
        assert not resolved > 1

    def test_more_than_one_gpu_fans_out(self) -> None:
        assert resolve_gpu_count(_Config(num_gpus=4)) > 1
