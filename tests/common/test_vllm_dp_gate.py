"""The data-parallel gate must not fire for models that can't replicate per GPU.

tests/ is gitignored — local-only. run_dp() starts one INDEPENDENT vLLM engine
per GPU; a quantised 31B wants the whole card once KV cache and vision
activations are counted, so N engines would OOM. On the locked 1xL40S target this
never triggers (one GPU), so these tests are the only thing exercising the guard.

See plans/2026-07-27-reintegrate-gemma4-vllm.md (gap G2).
"""

from dataclasses import dataclass

from common.vllm_dp import resolve_dp_gpus

_GEMMA = "gemma4-31b-w4a16-vllm"
_INTERNVL = "internvl3-vllm"


@dataclass
class _Config:
    """Minimal PipelineConfig stand-in for GPU-count resolution."""

    num_gpus: int = 0
    data_parallel_size: int | None = None


class TestDataParallelGate:
    def test_multi_gpu_capable_model_returns_the_rank_count(self) -> None:
        assert resolve_dp_gpus(_Config(num_gpus=4), _INTERNVL) == 4

    def test_multi_gpu_incapable_model_returns_none(self) -> None:
        assert resolve_dp_gpus(_Config(num_gpus=4), _GEMMA) is None

    def test_single_gpu_returns_none_for_any_model(self) -> None:
        # The locked 1xL40S case: DP is skipped because there's nothing to split.
        assert resolve_dp_gpus(_Config(num_gpus=1), _INTERNVL) is None
        assert resolve_dp_gpus(_Config(num_gpus=1), _GEMMA) is None

    def test_data_parallel_size_takes_priority(self) -> None:
        assert resolve_dp_gpus(_Config(num_gpus=1, data_parallel_size=2), _INTERNVL) == 2

    def test_explicit_data_parallel_size_cannot_force_an_incapable_model(self) -> None:
        # Capability wins over operator intent — the alternative is an OOM.
        assert resolve_dp_gpus(_Config(num_gpus=1, data_parallel_size=4), _GEMMA) is None


class TestGateLogging:
    def test_skipping_dp_for_capability_is_logged(self, caplog) -> None:
        # A silently slower run is indistinguishable from a normal one, so the
        # fall-through must always announce itself.
        with caplog.at_level("INFO", logger="common.vllm_dp"):
            resolve_dp_gpus(_Config(num_gpus=4), _GEMMA)
        assert _GEMMA in caplog.text
        assert "single-engine path" in caplog.text

    def test_single_gpu_does_not_log_a_capability_warning(self, caplog) -> None:
        with caplog.at_level("INFO", logger="common.vllm_dp"):
            resolve_dp_gpus(_Config(num_gpus=1), _GEMMA)
        assert "single-engine path" not in caplog.text
