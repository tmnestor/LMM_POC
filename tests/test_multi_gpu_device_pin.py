"""Tests for thread-local CUDA device pinning in MultiGPUOrchestrator.

Validates the fix for the OOM caused by worker threads inheriting the main
thread's current device (typically cuda:N-1, the last set_device during
Phase 1 loading). Without the pin, every worker's
``torch.cuda.empty_cache()`` / ``torch.cuda.synchronize()`` targets a
single GPU instead of the GPU its own model lives on, driving memory
fragmentation on non-current GPUs to OOM.

These tests mock torch to avoid requiring CUDA; they only verify that
``torch.cuda.set_device`` is called with the correct GPU id derived from
``gpu_config.device_map``.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from unittest.mock import MagicMock


@dataclass
class _FakeConfig:
    """Minimal stand-in for PipelineConfig -- only device_map is read."""

    device_map: str | None


def _install_fake_torch(monkeypatch, set_device_mock):
    """Install a fake torch module with cuda.set_device mocked.

    The real torch is too heavy to import in unit tests and on dev machines
    without GPUs it may not be importable at all.
    """
    fake_cuda = types.SimpleNamespace(set_device=set_device_mock)
    fake_torch = types.SimpleNamespace(cuda=fake_cuda)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def _stub_run_batch(monkeypatch):
    """Replace run_batch inside common.multi_gpu with a no-op mock.

    Returns the mock so tests can assert it was called.
    """
    import common.multi_gpu as multi_gpu_mod

    mock = MagicMock(return_value=([], [], {}, {}))
    monkeypatch.setattr(multi_gpu_mod, "run_batch", mock)
    return mock


# -- set_device is called with the correct GPU id for well-formed device_map --


def test_process_chunk_pins_cuda_0(monkeypatch):
    from common.multi_gpu import MultiGPUOrchestrator

    set_device = MagicMock()
    _install_fake_torch(monkeypatch, set_device)
    run_batch_mock = _stub_run_batch(monkeypatch)

    gpu_stack = (_FakeConfig(device_map="cuda:0"), MagicMock(), MagicMock())
    MultiGPUOrchestrator._process_chunk(gpu_stack, [], {})

    set_device.assert_called_once_with(0)
    run_batch_mock.assert_called_once()


def test_process_chunk_pins_cuda_1(monkeypatch):
    from common.multi_gpu import MultiGPUOrchestrator

    set_device = MagicMock()
    _install_fake_torch(monkeypatch, set_device)
    _stub_run_batch(monkeypatch)

    gpu_stack = (_FakeConfig(device_map="cuda:1"), MagicMock(), MagicMock())
    MultiGPUOrchestrator._process_chunk(gpu_stack, [], {})

    set_device.assert_called_once_with(1)


def test_process_chunk_pins_cuda_3(monkeypatch):
    """Ensure all 4 GPU indices work (4x A10G prod machine layout)."""
    from common.multi_gpu import MultiGPUOrchestrator

    set_device = MagicMock()
    _install_fake_torch(monkeypatch, set_device)
    _stub_run_batch(monkeypatch)

    gpu_stack = (_FakeConfig(device_map="cuda:3"), MagicMock(), MagicMock())
    MultiGPUOrchestrator._process_chunk(gpu_stack, [], {})

    set_device.assert_called_once_with(3)


# -- set_device is NOT called for non-cuda or malformed device_map -----------


def test_process_chunk_skips_pin_when_device_map_is_auto(monkeypatch):
    """Single-GPU fallback path (device_map='auto') must not call set_device."""
    from common.multi_gpu import MultiGPUOrchestrator

    set_device = MagicMock()
    _install_fake_torch(monkeypatch, set_device)
    _stub_run_batch(monkeypatch)

    gpu_stack = (_FakeConfig(device_map="auto"), MagicMock(), MagicMock())
    MultiGPUOrchestrator._process_chunk(gpu_stack, [], {})

    set_device.assert_not_called()


def test_process_chunk_skips_pin_when_device_map_is_none(monkeypatch):
    from common.multi_gpu import MultiGPUOrchestrator

    set_device = MagicMock()
    _install_fake_torch(monkeypatch, set_device)
    _stub_run_batch(monkeypatch)

    gpu_stack = (_FakeConfig(device_map=None), MagicMock(), MagicMock())
    MultiGPUOrchestrator._process_chunk(gpu_stack, [], {})

    set_device.assert_not_called()


def test_process_chunk_skips_pin_when_device_map_is_cpu(monkeypatch):
    from common.multi_gpu import MultiGPUOrchestrator

    set_device = MagicMock()
    _install_fake_torch(monkeypatch, set_device)
    _stub_run_batch(monkeypatch)

    gpu_stack = (_FakeConfig(device_map="cpu"), MagicMock(), MagicMock())
    MultiGPUOrchestrator._process_chunk(gpu_stack, [], {})

    set_device.assert_not_called()


# -- Malformed cuda: prefixes are tolerated, not fatal -----------------------


def test_process_chunk_tolerates_malformed_cuda_device_map(monkeypatch):
    """A malformed 'cuda:abc' string must not raise; it falls through."""
    from common.multi_gpu import MultiGPUOrchestrator

    set_device = MagicMock()
    _install_fake_torch(monkeypatch, set_device)
    run_batch_mock = _stub_run_batch(monkeypatch)

    gpu_stack = (_FakeConfig(device_map="cuda:abc"), MagicMock(), MagicMock())
    # Should not raise
    MultiGPUOrchestrator._process_chunk(gpu_stack, [], {})

    set_device.assert_not_called()
    run_batch_mock.assert_called_once()


def test_process_chunk_tolerates_runtime_error_from_set_device(monkeypatch):
    """If set_device raises (e.g. CUDA unavailable), the chunk still runs."""
    from common.multi_gpu import MultiGPUOrchestrator

    set_device = MagicMock(side_effect=RuntimeError("no CUDA"))
    _install_fake_torch(monkeypatch, set_device)
    run_batch_mock = _stub_run_batch(monkeypatch)

    gpu_stack = (_FakeConfig(device_map="cuda:0"), MagicMock(), MagicMock())
    MultiGPUOrchestrator._process_chunk(gpu_stack, [], {})

    # set_device was attempted once
    set_device.assert_called_once_with(0)
    # run_batch still ran despite the RuntimeError
    run_batch_mock.assert_called_once()
