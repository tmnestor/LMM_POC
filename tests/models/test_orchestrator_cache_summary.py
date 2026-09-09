from types import SimpleNamespace

from models.orchestrator import DocumentOrchestrator


def test_cache_hit_summary_proxies_backend():
    # Bypass the heavy __init__; we only exercise the proxy method.
    orch = DocumentOrchestrator.__new__(DocumentOrchestrator)
    orch._backend = SimpleNamespace(cache_hit_summary=lambda: {"available": True, "hit_ratio": 0.5})
    assert orch.cache_hit_summary() == {"available": True, "hit_ratio": 0.5}
