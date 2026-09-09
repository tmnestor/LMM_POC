from types import SimpleNamespace

from models.backends.vllm_backend import VllmBackend


def _backend() -> VllmBackend:
    return VllmBackend(engine=object())


def test_cache_summary_aggregates_cached_and_total_tokens():
    b = _backend()
    b._record_cache_stats(SimpleNamespace(num_cached_tokens=10, prompt_token_ids=list(range(40))))
    b._record_cache_stats(SimpleNamespace(num_cached_tokens=30, prompt_token_ids=list(range(60))))
    s = b.cache_hit_summary()
    assert s["available"] is True
    assert s["cached_prompt_tokens"] == 40
    assert s["total_prompt_tokens"] == 100
    assert abs(s["hit_ratio"] - 0.4) < 1e-9
    assert s["calls"] == 2  # one per _record_cache_stats


def test_cache_summary_counts_calls_even_when_unavailable():
    b = _backend()
    b._record_cache_stats(SimpleNamespace(prompt_token_ids=[1, 2, 3]))  # no num_cached_tokens
    b._record_cache_stats(SimpleNamespace(prompt_token_ids=[1, 2]))  # still counted
    assert b.cache_hit_summary() == {"available": False, "calls": 2}


def test_cache_summary_unavailable_when_attr_missing():
    b = _backend()
    b._record_cache_stats(SimpleNamespace(prompt_token_ids=[1, 2, 3]))  # no num_cached_tokens
    assert b.cache_hit_summary() == {"available": False, "calls": 1}


def test_cache_summary_unavailable_when_prompt_ids_missing():
    b = _backend()
    b._record_cache_stats(SimpleNamespace(num_cached_tokens=10))  # no prompt_token_ids
    assert b.cache_hit_summary() == {"available": False, "calls": 1}
