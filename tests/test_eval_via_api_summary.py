"""Latency summary math in scripts/eval_via_api.py — the numbers it prints
are what get quoted from a run, so they're worth pinning down."""
from scripts.eval_via_api import latency_summary, _percentile


def test_percentile_is_nearest_rank():
    vals = list(range(1, 21))  # 1..20
    assert _percentile(vals, 50) == 10
    assert _percentile(vals, 95) == 19
    assert _percentile(vals, 100) == 20
    assert _percentile([7], 95) == 7


def test_latency_summary():
    s = latency_summary([3000, 1000, 2000, 4000])
    assert s == {"n": 4, "mean_ms": 2500.0, "p50_ms": 2000, "p95_ms": 4000, "max_ms": 4000}


def test_latency_summary_empty():
    assert latency_summary([]) == {"n": 0}
