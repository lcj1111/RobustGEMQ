import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_concurrent_prefill",
    ROOT / "scripts" / "prefill" / "benchmark_concurrent_prefill.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_poisson_arrivals_are_reproducible_and_monotonic():
    first = MODULE.poisson_arrivals(16, 8.0, 20260829)
    second = MODULE.poisson_arrivals(16, 8.0, 20260829)
    assert first == second
    assert first[0] == 0.0
    assert all(left < right for left, right in zip(first, first[1:]))


def test_latency_summary_reports_tail_percentiles():
    summary = MODULE.summarize_ms([1.0, 2.0, 3.0, 4.0, 100.0])
    assert summary["count"] == 5
    assert summary["p50_ms"] == 3.0
    assert summary["p95_ms"] > summary["p50_ms"]
    assert summary["p99_ms"] >= summary["p95_ms"]


@pytest.mark.parametrize(
    "num_requests,request_rate", [(0, 1.0), (1, 0.0), (-1, 2.0)]
)
def test_invalid_arrival_configuration_is_rejected(num_requests, request_rate):
    with pytest.raises(ValueError):
        MODULE.poisson_arrivals(num_requests, request_rate, 1)
