from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_prefill", ROOT / "scripts" / "prefill" / "benchmark_prefill.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_parse_lengths_rejects_duplicates_and_non_positive_values():
    assert MODULE.parse_lengths("128, 512,2048") == [128, 512, 2048]
    with pytest.raises(Exception):
        MODULE.parse_lengths("128,128")
    with pytest.raises(Exception):
        MODULE.parse_lengths("0,128")


def test_latency_summary_is_recomputable():
    result = MODULE.summarize_latencies([4.0, 1.0, 3.0, 2.0, 5.0], tokens=10)
    assert result["median_ms"] == 3.0
    assert result["p95_ms"] == pytest.approx(4.8)
    assert result["median_tokens_per_second"] == pytest.approx(10 / 0.003)
