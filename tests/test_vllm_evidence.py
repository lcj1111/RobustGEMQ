import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_vllm_evidence", ROOT / "scripts" / "vllm" / "verify_evidence.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_nearest_rank_uses_observed_sample():
    values = [1.0, 2.0, 3.0, 4.0, 100.0]
    assert MODULE.nearest_rank(values, 0.50) == 3.0
    assert MODULE.nearest_rank(values, 0.95) == 100.0
    assert MODULE.nearest_rank(values, 0.99) == 100.0


def test_committed_vllm_evidence_is_recomputable():
    result = MODULE.verify(ROOT / "artifacts" / "vllm" / "evidence.json")
    assert result["status"] == "PASS"
    assert result["benchmark_requests"] == 144
    assert result["workload_identity_pairs"] == 3
