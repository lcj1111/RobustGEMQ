import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_prefill_evidence", ROOT / "scripts" / "prefill" / "verify_evidence.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_prefill_evidence_contract():
    summary = MODULE.verify(ROOT / "artifacts/prefill/evidence.json")
    assert summary["status"] == "PASS"
    assert summary["stages"] == ["baseline", "p1", "p2", "p3"]
    assert summary["verified_files"] == 24
