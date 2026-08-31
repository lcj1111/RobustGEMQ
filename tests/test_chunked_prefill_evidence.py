import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_chunked_evidence",
    ROOT / "scripts" / "prefill" / "verify_chunked_evidence.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_chunked_prefill_evidence_contract():
    result = MODULE.verify(ROOT / "artifacts" / "prefill" / "p4" / "manifest.json")
    assert result["status"] == "PASS"
    assert result["concurrent_requests"] == 400
