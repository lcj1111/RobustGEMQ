from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "07-release" / "evidence.json"


def verify(path: Path, *, check: bool = True):
    return subprocess.run(
        [sys.executable, "scripts/phase9/verify_public_evidence.py", "--evidence", str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=check,
    )


def test_committed_phase9_evidence_verifies_offline():
    result = verify(EVIDENCE)
    assert '"verified": true' in result.stdout


def test_public_evidence_rejects_bad_scenario_hash(tmp_path: Path):
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    evidence["scenario_provenance"]["code:seed-0"]["token_sha256"] = "0" * 63
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    result = verify(path, check=False)
    assert result.returncode != 0
    assert "SHA-256" in result.stderr


def test_public_evidence_rejects_inconsistent_aggregate(tmp_path: Path):
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    evidence["decision"]["point_metrics"]["concat"]["mean_domain_nll"] += 0.1
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    result = verify(path, check=False)
    assert result.returncode != 0
    assert "可重算结果不一致" in result.stderr
