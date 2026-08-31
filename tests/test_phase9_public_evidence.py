from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "07-release" / "manifest.json"


def verify(path: Path, *, check: bool = True):
    return subprocess.run(
        [sys.executable, "scripts/phase9/verify_public_evidence.py", "--manifest", str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=check,
    )


def test_committed_phase9_manifest_validates_offline():
    result = verify(MANIFEST)
    assert '"validated": true' in result.stdout


def test_public_manifest_rejects_missing_input_record(tmp_path: Path):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["inputs"].pop("data_manifest")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    result = verify(path, check=False)
    assert result.returncode != 0
    assert "输入 manifest 不完整" in result.stderr


def test_public_evidence_rejects_inconsistent_aggregate(tmp_path: Path):
    evidence = json.loads(MANIFEST.read_text(encoding="utf-8"))
    evidence["decision"]["point_metrics"]["concat"]["mean_domain_nll"] += 0.1
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    result = verify(path, check=False)
    assert result.returncode != 0
    assert "可重算结果不一致" in result.stderr
