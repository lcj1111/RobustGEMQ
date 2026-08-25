from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_summary(tmp_path: Path, exit_code: int) -> dict:
    junit = tmp_path / "junit.xml"
    junit.write_text(
        '<testsuite tests="2" failures="0" errors="0" skipped="0"></testsuite>',
        encoding="utf-8",
    )
    output = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/phase6/write_h6_summary.py",
            "--method",
            "concat",
            "--checkpoint",
            "/checkpoints/concat",
            "--exit-code",
            str(exit_code),
            "--junit",
            str(junit),
            "--log",
            str(tmp_path / "run.log"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def test_h6_summary_records_argmax_contract(tmp_path: Path):
    summary = write_summary(tmp_path, 0)
    assert summary["passed"] is True
    assert summary["pytest"] == {"errors": 0, "failures": 0, "skipped": 0, "tests": 2}
    assert summary["required_checks"]["decode_argmax_agreement_min"] == 0.95


def test_h6_summary_preserves_failure(tmp_path: Path):
    summary = write_summary(tmp_path, 1)
    assert summary["passed"] is False
    assert summary["exit_code"] == 1
