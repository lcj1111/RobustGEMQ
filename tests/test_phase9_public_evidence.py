from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_committed_phase9_evidence_verifies_offline():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/phase9/verify_public_evidence.py",
            "--evidence",
            "docs/07-release/evidence.json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert '"verified": true' in result.stdout
