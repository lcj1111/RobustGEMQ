from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_committed_phase10_public_evidence_is_valid():
    subprocess.run(
        [
            sys.executable,
            "scripts/phase10/verify_public_evidence.py",
            "--evidence",
            "docs/08-independent-confirmation/evidence.json",
        ],
        cwd=ROOT,
        check=True,
    )
