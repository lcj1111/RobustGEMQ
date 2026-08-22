#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
python_bin="${PYTHON_BIN:-$repo/.venv/bin/python}"
cd "$repo"
"$python_bin" scripts/phase6/finalize_phase6.py \
  --bootstrap artifacts/phase6/item-bootstrap/bootstrap.json \
  --h3 artifacts/phase3/h3-quality.json \
  --h6-root artifacts/phase6/h6-validation \
  --output artifacts/phase6/phase6_decision.json \
  --report artifacts/phase6/REPORT.md
