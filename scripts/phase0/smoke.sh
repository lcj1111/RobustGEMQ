#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

venv_dir="${VENV_DIR:-$repo_root/.venv}"
python_bin="$venv_dir/bin/python"
artifact_dir="$repo_root/artifacts/phase0"
mkdir -p "$artifact_dir"

if [[ ! -x "$python_bin" ]]; then
    echo "Missing $python_bin; run scripts/phase0/setup_env.sh first." >&2
    exit 2
fi

"$python_bin" scripts/phase0/collect_env.py --output "$artifact_dir/environment.json"

compile_status=0
"$python_bin" -m compileall -q gemq tests || compile_status=$?

import_status=0
"$python_bin" -c 'import gemq, torch, triton, transformers; assert torch.cuda.is_available(); print({"torch": torch.__version__, "cuda": torch.version.cuda, "gpus": torch.cuda.device_count(), "triton": triton.__version__, "transformers": transformers.__version__})' \
    2>&1 | tee "$artifact_dir/import-smoke.log" || import_status=${PIPESTATUS[0]}

pytest_status=0
"$python_bin" -m pytest \
    tests/test_quant_linear_equiv.py \
    tests/test_moe_block_equiv.py \
    -v --junitxml="$artifact_dir/synthetic-cuda.xml" \
    2>&1 | tee "$artifact_dir/synthetic-cuda.log" || pytest_status=${PIPESTATUS[0]}

PHASE0_COMPILE_STATUS="$compile_status" \
PHASE0_IMPORT_STATUS="$import_status" \
PHASE0_PYTEST_STATUS="$pytest_status" \
"$python_bin" -c '
import json, os
from datetime import datetime, timezone
from pathlib import Path
statuses = {
    "source_compile": int(os.environ["PHASE0_COMPILE_STATUS"]),
    "import_cuda": int(os.environ["PHASE0_IMPORT_STATUS"]),
    "synthetic_cuda_tests": int(os.environ["PHASE0_PYTEST_STATUS"]),
}
payload = {
    "schema_version": 1,
    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "pass" if all(value == 0 for value in statuses.values()) else "fail",
    "exit_codes": statuses,
}
Path("artifacts/phase0/smoke-summary.json").write_text(json.dumps(payload, indent=2) + "\n")
'

if (( compile_status != 0 || import_status != 0 || pytest_status != 0 )); then
    exit 1
fi

echo "Phase 0 smoke tests passed."
