#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python3}"
venv_dir="${VENV_DIR:-$repo_root/.venv}"

if [[ ! -x "$venv_dir/bin/python" ]]; then
    "$python_bin" -m venv "$venv_dir"
fi

"$venv_dir/bin/python" -m pip install --upgrade pip setuptools wheel
"$venv_dir/bin/python" -m pip install \
    --constraint requirements/phase0-constraints.txt \
    -e ".[dev]"

"$venv_dir/bin/python" scripts/phase0/collect_env.py \
    --output artifacts/phase0/environment.json

echo "Phase 0 environment ready: $venv_dir"
