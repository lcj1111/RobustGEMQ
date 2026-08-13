#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

tag="${BENCHMARK_TAG:-real}"
artifact_dir="artifacts/phase1/benchmark-${tag}"
mkdir -p "$artifact_dir"

set +e
CUDA_DEVICE="${CUDA_DEVICE:-0}" \
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}" \
NUM_SAMPLES="${NUM_SAMPLES:-3}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}" \
bash scripts/bench_generate_olmoe.sh 2>&1 | tee "$artifact_dir/run.log"
status="${PIPESTATUS[0]}"
set -e

printf 'exit_code=%s\n' "$status" > "$artifact_dir/status.txt"
exit "$status"
