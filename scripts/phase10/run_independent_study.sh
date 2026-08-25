#!/usr/bin/env bash
set -euo pipefail

# Phase 10 的单入口。每个子阶段都支持复用已验证的产物；任一门禁失败时立即停止。
repo="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
python_bin="${PYTHON_BIN:-$repo/.venv/bin/python}"
scenario_root="${SCENARIO_ROOT:-$repo/cache/phase10}"
config_root="${CONFIG_ROOT:-$repo/artifacts/phase10/configs/bpe-2.5}"
gemq_config_root="${GEMQ_CONFIG_ROOT:-$repo/configs/allenai/OLMoE-1B-7B-0924/GEMQ}"
status_root="${STATUS_ROOT:-$repo/artifacts/phase10/study-run}"
mkdir -p "$status_root"
cd "$repo"

# 允许接管一个已经启动的 calibration-A 进程，避免并发写同一场景。
if [[ -n "${WAIT_FOR_PID_FILE:-}" && -s "$WAIT_FOR_PID_FILE" ]]; then
  previous_pid="$(<"$WAIT_FOR_PID_FILE")"
  while kill -0 "$previous_pid" 2>/dev/null; do sleep 30; done
fi

started="$(date -u +%FT%TZ)"
printf 'started_utc=%s\nstage=calibration-a\n' "$started" > "$status_root/status.txt"

CUDA_DEVICES="${CUDA_DEVICES:-4,5,6,7}" bash scripts/phase10/run_calibration_a_stats.sh

printf 'started_utc=%s\nstage=method-configs\n' "$started" > "$status_root/status.txt"
"$python_bin" scripts/phase10/build_method_configs.py \
  --scenario-root "$scenario_root" --gemq-config-root "$gemq_config_root" \
  --output-root "$config_root" --bpe 2.5

printf 'started_utc=%s\nstage=validation-screen\n' "$started" > "$status_root/status.txt"
bash scripts/phase10/run_validation_screen.sh

printf 'started_utc=%s\nstage=final-checkpoint-seeds\n' "$started" > "$status_root/status.txt"
bash scripts/phase10/run_final_checkpoint_seeds.sh

printf 'started_utc=%s\nstage=independent-test\n' "$started" > "$status_root/status.txt"
bash scripts/phase10/run_independent_test.sh

printf 'started_utc=%s\nstage=downstream-suite\n' "$started" > "$status_root/status.txt"
bash scripts/phase10/run_downstream_suite.sh

printf 'started_utc=%s\nfinished_utc=%s\nstage=complete\nexit_code=0\n' \
  "$started" "$(date -u +%FT%TZ)" > "$status_root/status.txt"
