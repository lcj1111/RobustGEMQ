#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
profile="${PROFILE:-smoke}"
python_bin="${PYTHON_BIN:-$repo/.venv/bin/python}"

mapfile -t busy < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '$2 + 0 > 1024 {gsub(/ /, "", $1); print $1 ":" $2}')
if [[ ${#busy[@]} -gt 0 ]]; then
  echo "Refusing to start Phase 2 matrix; GPUs above 1 GiB used: ${busy[*]}" >&2
  exit 3
fi

domains=(general math code instruction)
jobs=()
labels=()
if [[ "$profile" == "smoke" ]]; then
  for index in 0 1 2 3; do
    DOMAIN="${domains[$index]}" SEED=0 CUDA_DEVICE="$index" PROFILE=smoke \
      PYTHON_BIN="$python_bin" bash "$repo/scripts/phase2/run_scenario.sh" &
    jobs+=("$!")
    labels+=("${domains[$index]}:0")
  done
elif [[ "$profile" == "pilot" ]]; then
  index=0
  for domain in "${domains[@]}"; do
    for seed in 0 1; do
      DOMAIN="$domain" SEED="$seed" CUDA_DEVICE="$index" PROFILE=pilot \
        PYTHON_BIN="$python_bin" bash "$repo/scripts/phase2/run_scenario.sh" &
      jobs+=("$!")
      labels+=("$domain:$seed")
      index=$((index + 1))
    done
  done
else
  echo "Unsupported PROFILE=$profile" >&2
  exit 2
fi

status=0
for index in "${!jobs[@]}"; do
  if ! wait "${jobs[$index]}"; then
    echo "Scenario failed: ${labels[$index]}" >&2
    status=1
  fi
done
exit "$status"
