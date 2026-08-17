#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
python_bin="${PYTHON_BIN:-$repo/.venv/bin/python}"
model="${MODEL_PATH:-/data/models/modelscope/LLM-Research/OLMoE-1B-7B-0924}"
scenario_root="$repo/cache/phase3/heldout"
config_root="$repo/artifacts/phase3/configs"
output_root="$repo/artifacts/phase3/fake-quality"
IFS=',' read -r -a gpus <<< "${GPU_LIST:-2,3,4,5,6,7}"
methods=(gemq-c4 concat domain-mean domain-worst domain-cvar-0.5 alphaq-style)
mkdir -p "$output_root"

if [[ ${#gpus[@]} -eq 0 ]]; then
  echo "GPU_LIST must contain at least one GPU" >&2
  exit 2
fi
for gpu in "${gpus[@]}"; do
  used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if (( used > 1024 )); then
    echo "Refusing GPU $gpu: ${used} MiB already used" >&2
    exit 3
  fi
done

jobs=()
labels=()
launch() {
  local gpu="$1" name="$2" config="$3"
  local config_args=()
  if [[ -n "$config" ]]; then config_args=(--config "$config"); fi
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" scripts/phase3/evaluate_quality.py \
    --model "$model" \
    "${config_args[@]}" \
    --name "$name" \
    --scenarios-root "$scenario_root" \
    --output "$output_root/$name/summary.json" \
    > "$output_root/$name.log" 2>&1 &
  jobs+=("$!")
  labels+=("$name")
}

wait_wave() {
  local status=0
  for index in "${!jobs[@]}"; do
    if ! wait "${jobs[$index]}"; then
      echo "Quality evaluation failed: ${labels[$index]}" >&2
      tail -100 "$output_root/${labels[$index]}.log" >&2 || true
      status=1
    fi
  done
  jobs=()
  labels=()
  return "$status"
}

tasks=("fp|")
for bpe in 2.5 2.0; do
  for method in "${methods[@]}"; do
    tasks+=("${method}-bpe-${bpe}|$config_root/bpe-${bpe}/${method}.pkl")
  done
done

status=0
for index in "${!tasks[@]}"; do
  IFS='|' read -r name config <<< "${tasks[$index]}"
  gpu="${gpus[$((index % ${#gpus[@]}))]}"
  launch "$gpu" "$name" "$config"
  if (( (${#jobs[@]} == ${#gpus[@]}) || (index + 1 == ${#tasks[@]}) )); then
    wait_wave || status=1
  fi
done
exit "$status"
