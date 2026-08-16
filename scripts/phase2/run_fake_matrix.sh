#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
python_bin="${PYTHON_BIN:-$repo/.venv/bin/python}"
model="${MODEL_PATH:-/data/models/modelscope/LLM-Research/OLMoE-1B-7B-0924}"
scenario_root="$repo/cache/phase2/pilot"
config_root="$repo/artifacts/phase2/pilot/configs"
output_root="$repo/artifacts/phase2/pilot/fake-eval"
mkdir -p "$output_root"

jobs=()
labels=()
gpu=0
for bpe in 2.5 2.0; do
  for domain in general math code instruction; do
    name="${domain}-bpe-${bpe}"
    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" scripts/phase2/evaluate_fake.py \
      --model "$model" \
      --config "$config_root/bpe-${bpe}/${domain}.pkl" \
      --name "$name" \
      --scenarios-root "$scenario_root" \
      --output-dir "$output_root/$name" \
      --quality-only \
      > "$output_root/$name.log" 2>&1 &
    jobs+=("$!")
    labels+=("$name")
    gpu=$((gpu + 1))
  done
done

status=0
for index in "${!jobs[@]}"; do
  if ! wait "${jobs[$index]}"; then
    echo "Fake evaluation failed: ${labels[$index]}" >&2
    tail -80 "$output_root/${labels[$index]}.log" >&2 || true
    status=1
  fi
done
exit "$status"
