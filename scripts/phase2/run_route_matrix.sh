#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
python_bin="${PYTHON_BIN:-$repo/.venv/bin/python}"
model="${MODEL_PATH:-/data/models/modelscope/LLM-Research/OLMoE-1B-7B-0924}"
scenario_root="$repo/cache/phase2/pilot"
config_root="$repo/artifacts/phase2/pilot/perturb-configs"
output_root="$repo/artifacts/phase2/pilot/route-eval"
mkdir -p "$output_root"

mapfile -t configs < <(find "$config_root" -maxdepth 1 -type f -name '*.pkl' | sort)
if [[ ${#configs[@]} -lt 20 ]]; then
  echo "Expected at least 20 perturbation configs, found ${#configs[@]}" >&2
  exit 2
fi

status=0
for ((offset=0; offset<${#configs[@]}; offset+=8)); do
  jobs=()
  labels=()
  for gpu in 0 1 2 3 4 5 6 7; do
    index=$((offset + gpu))
    [[ $index -lt ${#configs[@]} ]] || break
    config="${configs[$index]}"
    name=$(basename "$config" .pkl)
    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" scripts/phase2/evaluate_fake.py \
      --model "$model" \
      --config "$config" \
      --name "$name" \
      --scenarios-root "$scenario_root" \
      --output-dir "$output_root/$name" \
      > "$output_root/$name.log" 2>&1 &
    jobs+=("$!")
    labels+=("$name")
  done
  for index in "${!jobs[@]}"; do
    if ! wait "${jobs[$index]}"; then
      echo "Route evaluation failed: ${labels[$index]}" >&2
      tail -80 "$output_root/${labels[$index]}.log" >&2 || true
      status=1
    fi
  done
  [[ $status -eq 0 ]] || exit "$status"
done
