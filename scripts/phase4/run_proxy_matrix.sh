#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
python_bin="${PYTHON_BIN:-$repo/.venv/bin/python}"
model="${MODEL_PATH:-/data/models/modelscope/LLM-Research/OLMoE-1B-7B-0924}"
scenario_root="$repo/cache/phase2/pilot"
config_root="$repo/artifacts/phase4/proxy-configs"
output_root="$repo/artifacts/phase4/proxy-route-eval"
IFS=',' read -r -a gpus <<< "${GPU_LIST:-0,1,2,3,4,5,6,7}"
mkdir -p "$output_root"
mapfile -t configs < <(find "$config_root" -maxdepth 1 -type f -name '*.pkl' | sort)

if [[ ${#configs[@]} -lt 20 ]]; then
  echo "H4 requires at least 20 perturbation configs, found ${#configs[@]}" >&2
  exit 2
fi
for gpu in "${gpus[@]}"; do
  used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if (( used > 1024 )); then
    echo "Refusing GPU $gpu: ${used} MiB already used" >&2
    exit 3
  fi
done

status=0
for ((offset=0; offset<${#configs[@]}; offset+=${#gpus[@]})); do
  jobs=()
  labels=()
  for gpu_index in "${!gpus[@]}"; do
    index=$((offset + gpu_index))
    [[ $index -lt ${#configs[@]} ]] || break
    gpu="${gpus[$gpu_index]}"
    config="${configs[$index]}"
    name=$(basename "$config" .pkl)
    if [[ -s "$output_root/$name/summary.json" ]]; then
      echo "Reusing $name"
      continue
    fi
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
  for job_index in "${!jobs[@]}"; do
    if ! wait "${jobs[$job_index]}"; then
      echo "Proxy route evaluation failed: ${labels[$job_index]}" >&2
      tail -100 "$output_root/${labels[$job_index]}.log" >&2 || true
      status=1
    fi
  done
  [[ $status -eq 0 ]] || exit "$status"
done
