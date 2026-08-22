#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
python_bin="${PYTHON_BIN:-$repo/.venv/bin/python}"
model_path="${MODEL_PATH:-/data/models/modelscope/LLM-Research/OLMoE-1B-7B-0924}"
data_root="${DOMAIN_DATA_ROOT:-/data/models/datasets/robustgemq-phase2}"
registry="${DOMAIN_REGISTRY:-$repo/configs/domains/phase6_domains.json}"
output_root="${SCENARIO_ROOT:-$repo/cache/phase6/main}"
artifact_root="${ARTIFACT_ROOT:-$repo/artifacts/phase6/main-scenarios}"

mkdir -p "$output_root" "$artifact_root"
for domain in general math code instruction; do
  for seed in 0 1 2; do
    output="$output_root/$domain/seed-$seed"
    mkdir -p "$output"
    "$python_bin" "$repo/scripts/phase2/materialize_scenario.py" \
      --registry "$registry" \
      --data-root "$data_root" \
      --domain "$domain" \
      --model "$model_path" \
      --model-id allenai/OLMoE-1B-7B-0924 \
      --seed "$seed" --nsamples 128 --seqlen 2048 \
      --output-dir "$output" \
      > "$artifact_root/${domain}-seed-${seed}.json"
  done
done

"$python_bin" "$repo/scripts/phase6/validate_main_scenarios.py" "$output_root" \
  --output "$artifact_root/validation.json"
