#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
python_bin="${PYTHON_BIN:-$repo/.venv/bin/python}"
model_path="${MODEL_PATH:-/data/models/modelscope/LLM-Research/OLMoE-1B-7B-0924}"
data_root="${DOMAIN_DATA_ROOT:-/data/models/datasets/robustgemq-phase2}"
split_root="${SPLIT_ROOT:-$data_root/phase10}"
scenario_root="${SCENARIO_ROOT:-$repo/cache/phase10}"
artifact_root="${ARTIFACT_ROOT:-$repo/artifacts/phase10/data-contract}"
experiment="${EXPERIMENT:-$repo/configs/phase10/experiment.json}"
source_registry="${SOURCE_REGISTRY:-$repo/configs/domains/phase6_domains.json}"

mkdir -p "$artifact_root" "$scenario_root"
"$python_bin" scripts/phase10/build_record_splits.py \
  --registry "$source_registry" --data-root "$data_root" \
  --output-root "$split_root" --experiment "$experiment" \
  > "$artifact_root/build-splits.json"
"$python_bin" scripts/phase10/verify_record_splits.py \
  --manifest "$split_root/split-manifest.json" \
  > "$artifact_root/verify-splits.json"

materialize() {
  local registry_name="$1" output_split="$2" seed="$3" samples="$4"
  for domain in general math code instruction; do
    output="$scenario_root/$output_split/$domain/seed-$seed"
    mkdir -p "$output"
    "$python_bin" scripts/phase2/materialize_scenario.py \
      --registry "$split_root/registries/$registry_name.json" \
      --data-root "$data_root" --domain "$domain" \
      --model "$model_path" --model-id allenai/OLMoE-1B-7B-0924 \
      --seed "$seed" --nsamples "$samples" --seqlen 2048 --output-dir "$output" \
      > "$artifact_root/${output_split}-${domain}-seed-${seed}.json"
  done
}

for seed in 0 1 2; do
  materialize "calibration-a-seed-$seed" calibration-a "$seed" 24
done
materialize calibration-b calibration-b 0 24
materialize validation validation 0 48
materialize test test 0 96

"$python_bin" scripts/phase6/materialize_balanced_calibration.py \
  --scenario-root "$scenario_root/calibration-b" \
  --output-dir "$scenario_root/calibration-b-balanced" \
  --seed 0 --samples-per-domain 24 \
  > "$artifact_root/calibration-b-balanced.json"

"$python_bin" scripts/phase10/verify_materialized_protocol.py \
  --experiment "$experiment" --split-manifest "$split_root/split-manifest.json" \
  --scenario-root "$scenario_root" --output "$artifact_root/materialized-protocol.json"
