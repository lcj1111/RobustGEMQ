#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
python_bin="${PYTHON_BIN:-$repo/.venv/bin/python}"
model="${MODEL_PATH:-/data/models/modelscope/LLM-Research/OLMoE-1B-7B-0924}"
data_root="${DOMAIN_DATA_ROOT:-/data/models/datasets/robustgemq-phase2}"
registry="$repo/configs/domains/phase3_heldout.json"
output_root="$repo/cache/phase3/heldout"

"$python_bin" scripts/phase3/prepare_heldout.py \
  --data-root "$data_root" \
  --manifest "$repo/artifacts/phase3/heldout-sources.json"

for domain in general math code instruction; do
  for seed in 0 1; do
    "$python_bin" scripts/phase3/materialize_heldout.py \
      --registry "$registry" \
      --data-root "$data_root" \
      --domain "$domain" \
      --model "$model" \
      --seed "$seed" \
      --nsamples 32 \
      --seqlen 512 \
      --output-dir "$output_root/$domain/seed-$seed"
  done
done
