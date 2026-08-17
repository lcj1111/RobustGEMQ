#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

model_name="${MODEL_NAME:-allenai/OLMoE-1B-7B-0924}"
stats_dir="${STATS_DIR:-cache/$model_name}"
checkpoint="${MODEL_PATH:-results/real_quant_models/$model_name/GEMQ/C4-Seed0-WT2_A4-G16-D4-E2.0_RFT}"
bit_config="${BIT_CONFIG:-configs/$model_name/GEMQ/C4-Seed0_E2.0_B1,2,3_c2c3.pkl}"
output="${OUTPUT:-artifacts/phase1/checksums.txt}"

files=(
  "$stats_dir/LayerGrads_c4-N128-L2048-Seed0.pt"
  "$stats_dir/LayerRE_c4-N128-L2048-Seed0_B1,2,3_fast.pkl"
  "$bit_config"
  "$checkpoint/qmodel.pt"
)

for file in "${files[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "Missing required artifact: $file" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$output")"
sha256sum "${files[@]}" > "$output"
cat "$output"
