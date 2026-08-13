#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
IFS=',' read -r -a devices <<< "${CUDA_DEVICES:-2,3,6,7}"
if [[ ${#devices[@]} -ne 4 ]]; then
  echo "CUDA_DEVICES must contain exactly four comma-separated physical GPU indices" >&2
  exit 2
fi

ranges=("0 16" "16 32" "32 48" "48 64")
pids=()
for index in 0 1 2 3; do
  read -r start end <<< "${ranges[$index]}"
  CUDA_DEVICE="${devices[$index]}" EXPERT_START="$start" EXPERT_END="$end" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash "$repo/scripts/phase1/run_layer_re_shard.sh" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ $status -ne 0 ]]; then
  echo "At least one layer-re shard failed; inspect artifacts/phase1/layer-re-shards" >&2
  exit 1
fi

cache_dir="$repo/cache/allenai/OLMoE-1B-7B-0924"
"${PYTHON_BIN:-$repo/.venv/bin/python}" "$repo/scripts/phase1/merge_layer_re.py" \
  "$cache_dir/LayerRE_c4-N128-L2048-Seed0_B1,2,3_experts-0-16.pkl" \
  "$cache_dir/LayerRE_c4-N128-L2048-Seed0_B1,2,3_experts-16-32.pkl" \
  "$cache_dir/LayerRE_c4-N128-L2048-Seed0_B1,2,3_experts-32-48.pkl" \
  "$cache_dir/LayerRE_c4-N128-L2048-Seed0_B1,2,3_experts-48-64.pkl" \
  --output "$cache_dir/LayerRE_c4-N128-L2048-Seed0_B1,2,3_fast.pkl" \
  --layers 16 --experts 64 --bits 1,2,3
