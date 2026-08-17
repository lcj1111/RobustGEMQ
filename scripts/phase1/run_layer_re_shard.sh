#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_DEVICE:?set CUDA_DEVICE to one physical GPU index}"
: "${EXPERT_START:?set EXPERT_START (inclusive)}"
: "${EXPERT_END:?set EXPERT_END (exclusive)}"

repo="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
model_path="${MODEL_PATH:-/data/models/modelscope/LLM-Research/OLMoE-1B-7B-0924}"
data_root="${DATA_ROOT:-/data/models/datasets/gemq-phase1}"
artifact_dir="$repo/artifacts/phase1/layer-re-shards/experts-${EXPERT_START}-${EXPERT_END}"
output="$repo/cache/allenai/OLMoE-1B-7B-0924/LayerRE_c4-N128-L2048-Seed0_B1,2,3_experts-${EXPERT_START}-${EXPERT_END}.pkl"

mkdir -p "$artifact_dir"
cd "$repo"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits > "$artifact_dir/gpu-before.txt"
started=$(date -u +%FT%TZ)

set +e
/usr/bin/time -v env \
  PYTHON_BIN="${PYTHON_BIN:-$repo/.venv/bin/python}" \
  MODEL_PATH="$model_path" \
  DATA_ROOT="$data_root" \
  CUDA_DEVICE="$CUDA_DEVICE" \
  NSAMPLES=128 \
  SEQLEN=2048 \
  FORWARD_BATCH_SIZE="${FORWARD_BATCH_SIZE:-8}" \
  RUN_LAYER_GRADS=false \
  RUN_LAYER_RE=true \
  EXPERT_START="$EXPERT_START" \
  EXPERT_END="$EXPERT_END" \
  LAYER_RE_PATH="$output" \
  bash scripts/compute_stats_olmoe.sh 2>&1 | tee "$artifact_dir/run.log"
status=${PIPESTATUS[0]}
set -e

finished=$(date -u +%FT%TZ)
bytes=0
[[ -f "$output" ]] && bytes=$(stat -c %s "$output")
printf 'started_utc=%s\nfinished_utc=%s\nexit_code=%s\ngpu=%s\nexpert_start=%s\nexpert_end=%s\noutput=%s\nbytes=%s\n' \
  "$started" "$finished" "$status" "$CUDA_DEVICE" "$EXPERT_START" "$EXPERT_END" "$output" "$bytes" \
  > "$artifact_dir/status.txt"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits > "$artifact_dir/gpu-after.txt"
exit "$status"
