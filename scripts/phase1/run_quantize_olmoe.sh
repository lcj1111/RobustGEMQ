#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
artifact_dir="$repo/artifacts/phase1/quantize"
checkpoint="$repo/results/real_quant_models/allenai/OLMoE-1B-7B-0924/GEMQ/C4-Seed0-WT2_A4-G16-D4-E2.0_RFT"
mkdir -p "$artifact_dir"
cd "$repo"

nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits > "$artifact_dir/gpu-before.txt"
started=$(date -u +%FT%TZ)

set +e
/usr/bin/time -v env \
  HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTHON_BIN="${PYTHON_BIN:-$repo/.venv/bin/python}" \
  MODEL_PATH="${MODEL_PATH:-/data/models/modelscope/LLM-Research/OLMoE-1B-7B-0924}" \
  DATA_ROOT="${DATA_ROOT:-/data/models/datasets/gemq-phase1}" \
  CUDA_DEVICE="${CUDA_DEVICE:-2,3,6,7}" \
  NSAMPLES="${NSAMPLES:-128}" \
  SEQLEN="${SEQLEN:-2048}" \
  BPE=2.0 \
  MIXED_PREC=true \
  FINETUNE_ROUTERS=true \
  RFT_EPOCHS=1 \
  REAL_QUANT=true \
  SAVE_MODEL=true \
  bash scripts/quantize_olmoe.sh 2>&1 | tee "$artifact_dir/run.log"
status=${PIPESTATUS[0]}
set -e

finished=$(date -u +%FT%TZ)
bytes=0
if [[ -d "$checkpoint" ]]; then
  bytes=$(du -sb "$checkpoint" | cut -f1)
fi
printf 'started_utc=%s\nfinished_utc=%s\nexit_code=%s\ncheckpoint=%s\nbytes=%s\n' \
  "$started" "$finished" "$status" "$checkpoint" "$bytes" > "$artifact_dir/status.txt"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits > "$artifact_dir/gpu-after.txt"
exit "$status"
