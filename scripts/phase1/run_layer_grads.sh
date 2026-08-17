#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$repo"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

env \
  PYTHON_BIN="${PYTHON_BIN:-$repo/.venv/bin/python}" \
  MODEL_PATH="${MODEL_PATH:-/data/models/modelscope/LLM-Research/OLMoE-1B-7B-0924}" \
  DATA_ROOT="${DATA_ROOT:-/data/models/datasets/gemq-phase1}" \
  CUDA_DEVICE="${CUDA_DEVICE:-2,3,6,7}" \
  DEVICE_MAP="${DEVICE_MAP:-balanced}" \
  NSAMPLES=128 \
  SEQLEN=2048 \
  RUN_LAYER_GRADS=true \
  RUN_LAYER_RE=false \
  bash scripts/compute_stats_olmoe.sh
