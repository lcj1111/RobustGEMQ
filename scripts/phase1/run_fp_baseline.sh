#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

model_path="${MODEL_PATH:-/data/models/modelscope/LLM-Research/OLMoE-1B-7B-0924}"
data_root="${DATA_ROOT:-/data/models/datasets/gemq-phase1}"
python_bin="${PYTHON_BIN:-.venv/bin/python}"
cuda_device="${CUDA_DEVICE:-0}"
artifact_dir="artifacts/phase1/fp-baseline"

export GEMQ_C4_TRAIN_FILE="${GEMQ_C4_TRAIN_FILE:-$data_root/c4/en/c4-train.00000-of-01024.json}"
export GEMQ_C4_VALIDATION_FILE="${GEMQ_C4_VALIDATION_FILE:-$data_root/c4/en/c4-validation.00000-of-00008.json.gz}"
export GEMQ_WIKITEXT_DIR="${GEMQ_WIKITEXT_DIR:-$data_root/wikitext2}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

mkdir -p "$artifact_dir"
started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
set +e
CUDA_VISIBLE_DEVICES="$cuda_device" "$python_bin" -m gemq.quantize \
  --model "$model_path" \
  --model_name allenai/OLMoE-1B-7B-0924 \
  --use_fast \
  --model_dtype bfloat16 \
  --calib_dataset wikitext2 \
  --nsamples 128 \
  --seqlen 2048 \
  --eval_fp 2>&1 | tee "$artifact_dir/run.log"
status="${PIPESTATUS[0]}"
set -e
finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{
  printf 'started_utc=%s\n' "$started"
  printf 'finished_utc=%s\n' "$finished"
  printf 'exit_code=%s\n' "$status"
} > "$artifact_dir/status.txt"
exit "$status"
