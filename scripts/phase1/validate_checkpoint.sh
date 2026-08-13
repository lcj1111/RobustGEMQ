#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
checkpoint="${MODEL_PATH:-$repo/results/real_quant_models/allenai/OLMoE-1B-7B-0924/GEMQ/C4-Seed0-WT2_A4-G16-D4-E2.0_RFT}"
data_root="${DATA_ROOT:-/data/models/datasets/gemq-phase1}"
artifact_dir="$repo/artifacts/phase1/equivalence/${VALIDATION_TAG:-short}"
mkdir -p "$artifact_dir"
cd "$repo"

export GEMQ_WIKITEXT_DIR="$data_root/wikitext2"
export GEMQ_C4_TRAIN_FILE="$data_root/c4/en/c4-train.00000-of-01024.json"
export GEMQ_C4_VALIDATION_FILE="$data_root/c4/en/c4-validation.00000-of-00008.json.gz"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

started=$(date -u +%FT%TZ)
set +e
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-2}" "${PYTHON_BIN:-$repo/.venv/bin/python}" -m pytest \
  tests/test_real_vs_fake_ppl.py tests/test_decode_equiv.py \
  -v -s \
  --model-path "$checkpoint" \
  --model-name allenai/OLMoE-1B-7B-0924 \
  --nseq "${NSEQ:-8}" \
  --seqlen "${SEQLEN:-2048}" \
  --ndecode "${NDECODE:-32}" \
  --no-trust-remote-code \
  2>&1 | tee "$artifact_dir/run.log"
status=${PIPESTATUS[0]}
set -e
finished=$(date -u +%FT%TZ)
printf 'started_utc=%s\nfinished_utc=%s\nexit_code=%s\ncheckpoint=%s\n' \
  "$started" "$finished" "$status" "$checkpoint" > "$artifact_dir/status.txt"
exit "$status"
