#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
python_bin="${PYTHON_BIN:-$repo/.venv/bin/python}"
model_path="${MODEL_PATH:-/data/models/modelscope/LLM-Research/OLMoE-1B-7B-0924}"
scenario_root="${SCENARIO_ROOT:-$repo/cache/phase6/main}"
calibration_manifest="${CALIBRATION_MANIFEST:-$repo/cache/phase6/gptq-calibration/manifest.json}"
artifact_dir="${ARTIFACT_DIR:-$repo/artifacts/phase6/bf16-main}"
cuda_device="${CUDA_DEVICE:-3}"

mkdir -p "$artifact_dir"
tokens="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["tokens_path"])' "$calibration_manifest")"
cd "$repo"
CUDA_VISIBLE_DEVICES="$cuda_device" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$python_bin" -m gemq.quantize \
  --model "$model_path" --model_name allenai/OLMoE-1B-7B-0924 --use_fast --model_dtype bfloat16 \
  --calib_dataset phase6-balanced --scenario_tokens_path "$tokens" --nsamples 128 --seqlen 2048 \
  --eval_fp --phase6_eval_root "$scenario_root" --phase6_eval_seeds 0,1,2 \
  --phase6_eval_output "$artifact_dir/phase6-eval.json" --skip_builtin_eval \
  > "$artifact_dir/run.log" 2>&1
