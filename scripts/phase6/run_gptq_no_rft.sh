#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
python_bin="${PYTHON_BIN:-$repo/.venv/bin/python}"
model_path="${MODEL_PATH:-/data/models/modelscope/LLM-Research/OLMoE-1B-7B-0924}"
config_root="${CONFIG_ROOT:-$repo/artifacts/phase6/configs/bpe-2.5}"
scenario_root="${SCENARIO_ROOT:-$repo/cache/phase6/main}"
calibration_manifest="${CALIBRATION_MANIFEST:-$repo/cache/phase6/gptq-calibration/manifest.json}"
artifact_root="${ARTIFACT_ROOT:-$repo/artifacts/phase6/gptq-no-rft}"
IFS=',' read -r -a devices <<< "${CUDA_DEVICES:-0,1,2,3}"
methods=(gemq-c4 concat domain-mean alphaq-style)

[[ ${#devices[@]} -eq 4 ]] || { echo "CUDA_DEVICES must contain four GPU indices" >&2; exit 2; }
[[ -f "$calibration_manifest" ]] || { echo "missing $calibration_manifest" >&2; exit 2; }
tokens="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["tokens_path"])' "$calibration_manifest")"
mkdir -p "$artifact_root"
cd "$repo"

for index in 0 1 2 3; do
  gpu="${devices[$index]}"
  used="$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  (( used <= 1024 )) || { echo "GPU $gpu is busy (${used} MiB)" >&2; exit 3; }
done

pids=()
for index in 0 1 2 3; do
  method="${methods[$index]}"
  gpu="${devices[$index]}"
  method_dir="$artifact_root/$method"
  mkdir -p "$method_dir"
  if [[ -s "$method_dir/phase6-eval.json" ]]; then
    echo "reusing completed $method"
    continue
  fi
  (
    /usr/bin/time -v env CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$python_bin" -m gemq.quantize \
      --model "$model_path" --model_name allenai/OLMoE-1B-7B-0924 --use_fast --model_dtype bfloat16 \
      --calib_dataset phase6-balanced --scenario_tokens_path "$tokens" --nsamples 128 --seqlen 2048 \
      --quantizer gptq --mixed --bit_cfg "$config_root/$method.pkl" \
      --attn_wbits 4 --gate_wbits 16 --dense_wbits 4 --expert_wbits 2 \
      --groupsize 128 --blocksize 128 --percdamp 0.01 --mse --reproduce_mcmoe \
      --phase6_eval_root "$scenario_root" --phase6_eval_seeds 0,1,2 \
      --phase6_eval_output "$method_dir/phase6-eval.json" --skip_builtin_eval
  ) > "$method_dir/run.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ $status -eq 0 ]] || exit 4
"$python_bin" scripts/phase6/analyze_no_rft.py \
  --root "$artifact_root" --configs "$config_root" --output "$artifact_root/summary.json"
