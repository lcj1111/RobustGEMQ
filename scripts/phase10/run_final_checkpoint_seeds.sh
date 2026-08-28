#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
python_bin="${PYTHON_BIN:-$repo/.venv/bin/python}"
model_path="${MODEL_PATH:-/data/models/modelscope/LLM-Research/OLMoE-1B-7B-0924}"
config_root="${CONFIG_ROOT:-$repo/artifacts/phase10/configs/bpe-2.5}"
scenario_root="${SCENARIO_ROOT:-$repo/cache/phase10}"
calibration_manifest="${CALIBRATION_MANIFEST:-$scenario_root/calibration-b-balanced/manifest.json}"
# 方法筛选产物独立于 validation 明细；默认路径必须与筛选脚本的输出一致。
selection="${SELECTION:-$repo/artifacts/phase10/selection/seed-101/selection.json}"
artifact_root="${ARTIFACT_ROOT:-$repo/artifacts/phase10/final-checkpoints}"
checkpoint_root="${CHECKPOINT_ROOT:-$repo/results/phase10/checkpoints}"
h6_root="${H6_ROOT:-$repo/artifacts/phase10/h6}"
data_root="${DATA_ROOT:-/data/models/datasets/gemq-phase1}"
devices=(4 5 6)
tokens="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["tokens_path"])' "$calibration_manifest")"
mapfile -t methods < <("$python_bin" -c 'import json,sys; [print(x) for x in json.load(open(sys.argv[1]))["selected_methods"]]' "$selection")
[[ ${#methods[@]} -eq 3 ]] || { echo "selection must contain exactly three methods" >&2; exit 2; }
mkdir -p "$artifact_root" "$checkpoint_root" "$h6_root"
cd "$repo"

run_one() {
  local method="$1" seed="$2" gpu="$3"
  local run_dir="$artifact_root/$method/seed-$seed"
  local checkpoint="$checkpoint_root/$method/seed-$seed"
  local h6_dir="$h6_root/$method/seed-$seed"
  mkdir -p "$run_dir" "$checkpoint" "$h6_dir"
  if [[ ! -s "$checkpoint/config.json" ]]; then
    /usr/bin/time -v env CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$python_bin" -m gemq.quantize \
        --model "$model_path" --model_name allenai/OLMoE-1B-7B-0924 --use_fast --model_dtype bfloat16 \
        --seed "$seed" --calib_dataset phase10-calibration-b --scenario_tokens_path "$tokens" \
        --nsamples 96 --seqlen 2048 --quantizer gptq --mixed --bit_cfg "$config_root/$method.pkl" \
        --attn_wbits 4 --gate_wbits 16 --dense_wbits 4 --expert_wbits 2 \
        --groupsize 128 --blocksize 128 --percdamp 0.01 --mse --reproduce_mcmoe \
        --finetune_routers --rft_epochs 1 --rft_lr 0.0001 --rft_wd 0.0001 \
        --rft_audit_path "$run_dir/router-update.json" \
        --real_quant --save_path "$checkpoint" --skip_builtin_eval \
        > "$run_dir/quantize.log" 2>&1
  fi
  if [[ -s "$h6_dir/summary.json" ]] && "$python_bin" -c 'import json,sys; assert json.load(open(sys.argv[1]))["passed"] is True' "$h6_dir/summary.json"; then
    return
  fi
  set +e
  GEMQ_WIKITEXT_DIR="$data_root/wikitext2" \
  GEMQ_C4_TRAIN_FILE="$data_root/c4/en/c4-train.00000-of-01024.json" \
  GEMQ_C4_VALIDATION_FILE="$data_root/c4/en/c4-validation.00000-of-00008.json.gz" \
  HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$gpu" \
    "$python_bin" -m pytest tests/test_real_vs_fake_ppl.py tests/test_decode_equiv.py -q -s \
      --model-path "$checkpoint" --model-name allenai/OLMoE-1B-7B-0924 \
      --nseq 8 --seqlen 2048 --ndecode 32 --no-trust-remote-code \
      --junitxml "$h6_dir/junit.xml" > "$h6_dir/run.log" 2>&1
  exit_code=$?
  set -e
  "$python_bin" scripts/phase6/write_h6_summary.py --method "$method" \
    --checkpoint "$checkpoint" --exit-code "$exit_code" --junit "$h6_dir/junit.xml" \
    --log "$h6_dir/run.log" --output "$h6_dir/summary.json"
  [[ "$exit_code" -eq 0 ]] || return "$exit_code"
}

for seed in 202 303; do
  for gpu in "${devices[@]}"; do
    used="$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    (( used <= 1024 )) || { echo "GPU $gpu is busy (${used} MiB)" >&2; exit 3; }
  done
  pids=()
  for index in 0 1 2; do run_one "${methods[$index]}" "$seed" "${devices[$index]}" & pids+=("$!"); done
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  (( status == 0 )) || exit 4
done

"$python_bin" scripts/phase10/unlock_test.py --selection "$selection" \
  --h6-root "$h6_root" --output "$artifact_root/test-unlock.json"
