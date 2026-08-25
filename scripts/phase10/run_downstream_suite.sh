#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
python_bin="${PYTHON_BIN:-$repo/.venv/bin/python}"
selection="${SELECTION:-$repo/artifacts/phase10/validation-screen/seed-101/selection.json}"
unlock="${TEST_UNLOCK:-$repo/artifacts/phase10/final-checkpoints/test-unlock.json}"
checkpoint_root="${CHECKPOINT_ROOT:-$repo/results/phase10/checkpoints}"
output_root="${OUTPUT_ROOT:-$repo/artifacts/phase10/downstream}"
phase1_data="${PHASE1_DATA_ROOT:-/data/models/datasets/gemq-phase1}"
phase2_data="${PHASE2_DATA_ROOT:-/data/models/datasets/robustgemq-phase2}"
devices=(4 5 6)

"$python_bin" -c 'import json,sys; assert json.load(open(sys.argv[1]))["test_unlocked"] is True' "$unlock"
mapfile -t methods < <("$python_bin" -c 'import json,sys; [print(x) for x in json.load(open(sys.argv[1]))["selected_methods"]]' "$selection")
[[ ${#methods[@]} -eq 3 ]] || { echo "selection must contain exactly three methods" >&2; exit 2; }
mkdir -p "$output_root"
cd "$repo"

run_one() {
  local method="$1" seed="$2" gpu="$3" output="$output_root/$method/seed-$seed.json"
  [[ -s "$output" ]] && return
  CUDA_VISIBLE_DEVICES="$gpu" HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$python_bin" scripts/phase10/evaluate_downstream.py \
      --checkpoint "$checkpoint_root/$method/seed-$seed" --method "$method" --checkpoint-seed "$seed" \
      --wikitext "$phase1_data/wikitext2/test-00000-of-00001.parquet" \
      --gsm8k "$phase2_data/gsm8k/test.jsonl" --boolq "$phase2_data/boolq/BoolQ/val.jsonl" \
      --output "$output" > "$output_root/$method/seed-$seed.log" 2>&1
}

for seed in 101 202 303; do
  pids=()
  for index in 0 1 2; do run_one "${methods[$index]}" "$seed" "${devices[$index]}" & pids+=("$!"); done
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  (( status == 0 )) || exit 3
done

"$python_bin" scripts/phase10/analyze_downstream.py --root "$output_root" \
  --selection "$selection" --output "$output_root/summary.json"
