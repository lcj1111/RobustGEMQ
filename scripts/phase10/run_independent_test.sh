#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
python_bin="${PYTHON_BIN:-$repo/.venv/bin/python}"
scenario_root="${SCENARIO_ROOT:-$repo/cache/phase10}"
checkpoint_root="${CHECKPOINT_ROOT:-$repo/results/phase10/checkpoints}"
artifact_root="${ARTIFACT_ROOT:-$repo/artifacts/phase10/independent-test}"
unlock="${TEST_UNLOCK:-$repo/artifacts/phase10/final-checkpoints/test-unlock.json}"
mapfile -t methods < <("$python_bin" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["test_unlocked"] is True; [print(x) for x in p["selected_methods"]]' "$unlock")
[[ ${#methods[@]} -eq 3 ]] || exit 2
devices=(4 5 6 7)
mkdir -p "$artifact_root"
cd "$repo"

jobs=()
for method in "${methods[@]}"; do
  for seed in 101 202 303; do jobs+=("$method $seed"); done
done
for start in 0 4 8; do
  pids=()
  for offset in 0 1 2 3; do
    index=$((start + offset))
    (( index < ${#jobs[@]} )) || continue
    read -r method seed <<< "${jobs[$index]}"
    gpu="${devices[$offset]}"
    output="$artifact_root/$method/seed-$seed/test-items.json"
    mkdir -p "$(dirname "$output")"
    if [[ -s "$output" ]]; then continue; fi
    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" scripts/phase10/evaluate_checkpoint_items.py \
      --checkpoint "$checkpoint_root/$method/seed-$seed" --method "$method" \
      --checkpoint-seed "$seed" --scenario-root "$scenario_root" --split test \
      --output "$output" > "$(dirname "$output")/run.log" 2>&1 &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  (( status == 0 )) || exit 4
done
"$python_bin" scripts/phase10/analyze_independent_test.py --root "$artifact_root" \
  --unlock "$unlock" --output "$artifact_root/statistics.json"
