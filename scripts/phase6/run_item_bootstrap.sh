#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
python_bin="${PYTHON_BIN:-$repo/.venv/bin/python}"
scenario_root="${SCENARIO_ROOT:-$repo/cache/phase6/main}"
checkpoint_root="${CHECKPOINT_ROOT:-$repo/results/phase6/real-checkpoints}"
artifact_root="${ARTIFACT_ROOT:-$repo/artifacts/phase6/item-bootstrap}"
methods=(concat domain-mean gemq-c4)
devices=(0 1 2)

mkdir -p "$artifact_root"
cd "$repo"
pids=()
for index in 0 1 2; do
  method="${methods[$index]}"
  method_dir="$artifact_root/$method"
  mkdir -p "$method_dir"
  if [[ -s "$method_dir/item-nll.json" ]]; then
    echo "reusing completed $method"
    continue
  fi
  (
    CUDA_VISIBLE_DEVICES="${devices[$index]}" "$python_bin" scripts/phase6/evaluate_packed_items.py \
      --checkpoint "$checkpoint_root/$method" --scenario-root "$scenario_root" \
      --output "$method_dir/item-nll.json"
  ) > "$method_dir/run.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ $status -eq 0 ]] || exit 4
"$python_bin" scripts/phase6/analyze_item_bootstrap.py \
  --root "$artifact_root" --output "$artifact_root/bootstrap.json" --draws 10000
