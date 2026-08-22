#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
selection="${SELECTION:-$repo/artifacts/phase6/gptq-no-rft/summary.json}"
checkpoint_root="${CHECKPOINT_ROOT:-$repo/results/phase6/real-checkpoints}"
artifact_root="${ARTIFACT_ROOT:-$repo/artifacts/phase6/h6-validation}"
python_bin="${PYTHON_BIN:-$repo/.venv/bin/python}"
cuda_device="${CUDA_DEVICE:-3}"

mkdir -p "$artifact_root"
mapfile -t methods < <("$python_bin" -c 'import json,sys; [print(x) for x in json.load(open(sys.argv[1]))["selected_for_real_packing"]]' "$selection")
cd "$repo"
for method in "${methods[@]}"; do
  checkpoint="$checkpoint_root/$method"
  method_dir="$artifact_root/$method"
  mkdir -p "$method_dir"
  [[ -d "$checkpoint" ]] || { echo "missing checkpoint: $checkpoint" >&2; exit 2; }
  CUDA_VISIBLE_DEVICES="$cuda_device" "$python_bin" -m pytest \
    tests/test_real_vs_fake_ppl.py tests/test_decode_equiv.py -q -s \
    --model-path "$checkpoint" --model-name allenai/OLMoE-1B-7B-0924 \
    --nseq 8 --seqlen 2048 --ndecode 32 --no-trust-remote-code \
    > "$method_dir/run.log" 2>&1
  printf 'method=%s\ncheckpoint=%s\nexit_code=0\n' "$method" "$checkpoint" > "$method_dir/status.txt"
done
