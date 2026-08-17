#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
python_bin="${PYTHON_BIN:-$repo/.venv/bin/python}"
model="${MODEL_PATH:-/data/models/modelscope/LLM-Research/OLMoE-1B-7B-0924}"
scenario_root="$repo/cache/phase2/pilot"
fp_root="$repo/artifacts/phase2/pilot/fake-eval/fp"
output_root="$repo/cache/phase4/route-stats"
log_root="$repo/artifacts/phase4/route-stats"
IFS=',' read -r -a gpus <<< "${GPU_LIST:-0,1,2,3,4,5,6,7}"
domains=(general math code instruction)
seeds=(0 1)
mkdir -p "$output_root" "$log_root"

if [[ ${#gpus[@]} -lt 8 ]]; then
  echo "Phase 4 route statistics require eight listed GPUs for the frozen 4-domain x 2-seed matrix" >&2
  exit 2
fi
for gpu in "${gpus[@]}"; do
  used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if (( used > 1024 )); then
    echo "Refusing GPU $gpu: ${used} MiB already used" >&2
    exit 3
  fi
done

jobs=()
labels=()
index=0
for domain in "${domains[@]}"; do
  for seed in "${seeds[@]}"; do
    gpu="${gpus[$index]}"
    scenario="$scenario_root/$domain/seed-$seed/scenario.json"
    tokens=$(
      "$python_bin" - "$scenario" "$fp_root/summary.json" "$domain" "$seed" <<'PY'
import json, sys
scenario = json.load(open(sys.argv[1]))
summary = json.load(open(sys.argv[2]))
key = f"{sys.argv[3]}:seed-{sys.argv[4]}"
if summary["scenarios"][key]["token_sha256"] != scenario["token_sha256"]:
    raise SystemExit(f"FP route token hash mismatch for {key}")
print(scenario["tokens_path"])
PY
    )
    trace="$fp_root/route-$domain-seed-$seed.pt"
    out_dir="$output_root/$domain/seed-$seed"
    log_dir="$log_root/$domain/seed-$seed"
    out="$out_dir/RouteRE_B1,2,3.pkl"
    mkdir -p "$out_dir" "$log_dir"
    if [[ -s "$out" ]]; then
      echo "Reusing $out"
      index=$((index + 1))
      continue
    fi
    started=$(date -u +%FT%TZ)
    printf 'started_utc=%s\ngpu=%s\ndomain=%s\nseed=%s\n' "$started" "$gpu" "$domain" "$seed" \
      > "$log_dir/status.txt"
    CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$python_bin" -m gemq.compute_model_stats \
        --mode layer_re \
        --model "$model" \
        --model_name allenai/OLMoE-1B-7B-0924 \
        --model_dtype bfloat16 \
        --calib_dataset "$domain" \
        --scenario_tokens_path "$tokens" \
        --use_fast \
        --seed "$seed" \
        --nsamples 32 \
        --seqlen 512 \
        --wbits 1,2,3 \
        --route_margin_path "$trace" \
        --route_margin_eps 1e-6 \
        --route_vmax 100 \
        --layer_re_path "$out" \
        --forward_batch_size "${FORWARD_BATCH_SIZE:-8}" \
        > "$log_dir/run.log" 2>&1 &
    jobs+=("$!")
    labels+=("$domain:seed-$seed:$gpu:$log_dir")
    index=$((index + 1))
  done
done

status=0
for job_index in "${!jobs[@]}"; do
  IFS=':' read -r domain seed_label gpu log_dir <<< "${labels[$job_index]}"
  if wait "${jobs[$job_index]}"; then
    printf 'finished_utc=%s\nexit_code=0\n' "$(date -u +%FT%TZ)" >> "$log_dir/status.txt"
  else
    code=$?
    printf 'finished_utc=%s\nexit_code=%s\n' "$(date -u +%FT%TZ)" "$code" >> "$log_dir/status.txt"
    echo "Route statistics failed: $domain:$seed_label on GPU $gpu" >&2
    tail -100 "$log_dir/run.log" >&2 || true
    status=1
  fi
done
exit "$status"
