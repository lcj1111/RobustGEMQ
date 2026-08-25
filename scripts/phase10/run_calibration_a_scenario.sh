#!/usr/bin/env bash
set -euo pipefail

: "${DOMAIN:?set DOMAIN to general, math, code, or instruction}"
: "${SEED:?set SEED to 0, 1, or 2}"
case "$DOMAIN" in general|math|code|instruction) ;; *) echo "invalid DOMAIN=$DOMAIN" >&2; exit 2;; esac
case "$SEED" in 0|1|2) ;; *) echo "invalid SEED=$SEED" >&2; exit 2;; esac

repo="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
python_bin="${PYTHON_BIN:-$repo/.venv/bin/python}"
model_path="${MODEL_PATH:-/data/models/modelscope/LLM-Research/OLMoE-1B-7B-0924}"
scenario_dir="${SCENARIO_ROOT:-$repo/cache/phase10}/calibration-a/$DOMAIN/seed-$SEED"
artifact_dir="${ARTIFACT_ROOT:-$repo/artifacts/phase10/calibration-a-stats}/$DOMAIN/seed-$SEED"
grads="$scenario_dir/LayerGrads.pt"
merged="$scenario_dir/LayerRE_B1,2,3.pkl"
usage="$scenario_dir/ExpertUsage.pkl"
tokens="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["tokens_path"])' "$scenario_dir/scenario.json")"
IFS=',' read -r -a devices <<< "${CUDA_DEVICES:-4,5,6,7}"
ranges=("0 16" "16 32" "32 48" "48 64")
[[ ${#devices[@]} -eq 4 ]] || { echo "CUDA_DEVICES must contain four indices" >&2; exit 2; }

mkdir -p "$artifact_dir"
exec 9>"$artifact_dir/run.lock"
flock -n 9 || { echo "$DOMAIN seed-$SEED is already running" >&2; exit 3; }
cd "$repo"
for gpu in "${devices[@]}"; do
  used="$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  (( used <= 1024 )) || { echo "GPU $gpu is busy (${used} MiB)" >&2; exit 4; }
done

if [[ -s "$merged" && -s "$usage" ]]; then
  "$python_bin" scripts/phase6/validate_layer_re.py "$merged" --output "$artifact_dir/layer-re-summary.json"
  "$python_bin" scripts/phase10/validate_expert_usage.py "$usage" --samples 24 --output "$artifact_dir/usage-summary.json"
  exit 0
fi

started="$(date -u +%FT%TZ)"
if [[ ! -s "$grads" ]]; then
  /usr/bin/time -v env CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${devices[*]}")" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python_bin" -m gemq.compute_model_stats \
      --mode layer_grads --model "$model_path" --model_name allenai/OLMoE-1B-7B-0924 \
      --model_dtype bfloat16 --device_map auto --calib_dataset "$DOMAIN" \
      --scenario_tokens_path "$tokens" --use_fast --seed "$SEED" --nsamples 24 --seqlen 2048 \
      --layer_grads_path "$grads" > "$artifact_dir/layer-grads.log" 2>&1
fi
[[ -s "$grads" ]] || { echo "LayerGrads was not created" >&2; exit 5; }

pids=()
shards=()
for index in 0 1 2 3; do
  read -r expert_start expert_end <<< "${ranges[$index]}"
  shard="$scenario_dir/LayerRE_B1,2,3_experts-${expert_start}-${expert_end}.pkl"
  shards+=("$shard")
  [[ -s "$shard" ]] && continue
  usage_args=()
  [[ "$expert_start" -eq 0 ]] && usage_args=(--expert_usage_path "$usage")
  (
    /usr/bin/time -v env CUDA_VISIBLE_DEVICES="${devices[$index]}" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$python_bin" -m gemq.compute_model_stats \
        --mode layer_re --model "$model_path" --model_name allenai/OLMoE-1B-7B-0924 \
        --model_dtype bfloat16 --calib_dataset "$DOMAIN" --scenario_tokens_path "$tokens" \
        --use_fast --seed "$SEED" --nsamples 24 --seqlen 2048 --wbits 1,2,3 \
        --expert_start "$expert_start" --expert_end "$expert_end" \
        --layer_grads_path "$grads" --layer_re_path "$shard" --forward_batch_size 8 \
        "${usage_args[@]}"
  ) > "$artifact_dir/layer-re-${expert_start}-${expert_end}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
(( status == 0 )) || { echo "LayerRE shard failed; retaining resumable files" >&2; exit 6; }

"$python_bin" scripts/phase1/merge_layer_re.py "${shards[@]}" --output "$merged" \
  --layers 16 --experts 64 --bits 1,2,3 > "$artifact_dir/merge.log"
"$python_bin" scripts/phase6/validate_layer_re.py "$merged" --output "$artifact_dir/layer-re-summary.json"
"$python_bin" scripts/phase10/validate_expert_usage.py "$usage" --samples 24 --output "$artifact_dir/usage-summary.json"
rm -f -- "$grads" "${shards[@]}"
printf 'domain=%s\nseed=%s\nstarted_utc=%s\nfinished_utc=%s\nexit_code=0\n' \
  "$DOMAIN" "$SEED" "$started" "$(date -u +%FT%TZ)" > "$artifact_dir/status.txt"
