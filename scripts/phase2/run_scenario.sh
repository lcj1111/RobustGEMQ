#!/usr/bin/env bash
set -euo pipefail

: "${DOMAIN:?set DOMAIN to general, math, code, or instruction}"
: "${SEED:?set SEED}"
: "${CUDA_DEVICE:?set one physical CUDA device}"

repo="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
python_bin="${PYTHON_BIN:-$repo/.venv/bin/python}"
model_path="${MODEL_PATH:-/data/models/modelscope/LLM-Research/OLMoE-1B-7B-0924}"
model_id="allenai/OLMoE-1B-7B-0924"
data_root="${DOMAIN_DATA_ROOT:-/data/models/datasets/robustgemq-phase2}"
registry="${DOMAIN_REGISTRY:-$repo/configs/domains/phase2_domains.json}"
profile="${PROFILE:-smoke}"

case "$profile" in
  smoke)
    nsamples="${NSAMPLES:-8}"
    seqlen="${SEQLEN:-256}"
    ;;
  pilot)
    nsamples="${NSAMPLES:-32}"
    seqlen="${SEQLEN:-512}"
    ;;
  *)
    echo "Unsupported PROFILE=$profile; expected smoke or pilot" >&2
    exit 2
    ;;
esac

scenario_dir="$repo/cache/phase2/$profile/$DOMAIN/seed-$SEED"
artifact_dir="$repo/artifacts/phase2/$profile/$DOMAIN/seed-$SEED"
grads_path="$scenario_dir/LayerGrads.pt"
layer_re_path="$scenario_dir/LayerRE_B1,2,3.pkl"
mkdir -p "$scenario_dir" "$artifact_dir"
cd "$repo"

"$python_bin" scripts/phase2/materialize_scenario.py \
  --registry "$registry" \
  --data-root "$data_root" \
  --domain "$DOMAIN" \
  --model "$model_path" \
  --model-id "$model_id" \
  --seed "$SEED" \
  --nsamples "$nsamples" \
  --seqlen "$seqlen" \
  --output-dir "$scenario_dir" \
  > "$artifact_dir/materialize.json"

tokens_path=$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["tokens_path"])' "$scenario_dir/scenario.json")
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits > "$artifact_dir/gpu-before.txt"
started=$(date -u +%FT%TZ)

set +e
{
  if [[ ! -s "$grads_path" ]]; then
    /usr/bin/time -v env CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$python_bin" -m gemq.compute_model_stats \
        --mode layer_grads \
        --model "$model_path" \
        --model_name "$model_id" \
        --model_dtype bfloat16 \
        --device_map auto \
        --calib_dataset "$DOMAIN" \
        --scenario_tokens_path "$tokens_path" \
        --use_fast \
        --seed "$SEED" \
        --nsamples "$nsamples" \
        --seqlen "$seqlen" \
        --layer_grads_path "$grads_path"
  else
    echo "Reusing LayerGrads: $grads_path"
  fi

  if [[ ! -s "$layer_re_path" ]]; then
    /usr/bin/time -v env CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$python_bin" -m gemq.compute_model_stats \
        --mode layer_re \
        --model "$model_path" \
        --model_name "$model_id" \
        --model_dtype bfloat16 \
        --calib_dataset "$DOMAIN" \
        --scenario_tokens_path "$tokens_path" \
        --use_fast \
        --seed "$SEED" \
        --nsamples "$nsamples" \
        --seqlen "$seqlen" \
        --wbits 1,2,3 \
        --layer_grads_path "$grads_path" \
        --layer_re_path "$layer_re_path" \
        --forward_batch_size "${FORWARD_BATCH_SIZE:-8}"
  else
    echo "Reusing LayerRE: $layer_re_path"
  fi
} 2>&1 | tee "$artifact_dir/run.log"
status=${PIPESTATUS[0]}
set -e

finished=$(date -u +%FT%TZ)
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits > "$artifact_dir/gpu-after.txt"
printf 'profile=%s\ndomain=%s\nseed=%s\nstarted_utc=%s\nfinished_utc=%s\nexit_code=%s\ngpu=%s\nscenario=%s\nlayer_grads=%s\nlayer_re=%s\n' \
  "$profile" "$DOMAIN" "$SEED" "$started" "$finished" "$status" "$CUDA_DEVICE" \
  "$scenario_dir/scenario.json" "$grads_path" "$layer_re_path" > "$artifact_dir/status.txt"
exit "$status"
