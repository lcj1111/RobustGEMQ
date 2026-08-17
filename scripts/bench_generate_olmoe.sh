#!/usr/bin/env bash
set -euo pipefail

# These settings must match the quantization run unless MODEL_PATH is supplied.
model_name="${MODEL_NAME:-allenai/OLMoE-1B-7B-0924}"
bpe="${BPE:-2.0}"
finetune_routers="${FINETUNE_ROUTERS:-true}"
python_bin="${PYTHON_BIN:-python}"
cuda_device="${CUDA_DEVICE:-0}"

prompt="${PROMPT:-Although the experiment failed repeatedly, the researchers eventually}"
max_new_tokens="${MAX_NEW_TOKENS:-200}"
num_samples="${NUM_SAMPLES:-10}"
compile="${COMPILE:-true}"

rft_tag=""
if [[ "$finetune_routers" == "true" ]]; then
    rft_tag="_RFT"
fi
model_path="${MODEL_PATH:-results/real_quant_models/${model_name}/GEMQ/C4-Seed0-WT2_A4-G16-D4-E${bpe}${rft_tag}}"

if [[ ! -d "$model_path" ]]; then
    echo "Checkpoint not found: $model_path"
    echo "Check that bpe=${bpe} and finetune_routers=${finetune_routers} match scripts/quantize_olmoe.sh."
    exit 1
fi

compile_args=()
if [[ "$compile" == "true" ]]; then
    compile_args+=(--compile)
fi

CUDA_VISIBLE_DEVICES="$cuda_device" TORCH_LOGS="${TORCH_LOGS:-graph_breaks,recompiles}" \
  "$python_bin" -m gemq.benchmark_generate \
    --model_path "$model_path" \
    --model_name "$model_name" \
    --attn_impl eager \
    --prompt "$prompt" \
    --num_samples "$num_samples" \
    --max_new_tokens "$max_new_tokens" \
    --top_k 200 \
    "${compile_args[@]}"
