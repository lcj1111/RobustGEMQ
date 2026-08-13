#!/bin/bash
set -euo pipefail

# ===============================
#  Model settings
# ===============================
model_name="allenai/OLMoE-1B-7B-0924"
model="${MODEL_PATH:-allenai/OLMoE-1B-7B-0924}"
model_str="${MODEL_TAG:-}"  # optional suffix used in output filenames
model_dtype="${MODEL_DTYPE:-bfloat16}"  # OLMoE produces NaNs with float16
cuda_device="${CUDA_DEVICE:-0}"
python_bin="${PYTHON_BIN:-python}"

# ===============================
#  Dataset settings
# ===============================
dataset="${CALIB_DATASET:-c4}"  # c4 | math | math+c4
nsamples="${NSAMPLES:-128}"
seqlen="${SEQLEN:-2048}"
seed="${SEED:-0}"
forward_batch_size="${FORWARD_BATCH_SIZE:-8}"
wbits="${WBITS:-1,2,3}"
device_map="${DEVICE_MAP:-auto}"
expert_start="${EXPERT_START:-0}"
expert_end="${EXPERT_END:--1}"

if [[ -n "${DATA_ROOT:-}" ]]; then
    export GEMQ_C4_TRAIN_FILE="${GEMQ_C4_TRAIN_FILE:-${DATA_ROOT}/c4/en/c4-train.00000-of-01024.json}"
    export GEMQ_C4_VALIDATION_FILE="${GEMQ_C4_VALIDATION_FILE:-${DATA_ROOT}/c4/en/c4-validation.00000-of-00008.json.gz}"
    export GEMQ_WIKITEXT_DIR="${GEMQ_WIKITEXT_DIR:-${DATA_ROOT}/wikitext2}"
fi


# =============================================================================
#  Step1: Compute statistics - Layer output gradients
# =============================================================================
layer_grads_path="cache/${model_name}/LayerGrads_${dataset}-N${nsamples}-L${seqlen}-Seed${seed}${model_str}.pt"
if [[ "${RUN_LAYER_GRADS:-true}" == "true" ]]; then
CUDA_VISIBLE_DEVICES="$cuda_device" "$python_bin" -m gemq.compute_model_stats \
    --mode "layer_grads" \
    --model ${model} \
    --model_name ${model_name} \
    --model_dtype ${model_dtype} \
    --device_map ${device_map} \
    --calib_dataset ${dataset} \
    --use_fast \
    --seed ${seed} \
    --nsamples ${nsamples} \
    --seqlen ${seqlen} \
    --layer_grads_path ${layer_grads_path}
fi


# =============================================================================
#  Step2: Compute statistics - Weighted layer reconstruction errors
# =============================================================================
layer_re_path="${LAYER_RE_PATH:-cache/${model_name}/LayerRE_${dataset}-N${nsamples}-L${seqlen}-Seed${seed}_B${wbits}${model_str}_fast.pkl}"
if [[ "${RUN_LAYER_RE:-true}" == "true" ]]; then
CUDA_VISIBLE_DEVICES="$cuda_device" "$python_bin" -m gemq.compute_model_stats \
    --mode "layer_re" \
    --model ${model} \
    --model_name ${model_name} \
    --model_dtype ${model_dtype} \
    --calib_dataset ${dataset} \
    --use_fast \
    --seed ${seed} \
    --nsamples ${nsamples} \
    --seqlen ${seqlen} \
    --wbits ${wbits} \
    --expert_start ${expert_start} \
    --expert_end ${expert_end} \
    --layer_grads_path ${layer_grads_path} \
    --layer_re_path ${layer_re_path} \
    --forward_batch_size ${forward_batch_size}
fi
