#!/bin/bash
set -euo pipefail

# ===============================
#  Model settings
# ===============================
model_name="allenai/OLMoE-1B-7B-0924"
model="${MODEL_PATH:-allenai/OLMoE-1B-7B-0924}"
cuda_device="${CUDA_DEVICE:-0}"
python_bin="${PYTHON_BIN:-python}"

# ===============================
#  Dataset settings
# ===============================
calib_dataset="${CALIB_DATASET:-wikitext2}"
nsamples="${NSAMPLES:-128}"
seqlen="${SEQLEN:-2048}"

if [[ -n "${DATA_ROOT:-}" ]]; then
    export GEMQ_C4_TRAIN_FILE="${GEMQ_C4_TRAIN_FILE:-${DATA_ROOT}/c4/en/c4-train.00000-of-01024.json}"
    export GEMQ_C4_VALIDATION_FILE="${GEMQ_C4_VALIDATION_FILE:-${DATA_ROOT}/c4/en/c4-validation.00000-of-00008.json.gz}"
    export GEMQ_WIKITEXT_DIR="${GEMQ_WIKITEXT_DIR:-${DATA_ROOT}/wikitext2}"
fi

# ===============================
#  Quantization settings
# ===============================
quantizer="gptq"
bpe="${BPE:-2.0}"                    # bits per expert
mixed_prec="${MIXED_PREC:-true}"     # expert-level mixed precision
bit_cfg="${BIT_CFG:-configs/${model_name}/GEMQ/C4-Seed0_E${bpe}_B1,2,3_c2c3.pkl}"
reproduce_mcmoe="${REPRODUCE_MCMOE:-true}"

# ===============================
#  Router fine-tuning
# ===============================
finetune_routers="${FINETUNE_ROUTERS:-true}"
rft_epochs="${RFT_EPOCHS:-1}"
rft_lr="${RFT_LR:-1e-4}"

# ===============================
#  Evaluation settings
# ===============================
eval_downstream="${EVAL_DOWNSTREAM:-false}"
downstream_tasks="piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande,mathqa,mmlu"

# ===============================
#  I/O settings
# ===============================
real_quant="${REAL_QUANT:-true}"
save_model="${SAVE_MODEL:-true}"



# ===============================
#  AUTO argument construction
# ===============================
# NOTE: --use_fast is required for OLMoE tokenizer
# NOTE: float16 causes NaN in OLMoE
model_args=(--model "$model" --model_name "$model_name" --use_fast --model_dtype "bfloat16")

data_args=(--calib_dataset "$calib_dataset" --nsamples "$nsamples" --seqlen "$seqlen")

bpe_int=$(printf "%.0f" "$bpe")
quant_args=(--quantizer "$quantizer" --expert_wbits "$bpe_int" --groupsize 128 --mse)
if [[ "${reproduce_mcmoe}" == "true" ]]; then
    quant_args+=(--reproduce_mcmoe)
fi
if [[ "${mixed_prec}" == "true" ]]; then
    qtype="$(basename "$(dirname "$bit_cfg")")"
    quant_args+=(--mixed --bit_cfg "$bit_cfg")
else
    qtype="Uniform"
fi

rft_tag=""
if [[ "${finetune_routers}" == "true" ]]; then
    rft_tag="_RFT"
    quant_args+=(--finetune_routers --rft_epochs "$rft_epochs" --rft_lr "$rft_lr")
fi

eval_args=()
if [[ "${eval_downstream}" == "true" ]]; then
    eval_args=(--eval_downstream --downstream_tasks "$downstream_tasks")
fi

fname="${bit_cfg##*/}"
alloc_prefix="${fname%%_*}"
prefix="${alloc_prefix}-WT2"
if [[ "${save_model}" == "true" ]]; then
    if [[ "${real_quant}" == "true" ]]; then
        save_path="results/real_quant_models/${model_name}/${qtype}/${prefix}_A4-G16-D4-E${bpe}${rft_tag}"
        io_args=(--real_quant --save_path "$save_path")
    else
        save_path="results/fake_quant_models/${model_name}/${qtype}/${prefix}_A4-G16-D4-E${bpe}${rft_tag}"
        io_args=(--save_path "$save_path")
    fi
else
    save_path="None"
    io_args=()
fi


# ===============================
#  Run
# ===============================
echo "=============================================="
echo ">>> Quantization Job Summary"
echo "----------------------------------------------"
echo " Model:            ${model_name}"
echo " Dataset:          ${calib_dataset} (nsamples=${nsamples}, seqlen=${seqlen})"
echo "----------------------------------------------"
echo " Quantizer:        ${quantizer}"
echo " Expert bits:      ${bpe} (mixed: ${mixed_prec})"
echo " Bit config:       ${bit_cfg}"
echo " Finetune routers: ${finetune_routers} (epochs=${rft_epochs}, lr=${rft_lr})"
echo " Save path:        ${save_path}"
echo "----------------------------------------------"
echo ">>> Running quantization ..."
echo "=============================================="

CUDA_VISIBLE_DEVICES="$cuda_device" "$python_bin" -m gemq.quantize \
    "${model_args[@]}" \
    "${data_args[@]}" \
    "${quant_args[@]}" \
    "${eval_args[@]}" \
    "${io_args[@]}"
