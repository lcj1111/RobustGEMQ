#!/bin/bash
set -euo pipefail

# Settings
model_name="allenai/OLMoE-1B-7B-0924"
bits_per_expert="${BPE:-2.0}"            # target average bits-per-expert
wbits="${WBITS:-1,2,3}"                  # candidate bit-widths
ilp_solver="${ILP_SOLVER:-gemq}"         # bit allocation method
ilp_backend="${ILP_BACKEND:-highs}"      # highs (SciPy) or gurobi
extra_constr="${EXTRA_CONSTR:-c2c3}"     # extra constraints for bit allocation
python_bin="${PYTHON_BIN:-python}"
# path to the weighted layer reconstruction errors (i.e., ILP coefficients)
layer_re_path="${LAYER_RE_PATH:-cache/${model_name}/LayerRE_c4-N128-L2048-Seed0_B1,2,3_fast.pkl}"

"$python_bin" -m gemq.allocate_bits \
    --model_name ${model_name} \
    --layer_re_path ${layer_re_path} \
    --bit_budget ${bits_per_expert} \
    --bit_candidates ${wbits} \
    --ilp_solver ${ilp_solver} \
    --ilp_backend ${ilp_backend} \
    --extra_constr ${extra_constr}
