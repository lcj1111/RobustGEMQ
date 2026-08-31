#!/usr/bin/env python3
"""为 OLMoE 构建无需校准数据的 AlphaQ-style 专家代价张量。

评分沿用 AlphaQ 公开的无校准目标：
    sensitivity = (median(alpha) / alpha) ** gamma
    cost(bit) = sensitivity * weight_variance * 2 ** (-2 * bit)

为控制 OLMoE 试验开销，每个专家线性层使用确定性的 128×128 频谱草图估计
重尾指数。该方法仅作为 AlphaQ-style 对照，不宣称精确复现 AlphaQ。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from gemq.utils.model_utils import get_blocks, get_moe_block


MODEL_ID = "allenai/OLMoE-1B-7B-0924"
MODULES = ("gate_proj", "up_proj", "down_proj")
BITS = (1, 2, 3)
ALPHAQ_COMMIT = "3624976cfd800034156d4a39a3e5c04d23a02291"


def spectral_alpha(weights: torch.Tensor, sketch_size: int, tail_fraction: float) -> torch.Tensor:
    """在 CUDA 上批量计算 `[out, in]` 权重矩阵的 Hill 指数。"""
    rows = torch.linspace(0, weights.shape[1] - 1, min(sketch_size, weights.shape[1]), device=weights.device).long()
    cols = torch.linspace(0, weights.shape[2] - 1, min(sketch_size, weights.shape[2]), device=weights.device).long()
    sketches = weights.index_select(1, rows).index_select(2, cols).float()
    eigs = torch.linalg.svdvals(sketches).square()  # descending
    k = max(10, int(eigs.shape[1] * tail_fraction))
    k = min(k, eigs.shape[1] - 1)
    reference = eigs[:, k].clamp_min(1e-12)
    denominator = torch.log(eigs[:, :k].clamp_min(1e-12) / reference[:, None]).sum(dim=1).clamp_min(1e-12)
    return 1.0 + k / denominator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sketch-size", type=int, default=128)
    parser.add_argument("--tail-fraction", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=1.0)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="cuda",
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    linear_stats = []
    for layer_index, layer in enumerate(get_blocks(model, MODEL_ID)):
        experts = get_moe_block(layer, MODEL_ID).experts
        for module_name in MODULES:
            weights = torch.stack([getattr(expert, module_name).weight.detach() for expert in experts])
            alphas = spectral_alpha(weights, args.sketch_size, args.tail_fraction)
            variances = weights.float().var(dim=(1, 2), unbiased=False)
            for expert_index in range(len(experts)):
                linear_stats.append(
                    {
                        "layer": layer_index,
                        "expert": expert_index,
                        "module": module_name,
                        "alpha": float(alphas[expert_index].cpu()),
                        "variance": float(variances[expert_index].cpu()),
                    }
                )
            del weights, alphas, variances

    alpha_values = np.asarray([row["alpha"] for row in linear_stats], dtype=np.float64)
    if not np.isfinite(alpha_values).all() or (alpha_values <= 0).any():
        raise RuntimeError("Non-finite or non-positive spectral alpha encountered")
    alpha0 = float(np.median(alpha_values))
    expert_scores = np.zeros((16, 64), dtype=np.float64)
    for row in linear_stats:
        sensitivity = (alpha0 / row["alpha"]) ** args.gamma
        contribution = sensitivity * max(row["variance"], 1e-12)
        expert_scores[row["layer"], row["expert"]] += contribution
        row["sensitivity"] = sensitivity
        row["score_contribution"] = contribution
    result = {
        "schema_version": 1,
        "method": "AlphaQ-style deterministic spectral-sketch baseline",
        "model_id": MODEL_ID,
        "model_source": str(args.model),
        "upstream": {
            "repository": "https://github.com/Superone77/AlphaQ",
            "commit": ALPHAQ_COMMIT,
            "formula": "sum_module((median_alpha/alpha)^gamma * variance) * 2^(-2*bit)",
        },
        "approximation_boundary": (
            "Hill alpha uses a deterministic evenly-spaced spectral sketch; this is an "
            "AlphaQ-style comparator, not an exact reproduction of upstream FARMS."
        ),
        "sketch_size": args.sketch_size,
        "tail_fraction": args.tail_fraction,
        "gamma": args.gamma,
        "alpha_median": alpha0,
        "alpha_summary": {
            "min": float(alpha_values.min()),
            "median": alpha0,
            "max": float(alpha_values.max()),
        },
        "linear_stats_count": len(linear_stats),
        "candidate_bits": list(BITS),
        "expert_scores": expert_scores.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "linear_stats": len(linear_stats), **result["alpha_summary"]}))


if __name__ == "__main__":
    main()
