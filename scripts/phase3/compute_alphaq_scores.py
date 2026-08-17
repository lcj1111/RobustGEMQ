#!/usr/bin/env python3
"""Build a calibration-free AlphaQ-style expert cost tensor for OLMoE.

The score follows AlphaQ's public no-calibration objective:
    sensitivity = (median(alpha) / alpha) ** gamma
    cost(bit) = sensitivity * weight_variance * 2 ** (-2 * bit)

To keep the OLMoE pilot bounded, the heavy-tail exponent is estimated from a
deterministic 128x128 spectral sketch of each expert linear.  This is deliberately
reported as "AlphaQ-style", not as an exact reproduction of AlphaQ.
"""

from __future__ import annotations

import argparse
import hashlib
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spectral_alpha(weights: torch.Tensor, sketch_size: int, tail_fraction: float) -> torch.Tensor:
    """Compute Hill exponents for a batch of [out, in] matrices on CUDA."""
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
    stats_payload = json.dumps(linear_stats, sort_keys=True, separators=(",", ":"))

    index_path = args.model / "model.safetensors.index.json"
    result = {
        "schema_version": 1,
        "method": "AlphaQ-style deterministic spectral-sketch baseline",
        "model_id": MODEL_ID,
        "model_index_sha256": file_sha256(index_path),
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
        "linear_stats_sha256": hashlib.sha256(stats_payload.encode("utf-8")).hexdigest(),
        "candidate_bits": list(BITS),
        "expert_scores": expert_scores.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "linear_stats": len(linear_stats), **result["alpha_summary"]}))


if __name__ == "__main__":
    main()
