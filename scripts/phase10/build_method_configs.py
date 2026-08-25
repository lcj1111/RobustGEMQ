#!/usr/bin/env python3
"""用 calibration-A 构造并审计五个同预算 allocation。"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from gemq.allocate_bits import compute_total_bits
from gemq.allocation.control_allocations import (
    array_config,
    config_array,
    layer_balanced_config,
    usage_only_config,
)
from gemq.allocation.robust_solvers import RobustGEMQSolver


DOMAINS = ("general", "math", "code", "instruction")
SEEDS = (0, 1, 2)
BITS = (1, 2, 3)
METHODS = ("gemq-c4", "layer-balanced", "usage-only", "concat", "domain-mean")
MODEL_ID = "allenai/OLMoE-1B-7B-0924"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_layer_re(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        nested = pickle.load(handle)
    values = np.asarray(
        [[[nested[layer][expert][bit] for bit in BITS] for expert in range(64)] for layer in range(16)],
        dtype=np.float64,
    )
    if values.shape != (16, 64, 3) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"invalid LayerRE tensor: {path}")
    return values


def load_usage(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        nested = pickle.load(handle)
    values = np.asarray([nested[layer] for layer in range(16)], dtype=np.int64)
    if values.shape != (16, 64) or (values < 0).any():
        raise ValueError(f"invalid expert usage tensor: {path}")
    return values


def audit(config: dict, domain_tensors: dict[str, np.ndarray], budget: int) -> dict:
    values = config_array(config)
    if not np.isin(values, BITS).all() or int(values.sum()) != budget:
        raise ValueError("allocation violates candidate bits or exact budget")
    if any(not {2, 3}.issubset(set(values[layer])) for layer in range(16)):
        raise ValueError("allocation violates c2c3")
    losses = {
        domain: float(sum(tensor[layer, expert, BITS.index(int(values[layer, expert]))] for layer in range(16) for expert in range(64)))
        for domain, tensor in domain_tensors.items()
    }
    return {
        "verified": True,
        "used_bits": int(values.sum()),
        "actual_bpe": float(values.mean()),
        "bit_counts": {str(bit): int(np.count_nonzero(values == bit)) for bit in BITS},
        "c2c3_violations": 0,
        "domain_coefficient_loss": losses,
        "mean_domain_coefficient_loss": float(np.mean(list(losses.values()))),
        "worst_domain_coefficient_loss": float(np.max(list(losses.values()))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-root", type=Path, required=True)
    parser.add_argument("--gemq-config-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bpe", type=float, default=2.5)
    args = parser.parse_args()
    if args.bpe != 2.5:
        raise ValueError("independent study freezes the exact 2.5-bpe budget")

    normalized = {}
    per_token = {}
    usage_parts = []
    sources = {}
    for domain in DOMAINS:
        for seed in SEEDS:
            directory = args.scenario_root / "calibration-a" / domain / f"seed-{seed}"
            scenario_path = directory / "scenario.json"
            scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
            layer_re_path = directory / "LayerRE_B1,2,3.pkl"
            usage_path = directory / "ExpertUsage.pkl"
            raw = load_layer_re(layer_re_path)
            tokens = int(scenario["effective_tokens"])
            token_values = raw / tokens
            scale = float(np.median(token_values[:, :, 1]))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"invalid bit-2 normalization scale: {directory}")
            per_token[(domain, seed)] = token_values
            normalized[(domain, seed)] = token_values / scale
            usage_parts.append(load_usage(usage_path))
            sources[f"{domain}:seed-{seed}"] = {
                "token_sha256": scenario["token_sha256"],
                "selected_ids_sha256": scenario["selected_ids_sha256"],
                "layer_re_sha256": sha256(layer_re_path),
                "expert_usage_sha256": sha256(usage_path),
                "effective_tokens": tokens,
                "bit2_median_per_token": scale,
            }

    domain_tensors = {domain: np.mean([normalized[(domain, seed)] for seed in SEEDS], axis=0) for domain in DOMAINS}
    pooled = np.mean([per_token[(domain, seed)] for domain in DOMAINS for seed in SEEDS], axis=0)
    concat_tensor = pooled / float(np.median(pooled[:, :, 1]))
    usage = np.sum(usage_parts, axis=0)
    budget = compute_total_bits(MODEL_ID, args.bpe, BITS)

    gemq_path = args.gemq_config_root / "C4-Seed0_E2.5_B1,2,3_c2c3.pkl"
    with gemq_path.open("rb") as handle:
        gemq_config = pickle.load(handle)
    concat_solution = RobustGEMQSolver({"concat": concat_tensor}, objective="mean", bits=BITS, extra_constr="c2c3").solve(budget)
    normalized_solution = RobustGEMQSolver(domain_tensors, objective="mean", bits=BITS, extra_constr="c2c3").solve(budget)
    configs = {
        "gemq-c4": gemq_config,
        "layer-balanced": layer_balanced_config(),
        "usage-only": usage_only_config(usage),
        "concat": concat_solution.allocation,
        "domain-mean": normalized_solution.allocation,
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    arrays = {}
    method_manifest = {}
    for method in METHODS:
        config_path = args.output_root / f"{method}.pkl"
        with config_path.open("wb") as handle:
            pickle.dump(configs[method], handle)
        audit_payload = audit(configs[method], domain_tensors, budget)
        audit_path = args.output_root / f"{method}.audit.json"
        audit_path.write_text(json.dumps(audit_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        arrays[method] = config_array(configs[method])
        method_manifest[method] = {
            "config": str(config_path.resolve()),
            "config_sha256": sha256(config_path),
            "audit": str(audit_path.resolve()),
            "audit_sha256": sha256(audit_path),
        }
    hamming = {
        left: {right: float(np.mean(arrays[left] != arrays[right])) for right in METHODS}
        for left in METHODS
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": "robustgemq-independent-study-v1",
        "model": MODEL_ID,
        "bpe": args.bpe,
        "budget": budget,
        "methods": list(METHODS),
        "method_labels": {
            "gemq-c4": "GEMQ-C4",
            "layer-balanced": "Layer-Balanced",
            "usage-only": "Usage-Only",
            "concat": "Concat",
            "domain-mean": "Scenario-Normalized-Mean",
        },
        "definitions": {
            "layer-balanced": "32 alternating 2-bit and 32 alternating 3-bit experts per layer",
            "usage-only": "32 highest calibration-A Top-k counts use 3-bit per layer; remainder use 2-bit",
            "concat": "equal-token pooled per-token reconstruction coefficients",
            "domain-mean": "per-scenario bit-2 median normalization, seed mean within domain, then equal-domain mean",
        },
        "sources": sources,
        "usage_total_sha256": hashlib.sha256(usage.astype(np.int64).tobytes()).hexdigest(),
        "methods_manifest": method_manifest,
        "method_hamming_fraction": hamming,
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "methods": list(METHODS), "budget": budget}, sort_keys=True))


if __name__ == "__main__":
    main()
