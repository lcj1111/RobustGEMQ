#!/usr/bin/env python3
"""Build frozen Phase 6 2.5-bpe allocations from the 4-domain x 3-seed Main Statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from gemq.allocate_bits import compute_total_bits
from gemq.allocation.robust_solvers import RobustGEMQSolver, empirical_cvar


DOMAINS = ("general", "math", "code", "instruction")
SEEDS = (0, 1, 2)
BITS = (1, 2, 3)
MODEL_ID = "allenai/OLMoE-1B-7B-0924"
METHODS = ("gemq-c4", "concat", "domain-mean", "alphaq-style")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tensor(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        nested = pickle.load(handle)
    tensor = np.asarray(
        [
            [[nested[layer][expert][bit] for bit in BITS] for expert in range(64)]
            for layer in range(16)
        ],
        dtype=np.float64,
    )
    if tensor.shape != (16, 64, 3) or not np.isfinite(tensor).all() or (tensor < 0).any():
        raise ValueError(f"Invalid coefficient tensor: {path}")
    return tensor


def normalize(tensor: np.ndarray, effective_tokens: int) -> tuple[np.ndarray, np.ndarray, float]:
    per_token = tensor / float(effective_tokens)
    median_bit2 = float(np.median(per_token[:, :, 1]))
    if not np.isfinite(median_bit2) or median_bit2 <= 0:
        raise ValueError(f"Invalid median bit-2 coefficient: {median_bit2}")
    return per_token / median_bit2, per_token, median_bit2


def config_array(config: dict) -> np.ndarray:
    return np.asarray(
        [[int(config[layer][expert]) for expert in range(64)] for layer in range(16)],
        dtype=np.int8,
    )


def audit_config(config: dict, domain_tensors: dict[str, np.ndarray], budget: int) -> dict:
    array = config_array(config)
    used_bits = int(array.sum())
    layers_ok = set(config) == set(range(16)) and all(set(config[layer]) == set(range(64)) for layer in config)
    invalid_bits = int(np.count_nonzero(~np.isin(array, BITS)))
    c2c3_violations = int(sum(not {2, 3}.issubset(set(array[layer])) for layer in range(16)))
    if not layers_ok or invalid_bits or c2c3_violations or used_bits > budget:
        raise RuntimeError(
            "Allocation audit failed: "
            f"layers_ok={layers_ok}, invalid_bits={invalid_bits}, "
            f"c2c3_violations={c2c3_violations}, used_bits={used_bits}, budget={budget}"
        )
    losses = {
        domain: float(
            sum(
                tensor[layer, expert, BITS.index(int(array[layer, expert]))]
                for layer in range(16)
                for expert in range(64)
            )
        )
        for domain, tensor in domain_tensors.items()
    }
    values = np.asarray([losses[domain] for domain in DOMAINS], dtype=np.float64)
    counts = {str(bit): int(np.count_nonzero(array == bit)) for bit in BITS}
    return {
        "feasible": True,
        "candidate_bits": list(BITS),
        "constraint": "c2c3",
        "used_bits": used_bits,
        "actual_bpe": used_bits / (16 * 64),
        "budget": budget,
        "budget_slack": budget - used_bits,
        "one_hot_valid": layers_ok,
        "invalid_bit_count": invalid_bits,
        "c2c3_violations": c2c3_violations,
        "bit_counts": counts,
        "domain_losses": losses,
        "domain_mean": float(values.mean()),
        "domain_worst": float(values.max()),
        "domain_cvar_0.5": empirical_cvar(values, 0.5),
    }


def save_method(output_dir: Path, name: str, config: dict, audit: dict) -> dict:
    config_path = output_dir / f"{name}.pkl"
    audit_path = output_dir / f"{name}.audit.json"
    with config_path.open("wb") as handle:
        pickle.dump(config, handle)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "audit": str(audit_path),
        "used_bits": audit["used_bits"],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-root", type=Path, required=True)
    parser.add_argument("--alphaq-scores", type=Path, required=True)
    parser.add_argument("--gemq-c4-config-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bpe", type=float, default=2.5, choices=(2.5, 2.0))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normalized: dict[tuple[str, int], np.ndarray] = {}
    per_token: dict[tuple[str, int], np.ndarray] = {}
    source_manifest = {}
    for domain in DOMAINS:
        for seed in SEEDS:
            directory = args.scenario_root / domain / f"seed-{seed}"
            scenario = json.loads((directory / "scenario.json").read_text(encoding="utf-8"))
            source = directory / "LayerRE_B1,2,3.pkl"
            raw = load_tensor(source)
            norm, token_tensor, median = normalize(raw, int(scenario["effective_tokens"]))
            normalized[(domain, seed)] = norm
            per_token[(domain, seed)] = token_tensor
            source_manifest[f"{domain}:seed-{seed}"] = {
                "layer_re_sha256": sha256(source),
                "token_sha256": scenario["token_sha256"],
                "effective_tokens": scenario["effective_tokens"],
                "median_bit2_per_token": median,
            }

    domain_tensors = {
        domain: np.mean([normalized[(domain, seed)] for seed in SEEDS], axis=0)
        for domain in DOMAINS
    }
    pooled = np.mean([per_token[(domain, seed)] for domain in DOMAINS for seed in SEEDS], axis=0)
    concat_tensor = pooled / float(np.median(pooled[:, :, 1]))
    alphaq_payload = json.loads(args.alphaq_scores.read_text(encoding="utf-8"))
    if "expert_scores" in alphaq_payload:
        scores = np.asarray(alphaq_payload["expert_scores"], dtype=np.float64)
        alphaq_tensor = np.stack([scores * (2.0 ** (-2 * bit)) for bit in BITS], axis=2)
    else:
        alphaq_tensor = np.asarray(alphaq_payload["coefficients"], dtype=np.float64)
    if alphaq_tensor.shape != (16, 64, 3) or not np.isfinite(alphaq_tensor).all():
        raise ValueError("Invalid AlphaQ-style coefficient tensor")

    budget = compute_total_bits(MODEL_ID, args.bpe, BITS)
    output_dir = args.output_root / f"bpe-{args.bpe:.1f}"
    output_dir.mkdir(parents=True, exist_ok=True)
    c4_source = args.gemq_c4_config_root / f"C4-Seed0_E{args.bpe:.1f}_B1,2,3_c2c3.pkl"
    with c4_source.open("rb") as handle:
        c4_config = pickle.load(handle)
    solve_specs = {
        "concat": ({"concat": concat_tensor}, "mean"),
        "domain-mean": (domain_tensors, "mean"),
        "alphaq-style": ({"alphaq-style": alphaq_tensor}, "mean"),
    }
    methods: dict[str, tuple[dict, dict]] = {
        "gemq-c4": (c4_config, {"source_config_sha256": sha256(c4_source)})
    }
    for name, (scenarios, objective) in solve_specs.items():
        solved = RobustGEMQSolver(
            scenarios, objective=objective, bits=BITS, alpha=0.5, extra_constr="c2c3"
        ).solve(budget)
        methods[name] = (solved.allocation, {"solver": solved.audit})

    method_manifest = {}
    arrays = {}
    for name in METHODS:
        config, provenance = methods[name]
        audit = {**provenance, **audit_config(config, domain_tensors, budget)}
        method_manifest[name] = save_method(output_dir, name, config, audit)
        arrays[name] = config_array(config)
    hamming = {
        left: {right: float(np.mean(arrays[left] != arrays[right])) for right in METHODS}
        for left in METHODS
    }
    manifest = {
        "schema_version": 1,
        "phase": 6,
        "status": "frozen-main-statistics",
        "methods": list(METHODS),
        "bpe": args.bpe,
        "risk_hierarchy": "domain is the risk scenario; three seeds are averaged within each domain",
        "normalization": "per-token then each seed tensor divided by its own median bit-2 coefficient",
        "concat_definition": "equal-token pooling of all 12 per-token tensors, then one pooled bit-2 median scale",
        "source_scenarios": source_manifest,
        "alphaq_scores_sha256": sha256(args.alphaq_scores),
        "budget": budget,
        "method_hamming_fraction": hamming,
        "methods_manifest": method_manifest,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "methods": list(METHODS), "budget": budget}))


if __name__ == "__main__":
    main()
