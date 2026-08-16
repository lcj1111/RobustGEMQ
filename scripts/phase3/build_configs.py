#!/usr/bin/env python3
"""Build and audit the frozen Phase 3 OLMoE bit-allocation matrix."""

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
BITS = (1, 2, 3)
MODEL_ID = "allenai/OLMoE-1B-7B-0924"


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
        [[[nested[layer][expert][bit] for bit in BITS] for expert in range(64)] for layer in range(16)],
        dtype=np.float64,
    )
    if not np.isfinite(tensor).all() or (tensor < 0).any():
        raise ValueError(f"Invalid coefficient tensor: {path}")
    return tensor


def normalize(tensor: np.ndarray, effective_tokens: int) -> tuple[np.ndarray, float]:
    per_token = tensor / float(effective_tokens)
    scale = float(np.median(per_token[:, :, 1]))
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError(f"Invalid median bit-2 scale: {scale}")
    return per_token / scale, scale


def common_audit(config: dict, domain_tensors: dict[str, np.ndarray], budget: float) -> dict:
    layers = sorted(config)
    one_hot_ok = layers == list(range(16)) and all(sorted(config[layer]) == list(range(64)) for layer in layers)
    used = sum(int(bit) for experts in config.values() for bit in experts.values())
    invalid_bits = sum(bit not in BITS for experts in config.values() for bit in experts.values())
    c2c3_violations = sum(
        not all(bit in set(config[layer].values()) for bit in (2, 3)) for layer in layers
    )
    losses = {
        domain: float(
            sum(
                tensor[layer, expert, BITS.index(config[layer][expert])]
                for layer in range(16)
                for expert in range(64)
            )
        )
        for domain, tensor in domain_tensors.items()
    }
    values = np.asarray([losses[domain] for domain in DOMAINS])
    feasible = one_hot_ok and invalid_bits == 0 and c2c3_violations == 0 and used <= budget
    if not feasible:
        raise RuntimeError(
            f"Allocation audit failed: one_hot={one_hot_ok}, invalid_bits={invalid_bits}, "
            f"c2c3={c2c3_violations}, used={used}, budget={budget}"
        )
    return {
        "feasible": True,
        "used_bits": used,
        "actual_bpe": used / (16 * 64),
        "budget": budget,
        "budget_slack": budget - used,
        "candidate_bits": list(BITS),
        "one_hot_valid": one_hot_ok,
        "invalid_bit_count": invalid_bits,
        "c2c3_violations": c2c3_violations,
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
        "used_bits": audit["common"]["used_bits"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-root", type=Path, required=True)
    parser.add_argument("--alphaq-scores", type=Path, required=True)
    parser.add_argument("--gemq-c4-config-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    normalized = {}
    per_token = {}
    source_manifest = {}
    for domain in DOMAINS:
        for seed in (0, 1):
            directory = args.scenario_root / domain / f"seed-{seed}"
            scenario = json.loads((directory / "scenario.json").read_text(encoding="utf-8"))
            source = directory / "LayerRE_B1,2,3.pkl"
            raw = load_tensor(source)
            normalized[(domain, seed)], median = normalize(raw, scenario["effective_tokens"])
            per_token[(domain, seed)] = raw / float(scenario["effective_tokens"])
            source_manifest[f"{domain}:seed-{seed}"] = {
                "layer_re_sha256": sha256(source),
                "token_sha256": scenario["token_sha256"],
                "effective_tokens": scenario["effective_tokens"],
                "median_bit2_per_token": median,
            }
    domain_tensors = {
        domain: np.mean([normalized[(domain, 0)], normalized[(domain, 1)]], axis=0)
        for domain in DOMAINS
    }
    pooled = np.mean([per_token[key] for key in sorted(per_token)], axis=0)
    concat_tensor = pooled / float(np.median(pooled[:, :, 1]))
    alphaq_payload = json.loads(args.alphaq_scores.read_text(encoding="utf-8"))
    if "expert_scores" in alphaq_payload:
        scores = np.asarray(alphaq_payload["expert_scores"], dtype=np.float64)
        alphaq_tensor = np.stack([scores * (2.0 ** (-2 * bit)) for bit in BITS], axis=2)
    else:  # Backward-compatible with the initial Phase 3 artifact schema.
        alphaq_tensor = np.asarray(alphaq_payload["coefficients"], dtype=np.float64)
    if alphaq_tensor.shape != (16, 64, 3):
        raise ValueError(f"Unexpected AlphaQ tensor shape {alphaq_tensor.shape}")

    manifest = {
        "schema_version": 1,
        "phase": 3,
        "risk_hierarchy": "domain is the scenario; seed is averaged within domain",
        "normalization": "per-token then each seed tensor divided by its own median bit-2 coefficient",
        "concat_definition": (
            "equal-token pool of all eight per-token coefficient tensors, followed by one pooled bit-2 median scale"
        ),
        "source_scenarios": source_manifest,
        "alphaq_scores_sha256": sha256(args.alphaq_scores),
        "budgets": {},
    }
    for bpe in (2.5, 2.0):
        budget = compute_total_bits(MODEL_ID, bpe, BITS)
        output_dir = args.output_root / f"bpe-{bpe:.1f}"
        output_dir.mkdir(parents=True, exist_ok=True)
        methods = {}

        c4_source = args.gemq_c4_config_root / f"C4-Seed0_E{bpe:.1f}_B1,2,3_c2c3.pkl"
        with c4_source.open("rb") as handle:
            c4_config = pickle.load(handle)
        methods["gemq-c4"] = (c4_config, {"source_config_sha256": sha256(c4_source)})

        solve_specs = {
            "concat": ({"concat": concat_tensor}, "mean", 0.5),
            "domain-mean": (domain_tensors, "mean", 0.5),
            "domain-worst": (domain_tensors, "worst", 0.5),
            "domain-cvar-0.5": (domain_tensors, "cvar", 0.5),
            "alphaq-style": ({"alphaq-style": alphaq_tensor}, "mean", 0.5),
        }
        for name, (scenarios, objective, alpha) in solve_specs.items():
            result = RobustGEMQSolver(
                scenarios,
                objective=objective,
                bits=BITS,
                alpha=alpha,
                extra_constr="c2c3",
            ).solve(budget)
            methods[name] = (result.allocation, {"solver": result.audit})

        budget_manifest = {"target_bpe": bpe, "budget": budget, "methods": {}}
        for name, (config, method_audit) in methods.items():
            audit = {**method_audit, "common": common_audit(config, domain_tensors, budget)}
            budget_manifest["methods"][name] = save_method(output_dir, name, config, audit)
        manifest["budgets"][f"{bpe:.1f}"] = budget_manifest

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(manifest_path), "budgets": list(manifest["budgets"])}))


if __name__ == "__main__":
    main()
