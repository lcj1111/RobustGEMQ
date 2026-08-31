#!/usr/bin/env python3
"""Phase 2 coefficient stability and single-domain transfer diagnostics."""

from __future__ import annotations

import argparse
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from gemq.allocate_bits import compute_total_bits
from gemq.allocation.ilp_solvers import GEMQSolver


DOMAINS = ("general", "math", "code", "instruction")
BITS = (1, 2, 3)


def load_tensor(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        values = pickle.load(handle)
    tensor = np.empty((16, 64, 3), dtype=np.float64)
    for layer in range(16):
        for expert in range(64):
            for bit_index, bit in enumerate(BITS):
                tensor[layer, expert, bit_index] = float(values[layer][expert][bit])
    if not np.isfinite(tensor).all() or (tensor < 0).any():
        raise ValueError(f"Invalid coefficient tensor: {path}")
    return tensor


def normalize(tensor: np.ndarray, effective_tokens: int, mode: str) -> np.ndarray:
    per_token = tensor / float(effective_tokens)
    if mode == "per_token":
        return per_token
    if mode == "median_bit2":
        denominator = float(np.median(per_token[:, :, 1]))
        if denominator <= 0 or not np.isfinite(denominator):
            raise ValueError(f"Invalid bit-2 median normalization scale: {denominator}")
        return per_token / denominator
    raise ValueError(mode)


def sensitivity(tensor: np.ndarray) -> np.ndarray:
    return (tensor[:, :, 0] - tensor[:, :, 2]).reshape(-1)


def top_overlap(left: np.ndarray, right: np.ndarray, fraction: float = 0.10) -> float:
    count = max(1, int(round(left.size * fraction)))
    left_top = set(np.argpartition(left, -count)[-count:])
    right_top = set(np.argpartition(right, -count)[-count:])
    return len(left_top.intersection(right_top)) / count


def pair_metrics(left: np.ndarray, right: np.ndarray) -> dict:
    left_sensitivity = sensitivity(left)
    right_sensitivity = sensitivity(right)
    rho = float(spearmanr(left_sensitivity, right_sensitivity).statistic)
    return {"spearman": rho, "top10_overlap": top_overlap(left_sensitivity, right_sensitivity)}


def to_pickle_dict(tensor: np.ndarray) -> dict:
    return {
        layer: {
            expert: {bit: float(tensor[layer, expert, bit_index]) for bit_index, bit in enumerate(BITS)}
            for expert in range(tensor.shape[1])
        }
        for layer in range(tensor.shape[0])
    }


def solve_tensor(tensor: np.ndarray, bpe: float, directory: Path, label: str) -> dict:
    path = directory / f"{label}.pkl"
    with path.open("wb") as handle:
        pickle.dump(to_pickle_dict(tensor), handle)
    solver = GEMQSolver(path, x_space=BITS, extra_constr="c2c3", backend="highs")
    budget = compute_total_bits("allenai/OLMoE-1B-7B-0924", bpe, BITS)
    return solver.solve_all(total_bits=budget)


def config_array(config: dict) -> np.ndarray:
    return np.asarray([[config[layer][expert] for expert in range(64)] for layer in range(16)])


def evaluate(tensor: np.ndarray, config: dict) -> float:
    return float(
        sum(
            tensor[layer, expert, BITS.index(config[layer][expert])]
            for layer in range(16)
            for expert in range(64)
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configs-dir", type=Path)
    args = parser.parse_args()

    scenarios = {}
    manifests = {}
    for domain in DOMAINS:
        for seed in (0, 1):
            path = args.root / domain / f"seed-{seed}"
            manifest = json.loads((path / "scenario.json").read_text(encoding="utf-8"))
            raw = load_tensor(path / "LayerRE_B1,2,3.pkl")
            scenarios[(domain, seed)] = normalize(raw, manifest["effective_tokens"], "median_bit2")
            manifests[(domain, seed)] = manifest

    within = []
    for domain in DOMAINS:
        metrics = pair_metrics(scenarios[(domain, 0)], scenarios[(domain, 1)])
        within.append({"domain": domain, **metrics})

    domain_means = {
        domain: np.mean([scenarios[(domain, 0)], scenarios[(domain, 1)]], axis=0)
        for domain in DOMAINS
    }
    cross = []
    for left_index, left in enumerate(DOMAINS):
        for right in DOMAINS[left_index + 1 :]:
            cross.append({"left": left, "right": right, **pair_metrics(domain_means[left], domain_means[right])})

    within_rho = float(np.median([item["spearman"] for item in within]))
    cross_rho = float(np.median([item["spearman"] for item in cross]))
    within_overlap = float(np.median([item["top10_overlap"] for item in within]))
    cross_overlap = float(np.median([item["top10_overlap"] for item in cross]))
    h1_pass = (within_rho - cross_rho >= 0.05) or (within_overlap - cross_overlap >= 0.10)

    transfer = {}
    seed_stability = {}
    configs_dir = args.configs_dir or (args.output.parent / "configs")
    configs_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="robustgemq-phase2-") as temporary:
        temporary_path = Path(temporary)
        for bpe in (2.5, 2.0):
            configs = {
                domain: solve_tensor(domain_means[domain], bpe, temporary_path, f"{domain}-{bpe}")
                for domain in DOMAINS
            }
            bpe_dir = configs_dir / f"bpe-{bpe:.1f}"
            bpe_dir.mkdir(parents=True, exist_ok=True)
            for domain, config in configs.items():
                with (bpe_dir / f"{domain}.pkl").open("wb") as handle:
                    pickle.dump(config, handle)
                with (bpe_dir / f"{domain}-coefficients.pkl").open("wb") as handle:
                    pickle.dump(to_pickle_dict(domain_means[domain]), handle)
            target_optimum = {
                domain: evaluate(domain_means[domain], configs[domain]) for domain in DOMAINS
            }
            rows = []
            for source in DOMAINS:
                row = {}
                for target in DOMAINS:
                    cost = evaluate(domain_means[target], configs[source])
                    optimum = target_optimum[target]
                    row[target] = {
                        "cost": cost,
                        "relative_regret": (cost - optimum) / max(abs(optimum), 1e-12),
                    }
                rows.append({"source": source, "targets": row})
            transfer[str(bpe)] = rows

            per_domain_seed = {}
            for domain in DOMAINS:
                config0 = solve_tensor(scenarios[(domain, 0)], bpe, temporary_path, f"{domain}-0-{bpe}")
                config1 = solve_tensor(scenarios[(domain, 1)], bpe, temporary_path, f"{domain}-1-{bpe}")
                left, right = config_array(config0), config_array(config1)
                per_domain_seed[domain] = {
                    "hamming_fraction": float(np.mean(left != right)),
                    "exact_match": bool(np.array_equal(left, right)),
                }
            seed_stability[str(bpe)] = per_domain_seed

    max_proxy_regret = max(
        cell["relative_regret"]
        for rows in transfer.values()
        for row in rows
        for cell in row["targets"].values()
    )
    result = {
        "schema_version": 1,
        "normalization": "per-token then divide by scenario median bit-2 coefficient",
        "risk_hierarchy": "domain is the primary scenario; seed is within-domain uncertainty",
        "within_domain_seed": within,
        "cross_domain": cross,
        "summary": {
            "within_median_spearman": within_rho,
            "cross_median_spearman": cross_rho,
            "spearman_gap": within_rho - cross_rho,
            "within_median_top10_overlap": within_overlap,
            "cross_median_top10_overlap": cross_overlap,
            "top10_overlap_gap": within_overlap - cross_overlap,
            "h1_coefficient_gate_pass": h1_pass,
            "max_single_domain_proxy_regret": max_proxy_regret,
            "h2_status": "pending actual fake-quant NLL; coefficient regret is diagnostic only",
        },
        "single_domain_transfer": transfer,
        "seed_config_stability": seed_stability,
        "configs_dir": str(configs_dir),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
