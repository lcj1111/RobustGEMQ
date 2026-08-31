#!/usr/bin/env python3
"""用真实下游 Top-k 变化验证近边界路由代理。"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from gemq.routing.margin_proxy import (
    allocation_cost,
    bootstrap_partial_spearman,
    nested_to_tensor,
    normalize_scenario_tensor,
    partial_spearman,
)


DOMAINS = ("general", "math", "code", "instruction")
SEEDS = (0, 1)


def route_metrics(fp_path: Path, quant_path: Path) -> dict:
    fp = torch.load(fp_path, map_location="cpu", weights_only=True)
    quant = torch.load(quant_path, map_location="cpu", weights_only=True)
    flips = []
    low_margin_flips = []
    jaccards = []
    for layer in sorted(fp):
        fp_topk = fp[layer]["topk"].to(torch.int64)
        quant_topk = quant[layer]["topk"].to(torch.int64)
        if fp_topk.shape != quant_topk.shape:
            raise ValueError(f"Route shape mismatch at layer {layer}")
        intersection = (fp_topk.unsqueeze(2) == quant_topk.unsqueeze(1)).any(dim=2).sum(dim=1).float()
        flip = intersection < fp_topk.shape[1]
        margin = fp[layer]["margin"].float()
        low = margin <= torch.quantile(margin, 0.25)
        flips.append(flip.float())
        low_margin_flips.append(flip[low].float())
        union = 2 * fp_topk.shape[1] - intersection
        jaccards.append(intersection / union)
    return {
        "topk_flip_rate": float(torch.cat(flips).mean()),
        "low_margin_flip_rate": float(torch.cat(low_margin_flips).mean()),
        "topk_jaccard": float(torch.cat(jaccards).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--scenario-root", type=Path, required=True)
    parser.add_argument("--route-stats-root", type=Path, required=True)
    parser.add_argument("--fp-root", type=Path, required=True)
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=4000)
    args = parser.parse_args()

    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    if len(records) < 20:
        raise ValueError(f"H4 requires at least 20 configs, found {len(records)}")

    tensors = {}
    sources = {}
    for domain in DOMAINS:
        for seed in SEEDS:
            scenario_dir = args.scenario_root / domain / f"seed-{seed}"
            scenario = json.loads((scenario_dir / "scenario.json").read_text(encoding="utf-8"))
            source = args.route_stats_root / domain / f"seed-{seed}" / "RouteRE_B1,2,3.pkl"
            with source.open("rb") as handle:
                raw = nested_to_tensor(pickle.load(handle))
            tensors[(domain, seed)], scale = normalize_scenario_tensor(raw, scenario["effective_tokens"])
            sources[f"{domain}:seed-{seed}"] = {
                "median_bit2_per_token": scale,
            }

    rows = []
    for record in records:
        name = record["name"]
        with (args.config_root / f"{name}.pkl").open("rb") as handle:
            config = pickle.load(handle)
        proxy_by_seed = {}
        actual_by_seed = {}
        metrics = {}
        for seed in SEEDS:
            proxy_by_seed[str(seed)] = float(
                np.mean([allocation_cost(tensors[(domain, seed)], config) for domain in DOMAINS])
            )
            seed_metrics = []
            for domain in DOMAINS:
                key = f"{domain}:seed-{seed}"
                metrics[key] = route_metrics(
                    args.fp_root / f"route-{domain}-seed-{seed}.pt",
                    args.route_root / name / f"route-{domain}-seed-{seed}.pt",
                )
                seed_metrics.append(metrics[key]["topk_flip_rate"])
            actual_by_seed[str(seed)] = float(np.mean(seed_metrics))
        rows.append(
            {
                **record,
                "proxy_by_seed": proxy_by_seed,
                "proxy_mean": float(np.mean(list(proxy_by_seed.values()))),
                "actual_flip_by_seed": actual_by_seed,
                "actual_flip_mean": float(np.mean(list(actual_by_seed.values()))),
                "route": metrics,
            }
        )

    control = np.asarray([row["actual_hamming_fraction"] for row in rows])
    x = np.asarray([row["proxy_mean"] for row in rows])
    y = np.asarray([row["actual_flip_mean"] for row in rows])
    raw_rho = float(spearmanr(x, y).statistic)
    partial_rho = partial_spearman(x, y, control)
    ci = bootstrap_partial_spearman(
        x, y, control, iterations=args.bootstrap_iterations, seed=20260816
    )
    seed_rho = {}
    for seed in SEEDS:
        seed_x = [row["proxy_by_seed"][str(seed)] for row in rows]
        seed_y = [row["actual_flip_by_seed"][str(seed)] for row in rows]
        seed_rho[str(seed)] = partial_spearman(seed_x, seed_y, control)
    passed = partial_rho >= 0.4 and ci[0] > 0 and all(value > 0 for value in seed_rho.values())

    result = {
        "schema_version": 1,
        "phase": 4,
        "proxy": (
            "per-token, median-bit2-normalized sum of next-layer clipped reciprocal-margin-weighted "
            "expert output perturbation energy"
        ),
        "proxy_formula": "v=clip(1/max(margin,1e-6),0,100); final-layer route cost=0",
        "control": "actual config Hamming fraction from the frozen Scenario-Normalized-Mean base",
        "source_scenarios": sources,
        "configs": rows,
        "summary": {
            "config_count": len(rows),
            "raw_spearman": raw_rho,
            "partial_spearman": partial_rho,
            "bootstrap_95_ci": list(ci),
            "partial_spearman_by_scenario_seed": seed_rho,
            "thresholds": {
                "minimum_configs": 20,
                "minimum_partial_spearman": 0.4,
                "bootstrap_lower_bound_must_exceed": 0.0,
                "both_seeds_must_be_positive": True,
            },
            "h4_gate_pass": passed,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
