#!/usr/bin/env python3
"""Analyze coefficient-risk versus actual downstream route changes for 20 configs."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr


DOMAINS = ("general", "math", "code", "instruction")
BITS = (1, 2, 3)


def objective(coefficients: dict, config: dict) -> float:
    return float(
        sum(coefficients[layer][expert][config[layer][expert]] for layer in config for expert in config[layer])
    )


def route_metrics(fp_path: Path, quant_path: Path) -> dict:
    fp = torch.load(fp_path, map_location="cpu", weights_only=True)
    quant = torch.load(quant_path, map_location="cpu", weights_only=True)
    intersections = []
    margins = []
    flips = []
    low_margin_flips = []
    for layer in range(16):
        fp_topk = fp[layer]["topk"].to(torch.int64)
        quant_topk = quant[layer]["topk"].to(torch.int64)
        if fp_topk.shape != quant_topk.shape:
            raise ValueError(f"Route shape mismatch at layer {layer}")
        intersection = (fp_topk.unsqueeze(2) == quant_topk.unsqueeze(1)).any(dim=2).sum(dim=1).float()
        flip = intersection < 8
        margin = fp[layer]["margin"].float()
        threshold = torch.quantile(margin, 0.25)
        low = margin <= threshold
        intersections.append(intersection)
        flips.append(flip.float())
        margins.append(margin)
        low_margin_flips.append(flip[low].float())
    intersection = torch.cat(intersections)
    flip = torch.cat(flips)
    low_margin_flip = torch.cat(low_margin_flips)
    return {
        "topk_flip_rate": float(flip.mean()),
        "topk_jaccard": float((intersection / (16 - intersection)).mean()),
        "low_margin_flip_rate": float(low_margin_flip.mean()),
        "fp_margin_mean": float(torch.cat(margins).mean()),
    }


def partial_spearman(x, y, control) -> float:
    x_rank, y_rank, c_rank = rankdata(x), rankdata(y), rankdata(control)
    design = np.column_stack([np.ones(len(c_rank)), c_rank])
    x_residual = x_rank - design @ np.linalg.lstsq(design, x_rank, rcond=None)[0]
    y_residual = y_rank - design @ np.linalg.lstsq(design, y_rank, rcond=None)[0]
    return float(spearmanr(x_residual, y_residual).statistic)


def bootstrap_partial(x, y, control, iterations=2000, seed=20260816) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        indices = rng.integers(0, len(x), len(x))
        if len(np.unique(control[indices])) < 2:
            continue
        value = partial_spearman(x[indices], y[indices], control[indices])
        if np.isfinite(value):
            values.append(value)
    return tuple(map(float, np.quantile(values, [0.025, 0.975])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--coefficient-root", type=Path, required=True)
    parser.add_argument("--fp-root", type=Path, required=True)
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    coefficients = {}
    optimum = {}
    for domain in DOMAINS:
        with (args.coefficient_root / f"{domain}-coefficients.pkl").open("rb") as handle:
            coefficients[domain] = pickle.load(handle)
        with (args.coefficient_root / f"{domain}.pkl").open("rb") as handle:
            domain_config = pickle.load(handle)
        optimum[domain] = objective(coefficients[domain], domain_config)

    rows = []
    for record in manifest:
        name = record["name"]
        with (args.config_root / f"{name}.pkl").open("rb") as handle:
            config = pickle.load(handle)
        proxy_regrets = {
            domain: (objective(coefficients[domain], config) - optimum[domain]) / max(abs(optimum[domain]), 1e-12)
            for domain in DOMAINS
        }
        scenario_metrics = {}
        for domain in DOMAINS:
            for seed in (0, 1):
                key = f"{domain}:seed-{seed}"
                scenario_metrics[key] = route_metrics(
                    args.fp_root / f"route-{domain}-seed-{seed}.pt",
                    args.route_root / name / f"route-{domain}-seed-{seed}.pt",
                )
        rows.append(
            {
                **record,
                "proxy_regret_mean": float(np.mean(list(proxy_regrets.values()))),
                "proxy_regret_worst": float(np.max(list(proxy_regrets.values()))),
                "proxy_regrets": proxy_regrets,
                "route": scenario_metrics,
                "route_flip_mean": float(np.mean([m["topk_flip_rate"] for m in scenario_metrics.values()])),
            }
        )

    x = np.asarray([row["proxy_regret_mean"] for row in rows])
    y = np.asarray([row["route_flip_mean"] for row in rows])
    control = np.asarray([row["actual_hamming_fraction"] for row in rows])
    raw_rho = float(spearmanr(x, y).statistic)
    partial_rho = partial_spearman(x, y, control)
    ci_low, ci_high = bootstrap_partial(x, y, control)
    seed_rho = {}
    for seed in (0, 1):
        seed_y = np.asarray(
            [np.mean([row["route"][f"{domain}:seed-{seed}"]["topk_flip_rate"] for domain in DOMAINS]) for row in rows]
        )
        seed_rho[str(seed)] = partial_spearman(x, seed_y, control)

    gate_pass = partial_rho >= 0.4 and ci_low > 0 and all(value > 0 for value in seed_rho.values())
    result = {
        "schema_version": 1,
        "proxy": "mean domain-normalized GEMQ coefficient regret",
        "control": "config Hamming fraction",
        "configs": rows,
        "summary": {
            "raw_spearman": raw_rho,
            "partial_spearman": partial_rho,
            "bootstrap_95_ci": [ci_low, ci_high],
            "partial_spearman_by_scenario_seed": seed_rho,
            "h4_pilot_gate_pass": gate_pass,
            "interpretation": "GO only authorizes Phase 4 near-boundary proxy work; it is not itself a route-aware method claim",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
