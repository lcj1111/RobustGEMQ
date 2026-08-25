#!/usr/bin/env python3
"""Paired item-level bootstrap for the fixed Phase 6 real-checkpoint comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DOMAINS = ("general", "math", "code", "instruction")
BASELINES = ("concat", "gemq-c4")
TARGET = "domain-mean"


def load(path: Path) -> dict[tuple[str, int], np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[tuple[str, int], list[tuple[int, float]]] = {}
    for item in data["items"]:
        key = (item["domain"], int(item["seed"]))
        grouped.setdefault(key, []).append((int(item["item"]), float(item["nll"])))
    if set(grouped) != {(domain, seed) for domain in DOMAINS for seed in (0, 1, 2)}:
        raise ValueError(f"incomplete scenario matrix in {path}")
    output = {}
    for key, rows in grouped.items():
        rows.sort()
        if [item for item, _ in rows] != list(range(128)):
            raise ValueError(f"{path}: {key} is not exactly items 0..127")
        values = np.asarray([value for _, value in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: non-finite NLLs")
        output[key] = values
    return output


def point_metrics(values: dict[tuple[str, int], np.ndarray]) -> dict:
    domain = {
        name: float(np.mean([values[(name, seed)].mean() for seed in (0, 1, 2)]))
        for name in DOMAINS
    }
    aggregate = np.asarray(list(domain.values()))
    return {
        "domain_nll": domain,
        "mean_domain_nll": float(aggregate.mean()),
        "worst_domain_nll": float(aggregate.max()),
        "worst_domain": DOMAINS[int(aggregate.argmax())],
    }


def paired_bootstrap(target, baseline, target_point, baseline_point, draws, rng) -> dict:
    mean_diffs = np.empty(draws, dtype=np.float64)
    worst_diffs = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        target_domains, baseline_domains = [], []
        for domain in DOMAINS:
            target_seeds, baseline_seeds = [], []
            for seed in (0, 1, 2):
                indices = rng.integers(0, 128, size=128)
                target_seeds.append(target[(domain, seed)][indices].mean())
                baseline_seeds.append(baseline[(domain, seed)][indices].mean())
            target_domains.append(np.mean(target_seeds))
            baseline_domains.append(np.mean(baseline_seeds))
        target_domains = np.asarray(target_domains)
        baseline_domains = np.asarray(baseline_domains)
        mean_diffs[draw] = target_domains.mean() - baseline_domains.mean()
        worst_diffs[draw] = target_domains.max() - baseline_domains.max()
    def summary(samples, point_difference):
        return {
            "point_difference": float(point_difference),
            "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
            "probability_target_better": float(np.mean(samples < 0)),
        }
    return {
        "mean_domain_nll_difference_target_minus_baseline": summary(
            mean_diffs, target_point["mean_domain_nll"] - baseline_point["mean_domain_nll"]
        ),
        "worst_domain_nll_difference_target_minus_baseline": summary(
            worst_diffs, target_point["worst_domain_nll"] - baseline_point["worst_domain_nll"]
        ),
        "probability_target_strictly_dominates": float(np.mean((mean_diffs < 0) & (worst_diffs < 0))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    if args.draws < 1000:
        raise ValueError("at least 1000 bootstrap draws are required")
    data = {name: load(args.root / name / "item-nll.json") for name in (*BASELINES, TARGET)}
    point = {name: point_metrics(values) for name, values in data.items()}
    rng = np.random.default_rng(args.seed)
    comparisons = {
        baseline: paired_bootstrap(
            data[TARGET], data[baseline], point[TARGET], point[baseline], args.draws, rng
        )
        for baseline in BASELINES
    }
    result = {
        "schema_version": 1,
        "method": "descriptive stratified paired bootstrap within each fixed training domain/seed scenario over 128 items",
        "inference_scope": "fixed Phase 6 training scenarios; not an independent validation or test set",
        "draws": args.draws,
        "seed": args.seed,
        "point_metrics": point,
        "comparisons_target_domain_mean": comparisons,
        "interpretation": "Negative target-minus-baseline differences favor Scenario-Normalized-Mean (historical key: domain-mean).",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "draws": args.draws}))


if __name__ == "__main__":
    main()
