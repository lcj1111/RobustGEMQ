#!/usr/bin/env python3
"""Explain Phase 3 proxy/held-out disagreement without tuning on held-out data."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


DOMAINS = ("general", "math", "code", "instruction")
METHODS = ("gemq-c4", "concat", "domain-mean", "domain-worst", "domain-cvar-0.5", "alphaq-style")


def config_array(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        config = pickle.load(handle)
    return np.asarray([[config[layer][expert] for expert in range(64)] for layer in range(16)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    quality = json.loads(args.quality.read_text(encoding="utf-8"))
    result = {"schema_version": 1, "budgets": {}}
    for bpe in (2.5, 2.0):
        directory = args.configs / f"bpe-{bpe:.1f}"
        arrays = {name: config_array(directory / f"{name}.pkl") for name in METHODS}
        predicted = {
            name: json.loads((directory / f"{name}.audit.json").read_text(encoding="utf-8"))["common"]
            for name in METHODS
        }
        actual = quality["budgets"][f"{bpe:.1f}"]["metrics"]
        hamming = {
            left: {
                right: float(np.mean(arrays[left] != arrays[right])) for right in METHODS
            }
            for left in METHODS
        }
        correlations = {}
        for domain in DOMAINS:
            proxy = [predicted[name]["domain_losses"][domain] for name in METHODS]
            nll = [actual[name]["domain_nll_delta"][domain] for name in METHODS]
            correlations[domain] = float(spearmanr(proxy, nll).statistic)
        proxy_mean = [predicted[name]["domain_mean"] for name in METHODS]
        actual_mean = [actual[name]["mean_domain_nll_delta"] for name in METHODS]
        result["budgets"][f"{bpe:.1f}"] = {
            "config_hamming_fraction": hamming,
            "proxy_vs_heldout_spearman_by_domain": correlations,
            "proxy_mean_vs_heldout_mean_spearman": float(spearmanr(proxy_mean, actual_mean).statistic),
            "interpretation_boundary": (
                "Post-gate diagnostic only. These correlations cannot be used to change Phase 3 objectives "
                "or reselect configs on held-out data."
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "budgets": list(result["budgets"])}))


if __name__ == "__main__":
    main()
