#!/usr/bin/env python3
"""Post-Gate diagnostics for a failed H4 proxy; never used to retune the proxy."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from gemq.routing.margin_proxy import (
    allocation_cost,
    nested_to_tensor,
    normalize_scenario_tensor,
    partial_spearman,
)


DOMAINS = ("general", "math", "code", "instruction")


def correlations(x, y, control) -> dict:
    return {
        "raw_spearman": float(spearmanr(x, y).statistic),
        "hamming_controlled_partial_spearman": partial_spearman(x, y, control),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--scenario-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    if validation["summary"]["h4_gate_pass"]:
        raise RuntimeError("Failure analysis is only valid after H4 fails")
    rows = validation["configs"]
    control = np.asarray([row["actual_hamming_fraction"] for row in rows])
    route_proxy = np.asarray([row["proxy_mean"] for row in rows])
    actual_flip = np.asarray([row["actual_flip_mean"] for row in rows])
    actual_low_margin_flip = np.asarray(
        [np.mean([metric["low_margin_flip_rate"] for metric in row["route"].values()]) for row in rows]
    )
    actual_jaccard = np.asarray(
        [np.mean([metric["topk_jaccard"] for metric in row["route"].values()]) for row in rows]
    )

    quality_tensors = {}
    for domain in DOMAINS:
        seed_tensors = []
        for seed in (0, 1):
            directory = args.scenario_root / domain / f"seed-{seed}"
            scenario = json.loads((directory / "scenario.json").read_text(encoding="utf-8"))
            with (directory / "LayerRE_B1,2,3.pkl").open("rb") as handle:
                raw = nested_to_tensor(pickle.load(handle))
            seed_tensors.append(normalize_scenario_tensor(raw, scenario["effective_tokens"])[0])
        quality_tensors[domain] = np.mean(seed_tensors, axis=0)
    quality_proxy = []
    for row in rows:
        with (args.config_root / f"{row['name']}.pkl").open("rb") as handle:
            config = pickle.load(handle)
        quality_proxy.append(
            np.mean([allocation_cost(quality_tensors[domain], config) for domain in DOMAINS])
        )
    quality_proxy = np.asarray(quality_proxy)

    result = {
        "schema_version": 1,
        "phase": 4,
        "status": "post-gate diagnosis only; not eligible for proxy or lambda retuning",
        "near_boundary_proxy_vs_topk_flip": correlations(route_proxy, actual_flip, control),
        "near_boundary_proxy_vs_low_margin_flip": correlations(
            route_proxy, actual_low_margin_flip, control
        ),
        "near_boundary_proxy_vs_topk_jaccard": correlations(route_proxy, actual_jaccard, control),
        "quality_coefficient_proxy_vs_topk_flip": correlations(
            quality_proxy, actual_flip, control
        ),
        "interpretation": (
            "Raw correlations are confounded by perturbation size. The pre-registered H4 decision remains "
            "the Hamming-controlled Top-k flip result in h4-proxy-validation.json."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
