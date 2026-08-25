#!/usr/bin/env python3
"""Aggregate the frozen no-RFT GPTQ screen and nominate at most three pack candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DOMAINS = ("general", "math", "code", "instruction")
METHODS = ("gemq-c4", "concat", "domain-mean", "alphaq-style")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = {}
    for method in METHODS:
        payload = json.loads((args.root / method / "phase6-eval.json").read_text(encoding="utf-8"))
        domains = {
            domain: float(np.mean([payload["scenarios"][f"{domain}:seed-{seed}"]["nll"] for seed in (0, 1, 2)]))
            for domain in DOMAINS
        }
        values = np.asarray(list(domains.values()))
        audit = json.loads((args.configs / f"{method}.audit.json").read_text(encoding="utf-8"))
        metrics[method] = {
            "domain_nll": domains,
            "mean_domain_nll": float(values.mean()),
            "worst_domain_nll": float(values.max()),
            "worst_domain": DOMAINS[int(values.argmax())],
            "actual_bpe": audit["actual_bpe"],
            "used_bits": audit["used_bits"],
        }
    # Exact matched budget makes mean/worst NLL a sufficient Pareto coordinate here.
    pareto = []
    for name, item in metrics.items():
        dominated = any(
            other != name
            and metrics[other]["mean_domain_nll"] <= item["mean_domain_nll"]
            and metrics[other]["worst_domain_nll"] <= item["worst_domain_nll"]
            and (
                metrics[other]["mean_domain_nll"] < item["mean_domain_nll"]
                or metrics[other]["worst_domain_nll"] < item["worst_domain_nll"]
            )
            for other in METHODS
        )
        if not dominated:
            pareto.append(name)
    # Required coverage: both pooled baseline and robust candidate, then fill by Pareto order.
    selected = [name for name in ("concat", "domain-mean") if name in METHODS]
    for name in sorted(pareto, key=lambda item: (metrics[item]["worst_domain_nll"], metrics[item]["mean_domain_nll"])):
        if name not in selected and len(selected) < 3:
            selected.append(name)
    result = {
        "schema_version": 1,
        "stage": "no-RFT GPTQ screen",
        "selection_rule": "same 2.5-bpe allocation budget; retain Concat and Scenario-Normalized-Mean (historical key: domain-mean), then fill to at most three by no-RFT mean/worst Pareto order",
        "metrics": metrics,
        "pareto_methods": pareto,
        "selected_for_real_packing": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
