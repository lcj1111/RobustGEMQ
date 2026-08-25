#!/usr/bin/env python3
"""汇总独立 test 的 checkpoint variance 与逐样本配对 Bootstrap。"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


DOMAINS = ("general", "math", "code", "instruction")
SEEDS = (101, 202, 303)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity_hash(identities: list[tuple]) -> str:
    payload = json.dumps(sorted(identities), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_items(path: Path, method: str, seed: int) -> tuple[dict[tuple[str, int], float], list[tuple]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2 or payload.get("method") != method:
        raise ValueError(f"{method}/seed-{seed}: invalid item schema")
    if payload.get("checkpoint_seed") != seed or payload.get("split") != "test":
        raise ValueError(f"{method}/seed-{seed}: not the frozen independent test")
    values = {}
    identities = []
    for record in payload["items"]:
        key = (record["domain"], int(record["item"]))
        if key in values:
            raise ValueError(f"{method}/seed-{seed}: duplicate item {key}")
        nll = float(record["nll"])
        if not math.isfinite(nll):
            raise ValueError(f"{method}/seed-{seed}: non-finite NLL")
        values[key] = nll
        identities.append((
            record["domain"],
            int(record["scenario_seed"]),
            int(record["item"]),
            record["scenario_token_sha256"],
            record["item_token_sha256"],
        ))
    expected = {(domain, item) for domain in DOMAINS for item in range(96)}
    if set(values) != expected:
        raise ValueError(f"{method}/seed-{seed}: incomplete test grid")
    return values, sorted(identities)


def metrics(values: dict[tuple[str, int], float]) -> dict:
    domain_nll = {
        domain: float(np.mean([values[(domain, item)] for item in range(96)]))
        for domain in DOMAINS
    }
    return {
        "domain_nll": domain_nll,
        "mean_domain_nll": float(np.mean(list(domain_nll.values()))),
        "worst_domain_nll": float(np.max(list(domain_nll.values()))),
        "worst_domain": max(DOMAINS, key=domain_nll.__getitem__),
    }


def paired_bootstrap(left: dict, right: dict, draws: int, rng) -> dict:
    left_domains = np.asarray([[left[(domain, item)] for item in range(96)] for domain in DOMAINS])
    right_domains = np.asarray([[right[(domain, item)] for item in range(96)] for domain in DOMAINS])
    mean_differences = np.empty(draws)
    worst_differences = np.empty(draws)
    for draw in range(draws):
        indices = rng.integers(0, 96, size=(len(DOMAINS), 96))
        left_means = np.asarray([left_domains[index, indices[index]].mean() for index in range(4)])
        right_means = np.asarray([right_domains[index, indices[index]].mean() for index in range(4)])
        mean_differences[draw] = left_means.mean() - right_means.mean()
        worst_differences[draw] = left_means.max() - right_means.max()
    left_point, right_point = metrics(left), metrics(right)

    def summarize(samples: np.ndarray, point: float) -> dict:
        return {
            "point_difference": point,
            "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
            "probability_left_better": float(np.mean(samples < 0)),
        }

    return {
        "mean_domain_nll_left_minus_right": summarize(
            mean_differences, left_point["mean_domain_nll"] - right_point["mean_domain_nll"]
        ),
        "worst_domain_nll_left_minus_right": summarize(
            worst_differences, left_point["worst_domain_nll"] - right_point["worst_domain_nll"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--unlock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    if args.draws < 100:
        raise ValueError("at least 100 bootstrap draws are required")
    unlock = json.loads(args.unlock.read_text(encoding="utf-8"))
    if unlock.get("test_unlocked") is not True:
        raise ValueError("test is not unlocked")
    methods = unlock["selected_methods"]
    if len(methods) != 3:
        raise ValueError("expected three selected methods")

    data = {}
    sources = {}
    common_identity = None
    for method in methods:
        data[method] = {}
        sources[method] = {}
        for seed in SEEDS:
            path = args.root / method / f"seed-{seed}" / "test-items.json"
            data[method][seed], identities = load_items(path, method, seed)
            if common_identity is None:
                common_identity = identities
            elif identities != common_identity:
                raise ValueError(f"cross-checkpoint item identity mismatch: {method}/seed-{seed}")
            sources[method][str(seed)] = sha256(path)

    checkpoint_metrics = {
        method: {str(seed): metrics(data[method][seed]) for seed in SEEDS}
        for method in methods
    }
    checkpoint_variance = {}
    seed_mean_items = {}
    for method in methods:
        seed_mean_items[method] = {
            key: float(np.mean([data[method][seed][key] for seed in SEEDS]))
            for key in data[method][SEEDS[0]]
        }
        checkpoint_variance[method] = {
            "mean_domain_nll_sample_variance": float(np.var(
                [checkpoint_metrics[method][str(seed)]["mean_domain_nll"] for seed in SEEDS], ddof=1
            )),
            "worst_domain_nll_sample_variance": float(np.var(
                [checkpoint_metrics[method][str(seed)]["worst_domain_nll"] for seed in SEEDS], ddof=1
            )),
            "domain_nll_sample_variance": {
                domain: float(np.var(
                    [checkpoint_metrics[method][str(seed)]["domain_nll"][domain] for seed in SEEDS], ddof=1
                ))
                for domain in DOMAINS
            },
        }
    rng = np.random.default_rng(args.seed)
    comparisons = {}
    for left, right in itertools.combinations(methods, 2):
        comparisons[f"{left}__minus__{right}"] = paired_bootstrap(
            seed_mean_items[left], seed_mean_items[right], args.draws, rng
        )
    result = {
        "schema_version": 1,
        "experiment_id": "robustgemq-independent-study-v1",
        "split": "independent-test",
        "selected_methods": methods,
        "checkpoint_seeds": list(SEEDS),
        "items_per_checkpoint": len(common_identity or []),
        "item_identity_sha256": identity_hash(common_identity or []),
        "cross_method_and_checkpoint_item_identity_match": True,
        "checkpoint_metrics": checkpoint_metrics,
        "checkpoint_variance": checkpoint_variance,
        "seed_mean_point_metrics": {method: metrics(seed_mean_items[method]) for method in methods},
        "paired_item_bootstrap": {
            "method": "average each item across checkpoint seeds, then stratified paired resampling within independent test domains",
            "draws": args.draws,
            "seed": args.seed,
            "comparisons": comparisons,
        },
        "source_file_sha256": sources,
        "test_unlock_sha256": sha256(args.unlock),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "methods": methods, "draws": args.draws}, sort_keys=True))


if __name__ == "__main__":
    main()
