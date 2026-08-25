#!/usr/bin/env python3
"""汇总下游任务 checkpoint 方差，并核对所有方法使用相同样本。"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


TASKS = ("wikitext2-test", "gsm8k-test", "boolq-validation")


def load_results(root: Path, methods: list[str], seeds: list[int]) -> dict[tuple[str, int], dict]:
    results = {}
    reference = None
    reference_sources = None
    for method in methods:
        for seed in seeds:
            path = root / method / f"seed-{seed}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["method"] != method or payload["checkpoint_seed"] != seed:
                raise ValueError(f"metadata mismatch: {path}")
            identities = {task: [row["item_sha256"] for row in payload["tasks"][task]["items"]] for task in TASKS}
            if reference is None:
                reference = identities
                reference_sources = payload["source_sha256"]
            if identities != reference:
                raise ValueError(f"cross-checkpoint item identity mismatch: {path}")
            if payload["source_sha256"] != reference_sources:
                raise ValueError(f"cross-checkpoint source hash mismatch: {path}")
            results[(method, seed)] = payload
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    methods = selection["selected_methods"]
    seeds = [101, 202, 303]
    results = load_results(args.root, methods, seeds)
    summary = {"schema_version": 1, "methods": methods, "checkpoint_seeds": seeds, "tasks": {}}
    for task in TASKS:
        summary["tasks"][task] = {}
        for method in methods:
            values = [results[(method, seed)]["tasks"][task]["value"] for seed in seeds]
            summary["tasks"][task][method] = {
                "checkpoint_values": dict(zip(map(str, seeds), values)),
                "checkpoint_mean": statistics.fmean(values),
                "checkpoint_sample_variance": statistics.variance(values),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "methods": methods}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
