#!/usr/bin/env python3
"""将 Phase 10 私有评测产物导出为可提交 Git 的轻量证据包。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


METHOD_LABELS = {
    "gemq-c4": "GEMQ-C4",
    "concat": "Concat",
    "domain-mean": "Scenario-Normalized-Mean",
}


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--unlock", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = read_json(args.selection)
    unlock = read_json(args.unlock)
    statistics = read_json(args.statistics)
    methods = selection["selected_methods"]
    if methods != ["gemq-c4", "concat", "domain-mean"]:
        raise ValueError("Phase 10 public evidence expects the frozen three-method selection")
    if unlock.get("test_unlocked") is not True or unlock.get("selected_methods") != methods:
        raise ValueError("test is not unlocked for the frozen selected methods")
    if statistics.get("selected_methods") != methods:
        raise ValueError("statistics methods differ from frozen selection")
    if statistics.get("checkpoint_seeds") != [101, 202, 303]:
        raise ValueError("checkpoint seeds differ from the frozen protocol")
    if statistics.get("cross_method_and_checkpoint_item_identity_match") is not True:
        raise ValueError("statistics must prove cross-method and cross-checkpoint identity")

    point_metrics = statistics["seed_mean_point_metrics"]
    for method in methods:
        if method not in point_metrics:
            raise ValueError(f"missing point metrics for {method}")

    evidence = {
        "schema_version": 1,
        "project": "RobustGEMQ",
        "phase": 10,
        "phase_label": "independent-confirmation",
        "model": "allenai/OLMoE-1B-7B-0924",
        "method_labels": METHOD_LABELS,
        "protocol": {
            "methods_compared_in_validation": selection["methods_compared"],
            "screen_checkpoint_seed": selection["screen_checkpoint_seed"],
            "selection_split": selection["selection_split"],
            "selection_rule": selection["selection_rule"],
            "selected_methods": methods,
            "validation_item_identity_sha256": selection["validation_item_identity_sha256"],
            "validation_cross_method_item_identity_match": selection[
                "cross_method_item_identity_match"
            ],
            "test_checkpoint_seeds": statistics["checkpoint_seeds"],
            "test_items_per_checkpoint": statistics["items_per_checkpoint"],
            "test_item_identity_sha256": statistics["item_identity_sha256"],
            "test_cross_method_and_checkpoint_item_identity_match": statistics[
                "cross_method_and_checkpoint_item_identity_match"
            ],
        },
        "integrity": {
            "selection_sha256": selection["selection_sha256"],
            "test_unlock_sha256": statistics["test_unlock_sha256"],
            "test_unlocked_only_after_all_h6_passed": True,
            "h6_summary_sha256": unlock["h6_summary_sha256"],
        },
        "independent_test": {
            "seed_mean_point_metrics": point_metrics,
            "checkpoint_variance": statistics["checkpoint_variance"],
            "paired_item_bootstrap": statistics["paired_item_bootstrap"],
        },
        "conclusion": {
            "mean_domain_nll_winner": "concat",
            "worst_domain_nll_winner": "gemq-c4",
            "scenario_normalized_mean_strictly_dominates_baselines": False,
            "interpretation": (
                "独立 test 复核了平均质量与最差领域鲁棒性之间的权衡；"
                "不支持将 Scenario-Normalized-Mean 表述为通用质量提升。"
            ),
        },
        "scope": (
            "record-disjoint independent test after validation-only method selection; "
            "paired bootstrap averages each item across three checkpoint seeds and then resamples "
            "within domains. This confirms the OLMoE protocol only, not universal superiority."
        ),
        "source_file_sha256": {
            "selection": sha256(args.selection),
            "test_unlock": sha256(args.unlock),
            "independent_statistics": sha256(args.statistics),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "methods": methods}, ensure_ascii=False))


if __name__ == "__main__":
    main()
