#!/usr/bin/env python3
"""将阶段八私有评测产物导出为只含输入输出与结论的 manifest。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from scripts.phase10.select_validation_methods import selection_hash
except ModuleNotFoundError:
    from select_validation_methods import selection_hash


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
    recorded_selection_hash = selection.get("selection_sha256")
    canonical_selection = dict(selection)
    canonical_selection.pop("selection_sha256", None)
    if selection_hash(canonical_selection) != recorded_selection_hash:
        raise ValueError("selection hash does not match its frozen contents")
    if unlock.get("selection_sha256") != recorded_selection_hash:
        raise ValueError("test unlock refers to a different selection")
    if unlock.get("selection_file_sha256") != sha256(args.selection):
        raise ValueError("test unlock refers to a different selection file")
    if statistics.get("test_unlock_sha256") != sha256(args.unlock):
        raise ValueError("statistics refer to a different test unlock")
    methods = selection["selected_methods"]
    if methods != ["gemq-c4", "concat", "domain-mean"]:
        raise ValueError("Phase 10 public evidence expects the frozen three-method selection")
    if unlock.get("test_unlocked") is not True or unlock.get("selected_methods") != methods:
        raise ValueError("test is not unlocked for the frozen selected methods")
    if statistics.get("selected_methods") != methods:
        raise ValueError("statistics methods differ from frozen selection")
    if statistics.get("checkpoint_seeds") != [101, 202, 303]:
        raise ValueError("checkpoint seeds differ from the frozen protocol")
    if unlock.get("checkpoint_seeds") != statistics["checkpoint_seeds"]:
        raise ValueError("test unlock and statistics use different checkpoint seeds")
    if statistics.get("cross_method_and_checkpoint_item_identity_match") is not True:
        raise ValueError("statistics must prove cross-method and cross-checkpoint identity")

    point_metrics = statistics["seed_mean_point_metrics"]
    for method in methods:
        if method not in point_metrics:
            raise ValueError(f"missing point metrics for {method}")

    evidence = {
        "schema_version": 2,
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
            "validation_cross_method_item_identity_match": selection[
                "cross_method_item_identity_match"
            ],
            "test_checkpoint_seeds": statistics["checkpoint_seeds"],
            "test_items_per_checkpoint": statistics["items_per_checkpoint"],
            "test_cross_method_and_checkpoint_item_identity_match": statistics[
                "cross_method_and_checkpoint_item_identity_match"
            ],
        },
        "execution_order": {
            "test_unlocked_only_after_all_h6_passed": True,
        },
        "inputs": {
            "experiment": "configs/phase10/experiment.json",
            "data_manifest": "results/phase10/data/manifest.json",
            "checkpoint_root": "results/phase10/checkpoints",
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
        "outputs": {
            "report": "docs/08-independent-confirmation/report.md",
            "statistics": "results/phase10/independent-test/statistics.json",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "methods": methods}, ensure_ascii=False))


if __name__ == "__main__":
    main()
