#!/usr/bin/env python3
"""从完整实验产物导出可提交 Git 的阶段七 manifest。

输入目录可能包含大检查点和逐样本得分；输出只记录实验协议、输入、输出与结论。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


METHOD_LABELS = {
    "gemq-c4": "GEMQ-C4",
    "concat": "Concat",
    "domain-mean": "Scenario-Normalized-Mean",
    "alphaq-style": "AlphaQ-style",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_root
    # 这四个文件分别覆盖 allocation、统计结论、Gate 决策和离线验收。
    # 缺少任意一个文件时应直接失败，不能导出看似完整的证据包。
    paths = {
        "allocation_manifest": root / "configs" / "bpe-2.5" / "manifest.json",
        "bootstrap": root / "item-bootstrap" / "bootstrap.json",
        "decision": root / "phase6_decision.json",
        "release_verification": root / "release-verification.json",
    }
    documents = {name: read_json(path) for name, path in paths.items()}
    config = documents["allocation_manifest"]
    bootstrap = documents["bootstrap"]
    decision = documents["decision"]
    release = documents["release_verification"]
    if release.get("schema_version") != 2 or release.get("cross_method_item_identity_match") is not True:
        raise ValueError("release-verification.json must use schema v2 and prove cross-method item identity")
    source_scenarios = config["source_scenarios"]
    # 场景名是 ``domain:seed-N``；从名称反推域与种子，可避免维护第二份清单。
    domains = sorted({name.split(":", 1)[0] for name in source_scenarios})
    seeds = sorted({int(name.split("seed-", 1)[1]) for name in source_scenarios})
    evidence = {
        "schema_version": 3,
        "project": "RobustGEMQ",
        "phase": 6,
        "model": "allenai/OLMoE-1B-7B-0924",
        "inputs": {
            "allocation_manifest": "artifacts/phase3/configs/manifest.json",
            "data_manifest": "artifacts/phase2/data/source-manifest.json",
        },
        "study_design": {
            "domains": domains,
            "seeds": seeds,
            "scenarios": len(source_scenarios),
            "sequences_per_scenario": 128,
            "tokens_per_sequence": 2048,
            "effective_tokens_per_scenario": sorted(
                {entry["effective_tokens"] for entry in source_scenarios.values()}
            ),
        },
        "allocation": {
            "bpe": config["bpe"],
            "budget": config["budget"],
            "methods": config["methods"],
            "method_labels": METHOD_LABELS,
            "method_hamming_fraction": config["method_hamming_fraction"],
        },
        "real_checkpoint_validation": {
            "methods": release["methods_with_real_checkpoints"],
            "item_nlls_per_method": release["item_nlls_per_method"],
            "h6_passed": release["h6_passed"],
            "h6_evidence_format": release["h6_evidence_format"],
            "current_h6_contract": {
                "structured_json": True,
                "decode_argmax_agreement_min": 0.95,
            },
        },
        "item_identity": {
            "items_per_method": release["item_nlls_per_method"],
            "cross_method_match": release["cross_method_item_identity_match"],
        },
        "paired_bootstrap": {
            "method": bootstrap["method"],
            "draws": bootstrap["draws"],
            "seed": bootstrap["seed"],
            "comparisons_target_domain_mean": bootstrap["comparisons_target_domain_mean"],
            "inference_scope": bootstrap["inference_scope"],
        },
        "decision": {
            "gate": decision["gate"],
            "g6_status": decision["g6_status"],
            "h3_full_pass": decision["h3_full_pass"],
            "h6_pass": decision["h6_pass"],
            "domain_mean_on_matched_budget_mean_worst_pareto": decision[
                "domain_mean_on_matched_budget_mean_worst_pareto"
            ],
            "point_metrics": decision["point_metrics"],
            "reason": decision["reason"],
        },
        "result_scope": "descriptive bootstrap on the fixed Phase 6 training scenarios; not an independent validation or test set",
        "outputs": {"report": "docs/07-release/report.md"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "scenarios": len(source_scenarios)}, sort_keys=True))


if __name__ == "__main__":
    main()
