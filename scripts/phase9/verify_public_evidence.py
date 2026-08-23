#!/usr/bin/env python3
"""无需 GPU 或检查点，验证公开证据包是否仍满足发布约束。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXPECTED_METHODS = ["gemq-c4", "concat", "domain-mean", "alphaq-style"]
REAL_METHODS = ["concat", "domain-mean", "gemq-c4"]


def fail(message: str) -> None:
    raise ValueError(f"公开证据包无效：{message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    # 先锁定身份和实验设计，再检查指标；否则新实验可能误用这套发布结论。
    if evidence.get("schema_version") != 1 or evidence.get("project") != "RobustGEMQ":
        fail("unexpected identity/schema")
    if evidence.get("model") != "allenai/OLMoE-1B-7B-0924":
        fail("unexpected model")
    design = evidence["study_design"]
    if design["domains"] != ["code", "general", "instruction", "math"]:
        fail("frozen domain set changed")
    if design["seeds"] != [0, 1, 2] or design["scenarios"] != 12:
        fail("expected four domains by three seeds")
    if design["sequences_per_scenario"] != 128 or design["tokens_per_sequence"] != 2048:
        fail("unexpected scenario dimensions")
    if design["effective_tokens_per_scenario"] != [262144]:
        fail("unexpected effective token count")
    allocation = evidence["allocation"]
    if allocation["bpe"] != 2.5 or allocation["budget"] != 2560.0:
        fail("matched budget changed")
    if allocation["methods"] != EXPECTED_METHODS:
        fail("frozen allocation methods changed")
    if set(allocation["config_sha256"]) != set(EXPECTED_METHODS):
        fail("allocation hashes incomplete")
    provenance = evidence["scenario_provenance"]
    if len(provenance) != 12:
        fail("expected 12 scenario provenance records")
    # 公开的是摘要哈希而非 token 本身；固定长度检查可防止空值混入发布文件。
    for name, record in provenance.items():
        if len(record.get("token_sha256", "")) != 64 or len(record.get("layer_re_sha256", "")) != 64:
            fail(f"invalid scenario hash for {name}")
    validation = evidence["real_checkpoint_validation"]
    if validation["methods"] != REAL_METHODS or validation["item_nlls_per_method"] != 1536:
        fail("real-checkpoint coverage changed")
    if validation["h6_passed"] is not True:
        fail("H6 must pass before release")
    bootstrap = evidence["paired_bootstrap"]
    if bootstrap["draws"] < 10000:
        fail("bootstrap needs at least 10,000 draws")
    decision = evidence["decision"]
    if decision["gate"] != "G6" or decision["g6_status"] != "STOP_NO_LARGE_MODEL_EXPANSION":
        fail("frozen G6 decision changed")
    if decision["h3_full_pass"] or not decision["h6_pass"]:
        fail("inconsistent gate prerequisites")
    if decision["domain_mean_on_matched_budget_mean_worst_pareto"]:
        fail("public result cannot claim Domain-Mean Pareto entry")
    metrics = decision["point_metrics"]
    for method in REAL_METHODS:
        for value in (metrics[method]["mean_domain_nll"], metrics[method]["worst_domain_nll"]):
            if not math.isfinite(float(value)):
                fail(f"non-finite metric for {method}")
    # G6 的核心边界：若该置信区间不再完整为正，就不能继续写“均值稳定更差”。
    delta = bootstrap["comparisons_target_domain_mean"]["concat"]
    mean_ci = delta["mean_domain_nll_difference_target_minus_baseline"]["ci95"]
    if not (mean_ci[0] > 0 and mean_ci[1] > 0):
        fail("published Concat comparison no longer supports the stated boundary")
    print(json.dumps({"verified": True, "g6_status": decision["g6_status"], "scenarios": 12}, sort_keys=True))


if __name__ == "__main__":
    main()
