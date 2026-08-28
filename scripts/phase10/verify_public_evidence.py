#!/usr/bin/env python3
"""离线验证 Phase 10 公开证据的结构、指标自洽性和范围边界。"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


METHODS = ["gemq-c4", "concat", "domain-mean"]
DOMAINS = ["general", "math", "code", "instruction"]
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def fail(message: str) -> None:
    raise ValueError(f"Phase 10 公开证据无效：{message}")


def number(value: object, context: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        fail(f"{context} 不是有限数")
    return result


def require_hash(value: object, context: str) -> None:
    if SHA256_RE.fullmatch(str(value)) is None:
        fail(f"{context} 不是小写 SHA-256")


def close(value: object, expected: float, context: str) -> None:
    if not math.isclose(number(value, context), expected, rel_tol=1e-10, abs_tol=1e-10):
        fail(f"{context} 与领域指标不可重算")


def validate_metric(metric: dict, method: str) -> None:
    domain_nll = metric.get("domain_nll", {})
    if set(domain_nll) != set(DOMAINS):
        fail(f"{method} 缺少领域 NLL")
    values = {domain: number(domain_nll[domain], f"{method}.{domain}") for domain in DOMAINS}
    close(metric.get("mean_domain_nll"), sum(values.values()) / len(DOMAINS), f"{method}.mean")
    worst_domain = max(DOMAINS, key=values.__getitem__)
    if metric.get("worst_domain") != worst_domain:
        fail(f"{method} 最坏领域标签不一致")
    close(metric.get("worst_domain_nll"), values[worst_domain], f"{method}.worst")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    if evidence.get("schema_version") != 1 or evidence.get("project") != "RobustGEMQ":
        fail("项目 identity 或 schema 不正确")
    if evidence.get("phase") != 10 or evidence.get("phase_label") != "independent-confirmation":
        fail("阶段 identity 不正确")
    if evidence.get("model") != "allenai/OLMoE-1B-7B-0924":
        fail("模型 identity 不正确")

    protocol = evidence.get("protocol", {})
    if protocol.get("selected_methods") != METHODS:
        fail("冻结方法集合发生变化")
    if protocol.get("methods_compared_in_validation") != [
        "gemq-c4", "layer-balanced", "usage-only", "concat", "domain-mean"
    ]:
        fail("validation 方法集合不完整")
    if protocol.get("screen_checkpoint_seed") != 101 or protocol.get("selection_split") != "validation":
        fail("方法选择不再只依赖 validation seed 101")
    if protocol.get("test_checkpoint_seeds") != [101, 202, 303]:
        fail("checkpoint seed 不符合冻结协议")
    if protocol.get("test_items_per_checkpoint") != 384:
        fail("每个 checkpoint 的 test 样本数不正确")
    if protocol.get("validation_cross_method_item_identity_match") is not True:
        fail("validation 缺少跨方法 identity 验证")
    if protocol.get("test_cross_method_and_checkpoint_item_identity_match") is not True:
        fail("test 缺少跨方法/seed identity 验证")
    require_hash(protocol.get("validation_item_identity_sha256"), "validation identity")
    require_hash(protocol.get("test_item_identity_sha256"), "test identity")

    integrity = evidence.get("integrity", {})
    require_hash(integrity.get("selection_sha256"), "selection")
    require_hash(integrity.get("test_unlock_sha256"), "test unlock")
    if integrity.get("test_unlocked_only_after_all_h6_passed") is not True:
        fail("test unlock 缺少 H6 前置条件")
    h6 = integrity.get("h6_summary_sha256", {})
    if set(h6) != set(METHODS):
        fail("H6 方法集合不完整")
    for method in METHODS:
        if set(h6[method]) != {"101", "202", "303"}:
            fail(f"{method} 的 H6 seed 不完整")
        for digest in h6[method].values():
            require_hash(digest, f"{method} H6")

    result = evidence.get("independent_test", {})
    metrics = result.get("seed_mean_point_metrics", {})
    if set(metrics) != set(METHODS):
        fail("独立 test 的方法集合不完整")
    for method in METHODS:
        validate_metric(metrics[method], method)
    conclusion = evidence.get("conclusion", {})
    if conclusion.get("mean_domain_nll_winner") != min(METHODS, key=lambda name: metrics[name]["mean_domain_nll"]):
        fail("平均 NLL winner 与点估计不一致")
    if conclusion.get("worst_domain_nll_winner") != min(METHODS, key=lambda name: metrics[name]["worst_domain_nll"]):
        fail("最差领域 winner 与点估计不一致")
    if conclusion.get("scenario_normalized_mean_strictly_dominates_baselines") is not False:
        fail("不应将 Scenario-Normalized-Mean 标记为严格支配")

    bootstrap = result.get("paired_item_bootstrap", {})
    if bootstrap.get("draws", 0) < 10000:
        fail("Bootstrap 重采样次数不足")
    comparisons = bootstrap.get("comparisons", {})
    if set(comparisons) != {
        "gemq-c4__minus__concat",
        "gemq-c4__minus__domain-mean",
        "concat__minus__domain-mean",
    }:
        fail("Bootstrap 比较集合不完整")
    for comparison in comparisons.values():
        for record in comparison.values():
            ci = record.get("ci95", [])
            if len(ci) != 2 or number(ci[0], "bootstrap CI") > number(ci[1], "bootstrap CI"):
                fail("Bootstrap 区间非法")
            probability = number(record.get("probability_left_better"), "bootstrap probability")
            if not 0 <= probability <= 1:
                fail("Bootstrap 概率非法")

    for name, digest in evidence.get("source_file_sha256", {}).items():
        require_hash(digest, f"source {name}")
    if set(evidence.get("source_file_sha256", {})) != {
        "selection", "test_unlock", "independent_statistics"
    }:
        fail("源文件摘要集合不完整")
    if "record-disjoint independent test" not in evidence.get("scope", ""):
        fail("缺少独立 test 的范围边界")
    print(json.dumps({"verified": True, "phase": 10, "methods": METHODS}, ensure_ascii=False))


if __name__ == "__main__":
    main()
