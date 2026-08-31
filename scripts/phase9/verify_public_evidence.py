#!/usr/bin/env python3
"""无需 GPU 或检查点，验证阶段七 manifest 的协议、输入输出与结论。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXPECTED_METHODS = ["gemq-c4", "concat", "domain-mean", "alphaq-style"]
REAL_METHODS = ["concat", "domain-mean", "gemq-c4"]
DOMAINS = ["code", "general", "instruction", "math"]
METHOD_LABELS = {
    "gemq-c4": "GEMQ-C4",
    "concat": "Concat",
    "domain-mean": "Scenario-Normalized-Mean",
    "alphaq-style": "AlphaQ-style",
}
def fail(message: str) -> None:
    raise ValueError(f"阶段七 manifest 无效：{message}")


def finite(value: object, context: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        fail(f"{context} 不是有限数")
    return number


def close(observed: object, expected: float, context: str) -> None:
    if not math.isclose(finite(observed, context), expected, rel_tol=1e-10, abs_tol=1e-10):
        fail(f"{context} 与可重算结果不一致")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    evidence = json.loads(args.manifest.read_text(encoding="utf-8"))
    if evidence.get("schema_version") != 3 or evidence.get("project") != "RobustGEMQ":
        fail("identity/schema 必须为 RobustGEMQ v3")
    if evidence.get("model") != "allenai/OLMoE-1B-7B-0924":
        fail("模型身份发生变化")
    if set(evidence.get("inputs", {})) != {"allocation_manifest", "data_manifest"}:
        fail("输入 manifest 不完整")
    if evidence.get("outputs", {}).get("report") != "docs/07-release/report.md":
        fail("输出报告未记录")

    design = evidence["study_design"]
    if design["domains"] != DOMAINS or design["seeds"] != [0, 1, 2] or design["scenarios"] != 12:
        fail("冻结的 4 域 × 3 种子设计发生变化")
    if design["sequences_per_scenario"] != 128 or design["tokens_per_sequence"] != 2048:
        fail("场景尺寸发生变化")
    if design["effective_tokens_per_scenario"] != [262144]:
        fail("有效 token 数发生变化")

    allocation = evidence["allocation"]
    if allocation["bpe"] != 2.5 or allocation["budget"] != 2560.0:
        fail("匹配预算发生变化")
    if allocation["methods"] != EXPECTED_METHODS or allocation.get("method_labels") != METHOD_LABELS:
        fail("方法集合或公开名称发生变化")
    hamming = allocation["method_hamming_fraction"]
    if set(hamming) != set(EXPECTED_METHODS):
        fail("Hamming 矩阵行不完整")
    for left in EXPECTED_METHODS:
        if set(hamming[left]) != set(EXPECTED_METHODS):
            fail(f"Hamming 矩阵列不完整：{left}")
        for right in EXPECTED_METHODS:
            value = finite(hamming[left][right], f"Hamming {left}/{right}")
            if not 0 <= value <= 1 or (left == right and value != 0):
                fail(f"Hamming 值非法：{left}/{right}")
            close(value, float(hamming[right][left]), f"Hamming 对称性 {left}/{right}")

    identity = evidence["item_identity"]
    if identity["items_per_method"] != 1536:
        fail("逐样本数量发生变化")
    identity_status = identity.get("cross_method_match")
    if identity_status not in (True, "not-retroactively-verified"):
        fail("缺少跨方法逐样本身份检查状态")
    if identity_status == "not-retroactively-verified" and identity.get("future_release_requirement") is not True:
        fail("历史证据例外必须同时声明新发布强制校验")

    validation = evidence["real_checkpoint_validation"]
    if validation["methods"] != REAL_METHODS or validation["item_nlls_per_method"] != 1536:
        fail("真实检查点覆盖发生变化")
    if validation["h6_passed"] is not True:
        fail("H6 未通过")
    h6_contract = validation.get("current_h6_contract", {})
    if h6_contract.get("structured_json") is not True or h6_contract.get("decode_argmax_agreement_min") != 0.95:
        fail("当前 H6 契约缺少结构化 JSON 或 95% argmax 断言")
    if "h6_evidence_format" not in validation:
        fail("H6 证据格式未记录")

    scope = evidence.get("result_scope", "")
    if "descriptive bootstrap" not in scope or "not an independent" not in scope:
        fail("缺少固定训练场景内的描述性 Bootstrap 边界")
    bootstrap = evidence["paired_bootstrap"]
    if bootstrap["draws"] < 10000 or bootstrap.get("inference_scope") != scope:
        fail("Bootstrap 次数或推断范围不一致")

    decision = evidence["decision"]
    if decision["gate"] != "G6" or decision["g6_status"] != "STOP_NO_LARGE_MODEL_EXPANSION":
        fail("冻结的 G6 决策发生变化")
    if decision["h3_full_pass"] or not decision["h6_pass"]:
        fail("Gate 前置状态相互矛盾")
    metrics = decision["point_metrics"]
    if set(metrics) != set(REAL_METHODS):
        fail("点估计方法集合发生变化")
    for method in REAL_METHODS:
        values = metrics[method]
        if set(values["domain_nll"]) != set(DOMAINS):
            fail(f"{method} 的领域指标不完整")
        domains = {name: finite(value, f"{method}.{name}") for name, value in values["domain_nll"].items()}
        close(values["mean_domain_nll"], sum(domains.values()) / 4, f"{method}.mean")
        worst = max(DOMAINS, key=domains.__getitem__)
        if values["worst_domain"] != worst:
            fail(f"{method} 的最坏领域标签不一致")
        close(values["worst_domain_nll"], domains[worst], f"{method}.worst")

    target = metrics["domain-mean"]
    dominated = any(
        metrics[baseline]["mean_domain_nll"] <= target["mean_domain_nll"]
        and metrics[baseline]["worst_domain_nll"] <= target["worst_domain_nll"]
        and (
            metrics[baseline]["mean_domain_nll"] < target["mean_domain_nll"]
            or metrics[baseline]["worst_domain_nll"] < target["worst_domain_nll"]
        )
        for baseline in ("concat", "gemq-c4")
    )
    if decision["domain_mean_on_matched_budget_mean_worst_pareto"] is not (not dominated):
        fail("Pareto 标记与点估计不一致")

    comparisons = bootstrap["comparisons_target_domain_mean"]
    if set(comparisons) != {"concat", "gemq-c4"}:
        fail("Bootstrap 基线集合发生变化")
    for baseline, result in comparisons.items():
        for key, metric_name in (
            ("mean_domain_nll_difference_target_minus_baseline", "mean_domain_nll"),
            ("worst_domain_nll_difference_target_minus_baseline", "worst_domain_nll"),
        ):
            record = result[key]
            ci = [finite(value, f"{baseline}.{key}.ci95") for value in record["ci95"]]
            if len(ci) != 2 or ci[0] > ci[1]:
                fail(f"{baseline}.{key} 的置信区间非法")
            close(record["point_difference"], target[metric_name] - metrics[baseline][metric_name], f"{baseline}.{key}.point")
            probability = finite(record["probability_target_better"], f"{baseline}.{key}.probability")
            if not 0 <= probability <= 1:
                fail(f"{baseline}.{key} 的概率非法")
        dominance = finite(result["probability_target_strictly_dominates"], f"{baseline}.dominance")
        if not 0 <= dominance <= 1:
            fail(f"{baseline} 的支配概率非法")
    concat_ci = comparisons["concat"]["mean_domain_nll_difference_target_minus_baseline"]["ci95"]
    if not (concat_ci[0] > 0 and concat_ci[1] > 0):
        fail("Concat 比较不再支持已发布边界")

    print(json.dumps({"validated": True, "g6_status": decision["g6_status"], "scenarios": 12}, sort_keys=True))


if __name__ == "__main__":
    main()
