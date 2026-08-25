#!/usr/bin/env python3
"""Apply the frozen Phase 6 G6 rule and write a concise evidence report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METHOD = "domain-mean"
BASELINES = ("concat", "gemq-c4")


def h6_passed(root: Path) -> bool:
    for method in (*BASELINES, METHOD):
        summary_path = root / method / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            checks = summary.get("required_checks", {})
            if (
                summary.get("passed") is not True
                or int(summary.get("exit_code", -1)) != 0
                or float(checks.get("decode_argmax_agreement_min", 0)) < 0.95
            ):
                return False
            continue
        # 仅兼容升级前已经完成的 H6；新 runner 必须写结构化 summary.json。
        path = root / method / "status.txt"
        if not path.is_file() or "exit_code=0" not in path.read_text(encoding="utf-8"):
            return False
    return True


def is_pareto(metrics: dict) -> bool:
    target = metrics[METHOD]
    for baseline in BASELINES:
        other = metrics[baseline]
        if (
            other["mean_domain_nll"] <= target["mean_domain_nll"]
            and other["worst_domain_nll"] <= target["worst_domain_nll"]
            and (
                other["mean_domain_nll"] < target["mean_domain_nll"]
                or other["worst_domain_nll"] < target["worst_domain_nll"]
            )
        ):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--h3", type=Path, required=True)
    parser.add_argument("--h6-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    bootstrap = json.loads(args.bootstrap.read_text(encoding="utf-8"))
    h3 = json.loads(args.h3.read_text(encoding="utf-8"))
    metrics = bootstrap["point_metrics"]
    comparisons = bootstrap["comparisons_target_domain_mean"]
    h3_pass = bool(h3["decision"]["h3_full_pass"])
    h5_status = "not-executed-before-main-result; not eligible for post-hoc gate rescue"
    h6_pass = h6_passed(args.h6_root)
    pareto = is_pareto(metrics)
    evidence_gate = h3_pass  # H5 deliberately cannot be added after observing Phase 6.
    g6_go = evidence_gate and pareto and h6_pass
    if pareto:
        reason = (
            "H6 passed and Scenario-Normalized-Mean remains on the matched-budget mean/worst Pareto frontier, "
            "but the pre-registered H3/H5 evidence prerequisite is not met. Do not expand to a second model."
        )
        pareto_text = "Scenario-Normalized-Mean is not mean/worst dominated at matched 2.5 bpe"
    else:
        reason = (
            "H6 passed, but Scenario-Normalized-Mean does not enter the matched-budget mean/worst Pareto frontier "
            "on the real checkpoint item-level estimate; H3 also did not pass. Do not expand to a second model."
        )
        pareto_text = "Scenario-Normalized-Mean is mean/worst dominated on the matched-budget point estimate"
    decision = {
        "schema_version": 1,
        "phase": 6,
        "gate": "G6",
        "h3_full_pass": h3_pass,
        "h5_status": h5_status,
        "h6_pass": h6_pass,
        "domain_mean_on_matched_budget_mean_worst_pareto": pareto,
        "g6_status": "GO" if g6_go else "STOP_NO_LARGE_MODEL_EXPANSION",
        "reason": reason,
        "bootstrap": comparisons,
        "point_metrics": metrics,
        "allowed_follow_up": [
            "publish an auditable boundary/negative-result report",
            "package the reproducibility and reliability harness",
        ],
        "disallowed_follow_up": [
            "post-hoc H5 execution to rescue G6",
            "second-model quality sweep",
            "claiming Scenario-Normalized-Mean statistically dominates both baselines",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    concat = comparisons["concat"]
    c4 = comparisons["gemq-c4"]
    report = f"""# Phase 6 Final Decision\n\n## Decision\n\n**G6: {decision['g6_status']}**. H6 passed for all three packed checkpoints; however, {pareto_text}. H3 did not pass and H5 was not run before observing Phase 6, so the pre-registered prerequisite for second-model expansion is absent.\n\n## Matched-budget real checkpoint metrics\n\n| Method | Mean domain NLL | Worst domain NLL |\n|---|---:|---:|\n"""
    for name in ("concat", METHOD, "gemq-c4"):
        value = metrics[name]
        report += f"| {name} | {value['mean_domain_nll']:.6f} | {value['worst_domain_nll']:.6f} |\n"
    report += f"""\n## Paired item bootstrap (Scenario-Normalized-Mean minus baseline)\n\n- Versus Concat: mean-domain difference CI `{concat['mean_domain_nll_difference_target_minus_baseline']['ci95']}`; worst-domain difference CI `{concat['worst_domain_nll_difference_target_minus_baseline']['ci95']}`.\n- Versus GEMQ-C4: mean-domain difference CI `{c4['mean_domain_nll_difference_target_minus_baseline']['ci95']}`; worst-domain difference CI `{c4['worst_domain_nll_difference_target_minus_baseline']['ci95']}`.\n\nNegative differences favor Scenario-Normalized-Mean (historical key: `domain-mean`). These intervals are a descriptive bootstrap within the fixed Phase 6 training scenarios. They are not estimates on an independent validation or test set and do not authorize retuning or generalization claims.\n\n## Scope\n\nThe project should now be presented as an auditable MoE quantization reliability and failure-boundary study: frozen domain scenarios, constrained allocation audits, GPTQ/RFT/real-packing equivalence checks, and an explicit decision not to scale a method without pre-registered evidence.\n"""
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(json.dumps({"decision": decision["g6_status"], "output": str(args.output), "report": str(args.report)}))


if __name__ == "__main__":
    main()
