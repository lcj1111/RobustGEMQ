#!/usr/bin/env python3
"""Apply the frozen H3 held-out quality gate and select Phase 6 objectives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DOMAINS = ("general", "math", "code", "instruction")
BASELINES = ("gemq-c4", "concat")
ROBUST = ("domain-mean", "domain-worst", "domain-cvar-0.5")
METHODS = BASELINES + ROBUST + ("alphaq-style",)


def read(root: Path, name: str) -> dict:
    return json.loads((root / name / "summary.json").read_text(encoding="utf-8"))


def aggregate(summary: dict, fp: dict) -> dict:
    scenario_delta = {
        key: values["nll"] - fp["scenarios"][key]["nll"]
        for key, values in summary["scenarios"].items()
    }
    domains = {
        domain: float(np.mean([scenario_delta[f"{domain}:seed-{seed}"] for seed in (0, 1)]))
        for domain in DOMAINS
    }
    values = np.asarray(list(domains.values()))
    return {
        "scenario_nll_delta": scenario_delta,
        "domain_nll_delta": domains,
        "mean_domain_nll_delta": float(values.mean()),
        "worst_domain_nll_delta": float(values.max()),
        "worst_domain": DOMAINS[int(values.argmax())],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fp = read(args.root, "fp")
    result = {
        "schema_version": 1,
        "gate": "H3",
        "protocol": "fake RTN on frozen held-out tokens; selection evidence only",
        "regret_definition": "held-out quantized NLL minus matched full-precision NLL, averaged within domain over two seeds",
        "thresholds": {
            "worst_regret_reduction_vs_best_gemq_or_concat": 0.10,
            "max_mean_nll_increment_vs_best_baseline": 0.02,
            "must_not_exceed_alphaq_worst_domain": True,
        },
        "budgets": {},
    }
    primary_pass = False
    primary_candidates = []
    primary_limited = []
    for bpe in (2.5, 2.0):
        metrics = {
            method: aggregate(read(args.root, f"{method}-bpe-{bpe:.1f}"), fp) for method in METHODS
        }
        best_baseline_worst = min(metrics[name]["worst_domain_nll_delta"] for name in BASELINES)
        best_baseline_mean = min(metrics[name]["mean_domain_nll_delta"] for name in BASELINES)
        alphaq_worst = metrics["alphaq-style"]["worst_domain_nll_delta"]
        candidates = {}
        for name in ROBUST:
            worst = metrics[name]["worst_domain_nll_delta"]
            reduction = (best_baseline_worst - worst) / max(abs(best_baseline_worst), 1e-12)
            mean_increment = metrics[name]["mean_domain_nll_delta"] - best_baseline_mean
            improved_domains = [
                domain
                for domain in DOMAINS
                if metrics[name]["domain_nll_delta"][domain]
                < min(metrics[base]["domain_nll_delta"][domain] for base in BASELINES)
            ]
            passed = (
                reduction >= 0.10
                and mean_increment <= 0.02
                and worst <= alphaq_worst + 1e-12
                and bool(improved_domains)
            )
            candidates[name] = {
                "worst_regret_reduction": reduction,
                "mean_nll_increment": mean_increment,
                "no_worse_than_alphaq_worst": worst <= alphaq_worst + 1e-12,
                "directionally_improved_domains": improved_domains,
                "passed": passed,
                "limited_directional_evidence": (
                    mean_increment <= 0.02
                    and worst <= alphaq_worst + 1e-12
                    and bool(improved_domains)
                ),
            }
        passing = [name for name in ROBUST if candidates[name]["passed"]]
        selected = sorted(
            passing,
            key=lambda name: (
                metrics[name]["worst_domain_nll_delta"],
                metrics[name]["mean_domain_nll_delta"],
            ),
        )[:2]
        result["budgets"][f"{bpe:.1f}"] = {
            "metrics": metrics,
            "best_baseline_worst": best_baseline_worst,
            "best_baseline_mean": best_baseline_mean,
            "candidate_gates": candidates,
            "passing_robust_methods": passing,
            "selected_robust_methods": selected,
        }
        if bpe == 2.5:
            primary_pass = bool(passing)
            primary_candidates = selected
            primary_limited = sorted(
                [name for name in ROBUST if candidates[name]["limited_directional_evidence"]],
                key=lambda name: (
                    metrics[name]["worst_domain_nll_delta"],
                    metrics[name]["mean_domain_nll_delta"],
                ),
            )[:1]

    if primary_pass:
        retained_robust = primary_candidates[:1]
        g3_status = "GO"
    elif primary_limited:
        # The frozen plan requires a narrow pivot: retain only the best robust
        # objective and do not add a CVaR grid when evidence is directional/stability
        # only rather than a worst-domain quality win.
        retained_robust = primary_limited
        g3_status = "PIVOT"
    else:
        retained_robust = []
        g3_status = "STOP"
    selected_for_phase6 = list(BASELINES) + retained_robust + ["alphaq-style"]
    result["decision"] = {
        "h3_full_pass": primary_pass,
        "g3_status": g3_status,
        "primary_budget": 2.5,
        "retained_robust_objectives": retained_robust,
        "selected_for_phase6": selected_for_phase6,
        "additional_cvar_grid_authorized": bool(primary_pass and "domain-cvar-0.5" in retained_robust),
        "max_methods_respected": len(selected_for_phase6) <= 4,
        "restriction": "Must be re-tested with frozen GPTQ/RFT and real checkpoints before final claims.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
