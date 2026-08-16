#!/usr/bin/env python3
"""Build actual fake-quant NLL transfer matrices for the Phase 2 H2 gate."""

import argparse
import json
from pathlib import Path


DOMAINS = ("general", "math", "code", "instruction")


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def domain_mean(summary: dict, domain: str, metric: str = "nll") -> float:
    return sum(summary["scenarios"][f"{domain}:seed-{seed}"][metric] for seed in (0, 1)) / 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fp = load_summary(args.root / "fp" / "summary.json")

    result = {"schema_version": 1, "quantizer": "fake RTN blocksize=128", "budgets": {}}
    all_regrets = []
    for bpe in (2.5, 2.0):
        summaries = {
            source: load_summary(args.root / f"{source}-bpe-{bpe:.1f}" / "summary.json")
            for source in DOMAINS
        }
        matrix = []
        for source in DOMAINS:
            targets = {}
            for target in DOMAINS:
                nll = domain_mean(summaries[source], target)
                target_config_nll = domain_mean(summaries[target], target)
                fp_nll = domain_mean(fp, target)
                transfer_regret = nll - target_config_nll
                all_regrets.append(transfer_regret)
                targets[target] = {
                    "nll": nll,
                    "ppl": domain_mean(summaries[source], target, "ppl"),
                    "delta_vs_fp": nll - fp_nll,
                    "transfer_regret_vs_target_config": transfer_regret,
                }
            matrix.append({"source": source, "targets": targets})
        result["budgets"][f"{bpe:.1f}"] = {"matrix": matrix}

    maximum = max(all_regrets)
    result["summary"] = {
        "max_transfer_nll_regret": maximum,
        "h2_threshold": 0.10,
        "h2_fake_rtn_gate_pass": maximum >= 0.10,
        "scope": "pilot screening only; Phase 3/6 must repeat selected methods with the frozen GPTQ/RFT protocol",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
