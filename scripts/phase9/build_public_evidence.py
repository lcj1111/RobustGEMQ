#!/usr/bin/env python3
"""Create a compact, publishable Phase 6 evidence bundle from private artifacts.

The input directory may contain large checkpoints and per-item scores.  The
output intentionally contains only decision-relevant metrics, reproducibility
hashes and validation status, so it is suitable for version control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_root
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
    source_scenarios = config["source_scenarios"]
    domains = sorted({name.split(":", 1)[0] for name in source_scenarios})
    seeds = sorted({int(name.split("seed-", 1)[1]) for name in source_scenarios})
    evidence = {
        "schema_version": 1,
        "project": "RobustGEMQ",
        "phase": 6,
        "model": "allenai/OLMoE-1B-7B-0924",
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
            "method_hamming_fraction": config["method_hamming_fraction"],
            "config_sha256": {
                method: entry["config_sha256"]
                for method, entry in config["methods_manifest"].items()
            },
        },
        "scenario_provenance": {
            name: {
                "token_sha256": entry["token_sha256"],
                "layer_re_sha256": entry["layer_re_sha256"],
            }
            for name, entry in sorted(source_scenarios.items())
        },
        "real_checkpoint_validation": {
            "methods": release["methods_with_real_checkpoints"],
            "item_nlls_per_method": release["item_nlls_per_method"],
            "h6_passed": release["h6_passed"],
        },
        "paired_bootstrap": {
            "method": bootstrap["method"],
            "draws": bootstrap["draws"],
            "seed": bootstrap["seed"],
            "comparisons_target_domain_mean": bootstrap["comparisons_target_domain_mean"],
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
        "source_file_sha256": {name: sha256(path) for name, path in paths.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "scenarios": len(source_scenarios)}, sort_keys=True))


if __name__ == "__main__":
    main()
