#!/usr/bin/env python3
"""验证四路 token 场景与记录划分契约一致。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


DOMAINS = ("general", "math", "code", "instruction")


def token_hash(tokens: torch.Tensor) -> str:
    payload = tokens.to(dtype=torch.int64, device="cpu").contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--scenario-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    experiment = json.loads(args.experiment.read_text(encoding="utf-8"))
    split_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    protocol = experiment["data_protocol"]
    specs = [
        ("calibration-a", seed, protocol["calibration_a"]["sequences_per_domain_seed"])
        for seed in protocol["calibration_a"]["scenario_seeds"]
    ] + [
        ("calibration-b", 0, protocol["calibration_b"]["sequences_per_domain"]),
        ("validation", 0, protocol["validation"]["sequences_per_domain"]),
        ("test", 0, protocol["test"]["sequences_per_domain"]),
    ]
    records = []
    token_hashes = set()
    for split, seed, samples in specs:
        registry_key = f"calibration-a-seed-{seed}" if split == "calibration-a" else split
        registry_path = Path(split_manifest["registries"][registry_key]).resolve()
        registry_hash = hashlib.sha256(registry_path.read_bytes()).hexdigest()
        for domain in DOMAINS:
            directory = args.scenario_root / split / domain / f"seed-{seed}"
            scenario = json.loads((directory / "scenario.json").read_text(encoding="utf-8"))
            tokens = torch.load(scenario["tokens_path"], map_location="cpu", weights_only=True)
            if tuple(tokens.shape) != (samples, 2048):
                raise ValueError(f"{directory}: unexpected token shape {tuple(tokens.shape)}")
            observed = token_hash(tokens)
            if observed != scenario["token_sha256"]:
                raise ValueError(f"{directory}: token hash mismatch")
            if scenario["registry_sha256"] != registry_hash:
                raise ValueError(f"{directory}: registry identity mismatch")
            if observed in token_hashes:
                raise ValueError(f"{directory}: duplicate token scenario")
            token_hashes.add(observed)
            records.append({
                "split": split,
                "domain": domain,
                "seed": seed,
                "shape": list(tokens.shape),
                "token_sha256": observed,
                "selected_ids_sha256": scenario["selected_ids_sha256"],
                "allocation_sha256": scenario["allocation_sha256"],
            })
    summary = {
        "schema_version": 1,
        "verified": True,
        "record_split_overlap": split_manifest["pairwise_record_overlap"],
        "scenario_count": len(records),
        "unique_token_scenarios": len(token_hashes),
        "scenarios": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verified": True, "scenarios": len(records)}, sort_keys=True))


if __name__ == "__main__":
    main()
