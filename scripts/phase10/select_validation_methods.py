#!/usr/bin/env python3
"""只根据独立 validation 冻结三个进入正式 test 的方法。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


DOMAINS = ("general", "math", "code", "instruction")
METHODS = ("gemq-c4", "layer-balanced", "usage-only", "concat", "domain-mean")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_metrics(path: Path, method: str) -> tuple[dict, list[tuple]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2 or payload.get("method") != method:
        raise ValueError(f"{method}: invalid validation identity")
    if payload.get("checkpoint_seed") != 101 or payload.get("split") != "validation":
        raise ValueError(f"{method}: screening must use seed 101 validation only")
    observed = {}
    for record in payload["items"]:
        key = (record["domain"], int(record["item"]))
        if key in observed:
            raise ValueError(f"{method}: duplicate validation item {key}")
        identity = (
            record["domain"],
            int(record["scenario_seed"]),
            int(record["item"]),
            record["scenario_token_sha256"],
            record["item_token_sha256"],
        )
        nll = float(record["nll"])
        if not math.isfinite(nll):
            raise ValueError(f"{method}: non-finite validation NLL")
        observed[key] = (identity, nll)
    expected = {(domain, item) for domain in DOMAINS for item in range(48)}
    if set(observed) != expected:
        raise ValueError(f"{method}: validation grid is incomplete")
    identities = [observed[key][0] for key in sorted(observed)]
    domain_nll = {
        domain: sum(observed[(domain, item)][1] for item in range(48)) / 48
        for domain in DOMAINS
    }
    return {
        "domain_nll": domain_nll,
        "mean_domain_nll": sum(domain_nll.values()) / 4,
        "worst_domain_nll": max(domain_nll.values()),
        "worst_domain": max(DOMAINS, key=domain_nll.__getitem__),
    }, identities


def selection_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = {}
    source_hashes = {}
    common_identity = None
    for method in METHODS:
        path = args.root / method / "validation-items.json"
        metrics[method], identities = load_metrics(path, method)
        if common_identity is None:
            common_identity = identities
        elif identities != common_identity:
            raise ValueError(f"{method}: validation item identity differs across methods")
        source_hashes[method] = sha256(path)

    selected = ["gemq-c4"]
    mean_winner = min(
        (method for method in METHODS if method != "gemq-c4"),
        key=lambda method: (metrics[method]["mean_domain_nll"], method),
    )
    selected.append(mean_winner)
    worst_winner = min(
        (method for method in METHODS if method not in selected),
        key=lambda method: (metrics[method]["worst_domain_nll"], method),
    )
    selected.append(worst_winner)
    identity_digest = hashlib.sha256(
        json.dumps(common_identity, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    decision = {
        "schema_version": 1,
        "experiment_id": "robustgemq-independent-study-v1",
        "selection_split": "validation",
        "screen_checkpoint_seed": 101,
        "methods_compared": list(METHODS),
        "selection_rule": [
            "retain gemq-c4",
            "lowest validation mean-domain NLL among non-GEMQ methods",
            "lowest validation worst-domain NLL among remaining methods",
        ],
        "selected_methods": selected,
        "metrics": metrics,
        "cross_method_item_identity_match": True,
        "validation_item_identity_sha256": identity_digest,
        "source_file_sha256": source_hashes,
        "config_manifest_sha256": sha256(args.config_manifest),
        "test_opened": False,
    }
    decision["selection_sha256"] = selection_hash(decision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"selected_methods": selected, "selection_sha256": decision["selection_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
