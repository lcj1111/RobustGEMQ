#!/usr/bin/env python3
"""离线验证阶段六完整产物是否满足冻结的发布条件。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


DOMAINS = ("code", "general", "instruction", "math")
SEEDS = (0, 1, 2)
METHODS = ("concat", "domain-mean", "gemq-c4")
ALL_METHODS = ["gemq-c4", "concat", "domain-mean", "alphaq-style"]
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def require_sha256(value: object, context: str) -> str:
    text = str(value)
    if SHA256_RE.fullmatch(text) is None:
        raise ValueError(f"{context}: expected a lowercase SHA-256 digest")
    return text


def identity_sha256(rows: list[tuple[str, int, int, str]]) -> str:
    canonical = json.dumps(sorted(rows), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_items(items: object, method: str, scenario_hashes: dict[tuple[str, int], str]) -> tuple[list, dict]:
    if not isinstance(items, list):
        raise ValueError(f"{method}: items must be a list")
    expected_keys = {(domain, seed, item) for domain in DOMAINS for seed in SEEDS for item in range(128)}
    observed: dict[tuple[str, int, int], tuple[str, float]] = {}
    for position, record in enumerate(items):
        if not isinstance(record, dict):
            raise ValueError(f"{method}: item {position} is not an object")
        missing = {"domain", "seed", "item", "nll", "token_sha256"} - set(record)
        if missing:
            raise ValueError(f"{method}: item {position} missing fields {sorted(missing)}")
        key = (str(record["domain"]), int(record["seed"]), int(record["item"]))
        if key in observed:
            raise ValueError(f"{method}: duplicate item identity {key}")
        token_hash = require_sha256(record["token_sha256"], f"{method}:{key}")
        expected_hash = scenario_hashes.get((key[0], key[1]))
        if expected_hash is None or token_hash != expected_hash:
            raise ValueError(f"{method}: token identity mismatch for {key}")
        nll = float(record["nll"])
        if not math.isfinite(nll):
            raise ValueError(f"{method}: non-finite NLL for {key}")
        observed[key] = (token_hash, nll)
    if set(observed) != expected_keys:
        missing = expected_keys - set(observed)
        extra = set(observed) - expected_keys
        raise ValueError(f"{method}: incomplete item grid; missing={len(missing)}, extra={len(extra)}")

    identities = [(domain, seed, item, observed[(domain, seed, item)][0]) for domain, seed, item in sorted(observed)]
    domain_nll = {
        domain: sum(observed[(domain, seed, item)][1] for seed in SEEDS for item in range(128)) / (len(SEEDS) * 128)
        for domain in DOMAINS
    }
    mean_nll = sum(domain_nll.values()) / len(DOMAINS)
    worst_domain = max(DOMAINS, key=domain_nll.__getitem__)
    return identities, {
        "domain_nll": domain_nll,
        "mean_domain_nll": mean_nll,
        "worst_domain": worst_domain,
        "worst_domain_nll": domain_nll[worst_domain],
    }


def assert_close(observed: object, expected: float, context: str) -> None:
    if not math.isclose(float(observed), expected, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"{context}: {observed} != recomputed {expected}")


def validate_h6(root: Path, method: str) -> str:
    summary_path = root / "h6-validation" / method / "summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        if summary.get("schema_version") != 1 or summary.get("method") != method:
            raise ValueError(f"{method}: invalid H6 summary identity")
        if summary.get("passed") is not True or int(summary.get("exit_code", -1)) != 0:
            raise ValueError(f"{method}: H6 did not pass")
        checks = summary.get("required_checks", {})
        if float(checks.get("decode_argmax_agreement_min", 0)) < 0.95:
            raise ValueError(f"{method}: H6 summary lacks the 95% argmax contract")
        return "structured-json"
    # 兼容已完成但尚未重跑的历史产物；新 runner 一律生成 summary.json。
    legacy = root / "h6-validation" / method / "status.txt"
    if not legacy.is_file() or "exit_code=0" not in legacy.read_text(encoding="utf-8"):
        raise ValueError(f"{method}: H6 evidence is missing")
    return "legacy-status"


def pareto_status(metrics: dict) -> bool:
    target = metrics["domain-mean"]
    return not any(
        metrics[baseline]["mean_domain_nll"] <= target["mean_domain_nll"]
        and metrics[baseline]["worst_domain_nll"] <= target["worst_domain_nll"]
        and (
            metrics[baseline]["mean_domain_nll"] < target["mean_domain_nll"]
            or metrics[baseline]["worst_domain_nll"] < target["worst_domain_nll"]
        )
        for baseline in ("concat", "gemq-c4")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_root
    config = read_json(root / "configs" / "bpe-2.5" / "manifest.json")
    bootstrap = read_json(root / "item-bootstrap" / "bootstrap.json")
    decision = read_json(root / "phase6_decision.json")
    if config["methods"] != ALL_METHODS:
        raise ValueError("unexpected frozen Phase 6 method set")
    if config["budget"] != 2560.0 or config["bpe"] != 2.5:
        raise ValueError("Phase 6 must retain the matched 2.5-bpe budget")
    if bootstrap["draws"] < 10000:
        raise ValueError("item bootstrap must have at least 10,000 draws")

    source = config.get("source_scenarios", {})
    expected_scenarios = {f"{domain}:seed-{seed}" for domain in DOMAINS for seed in SEEDS}
    if set(source) != expected_scenarios:
        raise ValueError("frozen 12-scenario manifest changed")
    scenario_hashes = {
        (domain, seed): require_sha256(source[f"{domain}:seed-{seed}"]["token_sha256"], f"{domain}:seed-{seed}")
        for domain in DOMAINS
        for seed in SEEDS
    }

    common_identities = None
    recomputed_metrics = {}
    h6_formats = {}
    for method in METHODS:
        payload = read_json(root / "item-bootstrap" / method / "item-nll.json")
        identities, metrics = validate_items(payload.get("items"), method, scenario_hashes)
        if common_identities is None:
            common_identities = identities
        elif identities != common_identities:
            raise ValueError(f"{method}: item identities differ across methods")
        recomputed_metrics[method] = metrics
        h6_formats[method] = validate_h6(root, method)

    published_metrics = bootstrap.get("point_metrics", decision.get("point_metrics", {}))
    if decision.get("point_metrics") != published_metrics:
        raise ValueError("Gate decision metrics differ from bootstrap point metrics")
    for method in METHODS:
        if method not in published_metrics:
            raise ValueError(f"{method}: point metrics missing")
        for domain in DOMAINS:
            assert_close(published_metrics[method]["domain_nll"][domain], recomputed_metrics[method]["domain_nll"][domain], f"{method}:{domain}")
        assert_close(published_metrics[method]["mean_domain_nll"], recomputed_metrics[method]["mean_domain_nll"], f"{method}:mean")
        assert_close(published_metrics[method]["worst_domain_nll"], recomputed_metrics[method]["worst_domain_nll"], f"{method}:worst")
        if published_metrics[method]["worst_domain"] != recomputed_metrics[method]["worst_domain"]:
            raise ValueError(f"{method}: worst-domain label mismatch")

    recomputed_pareto = pareto_status(recomputed_metrics)
    if bool(decision["domain_mean_on_matched_budget_mean_worst_pareto"]) != recomputed_pareto:
        raise ValueError("G6 Pareto flag does not match item-level metrics")
    if decision["g6_status"] != "STOP_NO_LARGE_MODEL_EXPANSION":
        raise ValueError("unexpected G6 result; inspect the frozen gate implementation")
    scope = bootstrap.get("inference_scope")
    if scope is not None and "not an independent" not in scope:
        raise ValueError("bootstrap inference scope is ambiguous")

    digest = identity_sha256(common_identities or [])
    summary = {
        "schema_version": 2,
        "verified": True,
        "phase": 6,
        "methods_with_real_checkpoints": list(METHODS),
        "item_nlls_per_method": len(common_identities or []),
        "item_identity_sha256": digest,
        "cross_method_item_identity_match": True,
        "bootstrap_draws": bootstrap["draws"],
        "bootstrap_scope": "descriptive; fixed Phase 6 training scenarios; not an independent test set",
        "h6_passed": decision["h6_pass"],
        "h6_evidence_format": h6_formats,
        "g6_status": decision["g6_status"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
