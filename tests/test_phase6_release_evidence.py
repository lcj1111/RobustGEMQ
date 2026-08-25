from __future__ import annotations

import pytest

from scripts.phase6.verify_release import DOMAINS, SEEDS, identity_sha256, validate_items


def fixture_items():
    scenario_keys = [(domain, seed) for domain in DOMAINS for seed in SEEDS]
    hashes = {key: f"{index:064x}" for index, key in enumerate(scenario_keys, start=1)}
    items = [
        {
            "domain": domain,
            "seed": seed,
            "item": item,
            "nll": 1.0 + item / 1000,
            "token_sha256": hashes[(domain, seed)],
        }
        for domain in DOMAINS
        for seed in SEEDS
        for item in range(128)
    ]
    return items, hashes


def test_item_contract_is_order_independent_and_complete():
    items, hashes = fixture_items()
    left, metrics = validate_items(items, "concat", hashes)
    right, _ = validate_items(list(reversed(items)), "domain-mean", hashes)
    assert left == right
    assert identity_sha256(left) == identity_sha256(right)
    assert len(left) == 1536
    assert set(metrics["domain_nll"]) == set(DOMAINS)


def test_item_contract_rejects_duplicate_primary_key():
    items, hashes = fixture_items()
    items[-1] = dict(items[0])
    with pytest.raises(ValueError, match="duplicate item identity"):
        validate_items(items, "concat", hashes)


def test_item_contract_rejects_cross_scenario_token_hash():
    items, hashes = fixture_items()
    items[0]["token_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="token identity mismatch"):
        validate_items(items, "concat", hashes)
