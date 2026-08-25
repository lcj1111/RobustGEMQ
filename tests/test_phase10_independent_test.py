from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.phase10.select_validation_methods import selection_hash


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("gemq-c4", "usage-only", "domain-mean")
SEEDS = (101, 202, 303)
DOMAINS = ("general", "math", "code", "instruction")


def write_unlock_inputs(tmp_path: Path):
    selection = {
        "schema_version": 1,
        "selected_methods": list(METHODS),
        "test_opened": False,
    }
    selection["selection_sha256"] = selection_hash(selection)
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection))
    h6_root = tmp_path / "h6"
    for method in METHODS:
        for seed in SEEDS:
            path = h6_root / method / f"seed-{seed}" / "summary.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({
                "passed": True,
                "required_checks": {"decode_argmax_agreement_min": 0.95},
            }))
    return selection_path, h6_root


def test_unlock_and_independent_statistics_contract(tmp_path: Path):
    selection, h6_root = write_unlock_inputs(tmp_path)
    unlock = tmp_path / "unlock.json"
    subprocess.run(
        [sys.executable, "scripts/phase10/unlock_test.py", "--selection", str(selection), "--h6-root", str(h6_root), "--output", str(unlock)],
        cwd=ROOT,
        check=True,
    )
    assert json.loads(unlock.read_text())["test_unlocked"] is True

    item_root = tmp_path / "items"
    offsets = {"gemq-c4": 0.2, "usage-only": 0.1, "domain-mean": 0.15}
    for method in METHODS:
        for seed_index, seed in enumerate(SEEDS):
            items = []
            for domain_index, domain in enumerate(DOMAINS):
                for item in range(96):
                    items.append({
                        "domain": domain,
                        "scenario_seed": 0,
                        "item": item,
                        "nll": 1 + offsets[method] + seed_index * 0.01 + domain_index * 0.1 + item / 10000,
                        "scenario_token_sha256": f"{domain_index + 1:064x}",
                        "item_token_sha256": f"{domain_index * 96 + item + 1:064x}",
                    })
            path = item_root / method / f"seed-{seed}" / "test-items.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({
                "schema_version": 2,
                "method": method,
                "checkpoint_seed": seed,
                "split": "test",
                "items": items,
            }))
    output = tmp_path / "statistics.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/phase10/analyze_independent_test.py",
            "--root",
            str(item_root),
            "--unlock",
            str(unlock),
            "--output",
            str(output),
            "--draws",
            "100",
        ],
        cwd=ROOT,
        check=True,
    )
    result = json.loads(output.read_text())
    assert result["cross_method_and_checkpoint_item_identity_match"] is True
    assert result["items_per_checkpoint"] == 384
    assert result["checkpoint_variance"]["gemq-c4"]["mean_domain_nll_sample_variance"] > 0
    comparison = result["paired_item_bootstrap"]["comparisons"]["gemq-c4__minus__usage-only"]
    assert comparison["mean_domain_nll_left_minus_right"]["point_difference"] > 0
