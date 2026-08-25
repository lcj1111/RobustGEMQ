from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("gemq-c4", "layer-balanced", "usage-only", "concat", "domain-mean")
DOMAINS = ("general", "math", "code", "instruction")


def write_screen(root: Path):
    offsets = {
        "gemq-c4": 0.20,
        "layer-balanced": 0.30,
        "usage-only": 0.10,
        "concat": 0.05,
        "domain-mean": 0.15,
    }
    for method in METHODS:
        items = []
        for domain_index, domain in enumerate(DOMAINS):
            for item in range(48):
                items.append({
                    "method": method,
                    "checkpoint_seed": 101,
                    "split": "validation",
                    "domain": domain,
                    "scenario_seed": 0,
                    "item": item,
                    "nll": 1.0 + offsets[method] + domain_index * (0.5 if method == "concat" else 0.1),
                    "scenario_token_sha256": f"{domain_index + 1:064x}",
                    "item_token_sha256": f"{domain_index * 64 + item + 1:064x}",
                })
        directory = root / method
        directory.mkdir(parents=True)
        (directory / "validation-items.json").write_text(json.dumps({
            "schema_version": 2,
            "method": method,
            "checkpoint_seed": 101,
            "split": "validation",
            "items": items,
        }))


def test_selection_uses_only_frozen_validation_rule(tmp_path: Path):
    root = tmp_path / "screen"
    write_screen(root)
    config = tmp_path / "manifest.json"
    config.write_text("{}")
    output = tmp_path / "selection.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/phase10/select_validation_methods.py",
            "--root",
            str(root),
            "--config-manifest",
            str(config),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )
    decision = json.loads(output.read_text())
    assert decision["selected_methods"] == ["gemq-c4", "usage-only", "domain-mean"]
    assert decision["test_opened"] is False
    assert len(decision["selection_sha256"]) == 64
    assert decision["cross_method_item_identity_match"] is True


def test_selection_rejects_cross_method_item_mismatch(tmp_path: Path):
    root = tmp_path / "screen"
    write_screen(root)
    path = root / "concat" / "validation-items.json"
    payload = json.loads(path.read_text())
    payload["items"][0]["item_token_sha256"] = "f" * 64
    path.write_text(json.dumps(payload))
    config = tmp_path / "manifest.json"
    config.write_text("{}")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/phase10/select_validation_methods.py",
            "--root",
            str(root),
            "--config-manifest",
            str(config),
            "--output",
            str(tmp_path / "selection.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "identity differs" in result.stderr
