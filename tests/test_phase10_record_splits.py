from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_inputs(tmp_path: Path):
    data_root = tmp_path / "data"
    domains = {}
    for domain in ("general", "math", "code", "instruction"):
        relative = f"source/{domain}.jsonl"
        path = data_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for index in range(240):
                text = "shared exact record" if index == 0 else f"{domain} record {index} unique payload"
                handle.write(json.dumps({"id": index, "text": text}) + "\n")
        domains[domain] = {
            "source": f"test/{domain}",
            "revision": "deadbeef",
            "license": "test",
            "allocation": {
                "path": relative,
                "format": "jsonl",
                "id_field": "id",
                "template": "{text}",
            },
            "held_out": [],
        }
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"schema_version": 1, "registry_id": "test", "domains": domains}))
    experiment = tmp_path / "experiment.json"
    experiment.write_text(json.dumps({
        "experiment_id": "test",
        "data_protocol": {
            "split_salt": "fixed-test-salt",
            "record_fractions": {
                "calibration-a": 0.3,
                "calibration-b": 0.1,
                "validation": 0.2,
                "test": 0.4,
            },
        },
    }))
    return data_root, registry, experiment


def run_builder(data_root: Path, registry: Path, experiment: Path, output: Path):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [
            sys.executable,
            "scripts/phase10/build_record_splits.py",
            "--registry",
            str(registry),
            "--data-root",
            str(data_root),
            "--output-root",
            str(output),
            "--experiment",
            str(experiment),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def test_record_splits_are_deterministic_and_pairwise_disjoint(tmp_path: Path):
    data_root, registry, experiment = write_inputs(tmp_path)
    output = data_root / "phase10"
    run_builder(data_root, registry, experiment, output)
    first = (output / "split-manifest.json").read_bytes()
    run_builder(data_root, registry, experiment, output)
    assert (output / "split-manifest.json").read_bytes() == first

    result = subprocess.run(
        [
            sys.executable,
            "scripts/phase10/verify_record_splits.py",
            "--manifest",
            str(output / "split-manifest.json"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")},
    )
    assert '"pairwise_record_overlap": 0' in result.stdout


def test_record_splits_deduplicate_normalized_text(tmp_path: Path):
    data_root, registry, experiment = write_inputs(tmp_path)
    source = data_root / "source" / "general.jsonl"
    with source.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": 999, "text": "  GENERAL   RECORD 1 UNIQUE PAYLOAD "}) + "\n")
    output = data_root / "phase10"
    run_builder(data_root, registry, experiment, output)
    manifest = json.loads((output / "split-manifest.json").read_text())
    assert manifest["source"]["general"]["duplicates_removed"] == 1
