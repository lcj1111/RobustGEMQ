#!/usr/bin/env python3
"""只有方法选择冻结且 3×3 个 H6 全通过后才解锁 test。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from scripts.phase10.select_validation_methods import selection_hash
except ModuleNotFoundError:
    from select_validation_methods import selection_hash


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--h6-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    recorded_hash = selection.pop("selection_sha256")
    if selection_hash(selection) != recorded_hash or selection.get("test_opened") is not False:
        raise ValueError("selection contract changed before test unlock")
    methods = selection["selected_methods"]
    if len(methods) != 3:
        raise ValueError("exactly three selected methods are required")
    h6 = {}
    for method in methods:
        h6[method] = {}
        for seed in (101, 202, 303):
            path = args.h6_root / method / f"seed-{seed}" / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary.get("passed") is not True or summary.get("required_checks", {}).get("decode_argmax_agreement_min") != 0.95:
                raise ValueError(f"H6 failed or is incomplete: {method}/seed-{seed}")
            h6[method][str(seed)] = sha256(path)
    unlock = {
        "schema_version": 1,
        "test_unlocked": True,
        "selection_sha256": recorded_hash,
        "selection_file_sha256": sha256(args.selection),
        "selected_methods": methods,
        "checkpoint_seeds": [101, 202, 303],
        "h6_summary_sha256": h6,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(unlock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(unlock, sort_keys=True))


if __name__ == "__main__":
    main()
