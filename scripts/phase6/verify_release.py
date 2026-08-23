#!/usr/bin/env python3
"""离线验证阶段六完整产物是否满足冻结的发布条件。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


METHODS = ("concat", "domain-mean", "gemq-c4")


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_root
    config = read_json(root / "configs" / "bpe-2.5" / "manifest.json")
    bootstrap = read_json(root / "item-bootstrap" / "bootstrap.json")
    decision = read_json(root / "phase6_decision.json")
    if config["methods"] != ["gemq-c4", "concat", "domain-mean", "alphaq-style"]:
        raise ValueError("unexpected frozen Phase 6 method set")
    if config["budget"] != 2560.0 or config["bpe"] != 2.5:
        raise ValueError("Phase 6 must retain the matched 2.5-bpe budget")
    if bootstrap["draws"] < 10000:
        raise ValueError("item bootstrap must have at least 10,000 draws")
    # 4 域 × 3 种子 × 128 条样本是预先冻结的评测网格；数量变化即不再可比。
    for method in METHODS:
        item_path = root / "item-bootstrap" / method / "item-nll.json"
        items = read_json(item_path)["items"]
        if len(items) != 4 * 3 * 128:
            raise ValueError(f"{method}: expected 1536 item NLLs, got {len(items)}")
        if any(not math.isfinite(float(item["nll"])) for item in items):
            raise ValueError(f"{method}: non-finite item NLL")
        # 只有真实打包路径通过 H6，逐样本指标才能用于最终 Gate。
        status = (root / "h6-validation" / method / "status.txt").read_text(encoding="utf-8")
        if "exit_code=0" not in status:
            raise ValueError(f"{method}: H6 did not pass")
    if decision["g6_status"] != "STOP_NO_LARGE_MODEL_EXPANSION":
        raise ValueError("unexpected G6 result; inspect the frozen gate implementation")
    summary = {
        "verified": True,
        "phase": 6,
        "methods_with_real_checkpoints": list(METHODS),
        "item_nlls_per_method": 1536,
        "bootstrap_draws": bootstrap["draws"],
        "h6_passed": decision["h6_pass"],
        "g6_status": decision["g6_status"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
