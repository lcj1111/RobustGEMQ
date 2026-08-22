#!/usr/bin/env python3
"""Validate one complete OLMoE LayerRE tensor and emit a compact summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.path.open("rb") as handle:
        layer_re = pickle.load(handle)
    if set(layer_re) != set(range(16)):
        raise ValueError(f"expected layers 0..15, got {sorted(layer_re)}")
    values = []
    for layer, experts in layer_re.items():
        if set(experts) != set(range(64)):
            raise ValueError(f"layer {layer}: expected experts 0..63")
        for expert, costs in experts.items():
            if set(costs) != {1, 2, 3}:
                raise ValueError(f"layer {layer} expert {expert}: expected bits 1,2,3")
            values.extend(float(value) for value in costs.values())
    if len(values) != 16 * 64 * 3:
        raise ValueError(f"expected 3072 coefficients, got {len(values)}")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("coefficients must be finite and non-negative")
    digest = hashlib.sha256(args.path.read_bytes()).hexdigest()
    summary = {
        "path": str(args.path.resolve()),
        "bytes": args.path.stat().st_size,
        "sha256": digest,
        "coefficients": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
