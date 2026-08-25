#!/usr/bin/env python3
"""验证 OLMoE calibration-A 的逐层 expert usage 计数。"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("usage", type=Path)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.usage.open("rb") as handle:
        usage = pickle.load(handle)
    if set(usage) != set(range(16)):
        raise ValueError("usage must contain OLMoE layers 0..15")
    expected = args.samples * args.seqlen * args.top_k
    totals = {}
    for layer in range(16):
        values = np.asarray(usage[layer], dtype=np.int64)
        if values.shape != (64,) or (values < 0).any():
            raise ValueError(f"layer {layer}: expected 64 non-negative counts")
        if int(values.sum()) != expected:
            raise ValueError(f"layer {layer}: usage sum {values.sum()} != {expected}")
        totals[str(layer)] = int(values.sum())
    summary = {
        "schema_version": 1,
        "verified": True,
        "shape": [16, 64],
        "expected_routes_per_layer": expected,
        "routes_per_layer": totals,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
