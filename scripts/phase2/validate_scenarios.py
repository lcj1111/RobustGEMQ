#!/usr/bin/env python3
"""校验场景身份及 OLMoE 系数张量的完整性。"""

import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path

import torch


def token_hash(tokens: torch.Tensor) -> str:
    return hashlib.sha256(tokens.to(torch.int64).contiguous().numpy().tobytes()).hexdigest()


def validate_one(path: Path) -> dict:
    manifest = json.loads((path / "scenario.json").read_text(encoding="utf-8"))
    tokens = torch.load(manifest["tokens_path"], map_location="cpu", weights_only=True)
    expected_shape = (manifest["nsamples"], manifest["seqlen"])
    if tuple(tokens.shape) != expected_shape:
        raise ValueError(f"{path}: token shape {tuple(tokens.shape)} != {expected_shape}")
    if token_hash(tokens) != manifest["token_sha256"]:
        raise ValueError(f"{path}: token hash mismatch")

    layer_re_path = path / "LayerRE_B1,2,3.pkl"
    with layer_re_path.open("rb") as handle:
        layer_re = pickle.load(handle)
    if set(layer_re) != set(range(16)):
        raise ValueError(f"{path}: expected layers 0..15")
    values = []
    for layer, experts in layer_re.items():
        if set(experts) != set(range(64)):
            raise ValueError(f"{path}: layer {layer} does not contain experts 0..63")
        for expert, bit_costs in experts.items():
            if set(bit_costs) != {1, 2, 3}:
                raise ValueError(f"{path}: layer {layer} expert {expert} has invalid bits")
            values.extend(map(float, bit_costs.values()))
    if len(values) != 16 * 64 * 3 or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError(f"{path}: invalid coefficient tensor")
    return {
        "domain": manifest["domain"],
        "seed": manifest["seed"],
        "coefficients": len(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [validate_one(path) for path in sorted(args.root.glob("*/seed-*"))]
    if not results:
        raise ValueError(f"No scenarios found under {args.root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"scenarios": len(results), "output": str(args.output)}))


if __name__ == "__main__":
    main()
