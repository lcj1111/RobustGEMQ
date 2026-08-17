#!/usr/bin/env python3
"""Generate deterministic, budget- and per-layer-histogram-preserving bit perturbations."""

import argparse
import json
import pickle
import random
from pathlib import Path


def hamming(left: dict, right: dict) -> float:
    changed = sum(left[layer][expert] != right[layer][expert] for layer in left for expert in left[layer])
    total = sum(len(experts) for experts in left.values())
    return changed / total


def perturb(base: dict, fraction: float, seed: int) -> dict:
    rng = random.Random(seed)
    result = {layer: dict(experts) for layer, experts in base.items()}
    target_per_layer = max(2, round(64 * fraction))
    if target_per_layer % 2:
        target_per_layer += 1
    for layer, experts in result.items():
        candidates = list(experts)
        rng.shuffle(candidates)
        changed = 0
        attempts = 0
        while changed < target_per_layer and attempts < 64 * 64:
            attempts += 1
            left, right = rng.sample(candidates, 2)
            if experts[left] == experts[right]:
                continue
            experts[left], experts[right] = experts[right], experts[left]
            changed += 2
        if changed < target_per_layer:
            raise ValueError(f"Could not construct perturbation for layer {layer}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    with args.base.open("rb") as handle:
        base = pickle.load(handle)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for fraction in (0.05, 0.10, 0.20, 0.30, 0.40):
        for replicate in range(4):
            seed = int(fraction * 1000) + replicate
            config = perturb(base, fraction, seed)
            name = f"p{int(fraction * 100):02d}-r{replicate}"
            path = args.output_dir / f"{name}.pkl"
            with path.open("wb") as handle:
                pickle.dump(config, handle)
            base_bits = sum(bit for experts in base.values() for bit in experts.values())
            config_bits = sum(bit for experts in config.values() for bit in experts.values())
            if base_bits != config_bits:
                raise ValueError(f"{name} changed the actual bit budget")
            for layer in base:
                if sorted(base[layer].values()) != sorted(config[layer].values()):
                    raise ValueError(f"{name} changed layer {layer} bit histogram")
            records.append(
                {
                    "name": name,
                    "path": str(path),
                    "requested_fraction": fraction,
                    "actual_hamming_fraction": hamming(base, config),
                    "seed": seed,
                    "used_bits": config_bits,
                }
            )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"configs": len(records), "min_hamming": min(r["actual_hamming_fraction"] for r in records), "max_hamming": max(r["actual_hamming_fraction"] for r in records)}))


if __name__ == "__main__":
    main()
