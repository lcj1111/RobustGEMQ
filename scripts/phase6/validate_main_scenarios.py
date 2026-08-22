#!/usr/bin/env python3
"""Validate the complete 4-domain x 3-seed Phase 6 token matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def token_hash(tokens: torch.Tensor) -> str:
    payload = tokens.to(dtype=torch.int64, device="cpu").contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    identities = set()
    for domain in ("general", "math", "code", "instruction"):
        for seed in (0, 1, 2):
            directory = args.root / domain / f"seed-{seed}"
            manifest = json.loads((directory / "scenario.json").read_text(encoding="utf-8"))
            tokens = torch.load(manifest["tokens_path"], map_location="cpu", weights_only=True)
            if tuple(tokens.shape) != (128, 2048):
                raise ValueError(f"{directory}: shape {tuple(tokens.shape)} != (128, 2048)")
            observed_hash = token_hash(tokens)
            if observed_hash != manifest["token_sha256"]:
                raise ValueError(f"{directory}: token hash mismatch")
            identity = (domain, observed_hash)
            if identity in identities:
                raise ValueError(f"{directory}: duplicate token identity within domain")
            identities.add(identity)
            results.append(
                {
                    "domain": domain,
                    "seed": seed,
                    "shape": list(tokens.shape),
                    "token_sha256": observed_hash,
                    "allocation_sha256": manifest["allocation_sha256"],
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"scenarios": len(results), "unique_identities": len(identities)}))


if __name__ == "__main__":
    main()
