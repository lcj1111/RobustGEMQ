#!/usr/bin/env python3
"""Create one immutable 4-domain balanced GPTQ/RFT calibration tensor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


DOMAINS = ("general", "math", "code", "instruction")


def tensor_hash(tokens: torch.Tensor) -> str:
    return hashlib.sha256(tokens.to(torch.int64).contiguous().numpy().tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--samples-per-domain", type=int, default=32)
    args = parser.parse_args()
    if args.samples_per_domain <= 0:
        raise ValueError("samples-per-domain must be positive")
    chunks = []
    source = {}
    for domain in DOMAINS:
        directory = args.scenario_root / domain / f"seed-{args.seed}"
        manifest = json.loads((directory / "scenario.json").read_text(encoding="utf-8"))
        tokens = torch.load(manifest["tokens_path"], map_location="cpu", weights_only=True)
        if tokens.ndim != 2 or tokens.shape[0] < args.samples_per_domain:
            raise ValueError(f"{directory}: insufficient token rows")
        # A domain-specific deterministic offset avoids every domain always contributing its prefix.
        offset = int(hashlib.sha256(domain.encode()).hexdigest()[:8], 16) % tokens.shape[0]
        indices = [(offset + index) % tokens.shape[0] for index in range(args.samples_per_domain)]
        chunks.append(tokens[indices])
        source[domain] = {
            "seed": args.seed,
            "token_sha256": manifest["token_sha256"],
            "indices": indices,
        }
    balanced = torch.cat(chunks, dim=0).contiguous()
    expected_shape = (args.samples_per_domain * len(DOMAINS), 2048)
    if tuple(balanced.shape) != expected_shape:
        raise ValueError(f"expected {expected_shape}, got {tuple(balanced.shape)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    digest = tensor_hash(balanced)
    token_path = args.output_dir / f"tokens-{digest[:12]}.pt"
    torch.save(balanced, token_path)
    manifest = {
        "schema_version": 1,
        "purpose": "shared GPTQ and router-finetuning calibration; not an evaluation set",
        "domains": list(DOMAINS),
        "samples_per_domain": args.samples_per_domain,
        "shape": list(balanced.shape),
        "token_sha256": digest,
        "tokens_path": str(token_path.resolve()),
        "sources": source,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
