#!/usr/bin/env python3
"""Build immutable Phase 3 evaluation tokens from held-out dataset splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from gemq.utils.domain_data import build_domain_scenario, save_domain_scenario


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--nsamples", type=int, default=32)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokens, manifest = build_domain_scenario(
        registry_path=args.registry,
        data_root=args.data_root,
        domain=args.domain,
        tokenizer=tokenizer,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        seed=args.seed,
    )
    manifest.update(
        {
            "model_id": "allenai/OLMoE-1B-7B-0924",
            "tokenizer_path": str(args.model.resolve()),
            "split_role": "held-out evaluation only",
            "forbidden_use": "must not construct or tune coefficient objectives",
        }
    )
    saved = save_domain_scenario(tokens, manifest, args.output_dir)
    print(json.dumps(saved, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
