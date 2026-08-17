#!/usr/bin/env python3
"""Build one immutable domain/seed token scenario and its identity manifest."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from gemq.utils.domain_data import build_domain_scenario, save_domain_scenario


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--nsamples", type=int, required=True)
    parser.add_argument("--seqlen", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    manifest["model_id"] = args.model_id
    manifest["tokenizer_path"] = str(Path(args.model).resolve())
    saved = save_domain_scenario(tokens, manifest, args.output_dir)
    print(json.dumps(saved, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
