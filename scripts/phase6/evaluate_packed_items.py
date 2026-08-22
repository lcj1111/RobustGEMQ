#!/usr/bin/env python3
"""Evaluate every fixed Phase 6 item with a saved real-unpatched HQQ checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer

from gemq.utils.hf_loading import load_quantized_model


DOMAINS = ("general", "math", "code", "instruction")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenario-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=False)
    model = load_quantized_model(
        args.checkpoint, compute_dtype=torch.float16, device="cuda", trust_remote_code=False
    ).eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False
    records = []
    with torch.inference_mode():
        for domain in DOMAINS:
            for seed in (0, 1, 2):
                directory = args.scenario_root / domain / f"seed-{seed}"
                manifest = json.loads((directory / "scenario.json").read_text(encoding="utf-8"))
                tokens = torch.load(manifest["tokens_path"], map_location="cpu", weights_only=True)
                for index, row in enumerate(tokens):
                    batch = row.unsqueeze(0).to("cuda")
                    output = model(input_ids=batch, labels=batch, use_cache=False)
                    nll = float(output.loss)
                    if not math.isfinite(nll):
                        raise FloatingPointError(f"non-finite NLL for {domain}:seed-{seed}:item-{index}")
                    records.append(
                        {
                            "domain": domain,
                            "seed": seed,
                            "item": index,
                            "nll": nll,
                            "token_sha256": manifest["token_sha256"],
                        }
                    )
    model.config.use_cache = use_cache
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "model_path": str(args.checkpoint.resolve()),
        "evaluation_path": "real-unpatched HQQLinear; equivalence is established by H6",
        "items": records,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"output": str(args.output), "items": len(records)}))


if __name__ == "__main__":
    main()
