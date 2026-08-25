#!/usr/bin/env python3
"""在冻结的 validation/test token 上逐样本评估真实 HQQ 检查点。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer

from gemq.utils.hf_loading import load_quantized_model


DOMAINS = ("general", "math", "code", "instruction")


def row_hash(row: torch.Tensor) -> str:
    payload = row.to(dtype=torch.int64, device="cpu").contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--checkpoint-seed", type=int, required=True)
    parser.add_argument("--scenario-root", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--scenario-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=False)
    model = load_quantized_model(
        args.checkpoint, compute_dtype=torch.float16, device="cuda", trust_remote_code=False
    ).eval()
    del tokenizer
    original_cache = model.config.use_cache
    model.config.use_cache = False
    records = []
    with torch.inference_mode():
        for domain in DOMAINS:
            directory = args.scenario_root / args.split / domain / f"seed-{args.scenario_seed}"
            manifest = json.loads((directory / "scenario.json").read_text(encoding="utf-8"))
            tokens = torch.load(manifest["tokens_path"], map_location="cpu", weights_only=True)
            for index, row in enumerate(tokens):
                batch = row.unsqueeze(0).to("cuda")
                output = model(input_ids=batch, labels=batch, use_cache=False)
                nll = float(output.loss)
                if not math.isfinite(nll):
                    raise FloatingPointError(f"non-finite NLL for {domain}:item-{index}")
                records.append({
                    "method": args.method,
                    "checkpoint_seed": args.checkpoint_seed,
                    "split": args.split,
                    "domain": domain,
                    "scenario_seed": args.scenario_seed,
                    "item": index,
                    "nll": nll,
                    "predicted_tokens": int(row.numel() - 1),
                    "scenario_token_sha256": manifest["token_sha256"],
                    "item_token_sha256": row_hash(row),
                })
    model.config.use_cache = original_cache
    result = {
        "schema_version": 2,
        "model": "allenai/OLMoE-1B-7B-0924",
        "checkpoint": str(args.checkpoint.resolve()),
        "method": args.method,
        "checkpoint_seed": args.checkpoint_seed,
        "split": args.split,
        "evaluation_path": "real-unpatched HQQLinear; H6 required separately",
        "items": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"output": str(args.output), "items": len(records)}, sort_keys=True))


if __name__ == "__main__":
    main()
