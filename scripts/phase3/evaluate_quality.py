#!/usr/bin/env python3
"""Evaluate one Phase 3 fake-RTN allocation on frozen held-out scenarios."""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from gemq.quantizers.rtn import MCMoeRTNWeightQuantizer
from gemq.utils.model_utils import get_all_expert_names, get_blocks, get_named_linears, get_sublinear_names


MODEL_ID = "allenai/OLMoE-1B-7B-0924"


def apply_fake_quant(model, config: dict) -> None:
    expert_names = get_all_expert_names(MODEL_ID)
    sublinear_names = get_sublinear_names(MODEL_ID)
    with torch.no_grad():
        for layer_index, layer in enumerate(get_blocks(model, MODEL_ID)):
            linears = get_named_linears(layer)
            for expert_index, expert_name in enumerate(expert_names):
                bit = int(config[layer_index][expert_index])
                for linear_name in sublinear_names:
                    module = linears[f"{expert_name}.{linear_name}"]
                    quantizer = MCMoeRTNWeightQuantizer(module.weight.data, nbits=bit)
                    module.weight.data = quantizer.quantize()
                    del quantizer
    gc.collect()
    torch.cuda.empty_cache()


def evaluate(model, tokens: torch.Tensor, batch_size: int) -> dict:
    loss_sum = 0.0
    predicted_tokens = 0
    with torch.inference_mode():
        for start in range(0, tokens.shape[0], batch_size):
            batch = tokens[start : start + batch_size].to("cuda")
            output = model(input_ids=batch, labels=batch, use_cache=False)
            count = batch.numel() - batch.shape[0]
            loss_sum += float(output.loss) * count
            predicted_tokens += count
    nll = loss_sum / predicted_tokens
    return {"nll": nll, "ppl": float(torch.exp(torch.tensor(nll))), "predicted_tokens": predicted_tokens}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--scenarios-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    started = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="cuda",
        dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()
    model.config.use_cache = False
    used_bits = None
    if args.config:
        with args.config.open("rb") as handle:
            config = pickle.load(handle)
        used_bits = sum(bit for experts in config.values() for bit in experts.values())
        apply_fake_quant(model, config)

    result = {
        "schema_version": 1,
        "name": args.name,
        "config": str(args.config) if args.config else None,
        "used_bits": used_bits,
        "quantization": "fake RTN weight-only; screening, not final GPTQ/RFT evidence",
        "scenarios": {},
    }
    for domain in ("general", "math", "code", "instruction"):
        for seed in (0, 1):
            scenario_dir = args.scenarios_root / domain / f"seed-{seed}"
            manifest = json.loads((scenario_dir / "scenario.json").read_text(encoding="utf-8"))
            if manifest.get("split_role") != "held-out evaluation only":
                raise ValueError(f"Refusing non-held-out scenario: {scenario_dir}")
            tokens = torch.load(manifest["tokens_path"], map_location="cpu", weights_only=True)
            result["scenarios"][f"{domain}:seed-{seed}"] = {
                **evaluate(model, tokens, args.batch_size),
                "token_sha256": manifest["token_sha256"],
            }
    result["wall_time_seconds"] = time.time() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"name": args.name, "output": str(args.output), "seconds": result["wall_time_seconds"]}))


if __name__ == "__main__":
    main()
