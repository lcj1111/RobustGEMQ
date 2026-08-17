#!/usr/bin/env python3
"""Evaluate fake-RTN quality and capture OLMoE route traces for one bit config."""

from __future__ import annotations

import argparse
import gc
import json
import pickle
from functools import partial
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from gemq.quantizers.rtn import MCMoeRTNWeightQuantizer
from gemq.utils.model_utils import (
    get_all_expert_names,
    get_blocks,
    get_moe_block,
    get_named_linears,
    get_sublinear_names,
)


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


def gate_hook(layer_index: int, traces: dict, _module, _inputs, output) -> None:
    logits = output.detach().float()
    top_values, top_indices = torch.topk(logits, k=9, dim=-1)
    traces[layer_index]["topk"].append(top_indices[:, :8].to(torch.int16).cpu())
    traces[layer_index]["margin"].append((top_values[:, 7] - top_values[:, 8]).to(torch.float16).cpu())


def evaluate_scenario(model, tokens: torch.Tensor, batch_size: int) -> tuple[dict, dict]:
    route_parts = {layer: {"topk": [], "margin": []} for layer in range(16)}
    handles = []
    for layer_index, layer in enumerate(get_blocks(model, MODEL_ID)):
        gate = get_moe_block(layer, MODEL_ID).gate
        handles.append(gate.register_forward_hook(partial(gate_hook, layer_index, route_parts)))

    loss_sum = 0.0
    predicted_tokens = 0
    try:
        with torch.inference_mode():
            for start in range(0, tokens.shape[0], batch_size):
                batch = tokens[start : start + batch_size].to("cuda")
                output = model(input_ids=batch, labels=batch, use_cache=False)
                count = batch.numel() - batch.shape[0]
                loss_sum += float(output.loss) * count
                predicted_tokens += count
    finally:
        for handle in handles:
            handle.remove()

    trace = {
        layer: {
            "topk": torch.cat(parts["topk"], dim=0),
            "margin": torch.cat(parts["margin"], dim=0),
        }
        for layer, parts in route_parts.items()
    }
    metrics = {
        "nll": loss_sum / predicted_tokens,
        "ppl": float(torch.exp(torch.tensor(loss_sum / predicted_tokens))),
        "predicted_tokens": predicted_tokens,
    }
    return metrics, trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--scenarios-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--quality-only", action="store_true")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"name": args.name, "config": str(args.config) if args.config else None, "used_bits": used_bits, "scenarios": {}}
    for domain in ("general", "math", "code", "instruction"):
        for seed in (0, 1):
            scenario_dir = args.scenarios_root / domain / f"seed-{seed}"
            manifest = json.loads((scenario_dir / "scenario.json").read_text(encoding="utf-8"))
            tokens = torch.load(manifest["tokens_path"], map_location="cpu", weights_only=True)
            metrics, trace = evaluate_scenario(model, tokens, args.batch_size)
            key = f"{domain}:seed-{seed}"
            summary["scenarios"][key] = {**metrics, "token_sha256": manifest["token_sha256"]}
            if not args.quality_only:
                torch.save(trace, args.output_dir / f"route-{domain}-seed-{seed}.pt")

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"name": args.name, "scenarios": len(summary["scenarios"]), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
