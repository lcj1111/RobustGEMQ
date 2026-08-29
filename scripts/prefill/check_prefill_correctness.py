#!/usr/bin/env python3
"""对照原 one-hot 路径，验证 OLMoE 排序式 prefill 分发的数值等价性。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from gemq.inference.kv_cache import StaticCache
from gemq.inference.patch import prepare_for_inference
from gemq.utils.hf_loading import load_quantized_model
from gemq.utils.model_utils import get_blocks


@torch.inference_mode()
def one_hot_reference(block, hidden_states):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    x = hidden_states.view(-1, hidden_dim)
    router_logits = block.gate(x)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(
        routing_weights, block.top_k, dim=-1
    )
    if block.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(x.dtype)

    output = torch.zeros_like(x)
    expert_mask = F.one_hot(
        selected_experts, num_classes=block.num_experts
    ).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        slot, token = torch.where(expert_mask[expert_idx].squeeze(0))
        expert_input = x.index_select(0, token)
        expert_output = block.forward_single_expert(expert_idx, expert_input)
        expert_output *= routing_weights[token, slot, None]
        output.index_add_(0, token, expert_output.to(x.dtype))
    return output.reshape(batch_size, sequence_length, hidden_dim), router_logits


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-name", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--lengths", default="128,512")
    parser.add_argument("--block-index", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--atol", type=float, default=2e-3)
    parser.add_argument("--rtol", type=float, default=2e-3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = load_quantized_model(
        str(args.checkpoint),
        compute_dtype=torch.float16,
        device="cuda",
        trust_remote_code=False,
    ).eval()
    prepare_for_inference(model, args.model_name, is_fp=False)
    block = get_blocks(model, args.model_name)[args.block_index].mlp
    results = {}

    for length in [int(value) for value in args.lengths.split(",")]:
        input_ids = torch.randint(
            0, int(model.config.vocab_size), (1, length), device="cuda"
        )
        captured = {}

        def capture(_module, inputs):
            captured["hidden_states"] = inputs[0].detach().clone()

        handle = block.register_forward_pre_hook(capture)
        cache = StaticCache(model.config, max_cache_len=length)
        positions = torch.arange(length, device="cuda")
        model(input_ids, past_key_values=cache, cache_position=positions)
        handle.remove()
        hidden_states = captured["hidden_states"]

        reference, reference_router = one_hot_reference(block, hidden_states)
        candidate, candidate_router = block(hidden_states)
        torch.cuda.synchronize()
        difference = (candidate.float() - reference.float()).abs()
        router_equal = torch.equal(candidate_router, reference_router)
        allclose = torch.allclose(candidate, reference, atol=args.atol, rtol=args.rtol)
        results[str(length)] = {
            "allclose": bool(allclose),
            "router_exact": bool(router_equal),
            "max_abs_error": float(difference.max()),
            "mean_abs_error": float(difference.mean()),
        }
        if not allclose or not router_equal:
            raise AssertionError(f"length={length} 数值回归失败：{results[str(length)]}")

    payload = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "block_index": args.block_index,
        "atol": args.atol,
        "rtol": args.rtol,
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
