#!/usr/bin/env python3
"""以独立反量化实现核对 vLLM 插件的 attention 与 MoE 数值。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors import safe_open

from gemq.vllm_plugin.quantization import GEMQLinearMethod, GEMQMoEMethod


def dequantize_packed(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    bits: int,
    input_size: int,
) -> torch.Tensor:
    """独立实现 GemLite 32-bit over-K packing 的 q*scale+zero。"""

    elements_per_word = 32 // bits
    k = torch.arange(input_size, device=qweight.device, dtype=torch.int64)
    packed = qweight.index_select(0, torch.div(k, elements_per_word, rounding_mode="floor"))
    shifts = ((k % elements_per_word) * bits)[:, None]
    quantized = (packed.to(torch.int64) >> shifts) & ((1 << bits) - 1)
    group = torch.div(k, 128, rounding_mode="floor")
    return (
        quantized.float() * scales.index_select(0, group).float()
        + zeros.index_select(0, group).float()
    )


def tensor(handle, name: str) -> torch.Tensor:
    return handle.get_tensor(name).cuda()


def projection(handle, prefix: str, name: str) -> list[torch.Tensor]:
    return [
        tensor(handle, f"{prefix}.gemq_{name}_{suffix}")
        for suffix in (
            "qweight",
            "scales",
            "zeros",
            "nbits",
            "group_sizes",
            "qweight_offsets",
            "scale_offsets",
        )
    ]


def expert_weight(
    tensors: list[torch.Tensor], expert: int, input_size: int, output_size: int
) -> torch.Tensor:
    qweight, scales, zeros, nbits, group_sizes, qoffsets, soffsets = tensors
    bits = int(nbits[expert].item())
    if int(group_sizes[expert].item()) != 128:
        raise AssertionError("独立参考仅接受 group_size=128")
    q_start = int(qoffsets[expert].item())
    q_end = int(qoffsets[expert + 1].item()) if expert + 1 < len(qoffsets) else qweight.numel()
    s_start = int(soffsets[expert].item())
    s_end = int(soffsets[expert + 1].item()) if expert + 1 < len(soffsets) else scales.numel()
    packed = qweight.reshape(-1)[q_start:q_end].view(-1, output_size)
    scale = scales.reshape(-1)[s_start:s_end].view(-1, output_size)
    zero = zeros.reshape(-1)[s_start:s_end].view(-1, output_size)
    return dequantize_packed(packed, scale, zero, bits, input_size)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--attention-atol", type=float, default=4e-2)
    parser.add_argument("--moe-atol", type=float, default=8e-2)
    parser.add_argument("--mean-atol", type=float, default=5e-3)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    weights = args.checkpoint / "model.safetensors"

    with safe_open(weights, framework="pt", device="cpu") as handle:
        attention_prefix = "model.layers.0.self_attn.qkv_proj"
        attention_layer = SimpleNamespace(
            gemq_qweight=tensor(handle, f"{attention_prefix}.gemq_qweight"),
            gemq_scales=tensor(handle, f"{attention_prefix}.gemq_scales").half(),
            gemq_zeros=tensor(handle, f"{attention_prefix}.gemq_zeros").half(),
        )
        attention_input = torch.randn(4, 2048, device="cuda", dtype=torch.float16)
        attention_candidate = GEMQLinearMethod(attention_prefix).apply(
            attention_layer, attention_input
        )
        attention_weight = dequantize_packed(
            attention_layer.gemq_qweight,
            attention_layer.gemq_scales,
            attention_layer.gemq_zeros,
            bits=4,
            input_size=2048,
        )
        attention_reference = (attention_input.float() @ attention_weight).half()

        expert_prefix = "model.layers.0.mlp.experts"
        layer = SimpleNamespace(global_num_experts=64)
        projections = {}
        for name in ("w1", "w2", "w3"):
            values = projection(handle, expert_prefix, name)
            values[1] = values[1].half()
            values[2] = values[2].half()
            projections[name] = values
            for suffix, value in zip(
                (
                    "qweight",
                    "scales",
                    "zeros",
                    "nbits",
                    "group_sizes",
                    "qweight_offsets",
                    "scale_offsets",
                ),
                values,
            ):
                setattr(layer, f"gemq_{name}_{suffix}", value)

        moe_input = torch.randn(2, 2048, device="cuda", dtype=torch.float16) * 0.1
        topk_ids = torch.arange(8, device="cuda", dtype=torch.int32).repeat(2, 1)
        topk_weights = torch.softmax(
            torch.randn(2, 8, device="cuda", dtype=torch.float32), dim=-1
        ).half()
        method = object.__new__(GEMQMoEMethod)
        method.chunk_tokens = 128
        method.debug_validate = False
        method._debug_printed = False
        method._debug_chunk_printed = False
        moe_candidate = method.apply(
            layer, moe_input, topk_weights, topk_ids, None, None
        )
        moe_reference = torch.zeros_like(moe_input, dtype=torch.float32)
        for expert in range(8):
            w1 = expert_weight(projections["w1"], expert, 2048, 1024)
            w3 = expert_weight(projections["w3"], expert, 2048, 1024)
            w2 = expert_weight(projections["w2"], expert, 1024, 2048)
            hidden = F.silu(moe_input.float() @ w1) * (moe_input.float() @ w3)
            expert_output = hidden @ w2
            moe_reference += expert_output * topk_weights[:, expert, None].float()
        moe_reference = moe_reference.half()

    def metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict:
        difference = (candidate.float() - reference.float()).abs()
        return {
            "max_abs_error": float(difference.max().item()),
            "mean_abs_error": float(difference.mean().item()),
            "finite": bool(torch.isfinite(candidate).all().item()),
        }

    attention = metrics(attention_candidate, attention_reference)
    moe = metrics(moe_candidate, moe_reference)
    attention["pass"] = (
        attention["finite"]
        and attention["max_abs_error"] <= args.attention_atol
        and attention["mean_abs_error"] <= args.mean_atol
    )
    moe["pass"] = (
        moe["finite"]
        and moe["max_abs_error"] <= args.moe_atol
        and moe["mean_abs_error"] <= args.mean_atol
    )
    payload = {
        "schema_version": 1,
        "status": "pass" if attention["pass"] and moe["pass"] else "fail",
        "checkpoint": str(args.checkpoint.resolve()),
        "seed": args.seed,
        "reference": "independent-dense-dequantization",
        "attention": attention,
        "moe": moe,
        "thresholds": {
            "attention_max_abs": args.attention_atol,
            "moe_max_abs": args.moe_atol,
            "mean_abs": args.mean_atol,
        },
    }
    if payload["status"] != "pass":
        raise AssertionError(json.dumps(payload, ensure_ascii=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
