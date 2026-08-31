#!/usr/bin/env python3
"""不依赖 pytest 的 CUDA dispatch/reduce 精确正确性检查。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from gemq.triton_kernels.vllm_moe_dispatch import (
    fused_chunk_unpermute_reduce,
    stable_expert_dispatch,
    write_chunk_expert_offsets,
)


def check_dispatch(tokens: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260831 + tokens)
    topk_ids = torch.randint(
        0, 64, (tokens, 8), device="cuda", dtype=torch.int32, generator=generator
    )
    sorted_tokens, inverse_order, offsets = stable_expert_dispatch(topk_ids, 64)
    flat = topk_ids.view(-1)
    expected_order = torch.argsort(flat, stable=True)
    expected_inverse = torch.empty_like(expected_order)
    expected_inverse.scatter_(
        0, expected_order, torch.arange(flat.numel(), device="cuda")
    )
    counts = torch.bincount(flat.to(torch.int64), minlength=64).to(torch.int32)
    expected_offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    if not torch.equal(sorted_tokens, expected_order // 8):
        raise AssertionError(f"tokens={tokens}: sorted token 不一致")
    if not torch.equal(inverse_order, expected_inverse):
        raise AssertionError(f"tokens={tokens}: inverse order 不一致")
    if not torch.equal(offsets, expected_offsets):
        raise AssertionError(f"tokens={tokens}: expert offset 不一致")


def check_chunk_offsets(start: int, end: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260831)
    topk_ids = torch.randint(
        0, 64, (100, 8), device="cuda", dtype=torch.int32, generator=generator
    )
    _, _, global_offsets = stable_expert_dispatch(topk_ids, 64)
    chunk_offsets = torch.empty_like(global_offsets)
    write_chunk_expert_offsets(global_offsets, chunk_offsets, start, end)
    order = torch.argsort(topk_ids.view(-1), stable=True)
    sorted_experts = topk_ids.view(-1)[order]
    counts = torch.bincount(
        sorted_experts[start:end].to(torch.int64), minlength=64
    ).to(torch.int32)
    expected = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    if not torch.equal(chunk_offsets, expected):
        raise AssertionError(f"chunk=[{start}, {end}): offset 不一致")


def check_reduce() -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260831)
    tokens, top_k, hidden = 19, 8, 256
    topk_ids = torch.randint(
        0, 64, (tokens, top_k), device="cuda", dtype=torch.int32, generator=generator
    )
    weights = torch.rand(
        (tokens, top_k), device="cuda", dtype=torch.float16, generator=generator
    )
    weights /= weights.sum(dim=1, keepdim=True)
    _, inverse, _ = stable_expert_dispatch(topk_ids, 64)
    expert_output = torch.randn(
        (tokens * top_k, hidden),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    accumulator = torch.zeros((tokens, hidden), device="cuda", dtype=torch.float32)
    output = torch.empty((tokens, hidden), device="cuda", dtype=torch.float16)
    split = 67
    fused_chunk_unpermute_reduce(
        expert_output[:split], inverse, weights, accumulator, 0, split
    )
    fused_chunk_unpermute_reduce(
        expert_output[split:],
        inverse,
        weights,
        accumulator,
        split,
        tokens * top_k,
        final_output=output,
    )

    reference = torch.zeros_like(accumulator)
    for token in range(tokens):
        for slot in range(top_k):
            sorted_index = inverse[token * top_k + slot]
            reference[token] += (
                expert_output[sorted_index].float() * weights[token, slot].float()
            )
    if not torch.equal(output, reference.half()):
        raise AssertionError("融合 unpermute/reduce 与 FP32 固定顺序参考不一致")


def atomic_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_bytes(
            (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU")

    token_cases = [1, 17, 128, 513]
    chunk_cases = [(0, 64), (31, 193), (128, 512), (511, 700)]
    for tokens in token_cases:
        check_dispatch(tokens)
    for start, end in chunk_cases:
        check_chunk_offsets(start, end)
    check_reduce()
    torch.cuda.synchronize()

    payload = {
        "schema_version": 1,
        "status": "pass",
        "subject": "vLLM stable dispatch 与融合归并 CUDA 正确性",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "dispatch_token_cases": token_cases,
        "chunk_cases": [{"start": start, "end": end} for start, end in chunk_cases],
        "reduce": {
            "tokens": 19,
            "top_k": 8,
            "hidden_dim": 256,
            "chunks": [[0, 67], [67, 152]],
            "comparison": "exact FP16 equality after fixed-order FP32 accumulation",
        },
    }
    atomic_dump(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
