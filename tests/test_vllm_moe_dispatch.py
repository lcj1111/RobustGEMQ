import pytest
import torch


pytestmark = pytest.mark.cuda


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA")


@pytest.mark.parametrize("tokens", [1, 17, 128, 513])
def test_stable_dispatch_matches_pytorch(tokens):
    _require_cuda()
    from gemq.triton_kernels.vllm_moe_dispatch import stable_expert_dispatch

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
    expected_counts = torch.bincount(flat.to(torch.int64), minlength=64).to(torch.int32)
    expected_offsets = torch.cat(
        (expected_counts.new_zeros(1), expected_counts.cumsum(0))
    )
    assert torch.equal(sorted_tokens, expected_order // 8)
    assert torch.equal(inverse_order, expected_inverse)
    assert torch.equal(offsets, expected_offsets)


@pytest.mark.parametrize("start,end", [(0, 64), (31, 193), (128, 512), (511, 700)])
def test_chunk_offsets_match_sorted_slice(start, end):
    _require_cuda()
    from gemq.triton_kernels.vllm_moe_dispatch import (
        stable_expert_dispatch,
        write_chunk_expert_offsets,
    )

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
    assert torch.equal(chunk_offsets, expected)


def test_fused_unpermute_reduce_matches_fp32_reference():
    _require_cuda()
    from gemq.triton_kernels.vllm_moe_dispatch import (
        fused_chunk_unpermute_reduce,
        stable_expert_dispatch,
    )

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
    assert torch.equal(output, reference.half())
