"""vLLM 服务路径专用的稳定 expert dispatch 与融合归并。"""

from __future__ import annotations

import torch
from torch import Tensor

import triton
import triton.language as tl


@triton.jit
def stable_count_offsets_kernel(
    expert_ids_ptr,
    expert_offsets_ptr,
    num_assignments,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_A: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """一次扫描得到全部 expert count 和 exclusive offset。"""
    expert = tl.arange(0, BLOCK_E)
    counts = tl.zeros((BLOCK_E,), dtype=tl.int32)
    num_blocks = tl.cdiv(num_assignments, BLOCK_A)
    for block in range(0, num_blocks):
        assignment = block * BLOCK_A + tl.arange(0, BLOCK_A)
        valid_assignment = assignment < num_assignments
        ids = tl.load(
            expert_ids_ptr + assignment,
            mask=valid_assignment,
            other=-1,
        ).to(tl.int32)
        matches = (
            (ids[:, None] == expert[None, :])
            & valid_assignment[:, None]
            & (expert[None, :] < NUM_EXPERTS)
        )
        counts += tl.sum(matches.to(tl.int32), axis=0)

    inclusive = tl.cumsum(counts, axis=0)
    tl.store(expert_offsets_ptr, 0)
    tl.store(
        expert_offsets_ptr + expert + 1,
        inclusive,
        mask=expert < NUM_EXPERTS,
    )


@triton.jit
def stable_scatter_dispatch_kernel(
    expert_ids_ptr,
    expert_offsets_ptr,
    sorted_tokens_ptr,
    inverse_order_ptr,
    num_assignments,
    top_k,
    BLOCK_A: tl.constexpr,
):
    """每个 expert 稳定扫描原 assignment，并同时生成 inverse 映射。"""
    expert = tl.program_id(0)
    cursor = tl.load(expert_offsets_ptr + expert).to(tl.int32)
    num_blocks = tl.cdiv(num_assignments, BLOCK_A)
    for block in range(0, num_blocks):
        assignment = block * BLOCK_A + tl.arange(0, BLOCK_A)
        valid = assignment < num_assignments
        ids = tl.load(expert_ids_ptr + assignment, mask=valid, other=-1).to(tl.int32)
        matches = valid & (ids == expert)
        prefix = tl.cumsum(matches.to(tl.int32), axis=0) - 1
        sorted_assignment = cursor + prefix
        tl.store(
            sorted_tokens_ptr + sorted_assignment,
            assignment // top_k,
            mask=matches,
        )
        tl.store(
            inverse_order_ptr + assignment,
            sorted_assignment,
            mask=matches,
        )
        cursor += tl.sum(matches.to(tl.int32), axis=0)


@triton.jit
def chunk_offsets_from_global_kernel(
    global_offsets_ptr,
    chunk_offsets_ptr,
    chunk_start,
    chunk_size,
    NUM_OFFSETS: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """把全局 expert 边界裁剪为当前连续 assignment 分块的局部边界。"""
    index = tl.arange(0, BLOCK_E)
    global_offset = tl.load(
        global_offsets_ptr + index,
        mask=index < NUM_OFFSETS,
        other=0,
    ).to(tl.int32)
    local_offset = tl.minimum(
        tl.maximum(global_offset - chunk_start, 0),
        chunk_size,
    )
    tl.store(
        chunk_offsets_ptr + index,
        local_offset,
        mask=index < NUM_OFFSETS,
    )


@triton.jit
def fused_weighted_unpermute_reduce_kernel(
    expert_output_ptr,
    inverse_order_ptr,
    routing_weights_ptr,
    output_accumulator_ptr,
    final_output_ptr,
    hidden_dim,
    chunk_start,
    chunk_end,
    stride_em,
    stride_ed,
    stride_wt,
    stride_ws,
    stride_at,
    stride_ad,
    stride_ot,
    stride_od,
    TOP_K: tl.constexpr,
    FINAL_CHUNK: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """融合 route weight、unpermute、slot reduce，并在最后一块完成 FP16 cast。"""
    token = tl.program_id(0)
    column_tile = tl.program_id(1)
    columns = column_tile * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    column_mask = columns < hidden_dim
    accumulator_offsets = token * stride_at + columns * stride_ad
    accumulator = tl.load(
        output_accumulator_ptr + accumulator_offsets,
        mask=column_mask,
        other=0.0,
    ).to(tl.float32)

    for slot in range(TOP_K):
        flat_assignment = token * TOP_K + slot
        sorted_assignment = tl.load(inverse_order_ptr + flat_assignment)
        in_chunk = (sorted_assignment >= chunk_start) & (sorted_assignment < chunk_end)
        local_assignment = sorted_assignment - chunk_start
        weight = tl.load(
            routing_weights_ptr + token * stride_wt + slot * stride_ws
        )
        value = tl.load(
            expert_output_ptr
            + local_assignment * stride_em
            + columns * stride_ed,
            mask=column_mask & in_chunk,
            other=0.0,
        )
        accumulator += value.to(tl.float32) * weight.to(tl.float32)

    if FINAL_CHUNK:
        tl.store(
            final_output_ptr + token * stride_ot + columns * stride_od,
            accumulator.to(tl.float16),
            mask=column_mask,
        )
    else:
        tl.store(
            output_accumulator_ptr + accumulator_offsets,
            accumulator,
            mask=column_mask,
        )


def stable_expert_dispatch(
    topk_ids: Tensor,
    num_experts: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """返回 stable-sorted token、原位置到排序位置映射和全局 expert offsets。"""
    if topk_ids.ndim != 2 or not topk_ids.is_cuda:
        raise ValueError("topk_ids 必须为二维 CUDA tensor")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("topk_ids 必须为 int32 或 int64")
    if num_experts <= 0 or num_experts > 256:
        raise ValueError("num_experts 必须位于 [1, 256]")
    tokens, top_k = topk_ids.shape
    num_assignments = tokens * top_k
    if num_assignments <= 0:
        raise ValueError("至少需要一个 expert assignment")

    flat_experts = topk_ids.contiguous().view(-1)
    sorted_tokens = torch.empty(
        num_assignments, device=topk_ids.device, dtype=torch.int64
    )
    inverse_order = torch.empty_like(sorted_tokens)
    expert_offsets = torch.empty(
        num_experts + 1, device=topk_ids.device, dtype=torch.int32
    )
    block_a = 128
    block_e = triton.next_power_of_2(num_experts)
    stable_count_offsets_kernel[(1,)](
        flat_experts,
        expert_offsets,
        num_assignments,
        NUM_EXPERTS=num_experts,
        BLOCK_A=block_a,
        BLOCK_E=block_e,
        num_warps=4,
    )
    stable_scatter_dispatch_kernel[(num_experts,)](
        flat_experts,
        expert_offsets,
        sorted_tokens,
        inverse_order,
        num_assignments,
        top_k,
        BLOCK_A=block_a,
        num_warps=4,
    )
    return sorted_tokens, inverse_order, expert_offsets


def write_chunk_expert_offsets(
    global_offsets: Tensor,
    chunk_offsets: Tensor,
    chunk_start: int,
    chunk_end: int,
) -> None:
    """在预分配 tensor 中写入当前 chunk 的局部 expert offsets。"""
    if global_offsets.shape != chunk_offsets.shape:
        raise ValueError("global_offsets 与 chunk_offsets 形状必须一致")
    if global_offsets.dtype != torch.int32 or chunk_offsets.dtype != torch.int32:
        raise TypeError("expert offsets 必须为 int32")
    if chunk_start < 0 or chunk_end <= chunk_start:
        raise ValueError("chunk 边界非法")
    num_offsets = global_offsets.numel()
    block_e = triton.next_power_of_2(num_offsets)
    chunk_offsets_from_global_kernel[(1,)](
        global_offsets,
        chunk_offsets,
        chunk_start,
        chunk_end - chunk_start,
        NUM_OFFSETS=num_offsets,
        BLOCK_E=block_e,
        num_warps=1,
    )


def fused_chunk_unpermute_reduce(
    expert_output: Tensor,
    inverse_order: Tensor,
    routing_weights: Tensor,
    output_accumulator: Tensor,
    chunk_start: int,
    chunk_end: int,
    final_output: Tensor | None = None,
) -> None:
    """归并一个 chunk；最后一块可直接写最终 FP16 输出。"""
    if chunk_end <= chunk_start:
        raise ValueError("chunk_end 必须大于 chunk_start")
    if expert_output.shape[0] != chunk_end - chunk_start:
        raise ValueError("expert_output 行数必须与 chunk 长度一致")
    tokens, top_k = routing_weights.shape
    hidden_dim = expert_output.shape[1]
    if output_accumulator.shape != (tokens, hidden_dim):
        raise ValueError("output_accumulator 形状不匹配")
    if output_accumulator.dtype != torch.float32:
        raise TypeError("output_accumulator 必须为 FP32")
    if final_output is not None:
        if final_output.shape != (tokens, hidden_dim):
            raise ValueError("final_output 形状不匹配")
        if final_output.dtype != expert_output.dtype:
            raise TypeError("final_output dtype 必须与 expert_output 一致")

    destination = final_output if final_output is not None else output_accumulator
    block_size_d = 128
    fused_weighted_unpermute_reduce_kernel[
        (tokens, triton.cdiv(hidden_dim, block_size_d))
    ](
        expert_output,
        inverse_order,
        routing_weights,
        output_accumulator,
        destination,
        hidden_dim,
        chunk_start,
        chunk_end,
        expert_output.stride(0),
        expert_output.stride(1),
        routing_weights.stride(0),
        routing_weights.stride(1),
        output_accumulator.stride(0),
        output_accumulator.stride(1),
        destination.stride(0),
        destination.stride(1),
        TOP_K=top_k,
        FINAL_CHUNK=final_output is not None,
        BLOCK_SIZE_D=block_size_d,
    )
