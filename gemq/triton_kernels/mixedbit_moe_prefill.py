"""面向 MoE prefill 的 variable-M mixed-bit grouped GEMM。"""

from __future__ import annotations

import torch
from torch import Tensor

import triton
import triton.language as tl

from gemq.triton_kernels.utils import dequantize


def bucket_total_assignments(assignments: int) -> int:
    """按 2 的幂复用不同 token 数的 autotune 结果。"""
    if assignments <= 0:
        raise ValueError("assignment 数必须为正数")
    return max(128, 1 << (assignments - 1).bit_length())


def get_moe_prefill_autotune_config():
    return [
        triton.Config(
            {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=3,
            num_warps=4,
        ),
    ]


def get_fused_up_autotune_config():
    """双累加器会增加寄存器压力，因此使用较保守的 tile。"""
    return [
        triton.Config(
            {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=3,
            num_warps=8,
        ),
    ]


@triton.autotune(
    configs=get_moe_prefill_autotune_config(),
    key=["M_BUCKET", "N", "K", "E", "NUM_SM"],
)
@triton.jit
def mixedbit_variable_m_grouped_gemm_kernel(
    x_ptr,
    wq_ptr,
    output_ptr,
    expert_offsets_ptr,
    scales_ptr,
    zeros_ptr,
    nbits_ptr,
    group_sizes_ptr,
    wq_offsets_ptr,
    scale_offsets_ptr,
    total_m,
    N,
    K,
    E,
    M_BUCKET: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_om,
    stride_on,
    stride_sk,
    stride_sn,
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """一个 persistent grid 依次消费所有 expert 的 variable-M GEMM tile。"""
    tile_idx = tl.program_id(0)
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    # N 是运行时 int64，problem_tiles 因而也是 int64；循环累计量显式同型。
    problem_start = tl.full((), 0, tl.int64)

    for expert_idx in range(E):
        row_start = tl.load(expert_offsets_ptr + expert_idx)
        row_end = tl.load(expert_offsets_ptr + expert_idx + 1)
        expert_m = row_end - row_start
        num_m_tiles = tl.cdiv(expert_m, BLOCK_SIZE_M)
        problem_tiles = num_m_tiles * num_n_tiles

        nbit = tl.load(nbits_ptr + expert_idx)
        group_size = tl.load(group_sizes_ptr + expert_idx)
        wq_offset = tl.load(wq_offsets_ptr + expert_idx)
        scale_offset = tl.load(scale_offsets_ptr + expert_idx)
        expert_wq_ptr = wq_ptr + wq_offset
        expert_scale_ptr = scales_ptr + scale_offset
        expert_zero_ptr = zeros_ptr + scale_offset

        while tile_idx >= problem_start and tile_idx < problem_start + problem_tiles:
            local_tile = tile_idx - problem_start
            tile_m = local_tile // num_n_tiles
            tile_n = local_tile % num_n_tiles

            local_rows = tile_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            rows = row_start + local_rows
            cols = tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            k_offsets = tl.arange(0, BLOCK_SIZE_K)
            row_mask = local_rows < expert_m
            col_mask = cols < N

            accumulator = tl.zeros(
                (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
            )
            elements_per_word = 32 // nbit
            for k_tile in range(0, num_k_tiles):
                k = k_tile * BLOCK_SIZE_K + k_offsets
                k_mask = k < K
                x = tl.load(
                    x_ptr + rows[:, None] * stride_xm + k[None, :] * stride_xk,
                    mask=row_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
                packed = tl.load(
                    expert_wq_ptr
                    + (k[:, None] // elements_per_word) * stride_wk
                    + cols[None, :] * stride_wn,
                    mask=k_mask[:, None] & col_mask[None, :],
                    other=0,
                )
                scale_ptrs = (
                    expert_scale_ptr
                    + (k[:, None] // group_size) * stride_sk
                    + cols[None, :] * stride_sn
                )
                zero_ptrs = (
                    expert_zero_ptr
                    + (k[:, None] // group_size) * stride_sk
                    + cols[None, :] * stride_sn
                )
                scales = tl.load(
                    scale_ptrs,
                    mask=k_mask[:, None] & col_mask[None, :],
                    other=0.0,
                )
                zeros = tl.load(
                    zero_ptrs,
                    mask=k_mask[:, None] & col_mask[None, :],
                    other=0.0,
                )
                shifts = (k % elements_per_word * nbit).to(tl.int32)[:, None]
                unpack_mask = (1 << nbit) - 1
                weight = dequantize(packed, scales, zeros, shifts, unpack_mask)
                accumulator = tl.dot(x, weight, acc=accumulator)

            tl.store(
                output_ptr
                + rows[:, None] * stride_om
                + cols[None, :] * stride_on,
                accumulator.to(tl.float16),
                mask=row_mask[:, None] & col_mask[None, :],
            )
            tile_idx += NUM_SM

        problem_start += problem_tiles


@triton.autotune(
    configs=get_fused_up_autotune_config(),
    key=["M_BUCKET", "N", "K", "E", "NUM_SM"],
)
@triton.jit
def mixedbit_fused_up_activation_kernel(
    x_ptr,
    w1_ptr,
    w3_ptr,
    output_ptr,
    expert_offsets_ptr,
    w1_scales_ptr,
    w1_zeros_ptr,
    w3_scales_ptr,
    w3_zeros_ptr,
    nbits_ptr,
    group_sizes_ptr,
    wq_offsets_ptr,
    scale_offsets_ptr,
    total_m,
    N,
    K,
    E,
    M_BUCKET: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_om,
    stride_on,
    stride_sk,
    stride_sn,
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """共享输入 tile，联合计算 SiLU(xW1) * xW3。"""
    tile_idx = tl.program_id(0)
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    problem_start = tl.full((), 0, tl.int64)

    for expert_idx in range(E):
        row_start = tl.load(expert_offsets_ptr + expert_idx)
        row_end = tl.load(expert_offsets_ptr + expert_idx + 1)
        expert_m = row_end - row_start
        num_m_tiles = tl.cdiv(expert_m, BLOCK_SIZE_M)
        problem_tiles = num_m_tiles * num_n_tiles
        nbit = tl.load(nbits_ptr + expert_idx)
        group_size = tl.load(group_sizes_ptr + expert_idx)
        wq_offset = tl.load(wq_offsets_ptr + expert_idx)
        scale_offset = tl.load(scale_offsets_ptr + expert_idx)
        expert_w1_ptr = w1_ptr + wq_offset
        expert_w3_ptr = w3_ptr + wq_offset
        expert_w1_scale_ptr = w1_scales_ptr + scale_offset
        expert_w1_zero_ptr = w1_zeros_ptr + scale_offset
        expert_w3_scale_ptr = w3_scales_ptr + scale_offset
        expert_w3_zero_ptr = w3_zeros_ptr + scale_offset

        while tile_idx >= problem_start and tile_idx < problem_start + problem_tiles:
            local_tile = tile_idx - problem_start
            tile_m = local_tile // num_n_tiles
            tile_n = local_tile % num_n_tiles
            local_rows = tile_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            rows = row_start + local_rows
            cols = tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            k_offsets = tl.arange(0, BLOCK_SIZE_K)
            row_mask = local_rows < expert_m
            col_mask = cols < N
            w1_accumulator = tl.zeros(
                (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
            )
            w3_accumulator = tl.zeros(
                (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
            )
            elements_per_word = 32 // nbit

            for k_tile in range(0, num_k_tiles):
                k = k_tile * BLOCK_SIZE_K + k_offsets
                k_mask = k < K
                x = tl.load(
                    x_ptr + rows[:, None] * stride_xm + k[None, :] * stride_xk,
                    mask=row_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
                packed_offsets = (
                    (k[:, None] // elements_per_word) * stride_wk
                    + cols[None, :] * stride_wn
                )
                weight_mask = k_mask[:, None] & col_mask[None, :]
                packed_w1 = tl.load(
                    expert_w1_ptr + packed_offsets, mask=weight_mask, other=0
                )
                packed_w3 = tl.load(
                    expert_w3_ptr + packed_offsets, mask=weight_mask, other=0
                )
                scale_offsets = (
                    (k[:, None] // group_size) * stride_sk
                    + cols[None, :] * stride_sn
                )
                w1_scales = tl.load(
                    expert_w1_scale_ptr + scale_offsets,
                    mask=weight_mask,
                    other=0.0,
                )
                w1_zeros = tl.load(
                    expert_w1_zero_ptr + scale_offsets,
                    mask=weight_mask,
                    other=0.0,
                )
                w3_scales = tl.load(
                    expert_w3_scale_ptr + scale_offsets,
                    mask=weight_mask,
                    other=0.0,
                )
                w3_zeros = tl.load(
                    expert_w3_zero_ptr + scale_offsets,
                    mask=weight_mask,
                    other=0.0,
                )
                shifts = (k % elements_per_word * nbit).to(tl.int32)[:, None]
                unpack_mask = (1 << nbit) - 1
                w1 = dequantize(
                    packed_w1, w1_scales, w1_zeros, shifts, unpack_mask
                )
                w3 = dequantize(
                    packed_w3, w3_scales, w3_zeros, shifts, unpack_mask
                )
                w1_accumulator = tl.dot(x, w1, acc=w1_accumulator)
                w3_accumulator = tl.dot(x, w3, acc=w3_accumulator)

            # 显式模拟原路径的 FP16 GEMM 输出、FP16 SiLU 输出与 FP16 乘法。
            gate = w1_accumulator.to(tl.float16)
            up = w3_accumulator.to(tl.float16)
            gate_fp32 = gate.to(tl.float32)
            silu_gate = (gate_fp32 * tl.sigmoid(gate_fp32)).to(tl.float16)
            activated = (silu_gate * up).to(tl.float16)
            tl.store(
                output_ptr
                + rows[:, None] * stride_om
                + cols[None, :] * stride_on,
                activated,
                mask=row_mask[:, None] & col_mask[None, :],
            )
            tile_idx += NUM_SM

        problem_start += problem_tiles


@triton.jit
def deterministic_unpermute_reduce_kernel(
    expert_output_ptr,
    inverse_order_ptr,
    routing_weights_ptr,
    output_ptr,
    hidden_dim,
    stride_em,
    stride_ed,
    stride_wt,
    stride_ws,
    stride_ot,
    stride_od,
    TOP_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """按固定 top-k slot 顺序归并，避免 atomic index_add 的非确定性。"""
    token = tl.program_id(0)
    column_tile = tl.program_id(1)
    columns = column_tile * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    column_mask = columns < hidden_dim
    accumulator = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)
    for slot in range(TOP_K):
        flat_assignment = token * TOP_K + slot
        sorted_assignment = tl.load(inverse_order_ptr + flat_assignment)
        weight = tl.load(routing_weights_ptr + token * stride_wt + slot * stride_ws)
        value = tl.load(
            expert_output_ptr
            + sorted_assignment * stride_em
            + columns * stride_ed,
            mask=column_mask,
            other=0.0,
        )
        accumulator += value.to(tl.float32) * weight.to(tl.float32)
    tl.store(
        output_ptr + token * stride_ot + columns * stride_od,
        accumulator.to(tl.float16),
        mask=column_mask,
    )


def mixedbit_variable_m_grouped_gemm(
    x: Tensor,
    expert_offsets: Tensor,
    wq: Tensor,
    scales: Tensor,
    zeros: Tensor,
    nbits: Tensor,
    group_sizes: Tensor,
    wq_offsets: Tensor,
    scale_offsets: Tensor,
    compute_dtype=torch.float16,
) -> Tensor:
    """计算按 expert 连续排列、但每个 expert 行数不同的一组混合精度 GEMM。"""
    if x.ndim != 2 or expert_offsets.ndim != 1:
        raise ValueError("x 必须为二维，expert_offsets 必须为一维")
    total_m, K = x.shape
    if total_m <= 0:
        raise ValueError("至少需要一个 expert assignment")
    E = nbits.numel()
    if expert_offsets.numel() != E + 1:
        raise ValueError("expert_offsets 长度必须等于 expert 数加一")
    _, N = wq.shape
    output = torch.empty((total_m, N), device=x.device, dtype=compute_dtype)
    num_sm = torch.cuda.get_device_properties(x.device).multi_processor_count
    m_bucket = bucket_total_assignments(total_m)

    mixedbit_variable_m_grouped_gemm_kernel[(num_sm,)](
        x,
        wq,
        output,
        expert_offsets,
        scales,
        zeros,
        nbits,
        group_sizes,
        wq_offsets,
        scale_offsets,
        total_m,
        N,
        K,
        E,
        m_bucket,
        x.stride(0),
        x.stride(1),
        wq.stride(0),
        wq.stride(1),
        output.stride(0),
        output.stride(1),
        scales.stride(0),
        scales.stride(1),
        NUM_SM=num_sm,
    )
    return output


def mixedbit_fused_up_activation(
    x: Tensor,
    expert_offsets: Tensor,
    w1: Tensor,
    w3: Tensor,
    w1_scales: Tensor,
    w1_zeros: Tensor,
    w3_scales: Tensor,
    w3_zeros: Tensor,
    nbits: Tensor,
    group_sizes: Tensor,
    wq_offsets: Tensor,
    scale_offsets: Tensor,
) -> Tensor:
    """融合两个上投影和 SwiGLU 激活，只物化一个中间张量。"""
    total_m, K = x.shape
    _, N = w1.shape
    E = nbits.numel()
    output = torch.empty((total_m, N), device=x.device, dtype=torch.float16)
    num_sm = torch.cuda.get_device_properties(x.device).multi_processor_count
    m_bucket = bucket_total_assignments(total_m)
    mixedbit_fused_up_activation_kernel[(num_sm,)](
        x,
        w1,
        w3,
        output,
        expert_offsets,
        w1_scales,
        w1_zeros,
        w3_scales,
        w3_zeros,
        nbits,
        group_sizes,
        wq_offsets,
        scale_offsets,
        total_m,
        N,
        K,
        E,
        m_bucket,
        x.stride(0),
        x.stride(1),
        w1.stride(0),
        w1.stride(1),
        output.stride(0),
        output.stride(1),
        w1_scales.stride(0),
        w1_scales.stride(1),
        NUM_SM=num_sm,
    )
    return output


def deterministic_unpermute_reduce(
    expert_output: Tensor,
    assignment_order: Tensor,
    routing_weights: Tensor,
) -> Tensor:
    """恢复 token 顺序，并按 top-k slot 的固定顺序求和。"""
    tokens, top_k = routing_weights.shape
    hidden_dim = expert_output.shape[1]
    inverse_order = torch.empty_like(assignment_order)
    inverse_order.scatter_(
        0,
        assignment_order,
        torch.arange(assignment_order.numel(), device=assignment_order.device),
    )
    output = torch.empty(
        (tokens, hidden_dim), device=expert_output.device, dtype=expert_output.dtype
    )
    block_size_d = 128
    deterministic_unpermute_reduce_kernel[
        (tokens, triton.cdiv(hidden_dim, block_size_d))
    ](
        expert_output,
        inverse_order,
        routing_weights,
        output,
        hidden_dim,
        expert_output.stride(0),
        expert_output.stride(1),
        routing_weights.stride(0),
        routing_weights.stride(1),
        output.stride(0),
        output.stride(1),
        TOP_K=top_k,
        BLOCK_SIZE_D=block_size_d,
    )
    return output
