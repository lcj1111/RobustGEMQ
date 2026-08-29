import pytest

from gemq.triton_kernels.dequant_group_gemm import (
    bucket_expert_tokens,
    get_cuda_autotune_config,
)
from gemq.triton_kernels.mixedbit_moe_prefill import (
    bucket_total_assignments,
    get_fused_up_autotune_config,
    get_moe_prefill_autotune_config,
)


@pytest.mark.parametrize(
    ("tokens", "bucket"),
    [(1, 16), (16, 16), (17, 32), (32, 32), (33, 64), (511, 512), (513, 1024)],
)
def test_bucket_expert_tokens(tokens, bucket):
    assert bucket_expert_tokens(tokens) == bucket


def test_autotune_covers_short_and_long_prefill_tiles():
    block_m_values = {config.kwargs["BLOCK_SIZE_M"] for config in get_cuda_autotune_config()}
    assert block_m_values == {16, 32, 64}
    assert all("NUM_SM" not in config.kwargs for config in get_cuda_autotune_config())


def test_bucket_expert_tokens_rejects_empty_problem():
    with pytest.raises(ValueError, match="必须为正数"):
        bucket_expert_tokens(0)


@pytest.mark.parametrize(
    ("assignments", "bucket"),
    [(1, 128), (128, 128), (129, 256), (1024, 1024), (1025, 2048)],
)
def test_bucket_total_assignments(assignments, bucket):
    assert bucket_total_assignments(assignments) == bucket


def test_grouped_kernel_autotune_covers_variable_m_tiles():
    block_m_values = {
        config.kwargs["BLOCK_SIZE_M"]
        for config in get_moe_prefill_autotune_config()
    }
    assert block_m_values == {16, 32, 64}


def test_fused_up_configs_limit_register_pressure():
    configs = get_fused_up_autotune_config()
    assert {config.kwargs["BLOCK_SIZE_M"] for config in configs} == {16, 32, 64}
    assert all(config.kwargs["BLOCK_SIZE_N"] <= 128 for config in configs)
