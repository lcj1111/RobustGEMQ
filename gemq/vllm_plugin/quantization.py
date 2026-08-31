"""vLLM 0.28 的 GEMQ attention 与混合位宽 MoE 推理方法。"""

from __future__ import annotations

import math
import os
import re
import types
from typing import Any

import torch

from gemlite.core import DTYPE_TO_TORCH, GEMLITE_ACC_DTYPE, TORCH_TO_DTYPE, forward_functional
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs

from gemq.triton_kernels.mixedbit_moe_prefill import (
    mixedbit_fused_up_activation,
    mixedbit_variable_m_grouped_gemm,
)
from gemq.triton_kernels.vllm_moe_dispatch import (
    fused_chunk_unpermute_reduce,
    stable_expert_dispatch,
    write_chunk_expert_offsets,
)


LAYER_INDEX = re.compile(r"^model\.layers\.(\d+)\.")


def _exact_weight_loader(
    parameter: torch.Tensor,
    loaded_weight: torch.Tensor,
    shard_id: str | int | None = None,
    *args,
    **kwargs,
) -> None:
    """加载导出器已完成融合的张量，不允许 vLLM 再次切分。"""

    if shard_id is not None or args or kwargs:
        raise ValueError(f"GEMQ 精确张量不接受二次分片: shard_id={shard_id}")
    default_weight_loader(parameter, loaded_weight)


def _parameter(layer: torch.nn.Module, name: str, shape: tuple[int, ...], dtype: torch.dtype):
    parameter = torch.nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
    layer.register_parameter(name, parameter)
    set_weight_attrs(parameter, {"weight_loader": _exact_weight_loader})


def _layer_index(prefix: str) -> int:
    match = LAYER_INDEX.match(prefix)
    if match is None:
        raise ValueError(f"无法从 vLLM prefix 解析层号: {prefix}")
    return int(match.group(1))


def _load_gemq_expert_weights(self, weights):
    """让 RoutedExperts 接受导出器已融合、无需逐 expert 映射的张量。"""

    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    fallback = []
    for name, loaded_weight in weights:
        local_name = name.removeprefix("routed_experts.")
        if not local_name.startswith("gemq_"):
            fallback.append((name, loaded_weight))
            continue
        parameter = getattr(self, local_name, None)
        if parameter is None:
            raise AttributeError(f"{self.layer_name} 缺少导出参数 {local_name}")
        _exact_weight_loader(parameter, loaded_weight)
        yield local_name
    if fallback:
        yield from RoutedExperts.load_weights(self, fallback)


@register_quantization_config("gemq")
class GEMQConfig(QuantizationConfig):
    """只支持导出器生成的 OLMoE、单卡推理检查点。"""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        if config.get("quant_method") != "gemq" or config.get("schema_version") != 1:
            raise ValueError("GEMQ vLLM 配置版本不受支持")
        if config.get("group_size") != 128 or config.get("packing_bitwidth") != 32:
            raise ValueError("首版仅支持 group_size=128 与 32-bit packing")
        model = config.get("model")
        layers = config.get("layers")
        if not isinstance(model, dict) or not isinstance(layers, list):
            raise ValueError("GEMQ 配置缺少 model/layers 元数据")
        if len(layers) != int(model.get("num_layers", -1)):
            raise ValueError("GEMQ 层级元数据数量不一致")
        self.config = config
        self.model = model
        self.layers = {int(layer["index"]): layer for layer in layers}

    @classmethod
    def get_name(cls) -> str:
        return "gemq"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        # 当前 grouped/chunked kernel 的累加结果显式落为 FP16；BF16 需单独适配。
        return [torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GEMQConfig":
        return cls(config)

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        # RoutedExperts 延迟导入可避开 vLLM 模块初始化时的循环依赖。
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

        if isinstance(layer, RoutedExperts):
            return GEMQMoEMethod(self, layer.moe_config, prefix)
        if isinstance(layer, LinearBase):
            if prefix.endswith(".qkv_proj") or prefix.endswith(".o_proj"):
                return GEMQLinearMethod(prefix)
            return UnquantizedLinearMethod()
        return None


class GEMQLinearMethod(LinearMethodBase):
    def __init__(self, prefix: str):
        self.prefix = prefix

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        local_output = sum(output_partition_sizes)
        if input_size_per_partition != input_size or local_output != output_size:
            raise NotImplementedError("GEMQ vLLM 首版只支持 tensor_parallel_size=1")
        if input_size % 8:
            raise ValueError("W4 32-bit packing 要求输入维度可被 8 整除")
        _parameter(layer, "gemq_qweight", (input_size // 8, output_size), torch.int32)
        _parameter(layer, "gemq_scales", (input_size // 128, output_size), params_dtype)
        _parameter(layer, "gemq_zeros", (input_size // 128, output_size), params_dtype)

    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        data_type = TORCH_TO_DTYPE[x.dtype]
        if DTYPE_TO_TORCH[data_type.value] != x.dtype:
            raise TypeError(f"GemLite 不支持 activation dtype: {x.dtype}")
        metadata = [
            0,  # scaled_activations
            4,
            128,
            15,
            8,  # 一个 int32 存放 8 个 4-bit 值
            data_type.value,
            data_type.value,
            GEMLITE_ACC_DTYPE[data_type].value,
            data_type.value,
            0,  # channel_scale_mode
            4,  # q * scale + additive_zero
            1,  # packed tensor contiguous
        ]
        meta_scale = torch.tensor(0.0, dtype=torch.float32, device=x.device)
        return forward_functional(
            x,
            bias,
            [layer.gemq_qweight, layer.gemq_scales, layer.gemq_zeros, meta_scale],
            metadata,
        )


class GEMQMoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config: GEMQConfig, moe, prefix: str):
        super().__init__(moe)
        self.quant_config = quant_config
        self.prefix = prefix
        self.layer_manifest = quant_config.layers[_layer_index(prefix)]
        try:
            self.chunk_tokens = int(os.environ.get("GEMQ_PREFILL_CHUNK_TOKENS", "512"))
        except ValueError as exc:
            raise ValueError("GEMQ_PREFILL_CHUNK_TOKENS 必须为正整数") from exc
        if self.chunk_tokens <= 0:
            raise ValueError("GEMQ_PREFILL_CHUNK_TOKENS 必须为正整数")
        self.debug_validate = os.environ.get("GEMQ_VLLM_DEBUG_VALIDATE") == "1"
        self._debug_printed = False
        self._debug_chunk_printed = False

    @staticmethod
    def _packed_rows(input_size: int, bits: list[int]) -> int:
        return sum(math.ceil(input_size / (32 // bit)) for bit in bits)

    def create_weights(
        self,
        layer,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        expected_experts = int(self.quant_config.model["num_experts"])
        expected_intermediate = int(self.quant_config.model["intermediate_size"])
        if num_experts != expected_experts or intermediate_size_per_partition != expected_intermediate:
            raise NotImplementedError("GEMQ vLLM 首版只支持单卡、完整 expert 权重")
        projection_bits = {
            "w1": self.layer_manifest["experts"]["gate_proj"],
            "w2": self.layer_manifest["experts"]["down_proj"],
            "w3": self.layer_manifest["experts"]["up_proj"],
        }
        for projection, bits in projection_bits.items():
            if len(bits) != num_experts or any(bit not in {1, 2, 3, 4} for bit in bits):
                raise ValueError(f"{self.prefix} 的 {projection} 位宽元数据非法")
            input_size = hidden_size if projection in {"w1", "w3"} else expected_intermediate
            output_size = expected_intermediate if projection in {"w1", "w3"} else hidden_size
            _parameter(
                layer,
                f"gemq_{projection}_qweight",
                (self._packed_rows(input_size, bits), output_size),
                torch.int32,
            )
            _parameter(
                layer,
                f"gemq_{projection}_scales",
                (num_experts * input_size // 128, output_size),
                params_dtype,
            )
            _parameter(
                layer,
                f"gemq_{projection}_zeros",
                (num_experts * input_size // 128, output_size),
                params_dtype,
            )
            for suffix in ("nbits", "group_sizes", "qweight_offsets", "scale_offsets"):
                dtype = torch.int32 if suffix in {"nbits", "group_sizes"} else torch.int64
                _parameter(layer, f"gemq_{projection}_{suffix}", (num_experts,), dtype)
        # MoERunner 会把本层权重整体委托给 RoutedExperts；实例级分支只截获
        # gemq_*，其余名称仍进入 vLLM 原生逐 expert loader。
        layer.load_weights = types.MethodType(_load_gemq_expert_weights, layer)

    def get_fused_moe_quant_config(self, layer):
        # 本方法直接调用 RobustGEMQ kernel，不交给 vLLM 的统一 INT4/INT8 kernel。
        return None

    @staticmethod
    def _projection(layer, name: str):
        return (
            getattr(layer, f"gemq_{name}_qweight"),
            getattr(layer, f"gemq_{name}_scales"),
            getattr(layer, f"gemq_{name}_zeros"),
            getattr(layer, f"gemq_{name}_nbits"),
            getattr(layer, f"gemq_{name}_group_sizes"),
            getattr(layer, f"gemq_{name}_qweight_offsets"),
            getattr(layer, f"gemq_{name}_scale_offsets"),
        )

    def _run_chunk(self, layer, expert_input, expert_offsets):
        w1 = self._projection(layer, "w1")
        w2 = self._projection(layer, "w2")
        w3 = self._projection(layer, "w3")
        if self.debug_validate and not self._debug_chunk_printed:
            print(
                "GEMQ_VLLM_CHUNK_DEBUG",
                {
                    "prefix": self.prefix,
                    "expert_input": (
                        tuple(expert_input.shape),
                        tuple(expert_input.stride()),
                        str(expert_input.dtype),
                        bool(torch.isfinite(expert_input).all().item()),
                    ),
                    "offsets_head": expert_offsets[:10].tolist(),
                    "offsets_tail": expert_offsets[-3:].tolist(),
                    "w1": [(tuple(t.shape), tuple(t.stride()), str(t.dtype)) for t in w1],
                    "w1_nbits_head": w1[3][:10].tolist(),
                    "w1_qoffsets_head": w1[5][:10].tolist(),
                    "w1_soffsets_head": w1[6][:10].tolist(),
                },
                flush=True,
            )
            self._debug_chunk_printed = True
        activated = mixedbit_fused_up_activation(
            expert_input,
            expert_offsets,
            w1[0],
            w3[0],
            w1[1],
            w1[2],
            w3[1],
            w3[2],
            w1[3],
            w1[4],
            w1[5],
            w1[6],
        )
        return mixedbit_variable_m_grouped_gemm(
            activated, expert_offsets, *w2
        )

    def apply(
        self,
        layer,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts,
        shared_experts_input,
    ) -> torch.Tensor:
        if shared_experts is not None or shared_experts_input is not None:
            raise NotImplementedError("OLMoE GEMQ 路径不支持 shared experts")
        x = x.view(-1, x.shape[-1])
        topk_weights = topk_weights.to(x.dtype)
        if x.stride(-1) != 1:
            x = x.contiguous()
        if topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
            raise ValueError("vLLM 路由输出的 shape 不满足 [tokens, top_k]")
        if self.debug_validate and not self._debug_printed:
            torch.cuda.synchronize(x.device)
            print(
                "GEMQ_VLLM_DEBUG",
                {
                    "prefix": self.prefix,
                    "x_shape": tuple(x.shape),
                    "x_stride": tuple(x.stride()),
                    "x_dtype": str(x.dtype),
                    "topk_shape": tuple(topk_ids.shape),
                    "topk_dtype": str(topk_ids.dtype),
                    "topk_min": int(topk_ids.min().item()),
                    "topk_max": int(topk_ids.max().item()),
                    "weights_dtype": str(topk_weights.dtype),
                    "weights_finite": bool(torch.isfinite(topk_weights).all().item()),
                },
                flush=True,
            )
            self._debug_printed = True
        top_k = topk_ids.shape[-1]
        sorted_tokens, inverse_order, global_offsets = stable_expert_dispatch(
            topk_ids, layer.global_num_experts
        )
        output_accumulator = torch.zeros(
            (x.shape[0], x.shape[1]), dtype=torch.float32, device=x.device
        )
        final_output = torch.empty_like(x)
        chunk_offsets = torch.empty_like(global_offsets)
        assignment_limit = self.chunk_tokens * top_k
        num_assignments = sorted_tokens.numel()
        for start in range(0, num_assignments, assignment_limit):
            end = min(start + assignment_limit, num_assignments)
            write_chunk_expert_offsets(
                global_offsets, chunk_offsets, start, end
            )
            expert_input = x.index_select(0, sorted_tokens[start:end])
            expert_output = self._run_chunk(layer, expert_input, chunk_offsets)
            fused_chunk_unpermute_reduce(
                expert_output,
                inverse_order,
                topk_weights,
                output_accumulator,
                start,
                end,
                final_output=final_output if end == num_assignments else None,
            )
        return final_output
