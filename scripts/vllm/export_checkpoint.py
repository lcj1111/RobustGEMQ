#!/usr/bin/env python3
"""将 HQQ/GEMQ 检查点转换为 vLLM 插件可直接加载的格式。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import torch
from safetensors.torch import save_file

from gemq.inference.patch import prepare_for_inference
from gemq.utils.hf_loading import load_quantized_model


TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "generation_config.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """解除共享存储；safetensors 要求每个键拥有独立连续存储。"""

    return tensor.detach().to("cpu").contiguous().clone()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_quantized_tensors(model, config: dict) -> tuple[dict[str, torch.Tensor], list[dict]]:
    tensors: dict[str, torch.Tensor] = {}
    layers = []
    for index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        qkv = [attention.q_proj, attention.k_proj, attention.v_proj]
        for projection in (*qkv, attention.o_proj):
            if projection.W_nbits != 4 or projection.group_size != 128:
                raise ValueError(f"第 {index} 层 attention 不是预期的 W4-G128")

        qkv_prefix = f"model.layers.{index}.self_attn.qkv_proj"
        tensors[f"{qkv_prefix}.gemq_qweight"] = cpu_tensor(
            torch.cat([projection.W_q for projection in qkv], dim=1)
        )
        tensors[f"{qkv_prefix}.gemq_scales"] = cpu_tensor(
            torch.cat([projection.scales for projection in qkv], dim=1)
        )
        tensors[f"{qkv_prefix}.gemq_zeros"] = cpu_tensor(
            torch.cat([projection.zeros for projection in qkv], dim=1)
        )

        output_prefix = f"model.layers.{index}.self_attn.o_proj"
        tensors[f"{output_prefix}.gemq_qweight"] = cpu_tensor(attention.o_proj.W_q)
        tensors[f"{output_prefix}.gemq_scales"] = cpu_tensor(attention.o_proj.scales)
        tensors[f"{output_prefix}.gemq_zeros"] = cpu_tensor(attention.o_proj.zeros)

        block = layer.mlp
        expert_prefix = f"model.layers.{index}.mlp.experts"
        for projection in ("w1", "w2", "w3"):
            mapping = {
                "qweight": getattr(block, f"{projection}_wq"),
                "scales": getattr(block, f"{projection}_scales"),
                "zeros": getattr(block, f"{projection}_zeros"),
                "nbits": getattr(block, f"{projection}_nbits"),
                "group_sizes": getattr(block, f"{projection}_group_sizes"),
                "qweight_offsets": getattr(block, f"{projection}_wq_strides"),
                "scale_offsets": getattr(block, f"{projection}_zs_strides"),
            }
            for suffix, tensor in mapping.items():
                tensors[f"{expert_prefix}.gemq_{projection}_{suffix}"] = cpu_tensor(tensor)

        w1_bits = block.w1_nbits.tolist()
        w2_bits = block.w2_nbits.tolist()
        w3_bits = block.w3_nbits.tolist()
        if not (w1_bits == w2_bits == w3_bits):
            raise ValueError(f"第 {index} 层 expert 三组投影位宽不一致")
        if len(w1_bits) != int(config["num_experts"]):
            raise ValueError(f"第 {index} 层 expert 数量不一致")
        layers.append({
            "index": index,
            "attention": {
                "q_proj": 4,
                "k_proj": 4,
                "v_proj": 4,
                "o_proj": 4,
            },
            "experts": {
                "gate_proj": w1_bits,
                "up_proj": w3_bits,
                "down_proj": w2_bits,
            },
        })
    return tensors, layers


def add_full_precision_tensors(
    tensors: dict[str, torch.Tensor], source_checkpoint: Path
) -> None:
    state = torch.load(
        source_checkpoint / "qmodel.pt",
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    for module_name, module_state in state.items():
        if not isinstance(module_state, dict) or "W_q" in module_state:
            continue
        for name, value in module_state.items():
            if not isinstance(value, torch.Tensor):
                continue
            destination = f"{module_name}.{name}"
            if destination in tensors:
                raise ValueError(f"导出键重复: {destination}")
            tensors[destination] = cpu_tensor(value)


def export(args: argparse.Namespace) -> Path:
    source = args.checkpoint.resolve()
    destination = args.output.resolve()
    temporary = destination.with_name(destination.name + ".incomplete")
    if destination.exists() or temporary.exists():
        raise FileExistsError("输出目录或未完成目录已存在，请先明确处理后再导出")
    temporary.mkdir(parents=True)
    try:
        source_config_path = source / "config.json"
        config = json.loads(source_config_path.read_text(encoding="utf-8"))
        if config.get("architectures") != ["OlmoeForCausalLM"]:
            raise ValueError("首版导出仅支持 OlmoeForCausalLM")

        model = load_quantized_model(
            str(source),
            compute_dtype=torch.float16,
            device=args.device,
            trust_remote_code=False,
        )
        prepare_for_inference(model, args.model_name, is_fp=False)
        tensors, layer_manifest = collect_quantized_tensors(model, config)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        add_full_precision_tensors(tensors, source)

        weights_path = temporary / "model.safetensors"
        save_file(
            tensors,
            str(weights_path),
            metadata={"format": "pt", "gemq_schema_version": "1"},
        )
        del tensors
        gc.collect()

        for file in source.iterdir():
            if file.is_file() and file.name in TOKENIZER_FILES:
                shutil.copy2(file, temporary / file.name)

        embedded_quantization = {
            "quant_method": "gemq",
            "schema_version": 1,
            "group_size": 128,
            "packing_bitwidth": 32,
            "compute_dtype": "float16",
            "model": {
                "num_layers": int(config["num_hidden_layers"]),
                "num_experts": int(config["num_experts"]),
                "hidden_size": int(config["hidden_size"]),
                "intermediate_size": int(config["intermediate_size"]),
                "top_k": int(config["num_experts_per_tok"]),
            },
            "layers": layer_manifest,
            "manifest": "gemq_manifest.json",
        }
        exported_config = dict(config)
        exported_config["quantization_config"] = embedded_quantization
        (temporary / "config.json").write_text(
            json.dumps(exported_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        manifest = {
            "schema_version": 1,
            "format": "gemq-vllm",
            "base_model": {
                "source": args.base_model,
                "architecture": config["architectures"][0],
                "config_sha256": sha256_file(source_config_path),
            },
            "quantization": {
                "method": "gemq",
                "group_size": 128,
                "packing_bitwidth": 32,
                "compute_dtype": "float16",
            },
            "model": embedded_quantization["model"],
            "layers": layer_manifest,
            "artifacts": [{
                "path": weights_path.name,
                "size_bytes": weights_path.stat().st_size,
                "sha256": sha256_file(weights_path),
            }],
            "producer": {
                "robustgemq_commit": git_revision(),
                "source_checkpoint": str(source),
            },
        }
        (temporary / "gemq_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--model-name", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(export(args))


if __name__ == "__main__":
    main()
