#!/usr/bin/env python3
"""只读盘点 HQQ 检查点，为 vLLM 导出建立可验证输入清单。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import torch


ATTENTION = re.compile(
    r"^model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$"
)
EXPERT = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)$"
)
QUANT_KEYS = {"W_q", "nbits", "group_size", "shape", "scale", "zero", "packing"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quant_record(name: str, value: object) -> dict:
    if not isinstance(value, dict) or not QUANT_KEYS.issubset(value):
        raise ValueError(f"量化模块字段不完整: {name}")
    shape = [int(item) for item in value["shape"]]
    record = {
        "bits": int(value["nbits"]),
        "group_size": int(value["group_size"]),
        "shape": shape,
        "packing": str(value["packing"]),
        "qweight_shape": list(value["W_q"].shape),
        "scale_shape": list(value["scale"].shape),
        "zero_shape": list(value["zero"].shape),
        "qweight_dtype": str(value["W_q"].dtype).removeprefix("torch."),
        "scale_dtype": str(value["scale"].dtype).removeprefix("torch."),
        "zero_dtype": str(value["zero"].dtype).removeprefix("torch."),
    }
    if record["bits"] not in {1, 2, 3, 4}:
        raise ValueError(f"不支持的位宽: {name}={record['bits']}")
    if record["group_size"] != 128:
        raise ValueError(f"不支持的 group_size: {name}={record['group_size']}")
    if value.get("axis") != 1:
        raise ValueError(f"只支持 axis=1: {name}")
    return record


def build_inventory(checkpoint: Path, include_hash: bool) -> dict:
    config_path = checkpoint / "config.json"
    weights_path = checkpoint / "qmodel.pt"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError("检查点必须同时包含 config.json 与 qmodel.pt")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("architectures") != ["OlmoeForCausalLM"]:
        raise ValueError("首版导出仅支持 OlmoeForCausalLM")

    state = torch.load(weights_path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(state, dict):
        raise TypeError("qmodel.pt 根节点必须为字典")

    num_layers = int(config["num_hidden_layers"])
    num_experts = int(config["num_experts"])
    layers = [{
        "index": index,
        "attention": {},
        "experts": {
            projection: [None] * num_experts
            for projection in ("gate_proj", "up_proj", "down_proj")
        },
    } for index in range(num_layers)]
    packing = Counter()
    bits = Counter()
    quantized = 0
    full_precision = 0

    for name, value in state.items():
        attention = ATTENTION.match(name)
        expert = EXPERT.match(name)
        if attention or expert:
            record = quant_record(name, value)
            quantized += 1
            packing[record["packing"]] += 1
            bits[str(record["bits"])] += 1
            if attention:
                layer_index, projection = int(attention.group(1)), attention.group(2)
                layers[layer_index]["attention"][projection] = record
            else:
                layer_index = int(expert.group(1))
                expert_index = int(expert.group(2))
                projection = expert.group(3)
                layers[layer_index]["experts"][projection][expert_index] = record
        elif isinstance(value, dict) and "weight" in value:
            full_precision += 1

    for layer in layers:
        if set(layer["attention"]) != {"q_proj", "k_proj", "v_proj", "o_proj"}:
            raise ValueError(f"第 {layer['index']} 层 attention 投影不完整")
        expert_bits = []
        for projection, records in layer["experts"].items():
            if any(record is None for record in records):
                raise ValueError(f"第 {layer['index']} 层 {projection} expert 不完整")
            expert_bits.append([record["bits"] for record in records])
        if not (expert_bits[0] == expert_bits[1] == expert_bits[2]):
            raise ValueError(f"第 {layer['index']} 层三组 expert 位宽不一致")

    result = {
        "schema_version": 1,
        "status": "valid-input",
        "checkpoint": str(checkpoint.resolve()),
        "architecture": config["architectures"][0],
        "model": {
            "num_layers": num_layers,
            "num_experts": num_experts,
            "hidden_size": int(config["hidden_size"]),
            "intermediate_size": int(config["intermediate_size"]),
            "top_k": int(config["num_experts_per_tok"]),
        },
        "module_counts": {
            "state_entries": len(state),
            "quantized": quantized,
            "full_precision": full_precision,
        },
        "bit_histogram": dict(sorted(bits.items())),
        "packing_histogram": dict(sorted(packing.items())),
        "config_sha256": sha256_file(config_path),
        "qmodel": {
            "size_bytes": weights_path.stat().st_size,
            "sha256": sha256_file(weights_path) if include_hash else None,
        },
        "layers": layers,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-qmodel-hash", action="store_true")
    args = parser.parse_args()
    result = build_inventory(args.checkpoint, not args.skip_qmodel_hash)
    serialized = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(args.output)
    print(serialized, end="")


if __name__ == "__main__":
    main()
