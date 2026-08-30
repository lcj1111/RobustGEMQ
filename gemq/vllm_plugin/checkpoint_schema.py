"""GEMQ-vLLM 检查点清单的最小、可审计契约。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
FORMAT_NAME = "gemq-vllm"
SUPPORTED_BITS = {1, 2, 3, 4}
SUPPORTED_COMPUTE_DTYPES = {"float16"}


class ManifestError(ValueError):
    """检查点清单不满足格式契约。"""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def _positive_int(value: Any, field: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
             f"{field} 必须为正整数")
    return value


def _bits(values: Any, field: str, expected: int | None = None) -> list[int]:
    _require(isinstance(values, list) and values, f"{field} 必须为非空列表")
    _require(all(isinstance(v, int) and v in SUPPORTED_BITS for v in values),
             f"{field} 只能包含 1/2/3/4")
    if expected is not None:
        _require(len(values) == expected, f"{field} 应包含 {expected} 个 expert")
    return values


def _sha256(value: Any, field: str) -> str:
    _require(isinstance(value, str) and len(value) == 64, f"{field} 必须为 SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ManifestError(f"{field} 必须为十六进制 SHA-256") from exc
    return value.lower()


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """校验清单，并返回原对象以便加载端直接使用。"""

    _require(isinstance(manifest, dict), "清单根节点必须为对象")
    _require(manifest.get("schema_version") == SCHEMA_VERSION,
             f"schema_version 必须为 {SCHEMA_VERSION}")
    _require(manifest.get("format") == FORMAT_NAME, f"format 必须为 {FORMAT_NAME}")

    base = manifest.get("base_model")
    _require(isinstance(base, dict), "base_model 必须为对象")
    _require(isinstance(base.get("source"), str) and base["source"],
             "base_model.source 不能为空")
    _require(base.get("architecture") == "OlmoeForCausalLM",
             "首版仅支持 OlmoeForCausalLM")
    _sha256(base.get("config_sha256"), "base_model.config_sha256")

    quant = manifest.get("quantization")
    _require(isinstance(quant, dict), "quantization 必须为对象")
    _require(quant.get("method") == "gemq", "quantization.method 必须为 gemq")
    group_size = _positive_int(quant.get("group_size"), "quantization.group_size")
    _require(group_size == 128, "首版 kernel 仅支持 group_size=128")
    _require(quant.get("packing_bitwidth") == 32,
             "首版导出格式必须使用 32-bit packing")
    _require(quant.get("compute_dtype") in SUPPORTED_COMPUTE_DTYPES,
             "首版 vLLM kernel 仅支持 float16")

    model = manifest.get("model")
    _require(isinstance(model, dict), "model 必须为对象")
    num_layers = _positive_int(model.get("num_layers"), "model.num_layers")
    num_experts = _positive_int(model.get("num_experts"), "model.num_experts")
    _positive_int(model.get("hidden_size"), "model.hidden_size")
    _positive_int(model.get("intermediate_size"), "model.intermediate_size")
    _positive_int(model.get("top_k"), "model.top_k")

    layers = manifest.get("layers")
    _require(isinstance(layers, list) and len(layers) == num_layers,
             f"layers 应包含 {num_layers} 层")
    seen: set[int] = set()
    for pos, layer in enumerate(layers):
        _require(isinstance(layer, dict), f"layers[{pos}] 必须为对象")
        index = layer.get("index")
        _require(isinstance(index, int) and 0 <= index < num_layers,
                 f"layers[{pos}].index 越界")
        _require(index not in seen, f"layer index {index} 重复")
        seen.add(index)
        attention = layer.get("attention")
        _require(isinstance(attention, dict), f"layers[{pos}].attention 必须为对象")
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            _require(attention.get(projection) in SUPPORTED_BITS,
                     f"layers[{pos}].attention.{projection} 位宽非法")
        experts = layer.get("experts")
        _require(isinstance(experts, dict), f"layers[{pos}].experts 必须为对象")
        for projection in ("gate_proj", "up_proj", "down_proj"):
            _bits(experts.get(projection),
                  f"layers[{pos}].experts.{projection}", num_experts)

    artifacts = manifest.get("artifacts")
    _require(isinstance(artifacts, list) and artifacts, "artifacts 必须为非空列表")
    paths: set[str] = set()
    for pos, artifact in enumerate(artifacts):
        _require(isinstance(artifact, dict), f"artifacts[{pos}] 必须为对象")
        path = artifact.get("path")
        _require(isinstance(path, str) and path and not Path(path).is_absolute(),
                 f"artifacts[{pos}].path 必须为相对路径")
        _require(".." not in Path(path).parts, f"artifacts[{pos}].path 不得逃逸目录")
        _require(path not in paths, f"artifact path {path} 重复")
        paths.add(path)
        _positive_int(artifact.get("size_bytes"), f"artifacts[{pos}].size_bytes")
        _sha256(artifact.get("sha256"), f"artifacts[{pos}].sha256")

    return manifest


def load_manifest(path: str | Path, verify_files: bool = True) -> dict[str, Any]:
    """读取清单；正式加载默认同时核对文件大小与哈希。"""

    manifest_path = Path(path)
    manifest = validate_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
    if not verify_files:
        return manifest
    root = manifest_path.parent.resolve()
    for artifact in manifest["artifacts"]:
        candidate = (root / artifact["path"]).resolve()
        _require(candidate.is_relative_to(root), f"artifact 逃逸检查点目录: {candidate}")
        _require(candidate.is_file(), f"artifact 不存在: {candidate}")
        _require(candidate.stat().st_size == artifact["size_bytes"],
                 f"artifact 大小不符: {candidate}")
        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        _require(digest.hexdigest() == artifact["sha256"],
                 f"artifact 哈希不符: {candidate}")
    return manifest
