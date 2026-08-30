import copy
import hashlib
import json

import pytest

from gemq.vllm_plugin.checkpoint_schema import ManifestError, load_manifest, validate_manifest


def _manifest(artifact_sha: str, artifact_size: int):
    bits = [3, 2]
    return {
        "schema_version": 1,
        "format": "gemq-vllm",
        "base_model": {
            "source": "allenai/OLMoE-1B-7B-0924",
            "architecture": "OlmoeForCausalLM",
            "config_sha256": "0" * 64,
        },
        "quantization": {
            "method": "gemq",
            "group_size": 128,
            "packing_bitwidth": 32,
            "compute_dtype": "float16",
        },
        "model": {
            "num_layers": 1,
            "num_experts": 2,
            "hidden_size": 8,
            "intermediate_size": 4,
            "top_k": 1,
        },
        "layers": [{
            "index": 0,
            "attention": {"q_proj": 4, "k_proj": 4, "v_proj": 4, "o_proj": 4},
            "experts": {
                "gate_proj": bits,
                "up_proj": bits,
                "down_proj": bits,
            },
        }],
        "artifacts": [{
            "path": "model.safetensors",
            "size_bytes": artifact_size,
            "sha256": artifact_sha,
        }],
    }


def test_manifest_and_artifact_hash(tmp_path):
    payload = b"formal-checkpoint"
    artifact = tmp_path / "model.safetensors"
    artifact.write_bytes(payload)
    manifest = _manifest(hashlib.sha256(payload).hexdigest(), len(payload))
    path = tmp_path / "gemq_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert load_manifest(path)["format"] == "gemq-vllm"


def test_rejects_duplicate_layer_and_wrong_expert_count():
    manifest = _manifest("0" * 64, 1)
    manifest["model"]["num_layers"] = 2
    manifest["layers"].append(copy.deepcopy(manifest["layers"][0]))
    with pytest.raises(ManifestError, match="重复"):
        validate_manifest(manifest)

    manifest = _manifest("0" * 64, 1)
    manifest["layers"][0]["experts"]["up_proj"] = [3]
    with pytest.raises(ManifestError, match="2 个 expert"):
        validate_manifest(manifest)


def test_rejects_path_escape_and_unsupported_group_size():
    manifest = _manifest("0" * 64, 1)
    manifest["artifacts"][0]["path"] = "../model.safetensors"
    with pytest.raises(ManifestError, match="逃逸"):
        validate_manifest(manifest)

    manifest = _manifest("0" * 64, 1)
    manifest["quantization"]["group_size"] = 64
    with pytest.raises(ManifestError, match="group_size=128"):
        validate_manifest(manifest)


def test_rejects_bfloat16_until_kernel_supports_it():
    manifest = _manifest("0" * 64, 1)
    manifest["quantization"]["compute_dtype"] = "bfloat16"
    with pytest.raises(ManifestError, match="仅支持 float16"):
        validate_manifest(manifest)
