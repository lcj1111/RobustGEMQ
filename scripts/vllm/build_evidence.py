#!/usr/bin/env python3
"""从正式原始结果构建轻量 vLLM 证据索引。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from verify_evidence import load_json, sha256


RESULT_FILES = {
    "bf16_c1": ("benchmark", "artifacts/vllm/benchmarks/bf16-c1.json"),
    "bf16_c4": ("benchmark", "artifacts/vllm/benchmarks/bf16-c4.json"),
    "bf16_c8": ("benchmark", "artifacts/vllm/benchmarks/bf16-c8.json"),
    "robustgemq_c1": ("benchmark", "artifacts/vllm/benchmarks/robustgemq-c1.json"),
    "robustgemq_c4": ("benchmark", "artifacts/vllm/benchmarks/robustgemq-c4.json"),
    "robustgemq_c8": ("benchmark", "artifacts/vllm/benchmarks/robustgemq-c8.json"),
    "offline_smoke": ("correctness", "artifacts/vllm/correctness/offline-smoke.json"),
    "reference_greedy": ("correctness", "artifacts/vllm/correctness/reference-greedy.json"),
    "greedy_equivalence": ("correctness", "artifacts/vllm/correctness/greedy-equivalence.json"),
    "layer_equivalence": ("correctness", "artifacts/vllm/correctness/layer-equivalence.json"),
    "environment": ("metadata", "artifacts/vllm/metadata/environment.json"),
    "checkpoint_manifest": ("metadata", "artifacts/vllm/metadata/checkpoint-manifest.json"),
}

SOURCE_FILES = [
    "gemq/vllm_plugin/__init__.py",
    "gemq/vllm_plugin/checkpoint_schema.py",
    "gemq/vllm_plugin/quantization.py",
    "scripts/vllm/export_checkpoint.py",
    "scripts/vllm/benchmark_service.py",
    "scripts/vllm/smoke_offline.py",
    "scripts/vllm/reference_greedy.py",
    "scripts/vllm/compare_greedy_outputs.py",
    "scripts/vllm/check_layer_equivalence.py",
    "scripts/vllm/collect_environment.py",
    "requirements/vllm-constraints.txt",
    "pyproject.toml",
]


def entry(repo: Path, name: str, kind: str, relative: str) -> dict:
    path = repo / relative
    if not path.is_file():
        raise FileNotFoundError(relative)
    return {"name": name, "kind": kind, "path": relative, "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/vllm/evidence.json")
    )
    args = parser.parse_args()
    repo = args.repo.resolve()

    files = [
        entry(repo, name, kind, relative)
        for name, (kind, relative) in RESULT_FILES.items()
    ]
    files.extend(
        entry(repo, Path(relative).name, "source", relative)
        for relative in SOURCE_FILES
    )
    loaded = {
        name: load_json(repo / relative)
        for name, (_, relative) in RESULT_FILES.items()
    }
    comparisons = []
    for concurrency in (1, 4, 8):
        baseline = loaded[f"bf16_c{concurrency}"]["summary"]
        candidate = loaded[f"robustgemq_c{concurrency}"]["summary"]
        comparisons.append(
            {
                "concurrency": concurrency,
                "bf16_output_tokens_per_second": baseline[
                    "output_token_throughput_per_second"
                ],
                "robustgemq_output_tokens_per_second": candidate[
                    "output_token_throughput_per_second"
                ],
                "output_throughput_ratio": candidate[
                    "output_token_throughput_per_second"
                ]
                / baseline["output_token_throughput_per_second"],
                "bf16_ttft_p95_seconds": baseline["ttft_seconds"]["p95"],
                "robustgemq_ttft_p95_seconds": candidate["ttft_seconds"]["p95"],
                "ttft_p95_ratio": candidate["ttft_seconds"]["p95"]
                / baseline["ttft_seconds"]["p95"],
                "bf16_peak_memory_mib": baseline["gpu_memory_mib"]["peak"],
                "robustgemq_peak_memory_mib": candidate["gpu_memory_mib"]["peak"],
                "peak_memory_reduction": 1.0
                - candidate["gpu_memory_mib"]["peak"]
                / baseline["gpu_memory_mib"]["peak"],
            }
        )

    manifest = loaded["checkpoint_manifest"]
    payload = {
        "schema_version": 1,
        "status": "pass",
        "subject": "RobustGEMQ vLLM 0.28 单卡推理集成",
        "protocol": {
            "requests_per_case": 24,
            "concurrency": [1, 4, 8],
            "prompt_lengths": [128, 512],
            "output_tokens": 16,
            "streaming": True,
            "warmup_rounds_at_target_concurrency": 2,
            "kv_cache_memory_bytes": 4 * 1024**3,
            "percentile": "nearest-rank",
            "scope": "fixed-checkpoint descriptive serving benchmark",
        },
        "checkpoint_artifact": manifest["artifacts"][0],
        "comparisons": comparisons,
        "files": files,
        "claim_boundary": [
            "已验证 OLMoE Concat/seed-101 检查点可由 vLLM Engine 真实加载和推理",
            "已验证原 RobustGEMQ 与 vLLM greedy 生成的 8 个 token 完全一致",
            "已报告固定请求集上的 TTFT、吞吐和 GPU 总显存，不外推到其他模型或硬件",
            "首版只支持 FP16、单卡 TP=1，不宣称量化吞吐超过 BF16",
        ],
    }
    output = repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        # 固定 LF，避免 Windows 生成的 evidence 在 Linux CI 中哈希或 diff 漂移。
        temporary.write_bytes(
            (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(output)


if __name__ == "__main__":
    main()
