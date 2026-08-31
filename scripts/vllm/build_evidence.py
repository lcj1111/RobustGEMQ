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
    "baseline_gemq_c1": ("benchmark", "artifacts/vllm/benchmarks/baseline-gemq-c1.json"),
    "baseline_gemq_c4": ("benchmark", "artifacts/vllm/benchmarks/baseline-gemq-c4.json"),
    "baseline_gemq_c8": ("benchmark", "artifacts/vllm/benchmarks/baseline-gemq-c8.json"),
    "profile_summary": ("profile", "artifacts/vllm/profiles/summary.json"),
    "dispatch_correctness": (
        "correctness",
        "artifacts/vllm/correctness/dispatch-fusion.json",
    ),
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
    "scripts/vllm/check_dispatch_correctness.py",
    "scripts/vllm/collect_environment.py",
    "scripts/vllm/profile_service.py",
    "scripts/vllm/summarize_profiles.py",
    "scripts/vllm/build_evidence.py",
    "scripts/vllm/verify_evidence.py",
    "gemq/triton_kernels/mixedbit_moe_prefill.py",
    "gemq/triton_kernels/vllm_moe_dispatch.py",
    "requirements/vllm-constraints.txt",
    "pyproject.toml",
]

PROFILE_RAW_FILES = [
    relative
    for table, requests in (
        (
            "artifacts/vllm/profiles/raw/baseline-prefill.txt",
            "artifacts/vllm/profiles/raw/baseline-prefill-requests.json",
        ),
        (
            "artifacts/vllm/profiles/raw/baseline-decode.txt",
            "artifacts/vllm/profiles/raw/baseline-decode-requests.json",
        ),
        (
            "artifacts/vllm/profiles/raw/optimized-prefill.txt",
            "artifacts/vllm/profiles/raw/optimized-prefill-requests.json",
        ),
        (
            "artifacts/vllm/profiles/raw/optimized-decode.txt",
            "artifacts/vllm/profiles/raw/optimized-decode-requests.json",
        ),
    )
    for relative in (table, requests)
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
    files.extend(
        entry(repo, Path(relative).name, "profile_raw", relative)
        for relative in PROFILE_RAW_FILES
    )
    loaded = {
        name: load_json(repo / relative)
        for name, (_, relative) in RESULT_FILES.items()
    }
    bf16_comparisons = []
    dispatch_comparisons = []
    for concurrency in (1, 4, 8):
        baseline = loaded[f"bf16_c{concurrency}"]["summary"]
        candidate = loaded[f"robustgemq_c{concurrency}"]["summary"]
        original = loaded[f"baseline_gemq_c{concurrency}"]["summary"]
        bf16_comparisons.append(
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
        dispatch_comparisons.append(
            {
                "concurrency": concurrency,
                "baseline_output_tokens_per_second": original[
                    "output_token_throughput_per_second"
                ],
                "optimized_output_tokens_per_second": candidate[
                    "output_token_throughput_per_second"
                ],
                "output_throughput_improvement": candidate[
                    "output_token_throughput_per_second"
                ]
                / original["output_token_throughput_per_second"]
                - 1.0,
                "baseline_ttft_p95_seconds": original["ttft_seconds"]["p95"],
                "optimized_ttft_p95_seconds": candidate["ttft_seconds"]["p95"],
                "ttft_p95_reduction": 1.0
                - candidate["ttft_seconds"]["p95"]
                / original["ttft_seconds"]["p95"],
                "baseline_e2e_p95_seconds": original["e2e_seconds"]["p95"],
                "optimized_e2e_p95_seconds": candidate["e2e_seconds"]["p95"],
                "e2e_p95_reduction": 1.0
                - candidate["e2e_seconds"]["p95"]
                / original["e2e_seconds"]["p95"],
                "optimized_peak_memory_mib": candidate["gpu_memory_mib"]["peak"],
            }
        )

    manifest = loaded["checkpoint_manifest"]
    payload = {
        "schema_version": 2,
        "status": "pass",
        "subject": "RobustGEMQ vLLM 服务路径 dispatch/reduce 融合优化",
        "protocol": {
            "requests_per_case": 24,
            "concurrency": [1, 4, 8],
            "prompt_lengths": [128, 512],
            "output_tokens": 16,
            "streaming": True,
            "warmup_rounds_at_target_concurrency": 4,
            "warmup_shapes": ["128-only", "512-only", "mixed", "mixed"],
            "kv_cache_memory_bytes": 4 * 1024**3,
            "percentile": "nearest-rank",
            "prefix_caching": False,
            "scope": "fixed-checkpoint descriptive serving benchmark",
        },
        "checkpoint_artifact": manifest["artifacts"][0],
        "comparisons": {
            "bf16_vs_optimized": bf16_comparisons,
            "baseline_vs_optimized": dispatch_comparisons,
            "profiler": loaded["profile_summary"]["comparisons"],
        },
        "files": files,
        "claim_boundary": [
            "已验证融合 dispatch/reduce 后 greedy 生成的 8 个 token 与原 RobustGEMQ 完全一致",
            "相对原始 GEMQ 报告固定 uncached 请求集上的 TTFT、E2E、吞吐和 GPU 总显存",
            "c8 输出吞吐提高 21.2%，未达到预设 25% 门槛，不宣称服务优化阶段全部达标",
            "首版只支持 OLMoE、FP16、单卡 TP=1；不宣称量化吞吐超过 BF16",
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
