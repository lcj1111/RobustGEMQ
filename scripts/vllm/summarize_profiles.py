#!/usr/bin/env python3
"""将 vLLM Torch/CUPTI 原始表转换为可复算的服务路径分解。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any


CASES = {
    "baseline_prefill": (
        "artifacts/vllm/profiles/raw/baseline-prefill.txt",
        "artifacts/vllm/profiles/raw/baseline-prefill-requests.json",
    ),
    "baseline_decode": (
        "artifacts/vllm/profiles/raw/baseline-decode.txt",
        "artifacts/vllm/profiles/raw/baseline-decode-requests.json",
    ),
    "optimized_prefill": (
        "artifacts/vllm/profiles/raw/optimized-prefill.txt",
        "artifacts/vllm/profiles/raw/optimized-prefill-requests.json",
    ),
    "optimized_decode": (
        "artifacts/vllm/profiles/raw/optimized-decode.txt",
        "artifacts/vllm/profiles/raw/optimized-decode-requests.json",
    ),
}

OPERATIONS = (
    "vllm::moe_forward",
    "aten::sort",
    "aten::bincount",
    "aten::cumsum",
    "aten::cat",
    "aten::scatter_",
    "aten::div",
    "aten::arange",
    "aten::index_select",
    "stable_count_offsets_kernel",
    "stable_scatter_dispatch_kernel",
    "chunk_offsets_from_global_kernel",
    "mixedbit_fused_up_activation_kernel",
    "mixedbit_variable_m_grouped_gemm_kernel",
    "deterministic_chunk_reduce_kernel",
    "fused_weighted_unpermute_reduce_kernel",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def time_ms(value: str) -> float:
    match = re.fullmatch(r"([0-9.]+)(ns|us|ms|s)", value)
    if match is None:
        raise ValueError(f"无法解析 profiler 时间：{value}")
    number = float(match.group(1))
    return number * {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1000.0}[
        match.group(2)
    ]


def parse_table(path: Path) -> dict[str, dict[str, float | int]]:
    rows: dict[str, dict[str, float | int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = re.split(r"\s{2,}", line.strip())
        if len(fields) != 11 or not fields[-1].isdigit():
            continue
        name = fields[0]
        if name not in OPERATIONS:
            continue
        rows[name] = {
            "self_cpu_ms": time_ms(fields[2]),
            "cpu_total_ms": time_ms(fields[4]),
            "self_cuda_ms": time_ms(fields[6]),
            "cuda_total_ms": time_ms(fields[8]),
            "calls": int(fields[10]),
        }
    if "vllm::moe_forward" not in rows:
        raise ValueError(f"{path} 缺少 vllm::moe_forward")
    return rows


def nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(1, math.ceil(percentile * len(ordered))) - 1]


def parse_requests(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("schema_version") != 1 or result.get("status") != "pass":
        raise ValueError(f"{path} schema/status 非法")
    if "uncached" not in result.get("label", ""):
        raise ValueError(f"{path} 未标记为 uncached profiler")
    workload = result["workload"]
    requests = result["requests"]
    if workload["concurrency"] != 8 or len(requests) != 8:
        raise ValueError(f"{path} 不是固定 c8 profiler 负载")
    if [item["request_id"] for item in requests] != list(range(8)):
        raise ValueError(f"{path} 请求 identity 不完整")
    return result


def request_metrics(result: dict[str, Any]) -> dict[str, float]:
    ttft = [float(item["ttft_seconds"]) for item in result["requests"]]
    e2e = [float(item["e2e_seconds"]) for item in result["requests"]]
    return {
        "ttft_p50_ms": 1000.0 * nearest_rank(ttft, 0.50),
        "ttft_p95_ms": 1000.0 * nearest_rank(ttft, 0.95),
        "ttft_p99_ms": 1000.0 * nearest_rank(ttft, 0.99),
        "e2e_p95_ms": 1000.0 * nearest_rank(e2e, 0.95),
    }


def build_summary(repo: Path) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    source_files = []
    raw_requests: dict[str, dict[str, Any]] = {}
    for name, (table_relative, request_relative) in CASES.items():
        table_path = repo / table_relative
        request_path = repo / request_relative
        table = parse_table(table_path)
        requests = parse_requests(request_path)
        raw_requests[name] = requests
        cases[name] = {
            "workload": requests["workload"],
            "request_metrics": request_metrics(requests),
            "operations": table,
        }
        for relative, kind in ((table_relative, "torch_cupti_table"),
                               (request_relative, "request_records")):
            source_files.append(
                {"kind": kind, "path": relative, "sha256": sha256(repo / relative)}
            )

    for phase in ("prefill", "decode"):
        baseline = raw_requests[f"baseline_{phase}"]["workload"]
        optimized = raw_requests[f"optimized_{phase}"]["workload"]
        if baseline != optimized:
            raise ValueError(f"{phase} 的 baseline/optimized workload identity 不一致")

    comparisons = {}
    for phase in ("prefill", "decode"):
        baseline = cases[f"baseline_{phase}"]
        optimized = cases[f"optimized_{phase}"]
        baseline_moe = baseline["operations"]["vllm::moe_forward"]
        optimized_moe = optimized["operations"]["vllm::moe_forward"]
        baseline_ttft = baseline["request_metrics"]["ttft_p95_ms"]
        optimized_ttft = optimized["request_metrics"]["ttft_p95_ms"]
        comparisons[phase] = {
            "moe_cpu_total_reduction": 1.0
            - optimized_moe["cpu_total_ms"] / baseline_moe["cpu_total_ms"],
            "moe_cuda_total_reduction": 1.0
            - optimized_moe["cuda_total_ms"] / baseline_moe["cuda_total_ms"],
            "ttft_p95_reduction": 1.0 - optimized_ttft / baseline_ttft,
            "removed_eager_dispatch_ops": [
                name
                for name in (
                    "aten::sort",
                    "aten::bincount",
                    "aten::cumsum",
                    "aten::cat",
                    "aten::div",
                    "aten::arange",
                )
                if name in baseline["operations"]
                and name not in optimized["operations"]
            ],
        }

    return {
        "schema_version": 1,
        "status": "pass",
        "subject": "RobustGEMQ vLLM 服务路径 dispatch/reduce profiler 分解",
        "protocol": {
            "engine": "vLLM 0.28.0 real OpenAI-compatible service",
            "profiler": "torch.profiler with CUDA/CUPTI",
            "concurrency": 8,
            "prefix_caching": False,
            "prefill": {"requests": 8, "prompt_tokens": 512, "output_tokens": 1},
            "decode": {"requests": 8, "prompt_tokens": 128, "output_tokens": 16},
            "scope": "fixed-checkpoint descriptive single-run profile",
        },
        "cases": cases,
        "comparisons": comparisons,
        "source_files": source_files,
        "interpretation_boundary": [
            "CPU total 可能包含同步或嵌套算子，不能将各行直接相加",
            "profiler 只用于定位路径占比；正式 TTFT、吞吐与显存以 24 请求服务基准为准",
            "prefill/decode 均关闭 prefix cache，不外推到其他模型、GPU 或请求分布",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/vllm/profiles/summary.json"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    payload = build_summary(repo)
    output = repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        temporary.write_bytes(
            (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(payload["comparisons"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
