#!/usr/bin/env python3
"""离线校验 chunked prefill 的数值、workspace 与并发负载证据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{label} 无法由原始样本复算")


def verify_request_records(result: dict) -> None:
    records = result["requests"]
    expected_count = result["workload"]["num_requests"]
    request_ids = [record["request_id"] for record in records]
    if len(records) != expected_count or request_ids != list(range(expected_count)):
        raise ValueError("请求记录缺失、重复或顺序错误")
    arrivals = [record["arrival_offset_s"] for record in records]
    if arrivals[0] != 0.0 or any(
        left >= right for left, right in zip(arrivals, arrivals[1:])
    ):
        raise ValueError("到达时刻不是严格递增的开放环序列")
    ttft = [record["ttft_ms"] for record in records]
    summary = result["latency"]["ttft"]
    for key, probability in (("p50_ms", 0.50), ("p95_ms", 0.95), ("p99_ms", 0.99)):
        assert_close(summary[key], percentile(ttft, probability), f"TTFT {key}")
    assert_close(
        result["throughput"]["requests_per_second"],
        expected_count / result["duration_s"],
        "request throughput",
    )
    expected_tokens = expected_count * (
        result["workload"]["prompt_length"]
        + result["workload"]["output_tokens_per_request"]
    )
    assert_close(
        result["throughput"]["total_tokens_per_second"],
        expected_tokens / result["duration_s"],
        "total token throughput",
    )
    if result["memory"]["peak_allocated_bytes"] < result["memory"]["baseline_allocated_bytes"]:
        raise ValueError("峰值显存小于基线显存")


def verify(evidence_path: Path) -> dict:
    repo = evidence_path.resolve().parents[3]
    evidence = load_json(evidence_path)
    if evidence.get("schema_version") != 1:
        raise ValueError("不支持的 chunked evidence schema")

    loaded = {}
    entries = [*evidence["results"], *evidence["source_snapshot"]]
    for entry in entries:
        path = repo / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"缺少证据文件：{entry['path']}")
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"哈希不一致：{entry['path']}")
        if entry in evidence["results"]:
            loaded[entry["name"]] = load_json(path)

    revision = evidence["code_revision"]
    for name, result in loaded.items():
        if result.get("code_revision") != revision:
            raise ValueError(f"{name} 的 code_revision 不一致")

    correctness = loaded["correctness"]
    for length, case in correctness["cases"].items():
        if not case["allclose"] or not case["router_exact"]:
            raise ValueError(f"block 数值门槛未通过：{length}")
    end_to_end = loaded["end_to_end"]
    for length, case in end_to_end["cases"].items():
        if case["argmax_agreement"] < end_to_end["min_argmax_agreement"]:
            raise ValueError(f"整模型 argmax 门槛未通过：{length}")
        if case["mean_abs_error"] > end_to_end["max_mean_abs_error"]:
            raise ValueError(f"整模型平均误差门槛未通过：{length}")

    workspace_names = ["workspace_fused", "workspace_chunk128", "workspace_chunk256", "workspace_chunk512"]
    workspace = [loaded[name] for name in workspace_names]
    for result in workspace:
        if list(result["cases"]) != ["512", "2048", "4096"]:
            raise ValueError("workspace 扫描长度不一致")
        if result["seed"] != evidence["protocol"]["seed"]:
            raise ValueError("workspace 扫描 seed 不一致")
        if result["warmup"] != 2 or result["repeats"] != 5:
            raise ValueError("workspace 扫描预热/重复次数不是 2/5")

    concurrent_names = [
        "concurrent_fused_512",
        "concurrent_chunked_512",
        "concurrent_fused_2048",
        "concurrent_chunked_2048",
    ]
    for name in concurrent_names:
        result = loaded[name]
        verify_request_records(result)
        if result["workload"]["num_requests"] != 100:
            raise ValueError(f"{name} 请求数不是 100")
        if result["scheduler"]["policy"] != "fcfs":
            raise ValueError(f"{name} 不是 FCFS")

    for length in (512, 2048):
        fused = loaded[f"concurrent_fused_{length}"]
        chunked = loaded[f"concurrent_chunked_{length}"]
        for identity_key in ("prompt_token_ids_sha256", "arrival_schedule_sha256"):
            if fused["workload"][identity_key] != chunked["workload"][identity_key]:
                raise ValueError(f"{length} 的跨后端 workload identity 不一致")
        for key in ("request_rate_per_second", "max_num_seqs", "max_num_batched_tokens"):
            if fused["scheduler"][key] != chunked["scheduler"][key]:
                raise ValueError(f"{length} 的调度协议不一致：{key}")

    fused_4096 = loaded["workspace_fused"]["cases"]["4096"]["moe_block"]
    chunked_4096 = loaded["workspace_chunk512"]["cases"]["4096"]["moe_block"]
    if chunked_4096["peak_workspace_delta_bytes"] >= 0.30 * fused_4096["peak_workspace_delta_bytes"]:
        raise ValueError("4096-token 单层 workspace 未降低 70% 以上")

    fused_short = loaded["concurrent_fused_512"]
    chunked_short = loaded["concurrent_chunked_512"]
    if chunked_short["memory"]["peak_workspace_delta_bytes"] >= fused_short["memory"]["peak_workspace_delta_bytes"]:
        raise ValueError("短 prompt 并发负载没有降低峰值显存")
    throughput_ratio = (
        chunked_short["throughput"]["requests_per_second"]
        / fused_short["throughput"]["requests_per_second"]
    )
    if throughput_ratio < 0.95:
        raise ValueError("短 prompt 并发吞吐下降超过 5%")

    return {
        "status": "PASS",
        "code_revision": revision,
        "verified_files": len(entries),
        "workspace_cases": 12,
        "concurrent_requests": 400,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("artifacts/prefill/p4/evidence.json"),
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.evidence), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
