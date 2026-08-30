#!/usr/bin/env python3
"""离线复算 vLLM 集成的正确性、负载 identity 与服务指标。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gemq.vllm_plugin.checkpoint_schema import validate_manifest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def nearest_rank(values: list[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("无法复算空样本分位数")
    return ordered[max(1, math.ceil(probability * len(ordered))) - 1]


def assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{label} 无法由请求级记录复算：{actual} != {expected}")


def verify_distribution(summary: dict, values: list[float], label: str) -> None:
    assert_close(summary["mean"], sum(values) / len(values), f"{label}.mean")
    for key, probability in (("p50", 0.50), ("p95", 0.95), ("p99", 0.99)):
        assert_close(summary[key], nearest_rank(values, probability), f"{label}.{key}")
    assert_close(summary["max"], max(values), f"{label}.max")


def verify_benchmark(result: dict) -> None:
    if result.get("schema_version") != 1 or result.get("status") != "pass":
        raise ValueError("benchmark schema/status 非法")
    workload = result["workload"]
    records = result["requests"]
    expected_count = int(workload["num_requests"])
    request_ids = [record["request_id"] for record in records]
    if request_ids != list(range(expected_count)):
        raise ValueError("请求记录缺失、重复或顺序错误")
    if expected_count != 24:
        raise ValueError("正式 benchmark 每档必须包含 24 个请求")
    if workload["warmup_rounds"] != 2:
        raise ValueError("正式 benchmark 必须按目标并发度预热两轮")
    if workload["warmup_requests"] != 2 * workload["concurrency"]:
        raise ValueError("预热请求数与并发度不一致")

    workload_records = workload["requests"]
    if len(workload_records) != expected_count:
        raise ValueError("workload 请求清单数量错误")
    expected_hash = hashlib.sha256(
        json.dumps(
            workload_records, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    if workload["sha256"] != expected_hash:
        raise ValueError("workload SHA-256 无法由请求清单复算")

    for expected, record in zip(workload_records, records):
        for key in ("request_id", "prompt_tokens", "prompt_sha256"):
            if expected[key] != record[key]:
                raise ValueError(f"请求 {record['request_id']} 的 {key} identity 不一致")
        if record["completion_tokens"] != workload["max_tokens"]:
            raise ValueError(f"请求 {record['request_id']} 输出 token 不完整")
        if record["total_tokens"] != record["prompt_tokens"] + record["completion_tokens"]:
            raise ValueError(f"请求 {record['request_id']} usage 加和错误")
        if not 0.0 < record["ttft_seconds"] <= record["e2e_seconds"]:
            raise ValueError(f"请求 {record['request_id']} 时延非法")

    duration = float(result["summary"]["benchmark_seconds"])
    prompt_tokens = sum(record["prompt_tokens"] for record in records)
    completion_tokens = sum(record["completion_tokens"] for record in records)
    total_tokens = sum(record["total_tokens"] for record in records)
    summary = result["summary"]
    assert_close(summary["request_throughput_per_second"], expected_count / duration,
                 "request throughput")
    assert_close(summary["prompt_token_throughput_per_second"], prompt_tokens / duration,
                 "prompt token throughput")
    assert_close(summary["output_token_throughput_per_second"], completion_tokens / duration,
                 "output token throughput")
    assert_close(summary["total_token_throughput_per_second"], total_tokens / duration,
                 "total token throughput")
    verify_distribution(summary["ttft_seconds"],
                        [record["ttft_seconds"] for record in records], "TTFT")
    verify_distribution(summary["e2e_seconds"],
                        [record["e2e_seconds"] for record in records], "E2E")
    inter_token = [
        record["inter_token_seconds"]
        for record in records
        if record["inter_token_seconds"] is not None
    ]
    verify_distribution(summary["inter_token_seconds"], inter_token, "inter-token")

    memory = result["gpu_memory_samples"]
    if not memory or any(
        left["elapsed_seconds"] > right["elapsed_seconds"]
        for left, right in zip(memory, memory[1:])
    ):
        raise ValueError("显存采样缺失或时间顺序错误")
    if summary["gpu_memory_mib"]["samples"] != len(memory):
        raise ValueError("显存样本数无法复算")
    assert_close(summary["gpu_memory_mib"]["first"], memory[0]["memory_mib"],
                 "first GPU memory")
    assert_close(summary["gpu_memory_mib"]["peak"],
                 max(sample["memory_mib"] for sample in memory), "peak GPU memory")


def verify(evidence_path: Path) -> dict:
    repo = evidence_path.resolve().parents[2]
    evidence = load_json(evidence_path)
    if evidence.get("schema_version") != 1:
        raise ValueError("不支持的 vLLM evidence schema")

    loaded: dict[str, dict] = {}
    for entry in evidence["files"]:
        path = repo / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"缺少证据文件：{entry['path']}")
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"哈希不一致：{entry['path']}")
        if entry["kind"] != "source":
            loaded[entry["name"]] = load_json(path)

    environment = loaded["environment"]
    if environment.get("status") != "pass":
        raise ValueError("环境快照状态非法")
    protocol = environment["serving_protocol"]
    if protocol["kv_cache_memory_bytes"] != 4 * 1024**3:
        raise ValueError("BF16/GEMQ 未固定为 4 GiB KV Cache")
    if protocol["max_num_seqs"] != 16 or not protocol["enforce_eager"]:
        raise ValueError("服务协议与冻结配置不一致")

    manifest = validate_manifest(loaded["checkpoint_manifest"])
    artifact = manifest["artifacts"][0]
    if artifact != evidence["checkpoint_artifact"]:
        raise ValueError("evidence 中的检查点 artifact 与 manifest 不一致")

    smoke = loaded["offline_smoke"]
    equivalence = loaded["greedy_equivalence"]
    layer = loaded["layer_equivalence"]
    if smoke.get("status") != "pass" or smoke.get("engine") != "vllm":
        raise ValueError("vLLM 离线 smoke 未通过")
    if not equivalence.get("exact_match") or equivalence.get("agreement") != 1.0:
        raise ValueError("原推理路径与 vLLM 的 greedy token 不一致")
    if not layer["attention"]["pass"] or not layer["moe"]["pass"]:
        raise ValueError("层级独立反量化对照未通过")

    benchmarks: dict[tuple[str, int], dict] = {}
    for model in ("bf16", "robustgemq"):
        for concurrency in (1, 4, 8):
            name = f"{model}_c{concurrency}"
            result = loaded[name]
            verify_benchmark(result)
            if result["workload"]["concurrency"] != concurrency:
                raise ValueError(f"{name} 并发度字段错误")
            benchmarks[(model, concurrency)] = result

    comparisons = {item["concurrency"]: item for item in evidence["comparisons"]}
    for concurrency in (1, 4, 8):
        baseline = benchmarks[("bf16", concurrency)]
        candidate = benchmarks[("robustgemq", concurrency)]
        if baseline["workload"]["sha256"] != candidate["workload"]["sha256"]:
            raise ValueError(f"并发 {concurrency} 的跨方法 workload hash 不一致")
        if baseline["workload"]["requests"] != candidate["workload"]["requests"]:
            raise ValueError(f"并发 {concurrency} 的跨方法请求 identity 不一致")
        b = baseline["summary"]
        q = candidate["summary"]
        expected = {
            "output_throughput_ratio": q["output_token_throughput_per_second"]
            / b["output_token_throughput_per_second"],
            "ttft_p95_ratio": q["ttft_seconds"]["p95"] / b["ttft_seconds"]["p95"],
            "peak_memory_reduction": 1.0
            - q["gpu_memory_mib"]["peak"] / b["gpu_memory_mib"]["peak"],
        }
        actual = comparisons[concurrency]
        for key, value in expected.items():
            assert_close(actual[key], value, f"concurrency={concurrency} {key}")

    return {
        "status": "PASS",
        "verified_files": len(evidence["files"]),
        "benchmark_requests": sum(
            len(result["requests"]) for result in benchmarks.values()
        ),
        "workload_identity_pairs": 3,
        "checkpoint_sha256": artifact["sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence", type=Path, default=Path("artifacts/vllm/evidence.json")
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.evidence), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
