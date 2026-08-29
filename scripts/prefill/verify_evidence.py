#!/usr/bin/env python3
"""离线校验 prefill 优化证据的哈希、协议、数值门槛与性能结论。"""

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


def event_metric(profile: dict, name: str) -> dict:
    for event in profile["top_cuda_events"]:
        if event["name"] == name:
            return event
    return {"calls": 0, "device_time_us": 0.0}


def verify(evidence_path: Path) -> dict:
    repo = evidence_path.resolve().parents[2]
    evidence = load_json(evidence_path)
    if evidence.get("schema_version") != 1:
        raise ValueError("不支持的 prefill evidence schema")

    hashed_entries = [
        *evidence["correctness"],
        *evidence["end_to_end"],
        *evidence["traces"],
        *evidence["source_snapshot"],
    ]
    stage_entries = []
    stage_results = {}
    for stage, entry in evidence["stages"].items():
        stage_entries.append({"path": entry["result"], "sha256": entry["sha256"]})
        stage_results[stage] = load_json(repo / entry["result"])

    for entry in [*stage_entries, *hashed_entries]:
        path = repo / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"缺少证据文件：{entry['path']}")
        actual = sha256(path)
        if actual != entry["sha256"]:
            raise ValueError(f"哈希不一致：{entry['path']} expected={entry['sha256']} actual={actual}")

    protocol = evidence["protocol"]
    for stage, result in stage_results.items():
        if result["seed"] != protocol["seed"]:
            raise ValueError(f"{stage} seed 不一致")
        if result["warmup"] != protocol["warmup"] or result["repeats"] != protocol["repeats"]:
            raise ValueError(f"{stage} 预热/重复次数不一致")
        if [int(length) for length in result["cases"]] != protocol["lengths"]:
            raise ValueError(f"{stage} token 长度不一致")
        if result["device"]["name"] != protocol["device"]:
            raise ValueError(f"{stage} GPU 不一致")

    for length, expected in evidence["expected_medians_ms"].items():
        actual = {
            "baseline_full": stage_results["baseline"]["cases"][length]["full_model"]["median_ms"],
            "p3_full": stage_results["p3"]["cases"][length]["full_model"]["median_ms"],
            "baseline_block": stage_results["baseline"]["cases"][length]["moe_block"]["median_ms"],
            "p3_block": stage_results["p3"]["cases"][length]["moe_block"]["median_ms"],
        }
        for key, value in expected.items():
            if not math.isclose(actual[key], value, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"{length}/{key} 与冻结中位数不一致")
        if actual["baseline_full"] / actual["p3_full"] <= 5.0:
            raise ValueError(f"{length} 端到端加速未达到冻结结果")

    for entry in evidence["correctness"]:
        result = load_json(repo / entry["path"])
        for length, case in result["cases"].items():
            if not case["allclose"] or not case["router_exact"]:
                raise ValueError(f"数值门槛未通过：{entry['path']} length={length}")

    for entry in evidence["end_to_end"]:
        result = load_json(repo / entry["path"])
        for length, case in result["cases"].items():
            if case["argmax_agreement"] < result["min_argmax_agreement"]:
                raise ValueError(f"整模 argmax 门槛未通过：{entry['path']} length={length}")
            if case["mean_abs_error"] > result["max_mean_abs_error"]:
                raise ValueError(f"整模平均误差门槛未通过：{entry['path']} length={length}")

    baseline_profile = stage_results["baseline"]["cases"]["2048"]["moe_block"]["profile"]
    p1_profile = stage_results["p1"]["cases"]["2048"]["moe_block"]["profile"]
    p2_profile = stage_results["p2"]["cases"]["2048"]["moe_block"]["profile"]
    p3_profile = stage_results["p3"]["cases"]["2048"]["moe_block"]["profile"]
    if baseline_profile["dequant_group_gemm"]["calls"] != 159:
        raise ValueError("baseline 量化 GEMM 调用数不是 159")
    if p1_profile["dequant_group_gemm"]["calls"] != 159:
        raise ValueError("P1 不应改变量化 GEMM 调用数")
    if event_metric(p2_profile, "mixedbit_variable_m_grouped_gemm_kernel")["calls"] != 3:
        raise ValueError("P2 grouped GEMM 调用数不是 3")
    if p3_profile["fused_up_activation"]["calls"] != 1:
        raise ValueError("P3 fused-up 调用数不是 1")
    if p3_profile["variable_m_grouped_gemm"]["calls"] != 1:
        raise ValueError("P3 grouped-down 调用数不是 1")
    if p3_profile["deterministic_unpermute_reduce"]["calls"] != 1:
        raise ValueError("P3 确定性归并调用数不是 1")

    p2_workspace = stage_results["p2"]["cases"]["2048"]["moe_block"]["peak_workspace_delta_bytes"]
    p3_workspace = stage_results["p3"]["cases"]["2048"]["moe_block"]["peak_workspace_delta_bytes"]
    if not p3_workspace < p2_workspace:
        raise ValueError("P3 workspace 未低于 P2")

    return {
        "status": "PASS",
        "stages": list(stage_results),
        "lengths": protocol["lengths"],
        "verified_files": len(stage_entries) + len(hashed_entries),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence", type=Path, default=Path("artifacts/prefill/evidence.json")
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.evidence), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
