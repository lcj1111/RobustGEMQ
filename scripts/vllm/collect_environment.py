#!/usr/bin/env python3
"""采集 vLLM 正式实验所需、且不包含主机隐私信息的环境快照。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def atomic_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args()

    query = subprocess.run(
        [
            "nvidia-smi",
            f"--id={args.gpu_index}",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    name, driver, memory_mib = [part.strip() for part in query.split(",")]
    capability = torch.cuda.get_device_capability(0)
    payload = {
        "schema_version": 1,
        "status": "pass",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        # 实验代码尚未提交时，以父提交加 evidence 中的逐文件哈希共同定位源码。
        "base_revision": args.base_revision,
        "hardware": {
            "gpu": name,
            "compute_capability": f"{capability[0]}.{capability[1]}",
            "memory_total_mib": int(memory_mib),
            "driver": driver,
            "physical_gpu_index": args.gpu_index,
        },
        "runtime": {
            "python": ".".join(map(str, __import__("sys").version_info[:3])),
            "cuda_runtime": torch.version.cuda,
            "packages": {
                package: package_version(package)
                for package in (
                    "gemq",
                    "vllm",
                    "torch",
                    "triton",
                    "transformers",
                    "gemlite",
                    "aiohttp",
                )
            },
        },
        "serving_protocol": {
            "api": "OpenAI-compatible /v1/completions streaming",
            "dtype_bf16_baseline": "bfloat16",
            "dtype_robustgemq": "float16",
            "max_model_len": 2048,
            "max_num_seqs": 16,
            "kv_cache_memory_bytes": 4 * 1024**3,
            "enforce_eager": True,
            "gemq_prefill_chunk_tokens": 128,
            "prefix_caching": False,
        },
    }
    atomic_dump(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
