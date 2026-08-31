#!/usr/bin/env python3
"""采集阶段零所需的精简环境快照，不记录仓库状态和本机路径。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PACKAGES = (
    "torch",
    "triton",
    "transformers",
    "accelerate",
    "datasets",
    "hqq",
    "gemlite",
    "numpy",
    "scipy",
    "pytest",
)


def nvidia_runtime(repo: Path) -> dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if completed.returncode != 0:
            return {"available": False, "error": completed.stderr.strip()}
        devices = [line.split(", ") for line in completed.stdout.splitlines() if line.strip()]
        if not devices:
            return {"available": False, "error": "nvidia-smi 未返回设备"}
        first = devices[0]
        return {
            "available": True,
            "device_count": len(devices),
            "device_name": first[0],
            "memory_mib": int(first[1]),
            "driver_version": first[2],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc)}


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def torch_runtime() -> dict[str, object]:
    try:
        import torch

        device = None
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            device = {
                "name": props.name,
                "total_memory_bytes": props.total_memory,
                "compute_capability": f"{props.major}.{props.minor}",
            }
        return {
            "import_ok": True,
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": device,
        }
    except Exception as exc:  # 二进制依赖损坏时也必须生成诊断结果
        return {"import_ok": False, "error": repr(exc)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    report = {
        "schema_version": 2,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "python": {
            "version": sys.version,
        },
        "packages": package_versions(),
        "torch_runtime": torch_runtime(),
        "nvidia_runtime": nvidia_runtime(repo),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
