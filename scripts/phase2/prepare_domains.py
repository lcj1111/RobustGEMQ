#!/usr/bin/env python3
"""下载固定版本的数据源，并记录训练/留出文件及数据划分。"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.request
from pathlib import Path


SOURCES = {
    "gsm8k_train": {
        "url": "https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl",
        "path": "gsm8k/train.jsonl",
        "split": "allocation",
    },
    "gsm8k_test": {
        "url": "https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl",
        "path": "gsm8k/test.jsonl",
        "split": "held_out",
    },
    "mbpp_source": {
        "url": "https://raw.githubusercontent.com/google-research/google-research/589e977488f21a336a3d3da9b96da91ddbcf935e/mbpp/sanitized-mbpp.json",
        "path": "sources/sanitized-mbpp.json",
        "split": "source_only",
    },
    "dolly_train": {
        "url": "https://raw.githubusercontent.com/databrickslabs/dolly/2305eb7f2f4b3beb2379f34c6addf335b46c4b43/data/databricks-dolly-15k.jsonl",
        "path": "dolly/databricks-dolly-15k.jsonl",
        "split": "allocation",
    },
}


def download(url: str, target: Path, retries: int = 4) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0:
        return
    temporary = target.with_suffix(target.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "RobustGEMQ/phase2"})
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
                shutil.copyfileobj(response, out)
            if temporary.stat().st_size == 0:
                raise ValueError(f"Downloaded empty file from {url}")
            temporary.replace(target)
            return
        except Exception as error:  # retry transient GitHub/proxy failures
            last_error = error
            if temporary.exists():
                temporary.unlink()
            if attempt < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Failed to download {url} after {retries} attempts") from last_error


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def split_mbpp(source: Path, root: Path) -> dict[str, int]:
    records = json.loads(source.read_text(encoding="utf-8"))
    splits = {
        "prompt": [record for record in records if 1 <= int(record["task_id"]) <= 10],
        "test": [record for record in records if 11 <= int(record["task_id"]) <= 510],
        "validation": [record for record in records if 511 <= int(record["task_id"]) <= 600],
        "train": [record for record in records if 601 <= int(record["task_id"]) <= 974],
    }
    observed = {int(record["task_id"]) for values in splits.values() for record in values}
    source_ids = {int(record["task_id"]) for record in records}
    if observed != source_ids:
        raise ValueError(f"MBPP split rules do not cover source ids: {sorted(source_ids - observed)}")
    for name, values in splits.items():
        write_jsonl(root / "mbpp" / f"{name}.jsonl", values)
    return {name: len(values) for name, values in splits.items()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--c4-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_root.mkdir(parents=True, exist_ok=True)
    for spec in SOURCES.values():
        download(spec["url"], args.data_root / spec["path"])

    mbpp_counts = split_mbpp(args.data_root / SOURCES["mbpp_source"]["path"], args.data_root)

    c4_target = args.data_root / "c4"
    if not c4_target.exists():
        c4_target.symlink_to(args.c4_root.resolve(), target_is_directory=True)
    if c4_target.resolve() != args.c4_root.resolve():
        raise ValueError(f"Existing C4 link points to {c4_target.resolve()}, expected {args.c4_root.resolve()}")

    files = {}
    for path in sorted(args.data_root.rglob("*")):
        if path.is_file() and not path.name.endswith(".partial"):
            files[str(path.relative_to(args.data_root))] = {
                "bytes": path.stat().st_size,
            }
    # pathlib 不会递归主动创建的 C4 目录软链接，因此显式记录两个固定文件，
    # 保证清单覆盖分配数据与本地留出数据。
    for relative in (
        Path("c4/en/c4-train.00000-of-01024.json"),
        Path("c4/en/c4-validation.00000-of-00008.json.gz"),
    ):
        path = args.data_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned C4 file: {path}")
        files[str(relative)] = {"bytes": path.stat().st_size}
    allocation = {
        "general": "c4/en/c4-train.00000-of-01024.json",
        "math": "gsm8k/train.jsonl",
        "code": "mbpp/train.jsonl",
        "instruction": "dolly/databricks-dolly-15k.jsonl",
    }
    held_out = {
        "general": "c4/en/c4-validation.00000-of-00008.json.gz",
        "math": "gsm8k/test.jsonl",
        "code": "mbpp/test.jsonl",
    }
    manifest = {
        "schema_version": 2,
        "sources": SOURCES,
        "files": files,
        "mbpp_counts": mbpp_counts,
        "allocation_files": allocation,
        "held_out_files": held_out,
        "split_disjoint": not set(allocation.values()).intersection(held_out.values()),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(args.manifest), "mbpp_counts": mbpp_counts, "files": len(files)}))


if __name__ == "__main__":
    main()
