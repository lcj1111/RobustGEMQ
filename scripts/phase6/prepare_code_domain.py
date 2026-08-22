#!/usr/bin/env python3
"""Build the pinned, train-only Phase 6 CodeContests calibration corpus."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq


REVISION = "802411c3010cb00d1b05bad57ca77365a3c699d6"
MIRROR_ROOT = "https://hf-mirror.com/datasets/deepmind/code_contests/resolve"
SHARDS = (
    ("train-00000-of-00039-e991a271dbfa9925.parquet", "950bfdefde1f274edf93963e7e23c93ab16034de344aa7b7c32d14145e8d5232"),
    ("train-00001-of-00039-e092fe56fda18715.parquet", "7bcc72d98a3d97f07be90ea70bf8db834c8f29ac2c7626d7cfc657bae6f5ddd9"),
    ("train-00002-of-00039-9cea23812e920e41.parquet", "d67c9643019e632941b045f115c6bdf5925a4b32cd9f3fc991734970669b4a29"),
    ("train-00003-of-00039-e3822fccad6e083a.parquet", "f4952386b2958bddb71ff6713b273b5425b1191084ef75923731b111bc20485f"),
)
HUMANEVAL_REVISION = "6d43fb980f9fee3c892a914eda09951f772ad10d"
HUMANEVAL_URL = (
    "https://raw.githubusercontent.com/openai/human-eval/"
    f"{HUMANEVAL_REVISION}/data/HumanEval.jsonl.gz"
)
SOURCE_NAMES = {
    0: "UNKNOWN_SOURCE",
    1: "CODECHEF",
    2: "CODEFORCES",
    3: "HACKEREARTH",
    4: "CODEJAM",
    5: "ATCODER",
    6: "AIZU",
}
PYTHON3 = 3


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size:
        return
    partial = target.with_suffix(target.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "RobustGEMQ/phase6"})
    with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output)
    partial.replace(target)


def load_heldout_hashes(data_root: Path, humaneval_path: Path) -> tuple[set[str], dict]:
    hashes: set[str] = set()
    counts = {"mbpp": 0, "humaneval": 0}
    for path in sorted((data_root / "mbpp").glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                for field in ("prompt", "code"):
                    value = record.get(field)
                    if value:
                        hashes.add(normalized_hash(value))
                counts["mbpp"] += 1
    with gzip.open(humaneval_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            for field in ("prompt", "canonical_solution"):
                value = record.get(field)
                if value:
                    hashes.add(normalized_hash(value))
            counts["humaneval"] += 1
    return hashes, counts


def records_from_shard(path: Path, heldout_hashes: set[str]):
    table = pq.read_table(path, columns=["name", "description", "source", "solutions"])
    for row_index, record in enumerate(table.to_pylist()):
        description = (record.get("description") or "").strip()
        solutions = record.get("solutions") or {}
        candidates = {
            solution.strip()
            for language, solution in zip(
                solutions.get("language") or [], solutions.get("solution") or []
            )
            if language == PYTHON3 and solution and solution.strip()
        }
        if not description or not candidates:
            continue
        # Hash ordering makes the one-solution-per-problem choice independent of source ordering.
        solution = min(candidates, key=lambda value: (normalized_hash(value), value))
        description_hash = normalized_hash(description)
        solution_hash = normalized_hash(solution)
        if description_hash in heldout_hashes or solution_hash in heldout_hashes:
            continue
        source_id = int(record.get("source") or 0)
        identity = f"{path.name}:{row_index}:{record.get('name') or ''}"
        yield {
            "record_id": hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20],
            "name": record.get("name") or "",
            "source": SOURCE_NAMES.get(source_id, f"SOURCE_{source_id}"),
            "description": description,
            "solution": solution,
            "description_sha256": description_hash,
            "solution_sha256": solution_hash,
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--min-characters", type=int, default=2_000_000)
    parser.add_argument("--max-shards", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.max_shards <= len(SHARDS):
        raise ValueError(f"--max-shards must be between 1 and {len(SHARDS)}")
    source_dir = args.data_root / "sources" / "code_contests"
    output_path = args.data_root / "code" / "code_contests_train_python3.jsonl"
    humaneval_path = args.data_root / "sources" / "HumanEval.jsonl.gz"
    download(HUMANEVAL_URL, humaneval_path)
    heldout_hashes, heldout_counts = load_heldout_hashes(args.data_root, humaneval_path)

    records: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    source_files = []
    total_characters = 0
    leakage_rejections = 0
    duplicate_rejections = 0
    for filename, expected_sha256 in SHARDS[: args.max_shards]:
        path = source_dir / filename
        download(f"{MIRROR_ROOT}/{REVISION}/data/{filename}", path)
        observed_sha256 = sha256_file(path)
        if expected_sha256 and observed_sha256 != expected_sha256:
            raise ValueError(
                f"Checksum mismatch for {filename}: {observed_sha256}, expected {expected_sha256}"
            )
        before = len(records)
        for record in records_from_shard(path, heldout_hashes):
            pair = (record["description_sha256"], record["solution_sha256"])
            if pair in seen_pairs:
                duplicate_rejections += 1
                continue
            if pair[0] in heldout_hashes or pair[1] in heldout_hashes:
                leakage_rejections += 1
                continue
            seen_pairs.add(pair)
            records.append(record)
            total_characters += len(record["description"]) + len(record["solution"])
        source_files.append(
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": observed_sha256,
                "records_retained": len(records) - before,
            }
        )
        if total_characters >= args.min_characters:
            break
    if total_characters < args.min_characters:
        raise ValueError(
            f"Only retained {total_characters} characters; requested at least {args.min_characters}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1,
        "source": "deepmind/code_contests",
        "revision": REVISION,
        "split": "train",
        "language": "PYTHON3",
        "selection": "one accepted solution per problem; minimum normalized SHA-256",
        "license": "CC-BY-4.0; upstream acknowledgements apply",
        "source_files": source_files,
        "heldout": {
            "mbpp": "all locally materialized sanitized MBPP splits",
            "humaneval_revision": HUMANEVAL_REVISION,
            "counts": heldout_counts,
            "normalized_hash_count": len(heldout_hashes),
        },
        "leakage_rejections": leakage_rejections,
        "duplicate_rejections": duplicate_rejections,
        "records": len(records),
        "characters": total_characters,
        "output": str(output_path.resolve()),
        "output_bytes": output_path.stat().st_size,
        "output_sha256": sha256_file(output_path),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
