#!/usr/bin/env python3
"""Verify held-out sources and materialize the pinned SuperGLUE BoolQ archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
from zipfile import ZipFile


BOOLQ_URL = "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip"
BOOLQ_SHA256 = "853fbe7922f70c59629f06a39e8d9ca440c3d740e760fd3b87a5ddf3dcba2436"
BOOLQ_VAL_SHA256 = "0c86a5045886e5795fe9052003873f7d94b88ed3028a33007c51d99e44fd66d9"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected: str | None = None) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    actual = sha256(path)
    if expected and actual != expected:
        raise ValueError(f"Checksum mismatch for {path}: {actual} != {expected}")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": actual}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    archive = args.data_root / "boolq" / "BoolQ.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        temporary = archive.with_suffix(".zip.partial")
        request = urllib.request.Request(BOOLQ_URL, headers={"User-Agent": "RobustGEMQ/phase3"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        temporary.replace(archive)
    verify(archive, BOOLQ_SHA256)
    boolq_val = args.data_root / "boolq" / "BoolQ" / "val.jsonl"
    if not boolq_val.exists():
        with ZipFile(archive) as source:
            source.extract("BoolQ/val.jsonl", archive.parent)

    sources = {
        "general": verify(args.data_root / "c4/en/c4-validation.00000-of-00008.json.gz"),
        "math": verify(args.data_root / "gsm8k/test.jsonl"),
        "code": verify(args.data_root / "mbpp/test.jsonl"),
        "instruction": verify(boolq_val, BOOLQ_VAL_SHA256),
        "boolq_archive": verify(archive, BOOLQ_SHA256),
    }
    manifest = {
        "schema_version": 1,
        "split_role": "held-out evaluation only",
        "boolq_url": BOOLQ_URL,
        "sources": sources,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(args.manifest), "sources": len(sources)}))


if __name__ == "__main__":
    main()
