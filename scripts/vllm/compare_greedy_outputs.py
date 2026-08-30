#!/usr/bin/env python3
"""比较 HF 参考路径与 vLLM 插件路径的 greedy token。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    reference_ids = reference["token_ids"]
    candidate_ids = candidate["token_ids"]
    if len(reference_ids) != len(candidate_ids):
        raise AssertionError("两条路径的生成 token 数不同")
    matches = [left == right for left, right in zip(reference_ids, candidate_ids)]
    payload = {
        "schema_version": 1,
        "status": "pass" if all(matches) else "fail",
        "reference_engine": reference["engine"],
        "candidate_engine": candidate["engine"],
        "prompt": reference["prompt"],
        "token_count": len(matches),
        "exact_match": all(matches),
        "agreement": sum(matches) / len(matches),
        "reference_token_ids": reference_ids,
        "candidate_token_ids": candidate_ids,
    }
    if not all(matches):
        raise AssertionError(json.dumps(payload, ensure_ascii=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
