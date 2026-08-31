#!/usr/bin/env python3
"""通过 vLLM profiler API 采集固定 token 负载，并保存请求侧审计记录。"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

import aiohttp

from benchmark_service import (
    atomic_json_dump,
    build_workload,
    request_once,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--prompt-length", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument(
        "--prefix-caching", choices=("disabled", "enabled"), default="disabled"
    )
    parser.add_argument("--timeout", type=float, default=900.0)
    return parser.parse_args()


async def require_post(session: aiohttp.ClientSession, url: str) -> None:
    async with session.post(url) as response:
        body = await response.text()
        if response.status != 200:
            raise RuntimeError(f"{url} 返回 HTTP {response.status}: {body[:500]}")


async def run(args: argparse.Namespace) -> dict:
    if args.concurrency <= 0 or args.warmup_rounds <= 0:
        raise ValueError("concurrency 和 warmup_rounds 必须为正整数")
    workload = build_workload(
        args.tokenizer, [args.prompt_length], args.concurrency
    )
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # 预热覆盖目标并发 batch shape，不进入 profiler。
        for _ in range(args.warmup_rounds):
            await asyncio.gather(
                *(
                    request_once(
                        session,
                        args.endpoint,
                        args.model,
                        spec,
                        args.max_tokens,
                    )
                    for spec in workload
                )
            )

        await require_post(session, f"{args.endpoint.rstrip('/')}/start_profile")
        try:
            results = await asyncio.gather(
                *(
                    request_once(
                        session,
                        args.endpoint,
                        args.model,
                        spec,
                        args.max_tokens,
                    )
                    for spec in workload
                )
            )
        finally:
            await require_post(session, f"{args.endpoint.rstrip('/')}/stop_profile")

    return {
        "schema_version": 1,
        "status": "pass",
        "label": args.label,
        "endpoint": args.endpoint,
        "model": args.model,
        "workload": {
            "concurrency": args.concurrency,
            "prompt_length": args.prompt_length,
            "max_tokens": args.max_tokens,
            "warmup_rounds": args.warmup_rounds,
            "prefix_caching": args.prefix_caching,
            "request_identity": [
                {
                    "request_id": spec.request_id,
                    "prompt_tokens": len(spec.prompt_tokens),
                    "prompt_sha256": spec.prompt_sha256,
                }
                for spec in workload
            ],
        },
        "requests": [asdict(result) for result in results],
    }


def main() -> None:
    args = parse_args()
    payload = asyncio.run(run(args))
    atomic_json_dump(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
