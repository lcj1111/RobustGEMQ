#!/usr/bin/env python3
"""对兼容 OpenAI Completions API 的 vLLM 服务执行可审计并发基准。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiohttp
from transformers import AutoTokenizer


DEFAULT_SEED_TEXT = (
    "Mixture-of-Experts models route each token to a small subset of experts. "
    "A serving system should preserve model quality while controlling latency, "
    "throughput, and memory usage under concurrent requests. "
)


@dataclass(frozen=True)
class RequestSpec:
    request_id: int
    prompt_tokens: list[int]

    @property
    def prompt_sha256(self) -> str:
        payload = json.dumps(self.prompt_tokens, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class RequestResult:
    request_id: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_sha256: str
    ttft_seconds: float
    e2e_seconds: float
    inter_token_seconds: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True, help="例如 http://127.0.0.1:8100")
    parser.add_argument("--model", required=True, help="服务端暴露的模型名称")
    parser.add_argument("--tokenizer", required=True, help="本地 tokenizer 路径")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--num-requests", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[128, 512])
    parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=4,
        help="按目标并发度覆盖短、长、混合输入形状的完整预热轮数",
    )
    parser.add_argument(
        "--prefix-caching",
        choices=("disabled", "enabled"),
        default="disabled",
        help="记录服务端冻结的 prefix cache 状态；正式 uncached 基准必须为 disabled",
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--memory-sample-interval", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser.parse_args()


def atomic_json_dump(path: Path, payload: dict[str, Any]) -> None:
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


def build_workload(
    tokenizer_path: str,
    prompt_lengths: list[int],
    num_requests: int,
) -> list[RequestSpec]:
    if not prompt_lengths or min(prompt_lengths) <= 0:
        raise ValueError("prompt length 必须为正整数")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    seed = tokenizer.encode(DEFAULT_SEED_TEXT, add_special_tokens=False)
    if not seed:
        raise RuntimeError("固定文本未产生 token")

    workload: list[RequestSpec] = []
    for request_id in range(num_requests):
        target_length = prompt_lengths[request_id % len(prompt_lengths)]
        repeats = math.ceil(target_length / len(seed))
        prompt_tokens = (seed * repeats)[:target_length]
        workload.append(RequestSpec(request_id, prompt_tokens))
    return workload


async def sample_gpu_memory(
    gpu_index: int,
    interval: float,
    stop: asyncio.Event,
    samples: list[dict[str, float]],
) -> None:
    """采样物理 GPU 的已用显存；采样失败会使正式结果失败。"""
    while not stop.is_set():
        completed = await asyncio.to_thread(
            subprocess.run,
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        value = float(completed.stdout.strip().splitlines()[0])
        samples.append({"elapsed_seconds": time.perf_counter(), "memory_mib": value})
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except TimeoutError:
            pass


async def request_once(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    spec: RequestSpec,
    max_tokens: int,
) -> RequestResult:
    payload = {
        "model": model,
        "prompt": spec.prompt_tokens,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first_token_at: float | None = None
    usage: dict[str, int] | None = None

    async with session.post(f"{endpoint.rstrip('/')}/v1/completions", json=payload) as response:
        if response.status != 200:
            body = await response.text()
            raise RuntimeError(f"请求 {spec.request_id} 失败：HTTP {response.status}: {body[:500]}")
        async for raw_line in response.content:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                continue
            event = json.loads(data)
            choices = event.get("choices") or []
            if choices and first_token_at is None:
                first_token_at = time.perf_counter()
            if event.get("usage") is not None:
                usage = event["usage"]

    finished = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError(f"请求 {spec.request_id} 没有收到生成 token")
    if usage is None:
        raise RuntimeError(f"请求 {spec.request_id} 缺少 usage 统计")

    prompt_tokens = int(usage["prompt_tokens"])
    completion_tokens = int(usage["completion_tokens"])
    total_tokens = int(usage["total_tokens"])
    if prompt_tokens != len(spec.prompt_tokens):
        raise RuntimeError(
            f"请求 {spec.request_id} 输入 token 不一致：{prompt_tokens} != {len(spec.prompt_tokens)}"
        )
    if completion_tokens != max_tokens:
        raise RuntimeError(
            f"请求 {spec.request_id} 输出 token 不完整：{completion_tokens} != {max_tokens}"
        )
    e2e = finished - started
    ttft = first_token_at - started
    inter_token = None
    if completion_tokens > 1:
        inter_token = (e2e - ttft) / (completion_tokens - 1)
    return RequestResult(
        request_id=spec.request_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        prompt_sha256=spec.prompt_sha256,
        ttft_seconds=ttft,
        e2e_seconds=e2e,
        inter_token_seconds=inter_token,
    )


def nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("无法对空序列计算分位数")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "p50": nearest_rank(values, 0.50),
        "p95": nearest_rank(values, 0.95),
        "p99": nearest_rank(values, 0.99),
        "max": max(values),
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.concurrency <= 0 or args.num_requests <= 0 or args.warmup_rounds <= 0:
        raise ValueError("concurrency、num_requests 和 warmup_rounds 必须为正整数")
    workload = build_workload(args.tokenizer, args.prompt_lengths, args.num_requests)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=max(args.concurrency, 1))

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # 前两轮分别覆盖同长度短/长输入，后续轮覆盖交错输入。
        # 这样可以在测量前触发目标并发下的 Triton shape-specific autotune。
        for round_index in range(args.warmup_rounds):
            if round_index < len(args.prompt_lengths):
                target_length = args.prompt_lengths[round_index]
                candidates = [
                    item for item in workload if len(item.prompt_tokens) == target_length
                ]
                warmup_specs = [
                    candidates[index % len(candidates)]
                    for index in range(args.concurrency)
                ]
            else:
                offset = (round_index - len(args.prompt_lengths)) * args.concurrency
                warmup_specs = [
                    workload[(offset + index) % len(workload)]
                    for index in range(args.concurrency)
                ]
            await asyncio.gather(
                *(
                    request_once(
                        session,
                        args.endpoint,
                        args.model,
                        spec,
                        args.max_tokens,
                    )
                    for spec in warmup_specs
                )
            )

        memory_samples: list[dict[str, float]] = []
        stop_sampling = asyncio.Event()
        sampler = asyncio.create_task(
            sample_gpu_memory(
                args.gpu_index,
                args.memory_sample_interval,
                stop_sampling,
                memory_samples,
            )
        )
        await asyncio.sleep(args.memory_sample_interval)
        memory_origin = time.perf_counter()

        semaphore = asyncio.Semaphore(args.concurrency)

        async def bounded_request(spec: RequestSpec) -> RequestResult:
            async with semaphore:
                return await request_once(
                    session, args.endpoint, args.model, spec, args.max_tokens
                )

        benchmark_started = time.perf_counter()
        try:
            results = await asyncio.gather(*(bounded_request(spec) for spec in workload))
        finally:
            stop_sampling.set()
            await sampler
        benchmark_seconds = time.perf_counter() - benchmark_started

    if not memory_samples:
        raise RuntimeError("没有取得 GPU 显存样本")
    for sample in memory_samples:
        sample["elapsed_seconds"] -= memory_origin

    total_prompt_tokens = sum(item.prompt_tokens for item in results)
    total_completion_tokens = sum(item.completion_tokens for item in results)
    total_tokens = sum(item.total_tokens for item in results)
    ttft = [item.ttft_seconds for item in results]
    e2e = [item.e2e_seconds for item in results]
    inter_token = [
        item.inter_token_seconds
        for item in results
        if item.inter_token_seconds is not None
    ]
    workload_payload = [
        {
            "request_id": spec.request_id,
            "prompt_tokens": len(spec.prompt_tokens),
            "prompt_sha256": spec.prompt_sha256,
        }
        for spec in workload
    ]
    workload_sha256 = hashlib.sha256(
        json.dumps(workload_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    return {
        "schema_version": 1,
        "status": "pass",
        "benchmark": "vllm-openai-streaming-completions",
        "endpoint": args.endpoint,
        "model": args.model,
        "workload": {
            "num_requests": args.num_requests,
            "concurrency": args.concurrency,
            "prompt_lengths": args.prompt_lengths,
            "max_tokens": args.max_tokens,
            "warmup_rounds": args.warmup_rounds,
            "warmup_requests": args.warmup_rounds * args.concurrency,
            "prefix_caching": args.prefix_caching,
            "temperature": 0.0,
            "ignore_eos": True,
            "sha256": workload_sha256,
            "requests": workload_payload,
        },
        "summary": {
            "benchmark_seconds": benchmark_seconds,
            "request_throughput_per_second": args.num_requests / benchmark_seconds,
            "prompt_token_throughput_per_second": total_prompt_tokens / benchmark_seconds,
            "output_token_throughput_per_second": total_completion_tokens / benchmark_seconds,
            "total_token_throughput_per_second": total_tokens / benchmark_seconds,
            "ttft_seconds": distribution(ttft),
            "e2e_seconds": distribution(e2e),
            "inter_token_seconds": distribution(inter_token),
            "gpu_memory_mib": {
                "first": memory_samples[0]["memory_mib"],
                "peak": max(sample["memory_mib"] for sample in memory_samples),
                "samples": len(memory_samples),
            },
        },
        "requests": [asdict(item) for item in results],
        "gpu_memory_samples": memory_samples,
    }


def main() -> None:
    args = parse_args()
    payload = asyncio.run(run(args))
    atomic_json_dump(args.output, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
