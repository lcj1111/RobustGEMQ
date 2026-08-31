#!/usr/bin/env python3
"""用真实模型执行 vLLM 风格的并发 prefill 负载。

该脚本实现开放环 Poisson 到达、FCFS 排队、最大并发序列数与每轮 token
预算。为保证测量可解释，同一批请求使用相同 prompt 长度；每个批次可跨多个
scheduler chunk 执行，TTFT 从请求到达到最后一个 prompt chunk 完成计算。
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import re
import statistics
import subprocess
import time
from pathlib import Path

def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("不能统计空样本")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability 必须位于 [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_ms(samples: list[float]) -> dict:
    return {
        "count": len(samples),
        "mean_ms": statistics.fmean(samples),
        "p50_ms": percentile(samples, 0.50),
        "p95_ms": percentile(samples, 0.95),
        "p99_ms": percentile(samples, 0.99),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def poisson_arrivals(num_requests: int, request_rate: float, seed: int) -> list[float]:
    """生成从 0 开始的开放环到达时刻，间隔服从指数分布。"""
    if num_requests <= 0 or request_rate <= 0:
        raise ValueError("num_requests 与 request_rate 必须为正数")
    generator = random.Random(seed)
    arrivals = [0.0]
    for _ in range(1, num_requests):
        arrivals.append(arrivals[-1] + generator.expovariate(request_rate))
    return arrivals


def git_revision(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def device_metadata() -> dict:
    import torch

    properties = torch.cuda.get_device_properties(0)
    return {
        "name": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "sm_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python": platform.python_version(),
    }


def execute_prefill_batch(
    model,
    input_ids: torch.Tensor,
    prompt_length: int,
    max_num_batched_tokens: int,
) -> None:
    """按 scheduler token 预算执行一个同长度请求批次。"""
    import torch

    from gemq.inference.kv_cache import StaticCache

    batch_size = input_ids.shape[0]
    per_request_chunk = max_num_batched_tokens // batch_size
    if per_request_chunk <= 0:
        raise ValueError("max_num_batched_tokens 小于当前 batch size")
    cache = StaticCache(model.config, max_cache_len=prompt_length)
    for start in range(0, prompt_length, per_request_chunk):
        end = min(start + per_request_chunk, prompt_length)
        output = model(
            input_ids[:, start:end],
            past_key_values=cache,
            cache_position=torch.arange(start, end, device=input_ids.device),
            logits_to_keep=1,
        )
        del output


def main() -> None:
    import torch

    from gemq.inference.patch import prepare_for_inference
    from gemq.utils.hf_loading import load_quantized_model
    from gemq.utils.model_utils import get_blocks

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-name", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument(
        "--backend", choices=("fused", "chunked"), default="chunked"
    )
    parser.add_argument("--moe-workspace-chunk-tokens", type=int, default=512)
    parser.add_argument("--prompt-length", type=int, default=2048)
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--request-rate", type=float, default=8.0)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--code-revision",
        help="源码提交 SHA；服务器 checkout 与被测源码不一致时必须显式传入",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    positive_ints = {
        "moe-workspace-chunk-tokens": args.moe_workspace_chunk_tokens,
        "prompt-length": args.prompt_length,
        "num-requests": args.num_requests,
        "max-num-seqs": args.max_num_seqs,
        "max-num-batched-tokens": args.max_num_batched_tokens,
        "warmup-batches": args.warmup_batches,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"{name} 必须为正整数")
    if args.request_rate <= 0:
        raise ValueError("request-rate 必须为正数")
    if args.code_revision is not None and not re.fullmatch(
        r"[0-9a-f]{40}", args.code_revision
    ):
        raise ValueError("code-revision 必须是 40 位小写 Git SHA")
    if not torch.cuda.is_available():
        raise RuntimeError("并发 prefill benchmark 需要 CUDA")
    torch.set_grad_enabled(False)

    repo = Path(__file__).resolve().parents[2]
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = load_quantized_model(
        str(args.checkpoint),
        compute_dtype=torch.float16,
        device="cuda",
        trust_remote_code=False,
    ).eval()
    prepare_for_inference(model, args.model_name, is_fp=False)
    for block in get_blocks(model, args.model_name):
        block.mlp.prefill_backend = args.backend
        block.mlp.prefill_chunk_tokens = args.moe_workspace_chunk_tokens

    # 在计时前生成请求并覆盖最大 batch shape 的 autotune/缓存初始化。
    prompts = torch.randint(
        0,
        int(model.config.vocab_size),
        (args.num_requests, args.prompt_length),
        dtype=torch.long,
    )
    warmup_size = min(args.max_num_seqs, args.num_requests)
    # 开放环队列可能形成 1..max_num_seqs 的任意 batch。Triton 首次遇到新的
    # assignment bucket 会 autotune，因此必须逐个覆盖，不能把编译时间算入 TTFT。
    for batch_size in range(1, warmup_size + 1):
        warmup_prompts = prompts[:batch_size].to("cuda")
        for _ in range(args.warmup_batches):
            execute_prefill_batch(
                model,
                warmup_prompts,
                args.prompt_length,
                args.max_num_batched_tokens,
            )
            torch.cuda.synchronize()
        del warmup_prompts
    torch.cuda.empty_cache()

    arrival_offsets = poisson_arrivals(
        args.num_requests, args.request_rate, args.seed
    )
    records = [
        {
            "request_id": request_id,
            "arrival_offset_s": arrival,
        }
        for request_id, arrival in enumerate(arrival_offsets)
    ]

    baseline_bytes = int(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    next_request = 0
    completed = 0
    batch_records = []

    while completed < args.num_requests:
        now_offset = time.perf_counter() - start_time
        if next_request >= args.num_requests or arrival_offsets[next_request] > now_offset:
            if next_request < args.num_requests:
                time.sleep(min(arrival_offsets[next_request] - now_offset, 0.01))
                continue
            raise RuntimeError("仍有请求未完成，但等待队列为空")

        available = next_request
        while (
            available < args.num_requests
            and arrival_offsets[available] <= time.perf_counter() - start_time
        ):
            available += 1
        batch_end = min(available, next_request + args.max_num_seqs)
        request_ids = list(range(next_request, batch_end))
        next_request = batch_end
        batch_start = time.perf_counter()
        input_ids = prompts[request_ids].to("cuda")
        execute_prefill_batch(
            model,
            input_ids,
            args.prompt_length,
            args.max_num_batched_tokens,
        )
        torch.cuda.synchronize()
        batch_finish = time.perf_counter()
        del input_ids

        batch_start_offset = batch_start - start_time
        batch_finish_offset = batch_finish - start_time
        for request_id in request_ids:
            records[request_id].update(
                {
                    "batch_id": len(batch_records),
                    "queue_ms": 1000.0
                    * (batch_start_offset - arrival_offsets[request_id]),
                    "ttft_ms": 1000.0
                    * (batch_finish_offset - arrival_offsets[request_id]),
                }
            )
        batch_records.append(
            {
                "batch_id": len(batch_records),
                "size": len(request_ids),
                "request_ids": request_ids,
                "service_ms": 1000.0 * (batch_finish - batch_start),
            }
        )
        completed += len(request_ids)

    duration_s = time.perf_counter() - start_time
    peak_allocated_bytes = int(torch.cuda.max_memory_allocated())
    ttft_samples = [float(record["ttft_ms"]) for record in records]
    queue_samples = [float(record["queue_ms"]) for record in records]
    service_samples = [float(batch["service_ms"]) for batch in batch_records]
    if any(not math.isfinite(value) or value < 0 for value in ttft_samples):
        raise RuntimeError("TTFT 样本包含非法值")

    result = {
        "schema_version": 1,
        "benchmark": "vllm-style-concurrent-prefill",
        "scope": (
            "真实模型执行到首 token logits；同长度请求的开放环微批处理，"
            "不是 vLLM 引擎实测"
        ),
        "checkpoint": str(args.checkpoint.resolve()),
        "model_name": args.model_name,
        "code_revision": args.code_revision or git_revision(repo),
        "runtime_checkout_revision": git_revision(repo),
        "seed": args.seed,
        "backend": args.backend,
        "scheduler": {
            "policy": "fcfs",
            "arrival_process": "poisson",
            "request_rate_per_second": args.request_rate,
            "realized_arrival_rate_per_second": (
                (args.num_requests - 1) / arrival_offsets[-1]
                if args.num_requests > 1
                else None
            ),
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "prompt_chunking": True,
            "moe_workspace_chunk_tokens": args.moe_workspace_chunk_tokens,
        },
        "workload": {
            "num_requests": args.num_requests,
            "prompt_length": args.prompt_length,
            "output_tokens_per_request": 1,
            "prompt_generation": "torch.randint 使用实验 seed 生成固定 token 序列",
            "arrival_generation": "random.Random(seed) 生成 Poisson 到达序列",
        },
        "device": device_metadata(),
        "duration_s": duration_s,
        "throughput": {
            "requests_per_second": args.num_requests / duration_s,
            "input_tokens_per_second": (
                args.num_requests * args.prompt_length / duration_s
            ),
            "total_tokens_per_second": (
                args.num_requests * (args.prompt_length + 1) / duration_s
            ),
        },
        "latency": {
            "ttft": summarize_ms(ttft_samples),
            "queue": summarize_ms(queue_samples),
            "batch_service": summarize_ms(service_samples),
        },
        "memory": {
            "baseline_allocated_bytes": baseline_bytes,
            "peak_allocated_bytes": peak_allocated_bytes,
            "peak_workspace_delta_bytes": peak_allocated_bytes - baseline_bytes,
        },
        "batching": {
            "num_batches": len(batch_records),
            "mean_batch_size": statistics.fmean(
                batch["size"] for batch in batch_records
            ),
            "max_batch_size": max(batch["size"] for batch in batch_records),
            "batches": batch_records,
        },
        "requests": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "throughput": result["throughput"],
                "ttft": result["latency"]["ttft"],
                "memory": result["memory"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
