#!/usr/bin/env python3
"""为真实量化检查点采集 MoE prefill 的可复现性能基线。"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch

from gemq.inference.kv_cache import StaticCache
from gemq.inference.patch import prepare_for_inference
from gemq.utils.hf_loading import load_quantized_model
from gemq.utils.model_utils import get_blocks


def parse_lengths(value: str) -> list[int]:
    lengths = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("lengths 必须是逗号分隔的正整数")
    if len(set(lengths)) != len(lengths):
        raise argparse.ArgumentTypeError("lengths 不能重复")
    return lengths


def percentile(values: list[float], probability: float) -> float:
    """使用线性插值计算确定性的样本分位数。"""
    if not values:
        raise ValueError("cannot summarize an empty sample")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize_latencies(latencies_ms: list[float], tokens: int) -> dict:
    median_ms = statistics.median(latencies_ms)
    return {
        "samples_ms": latencies_ms,
        "median_ms": median_ms,
        "p95_ms": percentile(latencies_ms, 0.95),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "median_tokens_per_second": tokens / (median_ms / 1000.0),
    }


def git_revision(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def cuda_time(callable_, repeats: int) -> tuple[list[float], int, int]:
    latencies = []
    peak_bytes = 0
    peak_delta_bytes = 0
    for _ in range(repeats):
        baseline_bytes = int(torch.cuda.memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = callable_()
        end.record()
        end.synchronize()
        latencies.append(float(start.elapsed_time(end)))
        current_peak = int(torch.cuda.max_memory_allocated())
        peak_bytes = max(peak_bytes, current_peak)
        peak_delta_bytes = max(peak_delta_bytes, current_peak - baseline_bytes)
        del output
    return latencies, peak_bytes, peak_delta_bytes


def warmup(callable_, iterations: int) -> None:
    for _ in range(iterations):
        output = callable_()
        torch.cuda.synchronize()
        del output


def profile_call(callable_, trace_path: Path | None) -> dict:
    from torch.profiler import ProfilerActivity, profile, record_function

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        with record_function("robustgemq_prefill_profile"):
            output = callable_()
            torch.cuda.synchronize()
            del output
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(trace_path))

    cuda_events = []
    for event in profiler.events():
        if str(event.device_type).lower().endswith("cuda"):
            cuda_events.append(event)
    by_name: dict[str, dict] = {}
    for event in cuda_events:
        entry = by_name.setdefault(event.name, {"calls": 0, "device_time_us": 0.0})
        entry["calls"] += 1
        entry["device_time_us"] += float(getattr(event, "device_time", 0.0))
    top = sorted(by_name.items(), key=lambda item: item[1]["device_time_us"], reverse=True)[:25]
    empty_metric = {"calls": 0, "device_time_us": 0.0}
    dequant = by_name.get("dequant_group_gemm_kernel", empty_metric)
    variable_m = by_name.get(
        "mixedbit_variable_m_grouped_gemm_kernel", empty_metric
    )
    fused_up = by_name.get("mixedbit_fused_up_activation_kernel", empty_metric)
    deterministic_reduce = by_name.get(
        "deterministic_unpermute_reduce_kernel", empty_metric
    )
    dtoh = {"calls": 0, "device_time_us": 0.0}
    for name, metric in by_name.items():
        if name.startswith("Memcpy DtoH"):
            dtoh["calls"] += metric["calls"]
            dtoh["device_time_us"] += metric["device_time_us"]
    return {
        "cuda_event_count": len(cuda_events),
        "unique_cuda_event_names": len(by_name),
        "dequant_group_gemm": dequant,
        "variable_m_grouped_gemm": variable_m,
        "fused_up_activation": fused_up,
        "deterministic_unpermute_reduce": deterministic_reduce,
        "device_to_host_memcpy": dtoh,
        "top_cuda_events": [{"name": name, **values} for name, values in top],
        "trace_path": str(trace_path) if trace_path is not None else None,
    }


def make_cache(model, length: int) -> StaticCache:
    return StaticCache(model.config, max_cache_len=length)


def model_forward(model, input_ids: torch.Tensor, cache: StaticCache):
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    return model(input_ids, past_key_values=cache, cache_position=positions)


def capture_block_input(model, block, input_ids: torch.Tensor) -> torch.Tensor:
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, inputs):
        captured["hidden_states"] = inputs[0].detach().clone()

    handle = block.register_forward_pre_hook(hook)
    try:
        output = model_forward(model, input_ids, make_cache(model, input_ids.shape[1]))
        torch.cuda.synchronize()
        del output
    finally:
        handle.remove()
    if "hidden_states" not in captured:
        raise RuntimeError("没有捕获到目标 MoE block 的输入")
    return captured["hidden_states"]


def device_metadata() -> dict:
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


def write_result(path: Path, result: dict) -> None:
    """每完成一个阶段就原子式刷新结果，避免长基准中断后丢失全部数据。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-name", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--lengths", type=parse_lengths, default=parse_lengths("128,512,2048,4096"))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--block-index", type=int, default=7)
    parser.add_argument("--profile-length", type=int, default=2048)
    parser.add_argument(
        "--profile-scope", choices=("none", "block", "full", "both"), default="block"
    )
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 3:
        raise ValueError("warmup 至少为 1，repeats 至少为 3")
    if args.profile_length not in args.lengths:
        raise ValueError("profile-length 必须包含在 lengths 中")
    if not torch.cuda.is_available():
        raise RuntimeError("prefill benchmark 需要 CUDA")

    repo = Path(__file__).resolve().parents[2]
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    started_at = time.time()
    model = load_quantized_model(
        str(args.checkpoint), compute_dtype=torch.float16, device="cuda", trust_remote_code=False
    ).eval()
    prepare_for_inference(model, args.model_name, is_fp=False)
    blocks = get_blocks(model, args.model_name)
    if not 0 <= args.block_index < len(blocks):
        raise ValueError(f"block-index 超出范围：{args.block_index}/{len(blocks)}")
    block = blocks[args.block_index].mlp

    result = {
        "schema_version": 1,
        "benchmark": "mixed-bit-moe-prefill-baseline",
        "checkpoint": str(args.checkpoint.resolve()),
        "model_name": args.model_name,
        "git_revision": git_revision(repo),
        "seed": args.seed,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "block_index": args.block_index,
        "device": device_metadata(),
        "cases": {},
    }

    vocab_size = int(model.config.vocab_size)
    for length in args.lengths:
        print(f"[prefill] 开始 length={length}", flush=True)
        input_ids = torch.randint(0, vocab_size, (1, length), device="cuda", dtype=torch.long)
        captured = capture_block_input(model, block, input_ids)

        cache = make_cache(model, length)
        full_call = lambda: model_forward(model, input_ids, cache)
        block_call = lambda: block(captured)
        warmup(full_call, args.warmup)
        warmup(block_call, args.warmup)

        full_latencies, full_peak, full_delta = cuda_time(full_call, args.repeats)
        block_latencies, block_peak, block_delta = cuda_time(block_call, args.repeats)
        case = {
            "tokens": length,
            "full_model": {
                **summarize_latencies(full_latencies, length),
                "peak_allocated_bytes": full_peak,
                "peak_workspace_delta_bytes": full_delta,
            },
            "moe_block": {
                **summarize_latencies(block_latencies, length),
                "peak_allocated_bytes": block_peak,
                "peak_workspace_delta_bytes": block_delta,
            },
        }
        result["cases"][str(length)] = case
        result["wall_time_seconds"] = time.time() - started_at
        write_result(args.output, result)
        if length == args.profile_length and args.profile_scope != "none":
            trace_dir = args.trace_dir
            if args.profile_scope in ("block", "both"):
                case["moe_block"]["profile"] = profile_call(
                    block_call,
                    trace_dir / f"moe-block-{length}.json" if trace_dir else None,
                )
                write_result(args.output, result)
            if args.profile_scope in ("full", "both"):
                case["full_model"]["profile"] = profile_call(
                    full_call,
                    trace_dir / f"full-model-{length}.json" if trace_dir else None,
                )
                write_result(args.output, result)
        del cache, captured, input_ids
        torch.cuda.empty_cache()
        print(f"[prefill] 完成 length={length}", flush=True)

    result["wall_time_seconds"] = time.time() - started_at
    write_result(args.output, result)
    print(json.dumps({"output": str(args.output), "cases": list(result["cases"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
