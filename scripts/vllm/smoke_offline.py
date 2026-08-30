#!/usr/bin/env python3
"""通过 vLLM 真引擎执行一个最小离线生成，并原子写出结构化结果。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--prompt", default="The purpose of expert routing is")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    args = parser.parse_args()

    # 延迟导入，保证 spawn 子进程能先安全重载该脚本。
    from vllm import LLM, SamplingParams

    started = time.time()
    engine = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=4,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        trust_remote_code=False,
    )
    loaded = time.time()
    output = engine.generate(
        [args.prompt], SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    )[0]
    finished = time.time()
    result = {
        "schema_version": 1,
        "status": "pass",
        "engine": "vllm",
        "label": args.label,
        "model": args.model,
        "dtype": args.dtype,
        "enforce_eager": True,
        "max_model_len": args.max_model_len,
        "load_seconds": loaded - started,
        "generation_seconds": finished - loaded,
        "prompt": args.prompt,
        "text": output.outputs[0].text,
        "token_ids": output.outputs[0].token_ids,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
