#!/usr/bin/env python3
"""比较 P1 sorted 参考后端与最终 fused 后端的整模型 logits。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gemq.inference.kv_cache import StaticCache
from gemq.inference.patch import prepare_for_inference
from gemq.utils.hf_loading import load_quantized_model
from gemq.utils.model_utils import get_blocks


def set_backend(blocks, backend: str) -> None:
    for block in blocks:
        block.mlp.prefill_backend = backend


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-name", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--lengths", default="128,512")
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--candidate-backend", choices=("grouped", "fused"), default="fused"
    )
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-3)
    parser.add_argument("--min-argmax-agreement", type=float, default=0.95)
    parser.add_argument("--max-mean-abs-error", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = load_quantized_model(
        str(args.checkpoint),
        compute_dtype=torch.float16,
        device="cuda",
        trust_remote_code=False,
    ).eval()
    prepare_for_inference(model, args.model_name, is_fp=False)
    blocks = get_blocks(model, args.model_name)
    cases = {}

    for length in [int(value) for value in args.lengths.split(",")]:
        input_ids = torch.randint(
            0, int(model.config.vocab_size), (1, length), device="cuda"
        )
        positions = torch.arange(length, device="cuda")

        set_backend(blocks, "sorted")
        reference = model(
            input_ids,
            past_key_values=StaticCache(model.config, max_cache_len=length),
            cache_position=positions,
        ).logits.detach()
        set_backend(blocks, args.candidate_backend)
        candidate = model(
            input_ids,
            past_key_values=StaticCache(model.config, max_cache_len=length),
            cache_position=positions,
        ).logits.detach()
        torch.cuda.synchronize()

        difference = (candidate.float() - reference.float()).abs()
        argmax_agreement = (candidate.argmax(dim=-1) == reference.argmax(dim=-1)).float().mean()
        allclose = torch.allclose(candidate, reference, atol=args.atol, rtol=args.rtol)
        cases[str(length)] = {
            "allclose": bool(allclose),
            "argmax_agreement": float(argmax_agreement),
            "max_abs_error": float(difference.max()),
            "mean_abs_error": float(difference.mean()),
        }
        if (
            argmax_agreement < args.min_argmax_agreement
            or float(difference.mean()) > args.max_mean_abs_error
        ):
            raise AssertionError(f"length={length} 端到端回归失败：{cases[str(length)]}")

    payload = {
        "schema_version": 1,
        "reference_backend": "sorted",
        "candidate_backend": args.candidate_backend,
        "checkpoint": str(args.checkpoint.resolve()),
        "atol": args.atol,
        "rtol": args.rtol,
        "min_argmax_agreement": args.min_argmax_agreement,
        "max_mean_abs_error": args.max_mean_abs_error,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
