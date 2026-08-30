#!/usr/bin/env python3
"""用原 RobustGEMQ/HF 路径生成确定性 greedy token，供 vLLM 端到端对照。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from gemq.inference.kv_cache import StaticCache
from gemq.inference.patch import prepare_for_inference
from gemq.utils.hf_loading import load_quantized_model


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--prompt", default="The purpose of expert routing is")
    parser.add_argument("--max-tokens", type=int, default=8)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=False)
    model = load_quantized_model(
        str(args.checkpoint),
        compute_dtype=torch.float16,
        device="cuda",
        trust_remote_code=False,
    ).eval()
    prepare_for_inference(model, args.model_name, is_fp=False)
    tokens = tokenizer(args.prompt, return_tensors="pt").input_ids.to("cuda")
    prompt_length = tokens.shape[1]
    cache = StaticCache(model.config, max_cache_len=prompt_length + args.max_tokens)
    positions = torch.arange(prompt_length, device="cuda")
    outputs = model(tokens, past_key_values=cache, cache_position=positions)
    generated = [int(outputs.logits[0, -1].argmax().item())]
    position = torch.tensor([prompt_length], device="cuda")
    for _ in range(args.max_tokens - 1):
        token = torch.tensor([[generated[-1]]], device="cuda")
        outputs = model(token, past_key_values=cache, cache_position=position)
        generated.append(int(outputs.logits[0, -1].argmax().item()))
        position += 1

    payload = {
        "schema_version": 1,
        "status": "pass",
        "engine": "transformers-robustgemq",
        "checkpoint": str(args.checkpoint.resolve()),
        "dtype": "float16",
        "backend": "chunked",
        "prompt": args.prompt,
        "prompt_token_ids": tokens[0].tolist(),
        "token_ids": generated,
        "text": tokenizer.decode(generated, skip_special_tokens=True),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
