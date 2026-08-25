#!/usr/bin/env python3
"""在固定本地数据上评估一个真实 HQQ checkpoint 的小型下游任务集。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from gemq.utils.hf_loading import load_quantized_model


NUMBER_RE = re.compile(r"-?(?:\d+(?:\.\d*)?|\.\d+)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_hash(*parts: str) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) == limit:
                    break
    if len(rows) != limit:
        raise ValueError(f"{path} only contains {len(rows)} rows; expected {limit}")
    return rows


def normalize_number(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        number = Decimal(value.replace(",", ""))
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    normalized = format(number.normalize(), "f")
    return "0" if normalized in {"-0", "-0.0"} else normalized


def extract_gsm8k_gold(answer: str) -> str | None:
    marker = answer.rsplit("####", 1)[-1]
    matches = NUMBER_RE.findall(marker.replace(",", ""))
    return normalize_number(matches[-1]) if matches else None


def extract_gsm8k_prediction(text: str) -> str | None:
    matches = NUMBER_RE.findall(text.replace(",", ""))
    return normalize_number(matches[-1]) if matches else None


def score_completion(model, tokenizer, prompt: str, completion: str, device: str) -> float:
    """返回 completion 在给定 prompt 后的平均 token 对数似然。"""
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    completion_ids = tokenizer(completion, add_special_tokens=False).input_ids
    if not prompt_ids or not completion_ids:
        raise ValueError("prompt and completion must both contain at least one token")
    input_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, use_cache=False).logits.float()
    start = len(prompt_ids) - 1
    end = start + len(completion_ids)
    selected = F.log_softmax(logits[0, start:end], dim=-1)
    target = torch.tensor(completion_ids, dtype=torch.long, device=device)
    return float(selected.gather(1, target[:, None]).mean())


def evaluate_wikitext(model, tokenizer, path: Path, device: str, seqlen: int) -> dict:
    dataset = load_dataset("parquet", data_files={"test": str(path)}, split="test")
    input_ids = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt", add_special_tokens=False).input_ids[0]
    items = []
    total_nll = 0.0
    total_tokens = 0
    for index, start in enumerate(range(0, input_ids.numel() - 1, seqlen)):
        row = input_ids[start : start + seqlen]
        if row.numel() < 2:
            continue
        batch = row.unsqueeze(0).to(device)
        with torch.inference_mode():
            loss = float(model(input_ids=batch, labels=batch, use_cache=False).loss)
        predicted_tokens = int(row.numel() - 1)
        nll_sum = loss * predicted_tokens
        if not math.isfinite(nll_sum):
            raise FloatingPointError(f"non-finite WikiText NLL at window {index}")
        item_sha = hashlib.sha256(row.to(torch.int64).numpy().tobytes()).hexdigest()
        items.append({"item": index, "item_sha256": item_sha, "nll_sum": nll_sum,
                      "predicted_tokens": predicted_tokens})
        total_nll += nll_sum
        total_tokens += predicted_tokens
    return {
        "metric": "perplexity",
        "value": math.exp(total_nll / total_tokens),
        "total_nll": total_nll,
        "predicted_tokens": total_tokens,
        "items": items,
    }


def evaluate_boolq(model, tokenizer, path: Path, device: str, limit: int) -> dict:
    items = []
    for index, row in enumerate(read_jsonl(path, limit)):
        passage = row["passage"].strip()
        question = row["question"].strip()
        prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer:"
        scores = {
            "yes": score_completion(model, tokenizer, prompt, " yes", device),
            "no": score_completion(model, tokenizer, prompt, " no", device),
        }
        prediction = max(("no", "yes"), key=lambda choice: (scores[choice], choice))
        target = "yes" if bool(row["label"]) else "no"
        items.append({
            "item": index,
            "item_sha256": identity_hash(passage, question, target),
            "prediction": prediction,
            "target": target,
            "correct": int(prediction == target),
            "choice_logprob": scores,
        })
    return {"metric": "accuracy", "value": sum(x["correct"] for x in items) / len(items), "items": items}


def evaluate_gsm8k(model, tokenizer, path: Path, device: str, limit: int, batch_size: int) -> dict:
    rows = read_jsonl(path, limit)
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    items = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        prompts = [f"Question: {row['question'].strip()}\nAnswer: Let's think step by step." for row in batch_rows]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                            max_length=1536, add_special_tokens=False).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=128,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_width = encoded.input_ids.shape[1]
        continuations = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
        for offset, (row, continuation) in enumerate(zip(batch_rows, continuations)):
            question = row["question"].strip()
            target = extract_gsm8k_gold(row["answer"])
            prediction = extract_gsm8k_prediction(continuation)
            items.append({
                "item": start + offset,
                "item_sha256": identity_hash(question, target or ""),
                "prediction": prediction,
                "target": target,
                "correct": int(prediction is not None and prediction == target),
                "completion": continuation,
            })
    tokenizer.padding_side = old_side
    return {"metric": "exact_match", "value": sum(x["correct"] for x in items) / len(items), "items": items}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--checkpoint-seed", type=int, required=True)
    parser.add_argument("--wikitext", type=Path, required=True)
    parser.add_argument("--gsm8k", type=Path, required=True)
    parser.add_argument("--boolq", type=Path, required=True)
    parser.add_argument("--gsm8k-limit", type=int, default=128)
    parser.add_argument("--boolq-limit", type=int, default=256)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.wikitext, args.gsm8k, args.boolq):
        if not path.is_file():
            raise FileNotFoundError(path)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=False)
    model = load_quantized_model(
        args.checkpoint, compute_dtype=torch.float16, device="cuda", trust_remote_code=False
    ).eval()
    old_cache = model.config.use_cache
    model.config.use_cache = False
    result = {
        "schema_version": 1,
        "model": "allenai/OLMoE-1B-7B-0924",
        "checkpoint": str(args.checkpoint.resolve()),
        "method": args.method,
        "checkpoint_seed": args.checkpoint_seed,
        "protocol": {
            "wikitext2-test": "full fixed local split; non-overlapping windows",
            "gsm8k-test": "first 128 official-order records; zero-shot greedy; last-number exact match",
            "boolq-validation": "first 256 official-order records; yes/no mean conditional log-likelihood",
        },
        "source_sha256": {
            "wikitext2-test": sha256_file(args.wikitext),
            "gsm8k-test": sha256_file(args.gsm8k),
            "boolq-validation": sha256_file(args.boolq),
        },
        "tasks": {},
    }
    result["tasks"]["wikitext2-test"] = evaluate_wikitext(model, tokenizer, args.wikitext, "cuda", args.seqlen)
    result["tasks"]["gsm8k-test"] = evaluate_gsm8k(
        model, tokenizer, args.gsm8k, "cuda", args.gsm8k_limit, args.generation_batch_size
    )
    result["tasks"]["boolq-validation"] = evaluate_boolq(
        model, tokenizer, args.boolq, "cuda", args.boolq_limit
    )
    model.config.use_cache = old_cache
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "metrics": {
        key: value["value"] for key, value in result["tasks"].items()
    }}, ensure_ascii=False, sort_keys=True))
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
