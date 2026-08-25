#!/usr/bin/env python3
"""从训练来源构造记录级互斥、可审计的四路数据划分。"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import unicodedata
from pathlib import Path

SPLITS = ("calibration-a", "calibration-b", "validation", "test")
DOMAINS = ("general", "math", "code", "instruction")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_domain_registry(path: str | Path) -> dict:
    path = Path(path)
    registry = json.loads(path.resolve().read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1 or not isinstance(registry.get("domains"), dict):
        raise ValueError("domain registry must use schema v1 and contain domains")
    registry["_path"] = str(path.resolve())
    return registry


def load_records(path: Path, file_format: str) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    kwargs = {"mode": "rt", "encoding": "utf-8"} if path.suffix == ".gz" else {"mode": "r", "encoding": "utf-8"}
    with opener(path, **kwargs) as handle:
        records = [json.loads(line) for line in handle if line.strip()] if file_format == "jsonl" else json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError(f"empty or invalid record file: {path}")
    return records


def format_record(record: dict, template: str) -> str:
    values = {
        key: "\n".join(map(str, value)) if isinstance(value, list) else "" if value is None else value
        for key, value in record.items()
    }
    try:
        text = template.format_map(values).strip()
    except KeyError as error:
        raise ValueError(f"template field is absent: {error.args[0]}") from error
    if not text:
        raise ValueError("formatted record is empty")
    return text


def normalized_text(text: str) -> str:
    """用于泄漏检测的保守文本规范化；不改变最终写出的原始记录。"""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def digest_text(text: str) -> str:
    return hashlib.sha256(normalized_text(text).encode("utf-8")).hexdigest()


def stable_fraction(salt: str, record_hash: str) -> float:
    digest = hashlib.sha256(f"{salt}\0{record_hash}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def choose_split(value: float, fractions: dict[str, float]) -> str:
    cumulative = 0.0
    for name in SPLITS:
        cumulative += fractions[name]
        if value < cumulative:
            return name
    return SPLITS[-1]


def identity_set_sha256(values: set[str]) -> str:
    payload = json.dumps(sorted(values), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in sorted(records, key=lambda row: row["_robustgemq_record_id"]):
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_registry(base: dict, data_root: Path, output_root: Path, name: str) -> Path:
    registry = copy.deepcopy({key: value for key, value in base.items() if key != "_path"})
    registry["registry_id"] = f"robustgemq-{name}-records-v1"
    for domain in DOMAINS:
        record_path = output_root / "records" / name / f"{domain}.jsonl"
        registry["domains"][domain]["allocation"] = {
            **registry["domains"][domain]["allocation"],
            "path": record_path.relative_to(data_root).as_posix(),
            "format": "jsonl",
            "id_field": "_robustgemq_record_id",
        }
    path = output_root / "registries" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    if not output_root.is_relative_to(data_root):
        raise ValueError("output-root must be inside data-root so generated registries stay portable")
    experiment = json.loads(args.experiment.read_text(encoding="utf-8"))
    protocol = experiment["data_protocol"]
    fractions = {name: float(protocol["record_fractions"][name]) for name in SPLITS}
    if abs(sum(fractions.values()) - 1.0) > 1e-12 or any(value <= 0 for value in fractions.values()):
        raise ValueError("record fractions must be positive and sum to one")
    salt = str(protocol["split_salt"])
    registry = load_domain_registry(args.registry)

    buckets = {name: {domain: [] for domain in DOMAINS} for name in SPLITS}
    source_summary = {}
    for domain in DOMAINS:
        allocation = registry["domains"][domain]["allocation"]
        source_path = (data_root / allocation["path"]).resolve()
        records = load_records(source_path, allocation["format"])
        unique: dict[str, dict] = {}
        for index, record in enumerate(records):
            text = format_record(record, allocation["template"])
            record_hash = digest_text(text)
            if record_hash in unique:
                continue
            enriched = dict(record)
            enriched["_robustgemq_record_id"] = record_hash
            enriched["_robustgemq_source_index"] = index
            unique[record_hash] = enriched
        for record_hash, record in unique.items():
            split = choose_split(stable_fraction(salt, record_hash), fractions)
            buckets[split][domain].append(record)
        source_summary[domain] = {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "records": len(records),
            "unique_normalized_records": len(unique),
            "duplicates_removed": len(records) - len(unique),
        }

    # calibration-A 再按记录哈希分为三个互斥统计场景，避免把不同 seed 当成独立样本。
    calibration_shards = {seed: {domain: [] for domain in DOMAINS} for seed in (0, 1, 2)}
    for domain in DOMAINS:
        for record in buckets["calibration-a"][domain]:
            record_hash = record["_robustgemq_record_id"]
            digest = hashlib.sha256(f"{salt}\0calibration-a-shard\0{record_hash}".encode()).digest()
            seed = int.from_bytes(digest[:8], "big") % 3
            calibration_shards[seed][domain].append(record)

    for split in SPLITS:
        for domain in DOMAINS:
            write_jsonl(output_root / "records" / split / f"{domain}.jsonl", buckets[split][domain])
    for seed, domains in calibration_shards.items():
        for domain, records in domains.items():
            write_jsonl(output_root / "records" / f"calibration-a-seed-{seed}" / f"{domain}.jsonl", records)

    registries = {name: str(write_registry(registry, data_root, output_root, name)) for name in SPLITS}
    for seed in (0, 1, 2):
        name = f"calibration-a-seed-{seed}"
        registries[name] = str(write_registry(registry, data_root, output_root, name))

    identities = {
        split: {record["_robustgemq_record_id"] for domain in DOMAINS for record in buckets[split][domain]}
        for split in SPLITS
    }
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            if identities[left] & identities[right]:
                raise RuntimeError(f"record leakage between {left} and {right}")

    files = {}
    for path in sorted((output_root / "records").rglob("*.jsonl")):
        files[path.relative_to(output_root).as_posix()] = sha256_file(path)
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment["experiment_id"],
        "split_salt_sha256": hashlib.sha256(salt.encode()).hexdigest(),
        "normalization": "NFKC + casefold + whitespace collapse",
        "assignment": "SHA-256 hash partition by normalized record identity",
        "source": source_summary,
        "splits": {
            split: {
                "records_by_domain": {domain: len(buckets[split][domain]) for domain in DOMAINS},
                "unique_identity_count": len(identities[split]),
                "identity_set_sha256": identity_set_sha256(identities[split]),
            }
            for split in SPLITS
        },
        "calibration_a_shards": {
            str(seed): {
                "records_by_domain": {domain: len(calibration_shards[seed][domain]) for domain in DOMAINS}
            }
            for seed in (0, 1, 2)
        },
        "registries": registries,
        "record_file_sha256": files,
        "pairwise_record_overlap": 0,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "split-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "overlap": 0, "splits": manifest["splits"]}, sort_keys=True))


if __name__ == "__main__":
    main()
