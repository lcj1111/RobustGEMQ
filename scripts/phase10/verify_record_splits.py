#!/usr/bin/env python3
"""独立重算四路记录划分的哈希、唯一性和互斥性。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.phase10.build_record_splits import (
    DOMAINS,
    SPLITS,
    digest_text,
    format_record,
    identity_set_sha256,
    load_domain_registry,
    load_records,
    sha256_file,
)


def fail(message: str) -> None:
    raise ValueError(f"记录划分无效：{message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = args.manifest.parent
    identities = {split: set() for split in SPLITS}
    domain_counts = {split: {} for split in SPLITS}

    for split in SPLITS:
        registry = load_domain_registry(manifest["registries"][split])
        for domain in DOMAINS:
            allocation = registry["domains"][domain]["allocation"]
            # registry 路径相对于生成数据根；manifest 位于该根下的 phase10 子目录。
            data_root = root.parent
            path = (data_root / allocation["path"]).resolve()
            relative = path.relative_to(root).as_posix()
            if sha256_file(path) != manifest["record_file_sha256"][relative]:
                fail(f"文件哈希变化：{relative}")
            records = load_records(path, "jsonl")
            observed = set()
            for record in records:
                record_id = record.get("_robustgemq_record_id")
                if not isinstance(record_id, str) or len(record_id) != 64:
                    fail(f"{split}/{domain} 缺少记录 SHA-256")
                if record_id != digest_text(format_record(record, allocation["template"])):
                    fail(f"{split}/{domain} 记录内容与身份不一致")
                if record_id in observed:
                    fail(f"{split}/{domain} 存在重复记录")
                observed.add(record_id)
            identities[split].update(observed)
            domain_counts[split][domain] = len(records)
        expected = manifest["splits"][split]
        if domain_counts[split] != expected["records_by_domain"]:
            fail(f"{split} 记录数量变化")
        if identity_set_sha256(identities[split]) != expected["identity_set_sha256"]:
            fail(f"{split} identity 集合摘要变化")

    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            overlap = identities[left] & identities[right]
            if overlap:
                fail(f"{left} 与 {right} 泄漏 {len(overlap)} 条记录")

    # calibration-A 的三个统计 seed 必须互斥，且并集等于 calibration-A。
    shard_sets = []
    for seed in (0, 1, 2):
        registry = load_domain_registry(manifest["registries"][f"calibration-a-seed-{seed}"])
        current = set()
        for domain in DOMAINS:
            allocation = registry["domains"][domain]["allocation"]
            path = (root.parent / allocation["path"]).resolve()
            current.update(record["_robustgemq_record_id"] for record in load_records(path, "jsonl"))
        if any(current & previous for previous in shard_sets):
            fail(f"calibration-A seed-{seed} 与其他 seed 重叠")
        shard_sets.append(current)
    if set().union(*shard_sets) != identities["calibration-a"]:
        fail("calibration-A seed 分片并集不完整")
    print(json.dumps({"verified": True, "pairwise_record_overlap": 0, "counts": domain_counts}, sort_keys=True))


if __name__ == "__main__":
    main()
