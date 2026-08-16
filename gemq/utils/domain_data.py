"""Deterministic, auditable calibration scenarios for RobustGEMQ."""

from __future__ import annotations

import gzip
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

import torch


SCHEMA_VERSION = 1


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_domain_registry(path: str | Path) -> dict:
    path = Path(path).resolve()
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported domain registry schema {registry.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    domains = registry.get("domains")
    if not isinstance(domains, dict) or not domains:
        raise ValueError("Domain registry must contain a non-empty 'domains' mapping")
    for name, spec in domains.items():
        allocation = spec.get("allocation", {})
        for key in ("path", "format", "template"):
            if not allocation.get(key):
                raise ValueError(f"Domain {name!r} is missing allocation.{key}")
        if allocation["format"] not in {"json", "jsonl"}:
            raise ValueError(f"Domain {name!r} has unsupported format {allocation['format']!r}")
    registry["_path"] = str(path)
    return registry


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def load_records(path: Path, file_format: str) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Domain allocation file does not exist: {path}")
    with _open_text(path) as handle:
        if file_format == "jsonl":
            records = [json.loads(line) for line in handle if line.strip()]
        elif file_format == "json":
            records = json.load(handle)
        else:
            raise ValueError(f"Unsupported domain file format: {file_format}")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Domain allocation file is empty or not a record list: {path}")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Every record must be a JSON object: {path}")
    return records


def format_record(record: dict, template: str) -> str:
    values = dict(record)
    for key, value in list(values.items()):
        if isinstance(value, list):
            values[key] = "\n".join(map(str, value))
        elif value is None:
            values[key] = ""
    try:
        text = template.format_map(values)
    except KeyError as error:
        raise ValueError(f"Template field {error.args[0]!r} is absent from record") from error
    text = text.strip()
    if not text:
        raise ValueError("Formatted domain record is empty")
    return text


def _record_id(record: dict, index: int, id_field: str | None) -> str:
    if id_field and record.get(id_field) not in (None, ""):
        return str(record[id_field])
    return str(index)


def _hash_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hash_tokens(tokens: torch.Tensor) -> str:
    contiguous = tokens.to(dtype=torch.int64, device="cpu").contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def build_domain_scenario(
    *,
    registry_path: str | Path,
    data_root: str | Path,
    domain: str,
    tokenizer,
    nsamples: int,
    seqlen: int,
    seed: int,
) -> tuple[torch.Tensor, dict]:
    """Pack shuffled training records into deterministic fixed-length token blocks."""
    if nsamples <= 0 or seqlen <= 1:
        raise ValueError("nsamples must be positive and seqlen must be greater than one")
    registry = load_domain_registry(registry_path)
    if domain not in registry["domains"]:
        raise KeyError(f"Unknown domain {domain!r}; expected one of {sorted(registry['domains'])}")

    spec = registry["domains"][domain]
    allocation = spec["allocation"]
    relative_path = Path(allocation["path"])
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Domain path escapes data root: {relative_path}")
    # Resolve only after validating the registry path. The Phase 2 data root may
    # intentionally contain a pinned symlink to the already-audited Phase 1 C4 snapshot.
    source_path = (Path(data_root).resolve() / relative_path).resolve()
    records = load_records(source_path, allocation["format"])

    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    blocks: list[list[int]] = []
    buffer: list[int] = []
    selected_ids: list[str] = []
    cursor = 0
    while len(blocks) < nsamples:
        if cursor >= len(order):
            raise ValueError(
                f"Domain {domain!r} contains only enough tokens for {len(blocks)} blocks "
                f"of length {seqlen}; requested {nsamples}"
            )
        index = order[cursor]
        cursor += 1
        record = records[index]
        text = format_record(record, allocation["template"])
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        if not token_ids:
            continue
        selected_ids.append(_record_id(record, index, allocation.get("id_field")))
        buffer.extend(map(int, token_ids))
        while len(buffer) >= seqlen and len(blocks) < nsamples:
            blocks.append(buffer[:seqlen])
            del buffer[:seqlen]

    tokens = torch.tensor(blocks, dtype=torch.long)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "registry_id": registry.get("registry_id"),
        "registry_sha256": sha256_file(Path(registry["_path"])),
        "domain": domain,
        "source": spec["source"],
        "revision": spec["revision"],
        "license": spec["license"],
        "allocation_path": str(source_path),
        "allocation_sha256": sha256_file(source_path),
        "seed": seed,
        "nsamples": nsamples,
        "seqlen": seqlen,
        "effective_tokens": int(tokens.numel()),
        "selected_record_count": len(selected_ids),
        "selected_ids_sha256": _hash_json(selected_ids),
        "token_sha256": _hash_tokens(tokens),
        "held_out": spec.get("held_out", []),
    }
    return tokens, manifest


def save_domain_scenario(tokens: torch.Tensor, manifest: dict, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token_hash = manifest["token_sha256"]
    token_path = output_dir / f"tokens-{token_hash[:12]}.pt"
    manifest_path = output_dir / "scenario.json"

    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != {**manifest, "tokens_path": str(token_path)}:
            raise ValueError(
                f"Scenario identity collision in {output_dir}; remove or choose a new scenario directory"
            )
    if token_path.exists():
        existing_tokens = torch.load(token_path, map_location="cpu", weights_only=True)
        if _hash_tokens(existing_tokens) != token_hash:
            raise ValueError(f"Existing token cache failed identity check: {token_path}")
    else:
        torch.save(tokens, token_path)

    saved_manifest = {**manifest, "tokens_path": str(token_path)}
    manifest_path.write_text(
        json.dumps(saved_manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return saved_manifest


def load_scenario_tokens(path: str | Path) -> list[tuple[torch.Tensor, None]]:
    tokens = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.shape[0] == 0:
        raise ValueError(f"Scenario token cache must be a non-empty [samples, seqlen] tensor: {path}")
    return [(row.unsqueeze(0), None) for row in tokens]
