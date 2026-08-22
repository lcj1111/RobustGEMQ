from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.phase6.prepare_code_domain import normalized_hash, records_from_shard


def test_records_select_one_python3_and_reject_heldout(tmp_path: Path):
    path = tmp_path / "train.parquet"
    table = pa.Table.from_pylist(
        [
            {
                "name": "kept",
                "description": "Add two integers",
                "source": 2,
                "solutions": {"language": [2, 3, 3], "solution": ["cpp", "print(a+b)", "x=input()"]},
            },
            {
                "name": "heldout",
                "description": "Secret heldout prompt",
                "source": 5,
                "solutions": {"language": [3], "solution": ["print(1)"]},
            },
            {
                "name": "python2-only",
                "description": "Old Python",
                "source": 1,
                "solutions": {"language": [1], "solution": ["print 1"]},
            },
        ]
    )
    pq.write_table(table, path)
    records = list(records_from_shard(path, {normalized_hash("Secret heldout prompt")}))
    assert len(records) == 1
    assert records[0]["name"] == "kept"
    assert records[0]["source"] == "CODEFORCES"
    assert records[0]["solution"] in {"print(a+b)", "x=input()"}


def test_normalized_hash_ignores_case_and_whitespace():
    assert normalized_hash(" A\n  B ") == normalized_hash("a b")
