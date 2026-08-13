import pickle
import math
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "phase1" / "merge_layer_re.py"


def dump(path, value):
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def test_merge_expert_shards(tmp_path):
    left = {0: {0: {1: 0.1, 2: 0.2, 3: 0.3}}}
    right = {0: {1: {1: 0.4, 2: 0.5, 3: 0.6}}}
    dump(tmp_path / "left.pkl", left)
    dump(tmp_path / "right.pkl", right)
    output = tmp_path / "merged.pkl"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(tmp_path / "left.pkl"),
            str(tmp_path / "right.pkl"),
            "--output",
            str(output),
            "--layers",
            "1",
            "--experts",
            "2",
        ],
        check=True,
    )

    with output.open("rb") as handle:
        assert pickle.load(handle) == {0: {0: left[0][0], 1: right[0][1]}}


def test_merge_rejects_duplicate_experts(tmp_path):
    shard = {0: {0: {1: 0.1, 2: 0.2, 3: 0.3}}}
    dump(tmp_path / "one.pkl", shard)
    dump(tmp_path / "two.pkl", shard)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(tmp_path / "one.pkl"),
            str(tmp_path / "two.pkl"),
            "--output",
            str(tmp_path / "merged.pkl"),
            "--layers",
            "1",
            "--experts",
            "1",
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "duplicates" in result.stderr


def test_merge_rejects_nonfinite_coefficients(tmp_path):
    dump(tmp_path / "bad.pkl", {0: {0: {1: 0.1, 2: math.nan, 3: 0.3}}})

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(tmp_path / "bad.pkl"),
            "--output",
            str(tmp_path / "merged.pkl"),
            "--layers",
            "1",
            "--experts",
            "1",
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "invalid coefficient" in result.stderr
