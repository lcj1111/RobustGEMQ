import json

from scripts.phase10.analyze_downstream import load_results
from scripts.phase10.evaluate_downstream import (
    extract_gsm8k_gold,
    extract_gsm8k_prediction,
    normalize_number,
)


def test_gsm8k_number_extraction_is_deterministic():
    assert extract_gsm8k_gold("work\n#### 1,234.50") == "1234.5"
    assert extract_gsm8k_prediction("First 10, then the answer is -2.00") == "-2"
    assert normalize_number("-0.0") == "0"


def test_downstream_analyzer_rejects_identity_mismatch(tmp_path):
    methods = ["gemq-c4", "usage-only"]
    seeds = [101]
    for method in methods:
        target = tmp_path / method
        target.mkdir()
        tasks = {}
        for task in ("wikitext2-test", "gsm8k-test", "boolq-validation"):
            item_hash = "same" if method == "gemq-c4" else "different"
            tasks[task] = {"value": 1.0, "items": [{"item_sha256": item_hash}]}
        (target / "seed-101.json").write_text(json.dumps({
            "method": method,
            "checkpoint_seed": 101,
            "source_sha256": {"source": "hash"},
            "tasks": tasks,
        }), encoding="utf-8")
    try:
        load_results(tmp_path, methods, seeds)
    except ValueError as error:
        assert "identity mismatch" in str(error)
    else:
        raise AssertionError("identity mismatch was not rejected")
