#!/usr/bin/env python3
"""把 H6 pytest 结果整理为稳定、可供后续 Gate 消费的 JSON。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree


def junit_counts(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {"tests": 0, "failures": 0, "errors": 1, "skipped": 0}
    root = ElementTree.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        field: sum(int(suite.attrib.get(field, 0)) for suite in suites)
        for field in ("tests", "failures", "errors", "skipped")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    counts = junit_counts(args.junit)
    passed = (
        args.exit_code == 0
        and counts["tests"] > 0
        and counts["failures"] == 0
        and counts["errors"] == 0
        and counts["skipped"] == 0
    )
    summary = {
        "schema_version": 1,
        "method": args.method,
        "checkpoint": args.checkpoint,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "exit_code": args.exit_code,
        "passed": passed,
        "pytest": counts,
        "artifacts": {"junit_xml": str(args.junit), "run_log": str(args.log)},
        "required_checks": {
            "fake_real_ppl_relative_error_max": 0.01,
            "decode_logits_relative_error": "noise-aware; see test_decode_logits_match_fake",
            "decode_argmax_agreement_min": 0.95,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
