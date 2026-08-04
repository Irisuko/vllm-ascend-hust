#!/usr/bin/env python3
"""Merge process-local KV transfer traces into one ordered JSONL artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from m0_contract import merge_event_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    paths = args.trace_dir.glob("transfer-events-*.jsonl")
    rows = merge_event_files(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, separators=(",", ":"), sort_keys=True))
            output.write("\n")
    print(f"merged {len(rows)} events into {args.output}")


if __name__ == "__main__":
    main()
