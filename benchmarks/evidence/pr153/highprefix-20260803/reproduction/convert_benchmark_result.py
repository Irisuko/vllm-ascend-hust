#!/usr/bin/env python3
"""Convert ``vllm bench serve --save-detailed`` output to M0 JSONL."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class ConversionError(ValueError):
    """Raised when detailed benchmark output is incomplete or inconsistent."""


def _array(payload: Mapping[str, Any], name: str, count: int) -> Sequence[Any]:
    value = payload.get(name)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConversionError(f"{name} must be an array")
    if len(value) != count:
        raise ConversionError(f"{name} has {len(value)} rows; expected {count}")
    return value


def convert_result(
    payload: Mapping[str, Any],
    *,
    request_id_prefix: str,
    request_id_suffix: str = "",
    request_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    count = payload.get("num_prompts")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ConversionError("num_prompts must be a positive integer")
    if not request_id_prefix:
        raise ConversionError("request_id_prefix must not be empty")
    if request_ids is not None:
        if len(request_ids) != count:
            raise ConversionError(
                f"request_ids has {len(request_ids)} rows; expected {count}"
            )
        if any(
            not isinstance(request_id, str) or not request_id
            for request_id in request_ids
        ):
            raise ConversionError("request_ids values must be non-empty strings")
        if len(set(request_ids)) != count:
            raise ConversionError("request_ids values must be unique")

    ttfts = _array(payload, "ttfts", count)
    itls = _array(payload, "itls", count)
    input_lens = _array(payload, "input_lens", count)
    output_lens = _array(payload, "output_lens", count)
    errors = _array(payload, "errors", count)
    start_times = _array(payload, "start_times", count)

    rows: list[dict[str, Any]] = []
    for index in range(count):
        request_id = (
            request_ids[index]
            if request_ids is not None
            else f"{request_id_prefix}{index}{request_id_suffix}"
        )
        error = errors[index]
        if not isinstance(error, str):
            raise ConversionError(f"errors[{index}] must be a string")
        row: dict[str, Any] = {
            "request_id": request_id,
            "input_tokens": input_lens[index],
            "output_tokens": output_lens[index],
            "start_time_s": start_times[index],
        }
        if error:
            row.update(status="failed", error=error)
            rows.append(row)
            continue

        ttft = ttfts[index]
        request_itls = itls[index]
        if isinstance(ttft, bool) or not isinstance(ttft, (int, float)) or ttft <= 0:
            raise ConversionError(f"ttfts[{index}] must be positive")
        if not isinstance(request_itls, Sequence) or isinstance(
            request_itls, (str, bytes)
        ):
            raise ConversionError(f"itls[{index}] must be an array")
        if not request_itls:
            raise ConversionError(f"itls[{index}] must not be empty for a completed row")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value <= 0
            for value in request_itls
        ):
            raise ConversionError(f"itls[{index}] values must be positive")
        generation_s = sum(float(value) for value in request_itls)
        row.update(
            status="completed",
            ttft_ms=float(ttft) * 1000,
            tpot_ms=generation_s / len(request_itls) * 1000,
            latency_ms=(float(ttft) + generation_s) * 1000,
        )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--request-id-prefix", default="request-")
    parser.add_argument("--request-id-suffix", default="")
    parser.add_argument(
        "--request-set",
        type=Path,
        help="JSON request-set record whose ordered requests carry request_id",
    )
    args = parser.parse_args()

    payload = json.loads(args.result.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ConversionError("result must be a JSON object")
    request_ids = None
    if args.request_set is not None:
        request_set = json.loads(args.request_set.read_text(encoding="utf-8"))
        if not isinstance(request_set, Mapping):
            raise ConversionError("request set must be a JSON object")
        request_records = request_set.get("requests")
        if not isinstance(request_records, Sequence) or isinstance(
            request_records, (str, bytes)
        ):
            raise ConversionError("request set requests must be an array")
        request_ids = [
            record.get("request_id") if isinstance(record, Mapping) else None
            for record in request_records
        ]
    rows = convert_result(
        payload,
        request_id_prefix=args.request_id_prefix,
        request_id_suffix=args.request_id_suffix,
        request_ids=request_ids,
    )
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"converted {len(rows)} requests into {args.output}")


if __name__ == "__main__":
    main()
