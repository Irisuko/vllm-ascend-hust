#!/usr/bin/env python3
"""Validate and summarize matched PR 153 native/mapped pressure lifecycles."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from m0_contract import ContractError, load_json, load_jsonl, sha256_file

EXPECTED_MODES = {"native", "mapped"}
PARITY_FIELDS = (
    "vllm_commit",
    "vllm_ascend_commit",
    "vllm_ascend_head_tree",
    "model",
    "device_kv_bytes",
    "cpu_kv_bytes",
    "request_count",
    "request_rate",
    "prefix_tokens",
    "suffix_tokens",
    "output_tokens",
    "num_prefixes",
    "seed",
)
RESULT_FIELDS = (
    "duration",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p99_itl_ms",
)
PROCESS_MEMORY_RE = re.compile(r"Process memory\(MB\):\s*(\d+)")
HBM_USAGE_RE = re.compile(r"HBM Usage Rate\(%\)\s*:\s*(\d+)")


def nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise ContractError("cannot calculate a percentile of an empty sample")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile / 100.0 * len(ordered)))
    return ordered[rank - 1]


def finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{field} must be finite")
    return result


def transfer_summary(rows: list[dict[str, Any]], direction: str) -> dict[str, Any]:
    samples = [row for row in rows if row.get("event") == "transfer_done" and row.get("direction") == direction]
    if not samples:
        raise ContractError(f"no completed {direction} transfers")
    latencies_ms = [finite_number(row.get("transfer_time_ms"), "transfer_time_ms") for row in samples]
    byte_values = []
    for row in samples:
        value = row.get("bytes")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ContractError("transfer bytes must be a positive integer")
        byte_values.append(value)
    total_bytes = sum(byte_values)
    total_time_s = sum(latencies_ms) / 1000.0
    return {
        "count": len(samples),
        "bytes": total_bytes,
        "time_s": total_time_s,
        "aggregate_gbps": total_bytes / 1e9 / total_time_s,
        "mean_latency_ms": statistics.mean(latencies_ms),
        "median_latency_ms": statistics.median(latencies_ms),
        "p99_latency_ms": nearest_rank(latencies_ms, 99),
        "latency_samples_ms": latencies_ms,
    }


def load_run(run_dir: Path) -> dict[str, Any]:
    required = (
        "run-config.json",
        "result.json",
        "request_set.json",
        "raw_requests.jsonl",
        "transfer_events.jsonl",
        "resource-samples.log",
        "server.log",
        "npu-after-shutdown.txt",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise ContractError(f"{run_dir} is missing artifacts: {missing}")

    config = load_json(run_dir / "run-config.json")
    result = load_json(run_dir / "result.json")
    request_set = load_json(run_dir / "request_set.json")
    events = load_jsonl(run_dir / "transfer_events.jsonl")
    raw_requests = load_jsonl(run_dir / "raw_requests.jsonl")
    if not isinstance(config, dict) or not isinstance(result, dict):
        raise ContractError(f"{run_dir} config and result must be JSON objects")
    mode = config.get("mode")
    if mode not in EXPECTED_MODES:
        raise ContractError(f"{run_dir} mode must be native or mapped")
    if config.get("vllm_ascend_dirty") is not False:
        raise ContractError(f"{run_dir} backend source must be clean")
    if config.get("vllm_ascend_diff_sha256") != sha256_file(Path("/dev/null")):
        raise ContractError(f"{run_dir} backend source diff must be empty")

    request_count = config.get("request_count")
    if result.get("completed") != request_count or result.get("failed") != 0:
        raise ContractError(f"{run_dir} did not complete every request successfully")
    if len(raw_requests) != request_count:
        raise ContractError(f"{run_dir} raw request count does not match config")
    if any(row.get("status") != "completed" for row in raw_requests):
        raise ContractError(f"{run_dir} contains an incomplete request")
    request_set_sha = request_set.get("request_set_sha256")
    if not isinstance(request_set_sha, str) or len(request_set_sha) != 64:
        raise ContractError(f"{run_dir} request set lacks its canonical hash")

    resources = (run_dir / "resource-samples.log").read_text(encoding="utf-8")
    process_memory = [int(value) for value in PROCESS_MEMORY_RE.findall(resources)]
    hbm_usage = [int(value) for value in HBM_USAGE_RE.findall(resources)]
    if not process_memory or not hbm_usage:
        raise ContractError(f"{run_dir} has no NPU memory samples")
    physical_device_id = config.get("physical_device_id")
    if f"No running processes found in NPU {physical_device_id}" not in (run_dir / "npu-after-shutdown.txt").read_text(
        encoding="utf-8"
    ):
        raise ContractError(f"{run_dir} left a process on physical NPU {physical_device_id}")

    metrics = {field: finite_number(result.get(field), f"{run_dir}/{field}") for field in RESULT_FIELDS}
    return {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "mode": mode,
        "lifecycle_index": int(config["lifecycle_index"]),
        "config": config,
        "request_set_sha256": request_set_sha,
        "request_set_file_sha256": sha256_file(run_dir / "request_set.json"),
        "result": metrics,
        "completed": result["completed"],
        "failed": result["failed"],
        "d2h": transfer_summary(events, "d2h"),
        "h2d": transfer_summary(events, "h2d"),
        "peak_npu_process_memory_mb": max(process_memory),
        "peak_hbm_usage_percent": max(hbm_usage),
    }


def strip_samples(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "latency_samples_ms"}


def summarize_mode(runs: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"lifecycles": len(runs)}
    for field in RESULT_FIELDS:
        values = [run["result"][field] for run in runs]
        result[field] = {
            "mean": statistics.mean(values),
            "sample_stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    result["peak_npu_process_memory_mb"] = {
        "mean": statistics.mean(run["peak_npu_process_memory_mb"] for run in runs),
        "max": max(run["peak_npu_process_memory_mb"] for run in runs),
    }
    result["peak_hbm_usage_percent"] = {
        "mean": statistics.mean(run["peak_hbm_usage_percent"] for run in runs),
        "max": max(run["peak_hbm_usage_percent"] for run in runs),
    }
    for direction in ("d2h", "h2d"):
        total_bytes = sum(run[direction]["bytes"] for run in runs)
        total_time_s = sum(run[direction]["time_s"] for run in runs)
        latencies = [latency for run in runs for latency in run[direction]["latency_samples_ms"]]
        per_run_gbps = [run[direction]["aggregate_gbps"] for run in runs]
        result[direction] = {
            "count": sum(run[direction]["count"] for run in runs),
            "bytes": total_bytes,
            "time_s": total_time_s,
            "aggregate_gbps": total_bytes / 1e9 / total_time_s,
            "mean_run_gbps": statistics.mean(per_run_gbps),
            "sample_stddev_run_gbps": statistics.stdev(per_run_gbps),
            "mean_latency_ms": statistics.mean(latencies),
            "median_latency_ms": statistics.median(latencies),
            "p99_latency_ms": nearest_rank(latencies, 99),
        }
    return result


def percent_delta(mapped: float, native: float) -> float:
    return (mapped / native - 1.0) * 100.0


def build_summary(run_dirs: list[Path]) -> dict[str, Any]:
    runs = sorted((load_run(path) for path in run_dirs), key=lambda run: run["lifecycle_index"])
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        by_mode[run["mode"]].append(run)
    for mode in EXPECTED_MODES:
        if len(by_mode[mode]) < 3:
            raise ContractError(f"need at least three {mode} lifecycles")

    reference = runs[0]
    for run in runs[1:]:
        for field in PARITY_FIELDS:
            if run["config"].get(field) != reference["config"].get(field):
                raise ContractError(f"matched-run parity failed for {field}")
        if run["request_set_sha256"] != reference["request_set_sha256"]:
            raise ContractError("canonical request-set hashes differ")
        if run["request_set_file_sha256"] != reference["request_set_file_sha256"]:
            raise ContractError("request-set files differ")

    modes = {mode: summarize_mode(by_mode[mode]) for mode in sorted(EXPECTED_MODES)}
    native = modes["native"]
    mapped = modes["mapped"]
    deltas: dict[str, float] = {}
    for field in RESULT_FIELDS:
        deltas[f"{field}_percent"] = percent_delta(mapped[field]["mean"], native[field]["mean"])
    for direction in ("d2h", "h2d"):
        for field in ("aggregate_gbps", "mean_latency_ms", "p99_latency_ms"):
            deltas[f"{direction}_{field}_percent"] = percent_delta(mapped[direction][field], native[direction][field])
    deltas["peak_npu_process_memory_mb_percent"] = percent_delta(
        mapped["peak_npu_process_memory_mb"]["mean"],
        native["peak_npu_process_memory_mb"]["mean"],
    )

    pairs = []
    by_pair: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        by_pair[(run["lifecycle_index"] + 1) // 2].append(run)
    for pair_index, pair_runs in sorted(by_pair.items()):
        pair_modes = {run["mode"]: run for run in pair_runs}
        if set(pair_modes) != EXPECTED_MODES:
            raise ContractError(f"pair {pair_index} does not contain both modes")
        pair = {"pair": pair_index}
        for field in (
            "duration",
            "mean_ttft_ms",
            "p99_ttft_ms",
            "mean_tpot_ms",
            "p99_tpot_ms",
        ):
            pair[f"{field}_mapped_minus_native"] = (
                pair_modes["mapped"]["result"][field] - pair_modes["native"]["result"][field]
            )
        pairs.append(pair)

    public_runs = []
    for run in runs:
        item = {key: value for key, value in run.items() if key not in {"config", "run_dir"}}
        item["d2h"] = strip_samples(item["d2h"])
        item["h2d"] = strip_samples(item["h2d"])
        public_runs.append(item)
    return {
        "schema_version": "pr153-pressure-summary/v1",
        "request_set_sha256": reference["request_set_sha256"],
        "request_set_file_sha256": reference["request_set_file_sha256"],
        "source": {
            field: reference["config"][field]
            for field in (
                "vllm_commit",
                "vllm_ascend_commit",
                "vllm_ascend_head_tree",
            )
        },
        "runs": public_runs,
        "modes": modes,
        "mapped_vs_native": deltas,
        "reverse_order_pairs": pairs,
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# PR 153 pressure benchmark summary",
        "",
        "| Run | Mode | Duration (s) | Mean TTFT (ms) | P99 TTFT (ms) | "
        "Mean TPOT (ms) | P99 TPOT (ms) | D2H GB/s | D2H p99 (ms) | "
        "H2D GB/s | H2D p99 (ms) | Peak NPU MB |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in summary["runs"]:
        result = run["result"]
        lines.append(
            f"| {run['lifecycle_index']} | {run['mode']} | {result['duration']:.2f} | "
            f"{result['mean_ttft_ms']:.2f} | {result['p99_ttft_ms']:.2f} | "
            f"{result['mean_tpot_ms']:.2f} | {result['p99_tpot_ms']:.2f} | "
            f"{run['d2h']['aggregate_gbps']:.2f} | {run['d2h']['p99_latency_ms']:.2f} | "
            f"{run['h2d']['aggregate_gbps']:.2f} | {run['h2d']['p99_latency_ms']:.2f} | "
            f"{run['peak_npu_process_memory_mb']} |"
        )
    lines.extend(["", "## Three-lifecycle means", ""])
    lines.extend(
        [
            "| Mode | Duration (s) | Mean TTFT (ms) | P99 TTFT (ms) | "
            "Mean TPOT (ms) | P99 TPOT (ms) | D2H GB/s | D2H p99 (ms) | "
            "H2D GB/s | H2D p99 (ms) | Peak NPU MB |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode in ("native", "mapped"):
        data = summary["modes"][mode]
        lines.append(
            f"| {mode} | {data['duration']['mean']:.2f} | "
            f"{data['mean_ttft_ms']['mean']:.2f} | {data['p99_ttft_ms']['mean']:.2f} | "
            f"{data['mean_tpot_ms']['mean']:.2f} | {data['p99_tpot_ms']['mean']:.2f} | "
            f"{data['d2h']['aggregate_gbps']:.2f} | {data['d2h']['p99_latency_ms']:.2f} | "
            f"{data['h2d']['aggregate_gbps']:.2f} | {data['h2d']['p99_latency_ms']:.2f} | "
            f"{data['peak_npu_process_memory_mb']['mean']:.0f} |"
        )
    delta = summary["mapped_vs_native"]
    lines.extend(
        [
            "",
            "## Mapped versus native",
            "",
            f"- Duration: {delta['duration_percent']:+.2f}%",
            f"- Mean TTFT: {delta['mean_ttft_ms_percent']:+.2f}%",
            f"- P99 TTFT: {delta['p99_ttft_ms_percent']:+.2f}%",
            f"- Mean TPOT: {delta['mean_tpot_ms_percent']:+.2f}%",
            f"- P99 TPOT: {delta['p99_tpot_ms_percent']:+.2f}%",
            f"- D2H aggregate bandwidth: {delta['d2h_aggregate_gbps_percent']:+.2f}%",
            f"- D2H p99 latency: {delta['d2h_p99_latency_ms_percent']:+.2f}%",
            f"- H2D aggregate bandwidth: {delta['h2d_aggregate_gbps_percent']:+.2f}%",
            f"- H2D p99 latency: {delta['h2d_p99_latency_ms_percent']:+.2f}%",
            f"- Peak NPU process memory: {delta['peak_npu_process_memory_mb_percent']:+.2f}%",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    summary = build_summary(args.run_dirs)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.write_text(markdown(summary), encoding="utf-8")
    print(f"summarized {len(summary['runs'])} lifecycles")


if __name__ == "__main__":
    main()
