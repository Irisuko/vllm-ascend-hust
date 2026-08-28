#!/usr/bin/env python3
"""Compare mapped-host KV gather with a direct contiguous-span copy baseline.

The benchmark keeps the logical transfer size fixed while varying the number
of contiguous CPU/NPU block-pair runs.  It measures the two production-shaped
operations on the same NPU stream:

* span: ``npu[dst_start:dst_end].copy_(cpu[src_start:src_end])`` per run
* mapped: one ``kv_cache_block_gather`` call per K/V part

Host mapping registration is reported separately and excluded from the warm
steady-state samples.  Both paths share one PyTorch pinned host allocation so
the span baseline does not pay pageable-memory staging that production does
not.  Wall time is the primary metric; NPU event time is also reported to
expose stream-coverage mistakes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shlex
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BlockPair = tuple[int, int]
BlockSpan = tuple[int, int, int]


@dataclass(frozen=True)
class Case:
    block_bytes: int
    selected_blocks: int
    requested_span_len: int
    span_count: int
    mean_span_len: float
    max_span_len: int
    single_block_span_ratio: float
    block_pairs: tuple[BlockPair, ...]
    spans: tuple[BlockSpan, ...]


def coalesce_block_copy_spans(block_mapping: Sequence[BlockPair]) -> list[BlockSpan]:
    """Mirror main's adjacent CPU/GPU block-pair coalescing."""
    if not block_mapping:
        return []

    spans: list[BlockSpan] = []
    cpu_start, npu_start = block_mapping[0]
    previous_cpu, previous_npu = cpu_start, npu_start
    span_len = 1
    for cpu_block, npu_block in block_mapping[1:]:
        if cpu_block == previous_cpu + 1 and npu_block == previous_npu + 1:
            span_len += 1
        else:
            spans.append((cpu_start, npu_start, span_len))
            cpu_start, npu_start = cpu_block, npu_block
            span_len = 1
        previous_cpu, previous_npu = cpu_block, npu_block
    spans.append((cpu_start, npu_start, span_len))
    return spans


def build_fragmented_mapping(
    *,
    selected_blocks: int,
    requested_span_len: int,
    num_cpu_blocks: int,
    num_npu_blocks: int,
    seed: int,
) -> list[BlockPair]:
    """Build exact-length contiguous runs separated by at least one block."""
    if selected_blocks <= 0:
        raise ValueError("selected_blocks must be positive")
    if requested_span_len <= 0:
        raise ValueError("requested_span_len must be positive")

    run_lengths = []
    remaining = selected_blocks
    while remaining:
        run_len = min(requested_span_len, remaining)
        run_lengths.append(run_len)
        remaining -= run_len

    slot_width = requested_span_len + 1
    required_blocks = len(run_lengths) * slot_width
    if required_blocks > num_cpu_blocks or required_blocks > num_npu_blocks:
        raise ValueError(
            "not enough source/destination blocks for fragmented mapping: "
            f"required={required_blocks}, cpu={num_cpu_blocks}, npu={num_npu_blocks}"
        )

    cpu_starts = [index * slot_width for index in range(len(run_lengths))]
    npu_starts = list(cpu_starts)
    random.Random(seed).shuffle(cpu_starts)
    random.Random(seed ^ 0x5A17).shuffle(npu_starts)

    pairs: list[BlockPair] = []
    for run_len, cpu_start, npu_start in zip(run_lengths, cpu_starts, npu_starts):
        pairs.extend((cpu_start + offset, npu_start + offset) for offset in range(run_len))
    return pairs


def make_case(
    *,
    block_bytes: int,
    selected_blocks: int,
    requested_span_len: int,
    num_cpu_blocks: int,
    num_npu_blocks: int,
    seed: int,
) -> Case:
    pairs = build_fragmented_mapping(
        selected_blocks=selected_blocks,
        requested_span_len=requested_span_len,
        num_cpu_blocks=num_cpu_blocks,
        num_npu_blocks=num_npu_blocks,
        seed=seed,
    )
    spans = coalesce_block_copy_spans(pairs)
    lengths = [span_len for _, _, span_len in spans]
    return Case(
        block_bytes=block_bytes,
        selected_blocks=selected_blocks,
        requested_span_len=requested_span_len,
        span_count=len(spans),
        mean_span_len=statistics.fmean(lengths),
        max_span_len=max(lengths),
        single_block_span_ratio=sum(length == 1 for length in lengths) / len(lengths),
        block_pairs=tuple(pairs),
        spans=tuple(spans),
    )


def percentile(values: Sequence[float], percentage: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * percentage / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_ms(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(values),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--parts", type=int, default=2, help="KV tensor parts; 2 models K and V")
    parser.add_argument("--num-cpu-blocks", type=int, default=4096)
    parser.add_argument("--num-npu-blocks", type=int, default=4096)
    parser.add_argument("--selected-blocks", type=int, default=512)
    parser.add_argument("--block-bytes", type=int, nargs="+", default=[4096, 16384, 65536])
    parser.add_argument("--span-lengths", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 512])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument(
        "--backend-order",
        choices=("span-first", "mapped-first"),
        default="span-first",
        help="Run a reverse-order repeat to expose cache or ordering bias",
    )
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--decision-margin-percent", type=float, default=10.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark-results/kv-gather-vs-span"),
    )
    parser.add_argument("--op-lib", type=Path, default=None)
    parser.add_argument("--opapi-lib", type=Path, default=None)
    parser.add_argument(
        "--backend-repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="vLLM Ascend repository recorded in the evidence manifest",
    )
    parser.add_argument(
        "--vllm-repo",
        type=Path,
        default=None,
        help="paired vLLM repository; defaults to the imported package root",
    )
    return parser.parse_args()


def git_output(repo: Path, *args: str, strip: bool = True) -> str:
    try:
        output = subprocess.check_output(
            ["git", "-c", f"safe.directory={repo}", "-C", str(repo), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return output.strip() if strip else output.rstrip("\n")
    except Exception:
        return ""


def git_repo_state(repo: Path) -> dict[str, Any]:
    requested_path = repo.expanduser().resolve()
    root_text = git_output(requested_path, "rev-parse", "--show-toplevel")
    if not root_text:
        return {
            "requested_path": str(requested_path),
            "available": False,
        }

    root = Path(root_text)
    status = git_output(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        strip=False,
    )
    return {
        "requested_path": str(requested_path),
        "root": str(root),
        "available": True,
        "branch": git_output(root, "branch", "--show-current"),
        "commit": git_output(root, "rev-parse", "HEAD"),
        "tree": git_output(root, "rev-parse", "HEAD^{tree}"),
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
    }


def file_evidence(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "available": False}

    digest = hashlib.sha256()
    with resolved.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "available": True,
        "bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def load_runtime(args: argparse.Namespace):
    import torch
    import torch_npu

    if args.op_lib is not None:
        extension_path = args.op_lib.expanduser().resolve()
        torch.ops.load_library(str(extension_path))
    else:
        import vllm_ascend.vllm_ascend_C as extension_module

        extension_path = Path(extension_module.__file__).resolve()

    from vllm_ascend.custom_op_package import (
        activate_kv_cache_block_gather_runtime,
    )

    opapi_path = activate_kv_cache_block_gather_runtime(
        torch,
        opapi_library=args.opapi_lib,
    )

    gather = torch.ops._C_ascend.kv_cache_block_gather
    register = torch.ops._C_ascend.register_kv_cache_block_gather_host_pool
    inspect = torch.ops._C_ascend.inspect_kv_cache_block_gather_host_pool
    unregister = torch.ops._C_ascend.unregister_kv_cache_block_gather_host_pool
    return (
        torch,
        torch_npu,
        gather,
        register,
        inspect,
        unregister,
        extension_path,
        opapi_path,
    )


def validate_args(args: argparse.Namespace, element_size: int) -> None:
    if args.parts <= 0 or args.warmup < 0 or args.iters <= 0:
        raise ValueError("parts and iters must be positive; warmup must be non-negative")
    for block_bytes in args.block_bytes:
        if block_bytes <= 0 or block_bytes % element_size:
            raise ValueError(f"block_bytes={block_bytes} is not divisible by dtype size {element_size}")
    for span_len in args.span_lengths:
        if span_len <= 0 or span_len > args.selected_blocks:
            raise ValueError(f"invalid span length {span_len}")


def measure_operation(
    torch: Any,
    stream: Any,
    operation: Callable[[], None],
    *,
    warmup: int,
    iters: int,
) -> tuple[dict[str, float], dict[str, float], list[float], list[float]]:
    for _ in range(warmup):
        with torch.npu.stream(stream):
            operation()
        # Some custom-op adapters defer their handler through OpCommand.  A
        # stream event alone can therefore measure only submission if the
        # handler does not retain the Python-side stream context.  Use a
        # device-wide synchronization for the primary wall-clock measurement.
        torch.npu.synchronize()

    wall_samples = []
    event_samples = []
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    for _ in range(iters):
        torch.npu.synchronize()
        wall_start = time.perf_counter()
        with torch.npu.stream(stream):
            start_event.record(stream)
            operation()
            end_event.record(stream)
        torch.npu.synchronize()
        wall_samples.append((time.perf_counter() - wall_start) * 1000.0)
        event_samples.append(start_event.elapsed_time(end_event))
    return (
        summarize_ms(wall_samples),
        summarize_ms(event_samples),
        wall_samples,
        event_samples,
    )


def validate_output(torch: Any, source: Any, out: Any, case: Case, backend: str) -> None:
    cpu_ids = torch.tensor([cpu for cpu, _ in case.block_pairs], dtype=torch.long)
    npu_ids_cpu = torch.tensor([npu for _, npu in case.block_pairs], dtype=torch.long)
    expected = source.index_select(1, cpu_ids)
    actual = out.detach().cpu().index_select(1, npu_ids_cpu)
    if not torch.equal(actual, expected):
        max_diff = (actual.float() - expected.float()).abs().max().item()
        raise AssertionError(
            f"{backend} validation failed for block_bytes={case.block_bytes}, "
            f"span_len={case.requested_span_len}, max_diff={max_diff}"
        )


def run_case(
    torch: Any,
    gather: Any,
    stream: Any,
    source: Any,
    out: Any,
    case: Case,
    args: argparse.Namespace,
) -> dict[str, Any]:
    src_ids_cpu = torch.tensor([cpu for cpu, _ in case.block_pairs], dtype=torch.int32)
    dst_ids_cpu = torch.tensor([npu for _, npu in case.block_pairs], dtype=torch.int32)
    src_ids = src_ids_cpu.to(args.device)
    dst_ids = dst_ids_cpu.to(args.device)
    torch.npu.synchronize()

    def span_copy() -> None:
        for cpu_start, npu_start, span_len in case.spans:
            cpu_end = cpu_start + span_len
            npu_end = npu_start + span_len
            for part in range(args.parts):
                out[part, npu_start:npu_end].copy_(
                    source[part, cpu_start:cpu_end],
                    non_blocking=True,
                )

    def mapped_gather() -> None:
        for part in range(args.parts):
            gather(src_ids, source[part], dst_ids, out[part])

    measurements = {}
    operations = {
        "span": span_copy,
        "mapped": mapped_gather,
    }
    backend_order = ("span", "mapped") if args.backend_order == "span-first" else ("mapped", "span")
    for backend in backend_order:
        with torch.npu.stream(stream):
            out.zero_()
        torch.npu.synchronize()
        measurements[backend] = measure_operation(
            torch,
            stream,
            operations[backend],
            warmup=args.warmup,
            iters=args.iters,
        )
        validate_output(torch, source, out, case, backend)

    span_wall, span_event, span_wall_samples, span_event_samples = measurements["span"]
    gather_wall, gather_event, gather_wall_samples, gather_event_samples = measurements["mapped"]

    logical_bytes = args.parts * case.selected_blocks * case.block_bytes
    span_mean = span_wall["mean_ms"]
    gather_mean = gather_wall["mean_ms"]
    gain_percent = (span_mean - gather_mean) / span_mean * 100.0
    return {
        "block_bytes": case.block_bytes,
        "selected_blocks": case.selected_blocks,
        "requested_span_len": case.requested_span_len,
        "span_count": case.span_count,
        "mean_span_len": case.mean_span_len,
        "max_span_len": case.max_span_len,
        "single_block_span_ratio": case.single_block_span_ratio,
        "parts": args.parts,
        "logical_bytes": logical_bytes,
        "span_copy_ops": case.span_count * args.parts,
        "mapped_gather_ops": args.parts,
        "span_wall_mean_ms": span_mean,
        "span_wall_p50_ms": span_wall["p50_ms"],
        "span_wall_p95_ms": span_wall["p95_ms"],
        "span_event_mean_ms": span_event["mean_ms"],
        "span_wall_samples_ms": span_wall_samples,
        "span_event_samples_ms": span_event_samples,
        "mapped_wall_mean_ms": gather_mean,
        "mapped_wall_p50_ms": gather_wall["p50_ms"],
        "mapped_wall_p95_ms": gather_wall["p95_ms"],
        "mapped_event_mean_ms": gather_event["mean_ms"],
        "mapped_wall_samples_ms": gather_wall_samples,
        "mapped_event_samples_ms": gather_event_samples,
        "span_gbps": logical_bytes / span_mean / 1.0e6,
        "mapped_gbps": logical_bytes / gather_mean / 1.0e6,
        "mapped_gain_percent": gain_percent,
        "mapped_meets_margin": gain_percent >= args.decision_margin_percent,
        "validated": True,
    }


def write_results(output_dir: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with (output_dir / "results.jsonl").open("w") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    if rows:
        with (output_dir / "results.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# Mapped-host Gather vs Direct Span-copy",
        "",
        f"- git_branch: `{manifest['git_branch']}`",
        f"- git_sha: `{manifest['git_sha']}`",
        f"- device: `{manifest['device']}`",
        f"- dtype: `{manifest['dtype']}`",
        f"- decision_margin_percent: `{manifest['decision_margin_percent']}`",
        "",
        "| block B | blocks | span len | spans | span wall ms | mapped wall ms | "
        "span GB/s | mapped GB/s | mapped gain | pass |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in rows:
        lines.append(
            "| {block_bytes} | {selected_blocks} | {requested_span_len} | {span_count} | "
            "{span_wall_mean_ms:.3f} | {mapped_wall_mean_ms:.3f} | {span_gbps:.2f} | "
            "{mapped_gbps:.2f} | {mapped_gain_percent:+.1f}% | {mapped_meets_margin} |".format(**row)
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    (
        torch,
        torch_npu,
        gather,
        register,
        inspect,
        unregister,
        extension_path,
        opapi_path,
    ) = load_runtime(args)
    import vllm

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    element_size = torch.empty((), dtype=dtype).element_size()
    validate_args(args, element_size)

    torch.npu.set_device(args.device)
    # The connector runs on PyTorch's current stream.  Benchmark that path
    # directly rather than introducing a separate stream whose context may not
    # be propagated through an OpCommand custom handler.
    stream = torch.npu.current_stream(device=args.device)
    device = torch.device(args.device)
    device_index = torch.npu.current_device() if device.index is None else device.index
    vllm_repo = args.vllm_repo
    if vllm_repo is None:
        vllm_repo = Path(vllm.__file__).resolve().parents[1]

    backend_state = git_repo_state(args.backend_repo)
    vllm_state = git_repo_state(vllm_repo)

    manifest = {
        "schema_version": "mapped-host-gather-benchmark/v2",
        "benchmark_kind": "python-per-span-microbenchmark",
        "production_backend_equivalent": False,
        "argv": sys.argv,
        "command": shlex.join([sys.executable, *sys.argv]),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", ""),
        "cann": getattr(torch.version, "cann", ""),
        "vllm": getattr(vllm, "__version__", ""),
        "git_branch": backend_state.get("branch", ""),
        "git_sha": backend_state.get("commit", ""),
        "repositories": {
            "vllm_ascend": backend_state,
            "vllm": vllm_state,
        },
        "device": args.device,
        "device_index": device_index,
        "device_name": torch.npu.get_device_name(device_index),
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
        "artifacts": {
            "benchmark_script": file_evidence(Path(__file__)),
            "torch_extension": file_evidence(extension_path),
            "opapi_library": file_evidence(opapi_path),
        },
        "dtype": args.dtype,
        "parts": args.parts,
        "num_cpu_blocks": args.num_cpu_blocks,
        "num_npu_blocks": args.num_npu_blocks,
        "selected_blocks": args.selected_blocks,
        "block_bytes": args.block_bytes,
        "span_lengths": args.span_lengths,
        "warmup": args.warmup,
        "iters": args.iters,
        "backend_order": args.backend_order,
        "seed": args.seed,
        "decision_margin_percent": args.decision_margin_percent,
        "host_allocation": "torch_pinned",
        "mapping_registration": [],
    }

    rows = []
    live_sources = []
    mapping_handles = []
    try:
        for block_bytes in args.block_bytes:
            elements_per_block = block_bytes // element_size
            source = torch.empty(
                (args.parts, args.num_cpu_blocks, elements_per_block),
                dtype=dtype,
                pin_memory=True,
            )
            if not source.is_pinned():
                raise RuntimeError(
                    "benchmark requires a pinned CPU allocation so span copy and "
                    "mapped gather use the same production-shaped host pool"
                )
            for part in range(args.parts):
                # Keep every element non-zero so correctness validation catches
                # truncated copies, not just incorrect block indices.
                source[part].fill_(part + 1)
            source[:, :, 0] = torch.arange(args.num_cpu_blocks, dtype=dtype).remainder_(1024)
            out = torch.empty(
                (args.parts, args.num_npu_blocks, elements_per_block),
                dtype=dtype,
                device=args.device,
            )
            live_sources.append(source)

            for part in range(args.parts):
                started = time.perf_counter()
                handle = int(register(source[part]))
                mapping_handles.append(handle)
                registration = dict(inspect(handle))
                registration["wall_ms"] = (time.perf_counter() - started) * 1000.0
                registration["block_bytes"] = block_bytes
                registration["part"] = part
                manifest["mapping_registration"].append(registration)

            for index, span_len in enumerate(args.span_lengths):
                case = make_case(
                    block_bytes=block_bytes,
                    selected_blocks=args.selected_blocks,
                    requested_span_len=span_len,
                    num_cpu_blocks=args.num_cpu_blocks,
                    num_npu_blocks=args.num_npu_blocks,
                    seed=args.seed + index,
                )
                print(
                    f"block_bytes={block_bytes} span_len={span_len} spans={case.span_count}",
                    flush=True,
                )
                rows.append(run_case(torch, gather, stream, source, out, case, args))

        manifest["correctness"] = {
            "validated_cases": len(rows),
            "all_passed": bool(rows) and all(row["validated"] for row in rows),
        }
        write_results(args.output_dir, manifest, rows)
    finally:
        for handle in reversed(mapping_handles):
            if not unregister(handle):
                raise RuntimeError(f"failed to unregister mapped host pool {handle}")
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
