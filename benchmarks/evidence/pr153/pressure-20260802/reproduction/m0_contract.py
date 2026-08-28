#!/usr/bin/env python3
"""Shared validation helpers for the KV transfer-path M0 evidence contract."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DEVICE_KV_GIB = {8, 16, 24, 32}
WORKLOADS = {
    "random-online",
    "sharegpt-online",
    "prefix-repetition-online",
}
MODES = {"hbm-only", "tiering-disabled", "tiering-native", "tiering-mapped"}
EVIDENCE_LABELS = {"pilot-real-online", "retained-real-online"}
REQUIRED_EVENTS = (
    "preempt",
    "restore_start",
    "restore_done",
    "scheduler_wakeup",
    "admission",
    "first_prefill_or_decode",
)
REQUIRED_DECOMPOSITION = (
    "copy_ms",
    "restore_to_wakeup_ms",
    "wakeup_to_admission_ms",
    "restore_to_admission_ms",
    "admission_to_first_compute_ms",
    "requeue_count",
)
REQUIRED_ARTIFACTS = (
    "raw_requests.jsonl",
    "transfer_events.jsonl",
    "metrics_before.prom",
    "metrics_after.prom",
    "server.log",
)
REQUIRED_REPOSITORIES = ("parent", "core", "backend", "benchmark")


class ContractError(ValueError):
    """Raised when an artifact cannot support the declared claim."""


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read JSON {path}: {error}") from error


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ContractError(f"invalid JSONL {path}:{line_number}: {error}") from error
                if not isinstance(row, dict):
                    raise ContractError(f"JSONL row must be an object: {path}:{line_number}")
                rows.append(row)
    except OSError as error:
        raise ContractError(f"cannot read JSONL {path}: {error}") from error
    return rows


def merge_event_files(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Merge process-local trace files using the host-wide monotonic clock."""
    indexed_rows: list[tuple[int, str, int, dict[str, Any]]] = []
    path_list = sorted(set(paths))
    if not path_list:
        raise ContractError("no process-local transfer event files were found")
    for path in path_list:
        last_timestamp = -1
        for line_number, row in enumerate(load_jsonl(path), 1):
            if row.get("schema_version") != SCHEMA_VERSION:
                raise ContractError(f"{path}:{line_number} schema_version must be {SCHEMA_VERSION}")
            require_nonempty_string(row.get("role"), f"{path}:{line_number}.role")
            pid = row.get("pid")
            if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
                raise ContractError(f"{path}:{line_number}.pid must be positive")
            timestamp = _event_timestamp(row, line_number)
            if timestamp < last_timestamp:
                raise ContractError(f"process-local trace is out of order: {path}")
            last_timestamp = timestamp
            indexed_rows.append((timestamp, str(path), line_number, row))
    indexed_rows.sort(key=lambda item: item[:3])
    return [row for _, _, _, row in indexed_rows]


def require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{field} must be an object")
    return value


def require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field} must be a non-empty string")
    return value


def require_positive_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ContractError(f"{field} must be positive")
    return float(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ContractError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def validate_plan(payload: Mapping[str, Any]) -> None:
    expected_sets = {
        "device_kv_gib": DEVICE_KV_GIB,
        "workloads": WORKLOADS,
        "modes": MODES,
        "required_events": set(REQUIRED_EVENTS),
        "required_decomposition": set(REQUIRED_DECOMPOSITION),
    }
    for field, expected in expected_sets.items():
        value = payload.get(field, [])
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ContractError(f"{field} must be an array")
        missing = sorted(expected - set(value), key=str)
        if missing:
            raise ContractError(f"{field} missing: {missing}")
    if payload.get("repetitions", 0) < 3:
        raise ContractError("repetitions must be at least 3")
    if payload.get("evidence_label") != "planned-real-online":
        raise ContractError("plan must not claim completed real-online evidence")

    parity = require_mapping(payload.get("matched_run_parity"), "matched_run_parity")
    for field in (
        "model",
        "model_revision",
        "request_set_sha256",
        "request_count",
        "request_rate",
        "max_model_len",
        "dtype",
        "chip_count",
    ):
        if parity.get(field) is not True:
            raise ContractError(f"matched_run_parity.{field} must be true")

    stop = require_mapping(payload.get("stop_conditions"), "stop_conditions")
    if stop.get("minimum_attributable_regions_for_primitive") != 2:
        raise ContractError("stop_conditions.minimum_attributable_regions_for_primitive must be 2")
    if stop.get("stop_when_host_device_transfer_dominates") is not True:
        raise ContractError("stop_conditions.stop_when_host_device_transfer_dominates must be true")


def _validate_source(source: Mapping[str, Any]) -> None:
    for name in REQUIRED_REPOSITORIES:
        repository = require_mapping(source.get(name), f"source.{name}")
        sha = require_nonempty_string(repository.get("sha"), f"source.{name}.sha")
        if len(sha) != 40 or any(char not in "0123456789abcdef" for char in sha):
            raise ContractError(f"source.{name}.sha must be a full lowercase Git SHA")
        if repository.get("dirty") is not False:
            raise ContractError(f"source.{name}.dirty must be false")


def _validate_environment(environment: Mapping[str, Any]) -> None:
    for field in (
        "npu_model",
        "physical_device_id",
        "driver_version",
        "cann_version",
        "torch_version",
        "torch_npu_version",
        "model_revision",
    ):
        require_nonempty_string(environment.get(field), f"environment.{field}")
    if environment.get("npu_model") != "910B2":
        raise ContractError("environment.npu_model must be 910B2")
    if environment.get("chip_count") != 1:
        raise ContractError("environment.chip_count must be 1")


def _validate_workload(workload: Mapping[str, Any]) -> None:
    name = workload.get("name")
    if name not in WORKLOADS:
        raise ContractError(f"workload.name must be one of {sorted(WORKLOADS)}")
    require_nonempty_string(workload.get("request_set_sha256"), "workload.request_set_sha256")
    require_positive_number(workload.get("request_count"), "workload.request_count")
    require_positive_number(workload.get("request_rate"), "workload.request_rate")
    require_positive_number(workload.get("max_model_len"), "workload.max_model_len")
    if workload.get("max_model_len") != 32768:
        raise ContractError("workload.max_model_len must be 32768")
    require_nonempty_string(workload.get("model"), "workload.model")
    require_nonempty_string(workload.get("dtype"), "workload.dtype")


def _validate_declared_artifacts(run_dir: Path, payload: Mapping[str, Any]) -> None:
    artifacts = require_mapping(payload.get("artifacts"), "artifacts")
    for name in REQUIRED_ARTIFACTS:
        expected = require_nonempty_string(artifacts.get(name), f"artifacts.{name}")
        target = run_dir / name
        if not target.is_file():
            raise ContractError(f"missing required artifact: {target}")
        actual = sha256_file(target)
        if expected != actual:
            raise ContractError(f"artifact hash mismatch for {name}: expected {expected}, got {actual}")


def validate_manifest(run_dir: Path, payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ContractError(f"schema_version must be {SCHEMA_VERSION}")
    require_nonempty_string(payload.get("run_id"), "run_id")
    if payload.get("evidence_label") not in EVIDENCE_LABELS:
        raise ContractError(f"evidence_label must be one of {sorted(EVIDENCE_LABELS)}")
    if payload.get("mode") not in MODES:
        raise ContractError(f"mode must be one of {sorted(MODES)}")
    if payload.get("device_kv_gib") not in DEVICE_KV_GIB:
        raise ContractError(f"device_kv_gib must be one of {sorted(DEVICE_KV_GIB)}")
    lifecycle = payload.get("lifecycle_index")
    if isinstance(lifecycle, bool) or not isinstance(lifecycle, int) or lifecycle < 1:
        raise ContractError("lifecycle_index must be a positive integer")
    _validate_source(require_mapping(payload.get("source"), "source"))
    _validate_environment(require_mapping(payload.get("environment"), "environment"))
    _validate_workload(require_mapping(payload.get("workload"), "workload"))
    claim = require_mapping(payload.get("claim"), "claim")
    require_nonempty_string(claim.get("expected_observation"), "claim.expected_observation")
    require_nonempty_string(claim.get("falsifier"), "claim.falsifier")
    _validate_declared_artifacts(run_dir, payload)


def _event_timestamp(row: Mapping[str, Any], row_number: int) -> int:
    timestamp = row.get("timestamp_ns")
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
        raise ContractError(f"event row {row_number} timestamp_ns must be non-negative")
    return timestamp


def validate_events(rows: Sequence[Mapping[str, Any]], *, require_recovery: bool) -> dict[str, Any]:
    if not rows:
        raise ContractError("transfer_events.jsonl must not be empty")
    by_request: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    directions: Counter[str] = Counter()
    migrated_bytes: Counter[str] = Counter()
    transfer_starts: dict[tuple[str, str, str], int] = {}
    last_timestamp = -1
    for row_number, row in enumerate(rows, 1):
        request_id = require_nonempty_string(row.get("request_id"), f"event row {row_number}.request_id")
        event = require_nonempty_string(row.get("event"), f"event row {row_number}.event")
        timestamp = _event_timestamp(row, row_number)
        if timestamp < last_timestamp:
            raise ContractError("transfer events must be globally timestamp ordered")
        last_timestamp = timestamp
        by_request[request_id].append(row)
        if event in {"transfer_start", "transfer_done"}:
            direction = row.get("direction")
            if direction not in {"d2h", "h2d"}:
                raise ContractError(f"event row {row_number}.direction must be d2h or h2d")
            job_id = require_nonempty_string(row.get("job_id"), f"event row {row_number}.job_id")
            if event == "transfer_done":
                transfer_key = (request_id, job_id, str(direction))
                started_at = transfer_starts.get(transfer_key)
                if started_at is None:
                    raise ContractError(f"event row {row_number} transfer_done has no matching start")
                if timestamp < started_at:
                    raise ContractError(f"event row {row_number} transfer_done precedes its start")
                num_bytes = row.get("bytes")
                if isinstance(num_bytes, bool) or not isinstance(num_bytes, int) or num_bytes <= 0:
                    raise ContractError(f"event row {row_number}.bytes must be positive")
                directions[direction] += 1
                migrated_bytes[direction] += num_bytes
                del transfer_starts[transfer_key]
            else:
                transfer_key = (request_id, job_id, str(direction))
                if transfer_key in transfer_starts:
                    raise ContractError(f"event row {row_number} duplicates an active transfer start")
                transfer_starts[transfer_key] = timestamp

    if transfer_starts:
        raise ContractError("one or more transfer_start events have no matching completion")

    complete_timelines: list[str] = []
    decomposition: dict[str, dict[str, float | int]] = {}
    for request_id, request_rows in by_request.items():
        selected_indices: dict[str, int] | None = None
        for start_index, row in enumerate(request_rows):
            if row["event"] != REQUIRED_EVENTS[0]:
                continue
            candidate = {REQUIRED_EVENTS[0]: start_index}
            cursor = start_index
            for expected_event in REQUIRED_EVENTS[1:]:
                match = None
                for index in range(cursor + 1, len(request_rows)):
                    event = request_rows[index]["event"]
                    if event == REQUIRED_EVENTS[0]:
                        break
                    if event == expected_event:
                        match = index
                        break
                if match is None:
                    break
                candidate[expected_event] = match
                cursor = match
            if len(candidate) == len(REQUIRED_EVENTS):
                selected_indices = candidate
                break
        if selected_indices is None:
            continue
        selected_timestamps = {
            event: int(request_rows[index]["timestamp_ns"]) for event, index in selected_indices.items()
        }
        window_start = selected_indices[REQUIRED_EVENTS[0]]
        window_end = selected_indices[REQUIRED_EVENTS[-1]]
        complete_timelines.append(request_id)
        decomposition[request_id] = {
            "copy_ms": (selected_timestamps["restore_done"] - selected_timestamps["restore_start"]) / 1_000_000,
            "restore_to_wakeup_ms": (selected_timestamps["scheduler_wakeup"] - selected_timestamps["restore_done"])
            / 1_000_000,
            "wakeup_to_admission_ms": (selected_timestamps["admission"] - selected_timestamps["scheduler_wakeup"])
            / 1_000_000,
            "restore_to_admission_ms": (selected_timestamps["admission"] - selected_timestamps["restore_done"])
            / 1_000_000,
            "admission_to_first_compute_ms": (
                selected_timestamps["first_prefill_or_decode"] - selected_timestamps["admission"]
            )
            / 1_000_000,
            "requeue_count": sum(row["event"] == "requeue" for row in request_rows[window_start : window_end + 1]),
        }

    if require_recovery and not complete_timelines:
        raise ContractError("no request has a complete preempt-to-first-compute timeline")
    if require_recovery:
        for direction in ("d2h", "h2d"):
            if directions[direction] == 0 or migrated_bytes[direction] == 0:
                raise ContractError(f"no completed {direction} transfer was recorded")
    return {
        "event_request_ids": sorted(by_request),
        "complete_timeline_request_ids": sorted(complete_timelines),
        "decomposition": decomposition,
        "transfer_counts": dict(directions),
        "migrated_bytes": dict(migrated_bytes),
    }


def validate_requests(rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> dict[str, int]:
    if not rows:
        raise ContractError("raw_requests.jsonl must not be empty")
    request_ids: set[str] = set()
    completed = 0
    failed = 0
    for row_number, row in enumerate(rows, 1):
        request_id = require_nonempty_string(row.get("request_id"), f"request row {row_number}.request_id")
        if request_id in request_ids:
            raise ContractError(f"duplicate request_id in raw requests: {request_id}")
        request_ids.add(request_id)
        status = row.get("status")
        if status == "completed":
            completed += 1
            for field in ("ttft_ms", "tpot_ms", "latency_ms"):
                require_positive_number(row.get(field), f"request row {row_number}.{field}")
        elif status == "failed":
            failed += 1
        else:
            raise ContractError(f"request row {row_number}.status must be completed or failed")
    expected_count = int(require_mapping(manifest["workload"], "workload")["request_count"])
    if len(rows) != expected_count:
        raise ContractError(f"raw request count mismatch: expected {expected_count}, got {len(rows)}")
    if completed == 0:
        raise ContractError("no request completed")
    return {"total": len(rows), "completed": completed, "failed": failed}


def validate_run(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    manifest = require_mapping(load_json(manifest_path), "manifest")
    validate_manifest(run_dir, manifest)
    requests = load_jsonl(run_dir / "raw_requests.jsonl")
    events = load_jsonl(run_dir / "transfer_events.jsonl")
    request_result = validate_requests(requests, manifest)
    event_result = validate_events(
        events,
        require_recovery=manifest["mode"] in {"tiering-native", "tiering-mapped"},
    )
    raw_request_ids = {str(row["request_id"]) for row in requests}
    unknown_event_ids = set(event_result["event_request_ids"]) - raw_request_ids
    if unknown_event_ids:
        raise ContractError(
            f"transfer events contain request IDs absent from raw requests: {sorted(unknown_event_ids)}"
        )
    return {
        "run_id": manifest["run_id"],
        "evidence_label": manifest["evidence_label"],
        "mode": manifest["mode"],
        "device_kv_gib": manifest["device_kv_gib"],
        "requests": request_result,
        "events": event_result,
    }


def _parity_fingerprint(manifest: Mapping[str, Any]) -> tuple[Any, ...]:
    workload = require_mapping(manifest["workload"], "workload")
    environment = require_mapping(manifest["environment"], "environment")
    return (
        manifest["device_kv_gib"],
        workload["name"],
        workload["request_set_sha256"],
        workload["request_count"],
        workload["request_rate"],
        workload["model"],
        workload["max_model_len"],
        workload["dtype"],
        environment["npu_model"],
        environment["chip_count"],
        environment["model_revision"],
    )


def _request_metric(run_dir: Path, field: str) -> float:
    rows = load_jsonl(run_dir / "raw_requests.jsonl")
    values = [float(row[field]) for row in rows if row.get("status") == "completed"]
    if not values:
        raise ContractError(f"{run_dir} has no completed values for {field}")
    return statistics.median(values)


def validate_campaign(run_dirs: Iterable[Path]) -> dict[str, Any]:
    runs: list[tuple[Path, Mapping[str, Any]]] = []
    for run_dir in run_dirs:
        validate_run(run_dir)
        manifest = require_mapping(load_json(run_dir / "manifest.json"), "manifest")
        if manifest["evidence_label"] != "retained-real-online":
            raise ContractError(f"campaign run is not retained evidence: {run_dir}")
        runs.append((run_dir, manifest))
    if not runs:
        raise ContractError("campaign must contain run directories")

    fingerprints = {_parity_fingerprint(manifest) for _, manifest in runs}
    if len(fingerprints) != 1:
        raise ContractError("campaign runs do not have matched workload/environment parity")

    by_mode: dict[str, list[tuple[Path, Mapping[str, Any]]]] = defaultdict(list)
    for entry in runs:
        by_mode[str(entry[1]["mode"])].append(entry)
    if "tiering-native" not in by_mode or "tiering-mapped" not in by_mode:
        raise ContractError("campaign requires tiering-native and tiering-mapped modes")
    for mode in ("tiering-native", "tiering-mapped"):
        entries = by_mode[mode]
        indices = {int(manifest["lifecycle_index"]) for _, manifest in entries}
        if len(entries) < 3 or len(indices) < 3:
            raise ContractError(f"{mode} requires at least 3 independent lifecycles")

    order = [str(manifest["mode"]) for _, manifest in runs]
    compared = [mode for mode in order if mode in {"tiering-native", "tiering-mapped"}]
    if any(left == right for left, right in zip(compared, compared[1:])):
        raise ContractError("native/mapped lifecycles must alternate execution order")

    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for mode in ("tiering-native", "tiering-mapped"):
        metric_values: dict[str, list[float]] = {}
        for field in ("ttft_ms", "tpot_ms", "latency_ms"):
            metric_values[field] = [_request_metric(run_dir, field) for run_dir, _ in by_mode[mode]]
        metrics[mode] = {
            field: {
                "median": statistics.median(values),
                "iqr": statistics.quantiles(values, n=4, method="inclusive")[2]
                - statistics.quantiles(values, n=4, method="inclusive")[0],
            }
            for field, values in metric_values.items()
        }
    return {"run_count": len(runs), "order": order, "metrics": metrics}
