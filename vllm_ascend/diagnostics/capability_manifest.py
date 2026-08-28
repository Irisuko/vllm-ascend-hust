#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Structured capability/fallback manifest for Ascend Dense runtime paths.

Issue #198 requires every runtime downgrade on Dense hot paths to be reported as
a structured capability state (``enabled / fallback / unavailable /
disabled_by_policy``) instead of an anonymous one-off warning.  This enables:

* per-item A/B quantification (each loss is attributed to one capability),
* a CI gate that fails closed on non-whitelisted fallbacks,
* a whitelist that carries reason / owner / removal condition.

This module is intentionally dependency-free at import time (no torch, no CANN,
no vllm): hot-path modules and NPU-less CI containers can import it safely.
Runtime/version probes are resolved lazily inside :meth:`CapabilityManifest.as_dict`.

Enable with ``VLLM_ASCEND_CAPABILITY_MANIFEST_JSONL=/path/out.jsonl``.  Each
record is appended as one JSON line; the merged structured manifest is kept
current as a sibling ``*.json`` file on every new record and again in
``finalize()`` (also registered via ``atexit``).
"""

from __future__ import annotations

import atexit
import json
import os
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TextIO

SCHEMA_VERSION = 1
PRODUCER = "vllm-ascend-hust"

# Status values mandated by issue #198.
STATUS_ENABLED = "enabled"
STATUS_FALLBACK = "fallback"
STATUS_UNAVAILABLE = "unavailable"
STATUS_DISABLED_BY_POLICY = "disabled_by_policy"
ALL_STATUSES = (
    STATUS_ENABLED,
    STATUS_FALLBACK,
    STATUS_UNAVAILABLE,
    STATUS_DISABLED_BY_POLICY,
)

# Severity used when folding cross-process/rank observations of the same
# capability into the merged summary: a fallback/unavailable seen by any rank
# must never be hidden by a later ``enabled`` from another rank, so the most
# severe status wins (issue #198 review).
_STATUS_SEVERITY = {
    STATUS_ENABLED: 0,
    STATUS_DISABLED_BY_POLICY: 1,
    STATUS_FALLBACK: 2,
    STATUS_UNAVAILABLE: 3,
}


def _status_severity(status: str) -> int:
    """Map a status to its summary-merge severity (higher = worse)."""
    return _STATUS_SEVERITY.get(status, -1)


# Capability keys (component.<capability>) for the six tracked Dense hot paths.
CAP_RUNNER_V2_MODEL_RUNNER = "runner.v2_model_runner"
CAP_SAMPLER_TOP_K_TOP_P = "sampler.npu_apply_top_k_top_p"
CAP_SAMPLER_PENALTY_TRITON = "sampler.penalty_triton"
CAP_FUSION_ADD_RMS_NORM_BIAS = "fusion.npu_add_rms_norm_bias"
CAP_FUSION_QKNORM_ROPE = "fusion.qknorm_rope"
CAP_GRAPH_MODE = "graph.mode"

# Sampler fallback sub-states (issue #198 comment, 2026-08-03): distinguish
# "registered but the underlying ACLNN symbols are missing at runtime" from
# "the op was never registered at all", so one root cause is not shown as two.
SAMPLER_STATE_ENABLED = "enabled"
SAMPLER_STATE_NOT_REGISTERED = "not-registered"
SAMPLER_STATE_RUNTIME_SYMBOL_UNAVAILABLE = "registered-but-runtime-symbol-unavailable"

_MISSING = object()
_ENV_JSONL = "VLLM_ASCEND_CAPABILITY_MANIFEST_JSONL"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


@dataclass(frozen=True)
class CapabilityRecord:
    """One capability observation at a single fallback point."""

    capability: str
    status: str
    reason: str | None = None
    # Optional sub-state refining `status`, e.g. the sampler states above.
    state: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=_now_iso)
    # Producing process identity, so cross-process/rank aggregation (and any
    # lost-state reconstruction) can attribute a record to its writer.
    rank: int = 0
    pid: int = 0


class CapabilityManifest:
    """Thread-safe in-process registry of capability/fallback observations."""

    def __init__(
        self,
        jsonl_path: Path | None = None,
        *,
        rank: int = 0,
        world_size: int = 1,
        pid: int | None = None,
    ):
        self._lock = threading.Lock()
        self._records: dict[str, CapabilityRecord] = {}
        self._jsonl_path = jsonl_path
        self._file: TextIO | None = None
        self._finalized = False
        self.rank = rank
        self.world_size = world_size
        self.pid = os.getpid() if pid is None else pid

    # ------------------------------------------------------------------ API

    @classmethod
    def from_env(cls) -> CapabilityManifest | None:
        """Build a manifest from ``VLLM_ASCEND_CAPABILITY_MANIFEST_JSONL``.

        Returns ``None`` (disabled) when the env var is unset, mirroring the
        attention-path probe contract.
        """
        raw = os.getenv(_ENV_JSONL)
        if not raw:
            return None
        rank = int(os.getenv("RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        return cls(Path(raw), rank=rank, world_size=world_size)

    def record(
        self,
        capability: str,
        status: str,
        *,
        reason: str | None = None,
        state: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Record (or overwrite) the state of one capability."""
        if status not in ALL_STATUSES:
            raise ValueError(f"invalid capability status {status!r}, expected one of {ALL_STATUSES}")
        entry = CapabilityRecord(
            capability=capability,
            status=status,
            reason=reason,
            state=state,
            detail=dict(detail or {}),
            rank=self.rank,
            pid=self.pid,
        )
        with self._lock:
            # Hot-path fallback points re-report the same state on every step;
            # do not spam the JSONL with identical lines.
            prev = self._records.get(capability)
            if prev is not None:
                same = (
                    prev.status,
                    prev.state,
                    prev.reason,
                    prev.detail,
                ) == (
                    entry.status,
                    entry.state,
                    entry.reason,
                    entry.detail,
                )
                if same:
                    return
            self._records[capability] = entry
            appended = self._append_line(entry)
        # Keep the summary current while the run is live: vLLM EngineCore
        # workers are torn down via SIGTERM, so the exit-time finalize() is
        # not guaranteed to run. I/O stays out of the critical section.
        if appended:
            self._write_summary()

    def get(self, capability: str) -> CapabilityRecord | None:
        with self._lock:
            return self._records.get(capability)

    def capabilities(self) -> dict[str, CapabilityRecord]:
        with self._lock:
            return dict(self._records)

    def as_dict(self) -> dict[str, Any]:
        """Full structured manifest including lazily-probed runtime versions.

        This is the in-process view: only records this process observed. The
        sibling ``*.json`` written by :meth:`finalize` additionally folds in
        records appended to the shared JSONL by other processes (vLLM runs its
        workers in a separate EngineCore process).
        """
        with self._lock:
            records = [asdict(r) for r in self._records.values()]
        return self._summary_dict(records)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True)

    def _summary_dict(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "producer": PRODUCER,
            "created_at": _now_iso(),
            "pid": self.pid,
            "hostname": socket.gethostname(),
            "rank": self.rank,
            "world_size": self.world_size,
            "runtime": _runtime_versions(),
            "capabilities": records,
        }

    def _summary_records(self) -> list[dict[str, Any]]:
        """Merge in-memory records with JSONL lines from all processes.

        vLLM workers run in a separate EngineCore process that appends to the
        same JSONL path, so the final summary must reflect the whole process
        group. Each record carries its producing ``rank``/``pid`` (see
        :class:`CapabilityRecord`); when several processes observed the same
        capability, the most severe status wins (worst-status-wins), so a
        fallback on any rank is never hidden by a later ``enabled`` from
        another rank. The raw JSONL keeps every observation for audit.
        """
        with self._lock:
            merged: dict[str, dict[str, Any]] = {}
            for rec in (asdict(r) for r in self._records.values()):
                self._merge_record(merged, rec)
        if self._jsonl_path is not None:
            try:
                with open(self._jsonl_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        # Older JSONL lines may predate rank/pid attribution.
                        rec.setdefault("rank", -1)
                        rec.setdefault("pid", -1)
                        self._merge_record(merged, rec)
            except OSError:
                pass
        return list(merged.values())

    @staticmethod
    def _merge_record(merged: dict[str, dict[str, Any]], rec: dict[str, Any]) -> None:
        """Fold one record into the per-capability summary (worst status wins).

        A record is kept only when it is strictly worse than the current one,
        or ties its severity with a later timestamp (a re-report after a state
        change wins; when timestamps tie, the later JSONL line wins).
        """
        key = rec["capability"]
        current = merged.get(key)
        if current is None:
            merged[key] = rec
            return
        keep_current = _status_severity(rec["status"]) < _status_severity(current["status"]) or (
            rec["status"] == current["status"] and rec.get("ts", "") < current.get("ts", "")
        )
        if not keep_current:
            merged[key] = rec

    def finalize(self) -> None:
        """Flush the full manifest to ``<jsonl>.json`` (best effort)."""
        with self._lock:
            if self._finalized:
                return
            self._finalized = True
            if self._file is not None:
                self._file.flush()
        # Do the summary write outside the lock: _summary_records() re-acquires
        # the (non-reentrant) lock, so holding it here would deadlock. I/O also
        # stays out of the critical section.
        self._write_summary()

    def _write_summary(self) -> None:
        """Atomically publish the merged summary to ``<jsonl>.json``.

        Cross-process safe (issue #198 review): the snapshot is computed from a
        JSONL read serialized by a process-safe lock, then published via a
        unique temp file + atomic ``os.replace``. Concurrent ranks can never
        truncate or interleave the summary, and a reader always sees either the
        previous or the new complete snapshot -- never a partial file.
        """
        if self._jsonl_path is None:
            return
        try:
            with self._process_lock():
                payload = json.dumps(
                    self._summary_dict(self._summary_records()),
                    indent=2,
                    sort_keys=True,
                )
                self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                summary_path = self._jsonl_path.with_name(
                    f"{self._jsonl_path.name}.json"
                )
                tmp_path = self._jsonl_path.with_name(
                    f".{self._jsonl_path.name}.json."
                    f"{self.pid}.{threading.get_ident()}.tmp"
                )
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write(payload)
                os.replace(tmp_path, summary_path)
        except OSError:
            pass

    # ------------------------------------------------------------- internals

    @contextmanager
    def _process_lock(self) -> Iterator[None]:
        """Best-effort inter-process lock serializing JSONL append and reads.

        Uses ``fcntl.flock`` on the sibling ``<jsonl>.lock`` file so records
        from different ranks/processes are appended and snapshots computed
        one-at-a-time. Falls back to a no-op when ``fcntl`` is unavailable.
        """
        if self._jsonl_path is None:
            yield
            return
        try:
            import fcntl  # noqa: PLC0415 - platform-specific, imported lazily

            lock_path = self._jsonl_path.with_name(
                f"{self._jsonl_path.name}.lock"
            )
            lock_path.parent.mkdir(parents=True, exist_ok=True)
        except (ImportError, OSError):
            yield
            return
        try:
            with open(lock_path, "a", encoding="utf-8") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except OSError:
            yield

    def _append_line(self, entry: CapabilityRecord) -> bool:
        """Append one JSON line; return True when a line was written.

        The write is serialized by the process-safe lock so interleaved appends
        from concurrent ranks cannot corrupt the JSONL (issue #198 review).
        """
        if self._jsonl_path is None:
            return False
        if self._file is None:
            try:
                self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                self._file = self._jsonl_path.open("a", encoding="utf-8")
            except OSError:
                self._file = None
                return False
        try:
            with self._process_lock():
                self._file.write(json.dumps(asdict(entry), sort_keys=True) + "\n")
                self._file.flush()
            return True
        except OSError:
            return False


# --------------------------------------------------------------------- proxy

_manifest: Any = _MISSING
_manifest_lock = threading.Lock()


def get_capability_manifest() -> CapabilityManifest | None:
    """Return the process singleton, or ``None`` when disabled.

    Uses a sentinel so the common (disabled) case is a single attribute
    comparison after the first call instead of a per-call lock.
    """
    global _manifest
    if _manifest is _MISSING:
        with _manifest_lock:
            if _manifest is _MISSING:
                manifest = CapabilityManifest.from_env()
                _manifest = manifest if manifest is not None else None
                if manifest is not None:
                    atexit.register(manifest.finalize)
    return _manifest


def record_capability(
    capability: str,
    status: str,
    *,
    reason: str | None = None,
    state: str | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    """No-op helper used by instrumented fallback points.

    Cheap fast path: after the first call the singleton is cached, so hot-path
    instrumentation costs one attribute check when the manifest is disabled.
    """
    manifest = get_capability_manifest()
    if manifest is None:
        return
    manifest.record(capability, status, reason=reason, state=state, detail=detail)


# ------------------------------------------------------------- runtime probes


def _runtime_versions() -> dict[str, Any]:
    """Best-effort version snapshot; never raises on missing packages."""
    info: dict[str, Any] = {}
    try:
        import torch

        info["torch"] = torch.__version__
    except Exception:
        pass
    try:
        import torch_npu

        info["torch_npu"] = torch_npu.__version__
    except Exception:
        pass
    try:
        import vllm

        info["vllm"] = getattr(vllm, "__version__", None)
    except Exception:
        pass
    try:
        import vllm_ascend

        info["vllm_ascend"] = getattr(vllm_ascend, "__version__", None)
        info["vllm_ascend_commit"] = getattr(vllm_ascend, "__commit_id__", None)
        info["vllm_upstream"] = getattr(vllm_ascend, "__upstream_version__", None)
    except Exception:
        pass
    info["python"] = os.getenv("PYTHON_VERSION")
    return info
