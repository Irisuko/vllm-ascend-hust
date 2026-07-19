# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import atexit
import json
import os
import re
import threading
import time
from collections import Counter
from contextlib import suppress
from pathlib import Path
from typing import Any, TextIO

from vllm.logger import init_logger

from vllm_ascend import envs

logger = init_logger(__name__)

SCHEMA_VERSION = 1
EVENT_SEMANTICS = "python_dispatch_or_capture_event_not_graph_replay_execution"
_DEFAULT_EVERY_N_DISPATCHES = 64
_DEFAULT_MAX_RECORDS = 2048
_DEFAULT_MAX_BYTES = 1024 * 1024
_DEFAULT_BUFFER_RECORDS = 256
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_SUMMARY_LAYER_ID = "__summary__"
_SUMMARY_OPERATOR_ID = "__attention_dispatch__"


def classify_dispatch_coverage(*, capturing: bool, is_c8: bool, pooling: bool) -> str:
    """Describe what a Python-side dispatch event can prove."""
    if pooling:
        return "unsupported_pooling"
    if is_c8:
        return "unsupported_c8_capture" if capturing else "unsupported_c8_eager"
    if capturing:
        return "capture_dispatch_only_no_replay"
    return "eager_dispatch"


def _read_int_env(name: str, default: int, *, minimum: int) -> int:
    try:
        return max(minimum, int(getattr(envs, name)))
    except (TypeError, ValueError):
        logger.warning("Invalid %s; using %d", name, default)
        return default


def _rank_context() -> tuple[int, int] | None:
    raw_world_size = os.getenv("WORLD_SIZE", "1")
    try:
        world_size = int(raw_world_size)
    except ValueError:
        logger.warning(
            "Invalid WORLD_SIZE=%r; disabling attention-path probe",
            raw_world_size,
        )
        return None
    if world_size < 1:
        logger.warning("WORLD_SIZE must be positive; disabling attention-path probe")
        return None

    raw_rank = os.getenv("RANK")
    if raw_rank is None:
        if world_size > 1:
            logger.warning(
                "Attention-path probe requires RANK when WORLD_SIZE > 1; disabling to avoid ambiguous ownership"
            )
            return None
        return 0, world_size
    try:
        rank = int(raw_rank)
    except ValueError:
        logger.warning(
            "Invalid RANK=%r; disabling attention-path probe",
            raw_rank,
        )
        return None
    if not 0 <= rank < world_size:
        logger.warning(
            "RANK=%d is outside WORLD_SIZE=%d; disabling attention-path probe",
            rank,
            world_size,
        )
        return None
    return rank, world_size


class AttentionPathProbe:
    """Bounded telemetry for Python attention dispatch/capture events.

    These records prove which Python dispatch branch was entered. A capture
    record does not count later graph replays as physical operator executions.
    """

    def __init__(
        self,
        jsonl_path: Path,
        max_records: int,
        *,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        buffer_records: int = _DEFAULT_BUFFER_RECORDS,
        every_n_dispatches: int = _DEFAULT_EVERY_N_DISPATCHES,
        run_id: str,
        rank: int = 0,
        world_size: int = 1,
        pid: int | None = None,
    ):
        if not _RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError("run_id must be a non-secret telemetry identifier")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("valid world rank topology is required")

        self._max_records = max(0, max_records)
        self._max_bytes = max(0, max_bytes)
        self._buffer_records = max(1, buffer_records)
        self._every_n_dispatches = max(1, every_n_dispatches)
        self._run_id = run_id
        self._rank = rank
        self._world_size = world_size
        self._pid = os.getpid() if pid is None else pid

        suffix = jsonl_path.suffix or ".jsonl"
        self.output_path = jsonl_path.with_name(f"{jsonl_path.stem}.{run_id}.rank-{rank}.pid-{self._pid}{suffix}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO | None = self.output_path.open("x", encoding="utf-8", buffering=64 * 1024)

        self._counts: Counter[str] = Counter()
        self._event_step = 0
        self._accepted_records = 0
        self._accepted_bytes = 0
        self._dropped_cadence = 0
        self._dropped_record_limit = 0
        self._dropped_byte_limit = 0
        self._dropped_summary_only = 0
        self._dropped_io = 0
        self._buffer: list[str] = []
        self._disabled = False
        self._warned = False
        self._closed = False
        self._lock = threading.RLock()
        atexit.register(self.shutdown)

    @classmethod
    def from_env(cls) -> AttentionPathProbe | None:
        jsonl_path = envs.VLLM_ASCEND_ATTENTION_PATH_PROBE_JSONL.strip()
        if not jsonl_path:
            return None
        run_id = envs.VLLM_TELEMETRY_RUN_ID.strip()
        if not _RUN_ID_PATTERN.fullmatch(run_id):
            logger.warning("AttentionPathProbe requires a valid VLLM_TELEMETRY_RUN_ID; disabling probe")
            return None

        rank_context = _rank_context()
        if rank_context is None:
            return None
        rank, world_size = rank_context
        owner_rank = _read_int_env("VLLM_ASCEND_ATTENTION_PATH_PROBE_OWNER_RANK", 0, minimum=0)
        if owner_rank >= world_size:
            logger.warning(
                "Attention-path probe owner rank %d is outside WORLD_SIZE=%d; disabling",
                owner_rank,
                world_size,
            )
            return None
        if rank != owner_rank:
            return None

        try:
            probe = cls(
                Path(jsonl_path),
                max_records=_read_int_env(
                    "VLLM_ASCEND_ATTENTION_PATH_PROBE_MAX_RECORDS",
                    _DEFAULT_MAX_RECORDS,
                    minimum=0,
                ),
                max_bytes=_read_int_env(
                    "VLLM_ASCEND_ATTENTION_PATH_PROBE_MAX_BYTES",
                    _DEFAULT_MAX_BYTES,
                    minimum=0,
                ),
                buffer_records=_read_int_env(
                    "VLLM_ASCEND_ATTENTION_PATH_PROBE_BUFFER_RECORDS",
                    _DEFAULT_BUFFER_RECORDS,
                    minimum=1,
                ),
                every_n_dispatches=_read_int_env(
                    "VLLM_ASCEND_ATTENTION_PATH_PROBE_EVERY",
                    _DEFAULT_EVERY_N_DISPATCHES,
                    minimum=1,
                ),
                run_id=run_id,
                rank=rank,
                world_size=world_size,
            )
        except (OSError, ValueError):
            logger.warning(
                "Could not initialize attention-path probe at %s; disabling probe",
                jsonl_path,
                exc_info=True,
            )
            return None
        logger.info(
            "AttentionPathProbe enabled for Python dispatch events: "
            "every=%d max_records=%d max_bytes=%d owner_rank=%d world_size=%d",
            probe._every_n_dispatches,
            probe._max_records,
            probe._max_bytes,
            rank,
            world_size,
        )
        return probe

    @property
    def drop_counts(self) -> dict[str, int]:
        return {
            "cadence": self._dropped_cadence,
            "record_limit": self._dropped_record_limit,
            "byte_limit": self._dropped_byte_limit,
            "summary_only": self._dropped_summary_only,
            "io": self._dropped_io,
        }

    def _common_fields(
        self,
        *,
        layer_id: str,
        operator_id: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self._run_id,
            "rank": self._rank,
            "world_size": self._world_size,
            "pid": self._pid,
            "timestamp_ns": time.time_ns(),
            "monotonic_step": self._event_step,
            "layer_id": layer_id,
            "operator_id": operator_id,
        }

    def _disable_after_error(self) -> None:
        self._dropped_io += 1
        if not self._warned:
            logger.warning(
                "Attention-path probe write failed; disabling probe",
                exc_info=True,
            )
            self._warned = True
        self._disabled = True
        self._buffer.clear()
        if self._file is not None:
            with suppress(OSError):
                self._file.close()
            self._file = None

    def _flush_buffer_locked(self) -> None:
        if not self._buffer or self._disabled:
            return
        try:
            assert self._file is not None
            self._file.writelines(self._buffer)
            self._file.flush()
            self._buffer.clear()
        except (OSError, ValueError):
            self._disable_after_error()

    def record_dispatch(
        self,
        *,
        operator_id: str,
        layer_id: str,
        coverage: str,
        query: Any,
        attn_metadata: Any,
        sliding_window: int | None,
    ) -> None:
        if self._disabled or self._closed:
            return
        with self._lock:
            if self._disabled or self._closed:
                return
            self._event_step += 1
            if self._event_step % self._every_n_dispatches != 0:
                self._dropped_cadence += 1
                return

            try:
                attn_state = attn_metadata.attn_state.name
                seq_lens = attn_metadata.seq_lens_list or []
                self._counts[f"{coverage}:{operator_id}:{attn_state}"] += 1
                if self._max_records == 0 or self._max_bytes == 0:
                    self._dropped_summary_only += 1
                    return
                if self._accepted_records >= self._max_records:
                    self._dropped_record_limit += 1
                    return
                record = {
                    **self._common_fields(
                        layer_id=layer_id,
                        operator_id=operator_id,
                    ),
                    "record_type": "attention_dispatch",
                    "event_semantics": EVENT_SEMANTICS,
                    "coverage": coverage,
                    "attn_state": attn_state,
                    "query_tokens": int(query.shape[0]),
                    "num_actual_tokens": int(attn_metadata.num_actual_tokens or 0),
                    "num_decode_tokens": int(attn_metadata.num_decode_tokens or 0),
                    "num_prefills": int(attn_metadata.num_prefills or 0),
                    "num_decodes": int(attn_metadata.num_decodes or 0),
                    "seq_count": len(seq_lens),
                    "seq_lens_head": [int(value) for value in seq_lens[:8]],
                    "sliding_window": sliding_window,
                }
                line = json.dumps(record, sort_keys=True) + "\n"
                line_bytes = len(line.encode())
                if self._accepted_bytes + line_bytes > self._max_bytes:
                    self._dropped_byte_limit += 1
                    return
                self._buffer.append(line)
                self._accepted_records += 1
                self._accepted_bytes += line_bytes
                if len(self._buffer) >= self._buffer_records:
                    self._flush_buffer_locked()
            except (OSError, TypeError, ValueError):
                self._disable_after_error()

    def flush(self) -> None:
        with self._lock:
            self._flush_buffer_locked()

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            atexit.unregister(self.shutdown)
            if self._file is None:
                return
            try:
                summary = {
                    **self._common_fields(
                        layer_id=_SUMMARY_LAYER_ID,
                        operator_id=_SUMMARY_OPERATOR_ID,
                    ),
                    "record_type": "summary",
                    "event_semantics": EVENT_SEMANTICS,
                    "observed_dispatches": self._event_step,
                    "records_written": self._accepted_records,
                    "record_bytes": self._accepted_bytes,
                    "dropped": self.drop_counts,
                    "dispatch_counts": dict(sorted(self._counts.items())),
                }
                self._buffer.append(json.dumps(summary, sort_keys=True) + "\n")
                self._flush_buffer_locked()
                if self._file is not None:
                    self._file.close()
                    self._file = None
            except (OSError, TypeError, ValueError):
                self._disable_after_error()


ATTENTION_PATH_PROBE = AttentionPathProbe.from_env()


def shutdown_attention_path_probe() -> None:
    """Flush the process-owned probe from the normal worker shutdown path."""
    if ATTENTION_PATH_PROBE is not None:
        ATTENTION_PATH_PROBE.shutdown()
