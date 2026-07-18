# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import atexit
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

from vllm_ascend import envs

logger = init_logger(__name__)

_DEFAULT_MAX_RECORDS = 2048


class AttentionPathProbe:
    """Environment-gated diagnostics for executed attention paths."""

    def __init__(self, jsonl_path: Path, max_records: int):
        self._jsonl_path = jsonl_path
        self._max_records = max(0, max_records)
        self._counts: Counter[str] = Counter()
        self._records = 0
        self._disabled = False
        self._warned = False
        self._summary_written = False
        self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> AttentionPathProbe | None:
        jsonl_path = envs.VLLM_ASCEND_ATTENTION_PATH_PROBE_JSONL.strip()
        if not jsonl_path:
            return None
        try:
            max_records = max(0, envs.VLLM_ASCEND_ATTENTION_PATH_PROBE_MAX_RECORDS)
        except ValueError:
            logger.warning(
                "Invalid VLLM_ASCEND_ATTENTION_PATH_PROBE_MAX_RECORDS; using %d",
                _DEFAULT_MAX_RECORDS,
            )
            max_records = _DEFAULT_MAX_RECORDS
        try:
            probe = cls(Path(jsonl_path), max_records=max_records)
        except OSError:
            logger.warning(
                "Could not initialize attention-path probe at %s; disabling probe",
                jsonl_path,
                exc_info=True,
            )
            return None
        atexit.register(probe.flush_summary)
        return probe

    def _write(self, record: dict[str, Any]) -> None:
        with self._jsonl_path.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(record, sort_keys=True) + "\n")

    def _disable_after_error(self) -> None:
        if not self._warned:
            logger.warning(
                "Attention-path probe write failed; disabling probe",
                exc_info=True,
            )
            self._warned = True
        self._disabled = True

    def record(
        self,
        *,
        path: str,
        query: Any,
        attn_metadata: Any,
        sliding_window: int | None,
        capturing: bool,
    ) -> None:
        if self._disabled:
            return

        attn_state = attn_metadata.attn_state.name
        self._counts[path] += 1
        self._counts[f"{path}:{attn_state}"] += 1
        if self._records >= self._max_records:
            return

        seq_lens = attn_metadata.seq_lens_list or []
        record = {
            "event": "attention_path",
            "ts": time.time(),
            "path": path,
            "attn_state": attn_state,
            "query_tokens": int(query.shape[0]),
            "num_actual_tokens": int(attn_metadata.num_actual_tokens or 0),
            "num_decode_tokens": int(attn_metadata.num_decode_tokens or 0),
            "num_prefills": int(attn_metadata.num_prefills or 0),
            "num_decodes": int(attn_metadata.num_decodes or 0),
            "seq_count": len(seq_lens),
            "seq_lens_head": [int(value) for value in seq_lens[:8]],
            "sliding_window": sliding_window,
            "capturing": bool(capturing),
        }
        try:
            self._write(record)
            self._records += 1
        except OSError:
            self._disable_after_error()

    def flush_summary(self) -> None:
        if self._disabled or self._summary_written or not self._counts:
            return
        try:
            self._write(
                {
                    "event": "summary",
                    "ts": time.time(),
                    "counts": dict(sorted(self._counts.items())),
                }
            )
            self._summary_written = True
        except OSError:
            self._disable_after_error()


ATTENTION_PATH_PROBE = AttentionPathProbe.from_env()
