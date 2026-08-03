# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch.distributed as dist

from vllm_ascend.attention.path_probe import (
    EVENT_SEMANTICS,
    AttentionPathProbe,
    classify_dispatch_coverage,
    get_attention_path_probe,
)


def _metadata():
    return SimpleNamespace(
        attn_state=SimpleNamespace(name="DecodeOnly"),
        seq_lens_list=[16, 32],
        num_actual_tokens=2,
        num_decode_tokens=2,
        num_prefills=0,
        num_decodes=2,
    )


def _probe(path: Path, **kwargs) -> AttentionPathProbe:
    return AttentionPathProbe(
        path,
        max_records=kwargs.pop("max_records", 100),
        max_bytes=kwargs.pop("max_bytes", 1024 * 1024),
        buffer_records=kwargs.pop("buffer_records", 100),
        every_n_dispatches=kwargs.pop("every_n_dispatches", 1),
        run_id=kwargs.pop("run_id", "test-run"),
        pid=kwargs.pop("pid", 123),
        **kwargs,
    )


def _record(
    probe: AttentionPathProbe,
    *,
    operator_id: str = "paged_attention",
    coverage: str = "eager_dispatch",
    sliding_window=None,
) -> None:
    probe.record_dispatch(
        operator_id=operator_id,
        layer_id="model.layers.0.self_attn",
        coverage=coverage,
        query=SimpleNamespace(shape=(2, 8, 128)),
        attn_metadata=_metadata(),
        sliding_window=sliding_window,
    )


def _rows(probe: AttentionPathProbe) -> list[dict]:
    probe.shutdown()
    return [json.loads(line) for line in probe.output_path.read_text().splitlines()]


def test_from_env_is_disabled_without_path(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_ATTENTION_PATH_PROBE_JSONL", raising=False)
    assert AttentionPathProbe.from_env() is None


def test_from_env_requires_non_secret_run_id(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_ASCEND_ATTENTION_PATH_PROBE_JSONL", str(tmp_path / "attention"))
    monkeypatch.delenv("VLLM_TELEMETRY_RUN_ID", raising=False)
    assert AttentionPathProbe.from_env() is None

    monkeypatch.setenv("VLLM_TELEMETRY_RUN_ID", "contains a space")
    assert AttentionPathProbe.from_env() is None


def test_from_env_uses_single_owner_and_join_schema(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_ASCEND_ATTENTION_PATH_PROBE_JSONL", str(tmp_path / "attention"))
    monkeypatch.setenv("VLLM_ASCEND_ATTENTION_PATH_PROBE_EVERY", "1")
    monkeypatch.setenv("VLLM_TELEMETRY_RUN_ID", "run-128")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    assert AttentionPathProbe.from_env() is None

    monkeypatch.setenv("RANK", "0")
    probe = AttentionPathProbe.from_env()
    assert probe is not None
    _record(probe)
    dispatch, summary = _rows(probe)

    assert ".run-128.rank-0.pid-" in probe.output_path.name
    for row in (dispatch, summary):
        assert row["schema_version"] == 1
        assert row["run_id"] == "run-128"
        assert row["rank"] == 0
        assert row["world_size"] == 2
        assert isinstance(row["pid"], int)
        assert isinstance(row["timestamp_ns"], int)
        assert isinstance(row["monotonic_step"], int)
        assert isinstance(row["layer_id"], str)
        assert isinstance(row["operator_id"], str)


def test_missing_or_invalid_rank_context_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_ASCEND_ATTENTION_PATH_PROBE_JSONL", str(tmp_path / "attention"))
    monkeypatch.setenv("VLLM_TELEMETRY_RUN_ID", "rank-test")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.delenv("RANK", raising=False)
    assert AttentionPathProbe.from_env() is None

    monkeypatch.setenv("RANK", "2")
    assert AttentionPathProbe.from_env() is None
    monkeypatch.setenv("WORLD_SIZE", "invalid")
    assert AttentionPathProbe.from_env() is None


def test_from_env_prefers_initialized_distributed_rank(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_ASCEND_ATTENTION_PATH_PROBE_JSONL", str(tmp_path / "attention"))
    monkeypatch.setenv("VLLM_TELEMETRY_RUN_ID", "distributed-rank")
    monkeypatch.setenv("VLLM_ASCEND_ATTENTION_PATH_PROBE_EVERY", "1")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 1)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setenv("VLLM_ASCEND_ATTENTION_PATH_PROBE_OWNER_RANK", "1")

    probe = AttentionPathProbe.from_env()
    assert probe is not None
    assert ".rank-1." in probe.output_path.name
    _record(probe)
    dispatch, summary = _rows(probe)
    assert dispatch["rank"] == summary["rank"] == 1
    assert dispatch["world_size"] == summary["world_size"] == 2


def test_probe_initialization_is_lazy(monkeypatch):
    import vllm_ascend.attention.path_probe as path_probe

    sentinel = object()
    monkeypatch.setattr(path_probe, "_ATTENTION_PATH_PROBE", None)
    monkeypatch.setattr(path_probe, "_ATTENTION_PATH_PROBE_INITIALIZED", False)
    monkeypatch.setattr(path_probe.AttentionPathProbe, "from_env", lambda: sentinel)

    assert get_attention_path_probe() is sentinel
    assert get_attention_path_probe() is sentinel


def test_buffered_records_flush_at_threshold_and_shutdown(tmp_path):
    probe = _probe(tmp_path / "attention", buffer_records=2)
    _record(probe)
    assert probe.output_path.read_text() == ""

    _record(probe)
    assert len(probe.output_path.read_text().splitlines()) == 2
    rows = _rows(probe)

    assert rows[0]["record_type"] == "attention_dispatch"
    assert rows[0]["event_semantics"] == EVENT_SEMANTICS
    assert rows[0]["monotonic_step"] == 1
    assert rows[0]["coverage"] == "eager_dispatch"
    assert rows[-1]["record_type"] == "summary"
    assert rows[-1]["records_written"] == 2
    probe.shutdown()


def test_record_and_byte_limits_are_bounded_and_report_drops(tmp_path):
    record_limited = _probe(tmp_path / "record-limited", max_records=1)
    for _ in range(3):
        _record(record_limited)
    rows = _rows(record_limited)
    assert [row["record_type"] for row in rows] == [
        "attention_dispatch",
        "summary",
    ]
    assert rows[-1]["records_written"] == 1
    assert rows[-1]["dropped"]["record_limit"] == 2
    assert rows[-1]["dispatch_counts"]["eager_dispatch:paged_attention:DecodeOnly"] == 3

    byte_limited = _probe(tmp_path / "byte-limited", max_bytes=1)
    _record(byte_limited)
    rows = _rows(byte_limited)
    assert [row["record_type"] for row in rows] == ["summary"]
    assert rows[0]["dropped"]["byte_limit"] == 1


def test_zero_budget_means_summary_only(tmp_path):
    probe = _probe(tmp_path / "summary-only", max_records=0)
    _record(probe)
    _record(probe)
    rows = _rows(probe)

    assert [row["record_type"] for row in rows] == ["summary"]
    assert rows[0]["dropped"]["summary_only"] == 2
    assert rows[0]["dispatch_counts"]["eager_dispatch:paged_attention:DecodeOnly"] == 2


def test_cadence_uses_monotonic_dispatch_steps(tmp_path):
    probe = _probe(tmp_path / "cadence", every_n_dispatches=3)
    for _ in range(7):
        _record(probe)
    rows = _rows(probe)

    samples = [row for row in rows if row["record_type"] == "attention_dispatch"]
    assert [row["monotonic_step"] for row in samples] == [3, 6]
    assert rows[-1]["observed_dispatches"] == 7
    assert rows[-1]["dropped"]["cadence"] == 5


def test_coverage_labels_do_not_claim_graph_replay_or_unsupported_paths():
    assert classify_dispatch_coverage(capturing=False, is_c8=False, pooling=False) == "eager_dispatch"
    assert classify_dispatch_coverage(capturing=True, is_c8=False, pooling=False) == "capture_dispatch_only_no_replay"
    assert classify_dispatch_coverage(capturing=False, is_c8=True, pooling=False) == "unsupported_c8_eager"
    assert classify_dispatch_coverage(capturing=True, is_c8=True, pooling=False) == "unsupported_c8_capture"
    assert classify_dispatch_coverage(capturing=False, is_c8=False, pooling=True) == "unsupported_pooling"


def test_concurrent_recording_has_unique_monotonic_steps(tmp_path):
    probe = _probe(tmp_path / "threaded")
    threads = [threading.Thread(target=_record, args=(probe,)) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    rows = _rows(probe)

    steps = [row["monotonic_step"] for row in rows if row["record_type"] == "attention_dispatch"]
    assert sorted(steps) == list(range(1, 17))


def test_serialization_or_write_failure_disables_probe(tmp_path):
    serialization_probe = _probe(tmp_path / "serialization")
    _record(serialization_probe, sliding_window=object())
    assert serialization_probe._disabled is True
    assert serialization_probe.drop_counts["io"] == 1

    write_probe = _probe(tmp_path / "write", buffer_records=1)
    assert write_probe._file is not None
    write_probe._file.close()
    _record(write_probe)
    assert write_probe._disabled is True
    assert write_probe.drop_counts["io"] == 1


def test_constructor_rejects_ambiguous_provenance(tmp_path):
    with pytest.raises(ValueError, match="run_id"):
        _probe(tmp_path / "bad-run", run_id="bad run")
    with pytest.raises(ValueError, match="topology"):
        _probe(tmp_path / "bad-rank", rank=2, world_size=2)


def test_runtime_integration_records_base_and_c8_and_flushes_on_shutdown():
    root = Path(__file__).resolve().parents[3]
    attention_source = (root / "vllm_ascend/attention/attention_v1.py").read_text()
    worker_source = (root / "vllm_ascend/worker/worker.py").read_text()

    assert attention_source.count("self._record_python_dispatch(") >= 2
    assert 'operator_id="c8_attention_dispatch"' in attention_source
    assert "capture_dispatch_only_no_replay" not in attention_source
    assert "shutdown_attention_path_probe()" in worker_source
