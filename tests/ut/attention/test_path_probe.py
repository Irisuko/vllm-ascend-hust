# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

import json
from types import SimpleNamespace

from vllm_ascend.attention.path_probe import AttentionPathProbe


def _metadata():
    return SimpleNamespace(
        attn_state=SimpleNamespace(name="DecodeOnly"),
        seq_lens_list=[16, 32],
        num_actual_tokens=2,
        num_decode_tokens=2,
        num_prefills=0,
        num_decodes=2,
    )


def _record(probe: AttentionPathProbe, path: str = "paged_attention") -> None:
    probe.record(
        path=path,
        query=SimpleNamespace(shape=(2, 8, 128)),
        attn_metadata=_metadata(),
        sliding_window=None,
        capturing=True,
    )


def test_from_env_is_disabled_without_path(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_ATTENTION_PATH_PROBE_JSONL", raising=False)
    assert AttentionPathProbe.from_env() is None


def test_probe_limits_details_but_preserves_counts(tmp_path):
    output_path = tmp_path / "attention.jsonl"
    probe = AttentionPathProbe(output_path, max_records=1)

    _record(probe)
    _record(probe)
    probe.flush_summary()
    probe.flush_summary()

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["attention_path", "summary"]
    assert rows[0]["path"] == "paged_attention"
    assert rows[0]["query_tokens"] == 2
    assert rows[1]["counts"]["paged_attention"] == 2
    assert rows[1]["counts"]["paged_attention:DecodeOnly"] == 2


def test_write_failure_disables_probe(tmp_path):
    probe = AttentionPathProbe(tmp_path, max_records=1)

    _record(probe)
    _record(probe)

    assert probe._disabled is True
    assert probe._counts["paged_attention"] == 1
