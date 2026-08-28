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
"""Unit tests for the capability/fallback manifest (issue #198)."""

import json
import multiprocessing as mp
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import vllm_ascend.diagnostics.capability_manifest as cm


def _manifest(tmp_path, name="manifest.jsonl", **kwargs):
    return cm.CapabilityManifest(tmp_path / name, pid=123, **kwargs)


# --------------------------------------------------------------------- schema


def test_status_and_capability_contract():
    assert set(cm.ALL_STATUSES) == {
        cm.STATUS_ENABLED,
        cm.STATUS_FALLBACK,
        cm.STATUS_UNAVAILABLE,
        cm.STATUS_DISABLED_BY_POLICY,
    }
    assert cm.CAP_RUNNER_V2_MODEL_RUNNER == "runner.v2_model_runner"
    assert cm.CAP_SAMPLER_TOP_K_TOP_P == "sampler.npu_apply_top_k_top_p"
    assert cm.CAP_SAMPLER_PENALTY_TRITON == "sampler.penalty_triton"
    assert cm.CAP_FUSION_ADD_RMS_NORM_BIAS == "fusion.npu_add_rms_norm_bias"
    assert cm.CAP_FUSION_QKNORM_ROPE == "fusion.qknorm_rope"
    assert cm.CAP_GRAPH_MODE == "graph.mode"


def test_invalid_status_rejected(tmp_path):
    manifest = _manifest(tmp_path)
    with pytest.raises(ValueError, match="invalid capability status"):
        manifest.record(cm.CAP_GRAPH_MODE, "bogus")


# ------------------------------------------------------------------- recording


def test_record_get_and_overwrite(tmp_path):
    manifest = _manifest(tmp_path)
    manifest.record(cm.CAP_GRAPH_MODE, cm.STATUS_ENABLED, reason="graph", detail={"a": 1})
    record = manifest.get(cm.CAP_GRAPH_MODE)
    assert record.status == cm.STATUS_ENABLED
    assert record.reason == "graph"
    assert record.detail == {"a": 1}

    manifest.record(cm.CAP_GRAPH_MODE, cm.STATUS_DISABLED_BY_POLICY, reason="eager")
    assert manifest.get(cm.CAP_GRAPH_MODE).status == cm.STATUS_DISABLED_BY_POLICY


def test_jsonl_dedupes_identical_records(tmp_path):
    manifest = _manifest(tmp_path)
    manifest.record(
        cm.CAP_SAMPLER_PENALTY_TRITON,
        cm.STATUS_ENABLED,
        detail={"triton_available": True},
    )
    manifest.record(
        cm.CAP_SAMPLER_PENALTY_TRITON,
        cm.STATUS_ENABLED,
        detail={"triton_available": True},
    )
    manifest.record(
        cm.CAP_SAMPLER_PENALTY_TRITON,
        cm.STATUS_FALLBACK,
        detail={"triton_available": False},
    )
    lines = manifest._jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["status"] == cm.STATUS_FALLBACK


def test_finalize_writes_manifest_json_and_is_idempotent(tmp_path):
    manifest = _manifest(tmp_path)
    manifest.record(cm.CAP_GRAPH_MODE, cm.STATUS_DISABLED_BY_POLICY, reason="eager")
    # _runtime_versions() lazily imports torch/torch_npu/vllm (several seconds each
    # on NPU hosts); stub it so this test stays fast and hermetic.
    with patch.object(cm, "_runtime_versions", return_value={"torch": "0.0.0"}):
        manifest.finalize()
        manifest.finalize()

    summary = json.loads((tmp_path / "manifest.jsonl.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == cm.SCHEMA_VERSION
    assert summary["producer"] == cm.PRODUCER
    assert summary["pid"] == 123
    assert summary["hostname"]
    capabilities = {entry["capability"]: entry for entry in summary["capabilities"]}
    assert capabilities[cm.CAP_GRAPH_MODE]["status"] == cm.STATUS_DISABLED_BY_POLICY
    assert isinstance(summary["runtime"], dict)


def test_record_keeps_summary_current_without_finalize(tmp_path):
    """vLLM EngineCore workers are SIGTERM-killed on shutdown, so the atexit
    finalize() may never run; the summary must be refreshed on every new
    record instead of only at process exit."""
    manifest = _manifest(tmp_path)
    with patch.object(cm, "_runtime_versions", return_value={"torch": "0.0.0"}):
        manifest.record(cm.CAP_GRAPH_MODE, cm.STATUS_DISABLED_BY_POLICY, reason="eager")
        # A second new record refreshes (and must not corrupt) the live summary.
        manifest.record(cm.CAP_SAMPLER_PENALTY_TRITON, cm.STATUS_FALLBACK, reason="no triton")

    summary = json.loads((tmp_path / "manifest.jsonl.json").read_text(encoding="utf-8"))
    caps = {entry["capability"]: entry for entry in summary["capabilities"]}
    assert set(caps) == {cm.CAP_GRAPH_MODE, cm.CAP_SAMPLER_PENALTY_TRITON}
    assert caps[cm.CAP_SAMPLER_PENALTY_TRITON]["status"] == cm.STATUS_FALLBACK


def test_finalize_folds_in_records_from_other_processes(tmp_path):
    """vLLM workers run in a separate EngineCore process that appends to the
    shared JSONL; the final summary must include those records too."""
    manifest = _manifest(tmp_path)
    manifest.record(cm.CAP_GRAPH_MODE, cm.STATUS_DISABLED_BY_POLICY, reason="eager")
    # Simulate the worker process writing a capability into the shared JSONL.
    with manifest._jsonl_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "capability": cm.CAP_RUNNER_V2_MODEL_RUNNER,
                    "status": cm.STATUS_FALLBACK,
                    "reason": "worker",
                    "ts": "2026-01-01T00:00:00",
                }
            )
            + "\n"
        )
        f.flush()
    with patch.object(cm, "_runtime_versions", return_value={"torch": "0.0.0"}):
        manifest.finalize()
    summary = json.loads((tmp_path / "manifest.jsonl.json").read_text(encoding="utf-8"))
    caps = {entry["capability"] for entry in summary["capabilities"]}
    assert caps == {cm.CAP_GRAPH_MODE, cm.CAP_RUNNER_V2_MODEL_RUNNER}


# --------------------------------------------------- multi-process aggregation


def test_summary_worst_status_wins_across_ranks(tmp_path):
    """Regression (issue #198 review): if rank 0 falls back and rank 1 later
    succeeds, the merged summary must still report the fallback instead of
    letting the later ``enabled`` line hide it."""
    jsonl = tmp_path / "multi.jsonl"
    rank0 = cm.CapabilityManifest(jsonl, rank=0, world_size=2, pid=1000)
    rank1 = cm.CapabilityManifest(jsonl, rank=1, world_size=2, pid=1001)
    rank0.record(cm.CAP_SAMPLER_PENALTY_TRITON, cm.STATUS_FALLBACK, reason="no triton")
    rank1.record(cm.CAP_SAMPLER_PENALTY_TRITON, cm.STATUS_ENABLED, reason="triton ok")
    with patch.object(cm, "_runtime_versions", return_value={"torch": "0.0.0"}):
        rank1.finalize()
    summary = json.loads(jsonl.with_name(jsonl.name + ".json").read_text(encoding="utf-8"))
    caps = {entry["capability"]: entry for entry in summary["capabilities"]}
    entry = caps[cm.CAP_SAMPLER_PENALTY_TRITON]
    assert entry["status"] == cm.STATUS_FALLBACK  # worst status wins
    assert entry["rank"] == 0  # producer identity retained
    assert entry["pid"] == 1000


def test_summary_worst_status_wins_regardless_of_order(tmp_path):
    """The failing rank may record after the healthy one; enabled must not win."""
    jsonl = tmp_path / "order.jsonl"
    healthy = cm.CapabilityManifest(jsonl, rank=1, world_size=2, pid=1001)
    failing = cm.CapabilityManifest(jsonl, rank=0, world_size=2, pid=1000)
    healthy.record(cm.CAP_SAMPLER_PENALTY_TRITON, cm.STATUS_ENABLED, reason="triton ok")
    failing.record(cm.CAP_SAMPLER_PENALTY_TRITON, cm.STATUS_FALLBACK, reason="no triton")
    with patch.object(cm, "_runtime_versions", return_value={"torch": "0.0.0"}):
        failing.finalize()
    summary = json.loads(jsonl.with_name(jsonl.name + ".json").read_text(encoding="utf-8"))
    caps = {entry["capability"]: entry for entry in summary["capabilities"]}
    entry = caps[cm.CAP_SAMPLER_PENALTY_TRITON]
    assert entry["status"] == cm.STATUS_FALLBACK
    assert entry["rank"] == 0


def test_jsonl_records_carry_rank_and_pid(tmp_path):
    """Each JSONL line must retain producer identity for audit/reconstruction."""
    manifest = cm.CapabilityManifest(tmp_path / "who.jsonl", rank=3, world_size=8, pid=4242)
    with patch.object(cm, "_runtime_versions", return_value={"torch": "0.0.0"}):
        manifest.record(cm.CAP_GRAPH_MODE, cm.STATUS_ENABLED, reason="graph")
    line = json.loads(manifest._jsonl_path.read_text(encoding="utf-8").splitlines()[0])
    assert line["rank"] == 3
    assert line["pid"] == 4242


def test_summary_worst_status_ordering(tmp_path):
    """Severity order: unavailable > fallback > disabled_by_policy > enabled."""
    jsonl = tmp_path / "severity.jsonl"
    manifests = [cm.CapabilityManifest(jsonl, rank=rank, world_size=4, pid=1000 + rank) for rank in range(4)]
    statuses = [
        cm.STATUS_DISABLED_BY_POLICY,
        cm.STATUS_ENABLED,
        cm.STATUS_UNAVAILABLE,
        cm.STATUS_FALLBACK,
    ]
    for manifest, status in zip(manifests, statuses):
        manifest.record(cm.CAP_RUNNER_V2_MODEL_RUNNER, status)
    with patch.object(cm, "_runtime_versions", return_value={"torch": "0.0.0"}):
        manifests[-1].finalize()
    summary = json.loads(jsonl.with_name(jsonl.name + ".json").read_text(encoding="utf-8"))
    caps = {entry["capability"]: entry for entry in summary["capabilities"]}
    entry = caps[cm.CAP_RUNNER_V2_MODEL_RUNNER]
    assert entry["status"] == cm.STATUS_UNAVAILABLE  # worst of the four


def _contention_worker(jsonl_path, rank, workers, fallback_ready, iterations):
    """Worker for the multiprocessing contention test (fork- and spawn-safe).

    Rank 0 is the designated failing rank: it appends its ``fallback`` first,
    then signals the other ranks so every snapshot published afterwards must
    fold in that worst status.
    """
    manifest = cm.CapabilityManifest(
        Path(jsonl_path), rank=rank, world_size=workers, pid=1000 + rank
    )
    if rank == 0:
        manifest.record(cm.CAP_GRAPH_MODE, cm.STATUS_FALLBACK, reason=f"rank{rank}")
        fallback_ready.set()
    else:
        fallback_ready.wait(timeout=60)
    for i in range(iterations):
        manifest.record(
            cm.CAP_GRAPH_MODE, cm.STATUS_ENABLED, reason=f"rank{rank}-{i}"
        )
        manifest.record(
            cm.CAP_RUNNER_V2_MODEL_RUNNER,
            cm.STATUS_ENABLED,
            reason=f"rank{rank}-{i}",
        )
        manifest.finalize()
    manifest.finalize()


def test_summary_multiprocessing_contention_never_loses_worst(tmp_path):
    """Real multiprocessing contention (issue #198 review).

    Concurrent ranks append to the same JSONL and publish snapshots while a
    reader repeatedly parses the live summary. The summary must always be
    valid JSON (atomic publish -- no truncation or interleaving) and, once rank
    0's fallback is in the JSONL, no published snapshot may lose the worst
    status.
    """
    jsonl = tmp_path / "mp.jsonl"
    summary_path = jsonl.with_name(jsonl.name + ".json")
    workers = 4
    iterations = 40
    fallback_ready = mp.Event()

    procs = [
        mp.Process(
            target=_contention_worker,
            args=(str(jsonl), rank, workers, fallback_ready, iterations),
        )
        for rank in range(workers)
    ]
    for proc in procs:
        proc.start()

    observations = 0
    try:
        while any(proc.is_alive() for proc in procs):
            if not summary_path.exists():
                time.sleep(0.001)
                continue
            # Atomic publish guarantees this is always complete, valid JSON.
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            caps = {c["capability"]: c["status"] for c in data["capabilities"]}
            if caps.get(cm.CAP_GRAPH_MODE) is not None:
                assert caps[cm.CAP_GRAPH_MODE] == cm.STATUS_FALLBACK, caps
            observations += 1
    finally:
        for proc in procs:
            proc.join(timeout=60)
    assert not any(proc.is_alive() for proc in procs)

    # The reader must have observed live snapshots while workers were running.
    assert observations > 0
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    caps = {c["capability"]: c for c in data["capabilities"]}
    entry = caps[cm.CAP_GRAPH_MODE]
    assert entry["status"] == cm.STATUS_FALLBACK  # worst status never lost
    assert entry["rank"] == 0
    assert entry["pid"] == 1000


# --------------------------------------------------------------------- from_env


def test_from_env_disabled_without_path(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_CAPABILITY_MANIFEST_JSONL", raising=False)
    assert cm.CapabilityManifest.from_env() is None


def test_from_env_enabled_with_path(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_ASCEND_CAPABILITY_MANIFEST_JSONL", str(tmp_path / "out.jsonl"))
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "4")
    manifest = cm.CapabilityManifest.from_env()
    assert manifest is not None
    assert manifest.rank == 1
    assert manifest.world_size == 4


# ----------------------------------------------------------------- proxy / noop


def test_record_capability_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_CAPABILITY_MANIFEST_JSONL", raising=False)
    monkeypatch.setattr(cm, "_manifest", cm._MISSING)
    cm.record_capability(cm.CAP_GRAPH_MODE, cm.STATUS_ENABLED)
    assert cm.get_capability_manifest() is None


def test_record_capability_forwards_to_manifest(monkeypatch, tmp_path):
    manifest = _manifest(tmp_path)
    monkeypatch.setattr(cm, "_manifest", manifest)
    cm.record_capability(cm.CAP_GRAPH_MODE, cm.STATUS_ENABLED, reason="ok")
    assert manifest.get(cm.CAP_GRAPH_MODE).reason == "ok"


# ------------------------------------------------------------------ integration


def test_runtime_integration_covers_all_six_fallback_points():
    root = Path(__file__).resolve().parents[3]
    expectations = {
        "vllm_ascend/worker/worker.py": [
            cm.CAP_RUNNER_V2_MODEL_RUNNER,
            cm.CAP_GRAPH_MODE,
        ],
        "vllm_ascend/sample/sampler.py": [
            cm.CAP_SAMPLER_TOP_K_TOP_P,
            cm.CAP_SAMPLER_PENALTY_TRITON,
        ],
        "vllm_ascend/utils.py": [cm.CAP_FUSION_ADD_RMS_NORM_BIAS],
        "vllm_ascend/compilation/passes/qknorm_rope_fusion_pass.py": [
            cm.CAP_FUSION_QKNORM_ROPE,
        ],
    }
    # Resolve each capability value (e.g. "runner.v2_model_runner") to the
    # CAP_* constant name actually used in the instrumented source files.
    value_to_name = {value: name for name, value in vars(cm).items() if name.startswith("CAP_")}
    for rel_path, keys in expectations.items():
        source = (root / rel_path).read_text(encoding="utf-8")
        assert "record_capability(" in source, f"no instrumentation in {rel_path}"
        for key in keys:
            assert value_to_name[key] in source, f"missing {key} ({value_to_name[key]}) in {rel_path}"


# ------------------------------------------------- sampler fallback sub-states


def test_apply_top_k_top_p_disabled_by_policy_on_unsupported_device(tmp_path):
    from vllm_ascend.sample import sampler as sampler_module

    manifest = _manifest(tmp_path)
    ascend_config = SimpleNamespace(enable_reduce_sample=False)
    with (
        patch.object(
            sampler_module,
            "get_ascend_device_type",
            return_value=sampler_module.AscendDeviceType._310P,
        ),
        patch.object(
            sampler_module,
            "get_ascend_config",
            return_value=ascend_config,
        ),
        patch.object(
            sampler_module,
            "record_capability",
            side_effect=manifest.record,
        ),
    ):
        sampler_module.apply_top_k_top_p(
            torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
            torch.tensor([2], dtype=torch.int32),
            None,
        )
    record = manifest.get(cm.CAP_SAMPLER_TOP_K_TOP_P)
    assert record is not None
    assert record.status == cm.STATUS_DISABLED_BY_POLICY


def test_apply_top_k_top_p_not_registered(tmp_path):
    from vllm_ascend.sample import sampler as sampler_module

    manifest = _manifest(tmp_path)
    ascend_config = SimpleNamespace(enable_reduce_sample=False)
    with (
        patch.object(
            sampler_module,
            "get_ascend_device_type",
            return_value=sampler_module.AscendDeviceType.A2,
        ),
        patch.object(
            sampler_module,
            "get_ascend_config",
            return_value=ascend_config,
        ),
        patch.object(
            sampler_module.torch.ops,
            "_C_ascend",
            SimpleNamespace(),
            create=True,
        ),
        patch.object(
            sampler_module,
            "record_capability",
            side_effect=manifest.record,
        ),
    ):
        sampler_module._MISSING_TOP_K_TOP_P_OP_WARNED = False
        sampler_module._DISABLE_TOP_K_TOP_P_CUSTOM_OP = False
        sampler_module.apply_top_k_top_p(
            torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
            torch.tensor([2], dtype=torch.int32),
            None,
        )
    record = manifest.get(cm.CAP_SAMPLER_TOP_K_TOP_P)
    assert record.status == cm.STATUS_FALLBACK
    assert record.state == cm.SAMPLER_STATE_NOT_REGISTERED


def test_apply_top_k_top_p_runtime_symbol_unavailable_not_overwritten(tmp_path):
    """Regression: a broken-but-registered op must keep the
    RUNTIME_SYMBOL_UNAVAILABLE state and not be overwritten by NOT_REGISTERED."""

    from vllm_ascend.sample import sampler as sampler_module

    def broken_custom_op(*args, **kwargs):
        raise RuntimeError("aclnnApplyTopKTopPCustom or aclnnApplyTopKTopPCustomGetWorkspaceSize not in libopapi.so")

    manifest = _manifest(tmp_path)
    ascend_config = SimpleNamespace(enable_reduce_sample=False)
    with (
        patch.object(
            sampler_module,
            "get_ascend_device_type",
            return_value=sampler_module.AscendDeviceType.A2,
        ),
        patch.object(
            sampler_module,
            "get_ascend_config",
            return_value=ascend_config,
        ),
        patch.object(
            sampler_module.torch.ops,
            "_C_ascend",
            SimpleNamespace(npu_apply_top_k_top_p=broken_custom_op),
            create=True,
        ),
        patch.object(
            sampler_module,
            "record_capability",
            side_effect=manifest.record,
        ),
    ):
        sampler_module._MISSING_TOP_K_TOP_P_OP_WARNED = False
        sampler_module._BROKEN_TOP_K_TOP_P_OP_WARNED = False
        sampler_module._DISABLE_TOP_K_TOP_P_CUSTOM_OP = False
        sampler_module.apply_top_k_top_p(
            torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
            torch.tensor([2], dtype=torch.int32),
            None,
        )

    record = manifest.get(cm.CAP_SAMPLER_TOP_K_TOP_P)
    assert record is not None
    assert record.status == cm.STATUS_FALLBACK
    assert record.state == cm.SAMPLER_STATE_RUNTIME_SYMBOL_UNAVAILABLE
    assert "aclnnApplyTopKTopPCustom" in record.detail["error"]


def test_apply_penalties_triton_fallback(tmp_path):
    from vllm_ascend.sample import sampler as sampler_module

    manifest = _manifest(tmp_path)
    logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    with (
        patch.object(
            sampler_module,
            "HAS_TRITON",
            False,
        ),
        patch.object(
            sampler_module.Sampler,
            "apply_penalties",
            return_value=logits,
        ),
        patch.object(
            sampler_module,
            "record_capability",
            side_effect=manifest.record,
        ),
    ):
        sampler_module.AscendSampler.apply_penalties(logits, SimpleNamespace(), [])

    record = manifest.get(cm.CAP_SAMPLER_PENALTY_TRITON)
    assert record is not None
    assert record.status == cm.STATUS_FALLBACK


def test_apply_penalties_triton_enabled(tmp_path):
    from vllm_ascend.sample import sampler as sampler_module

    manifest = _manifest(tmp_path)
    logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    metadata = SimpleNamespace(no_penalties=True)
    with (
        patch.object(
            sampler_module,
            "HAS_TRITON",
            True,
        ),
        patch.object(
            sampler_module,
            "record_capability",
            side_effect=manifest.record,
        ),
    ):
        out = sampler_module.AscendSampler.apply_penalties(logits, metadata, [])
        assert torch.equal(out, logits)

    record = manifest.get(cm.CAP_SAMPLER_PENALTY_TRITON)
    assert record is not None
    assert record.status == cm.STATUS_ENABLED
