# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

from vllm_ascend.kv_offload import experimental_mapped
from vllm_ascend.kv_offload.npu import (
    CPUOffloadingSpec,
    NPUOffloadingSpec,
    _set_cpu_bytes_from_legacy_num_blocks,
)


def _canonical_caches(page_size_bytes: int = 32) -> CanonicalKVCaches:
    tensor = torch.empty((4, page_size_bytes), dtype=torch.int8)
    return CanonicalKVCaches(
        tensors=[CanonicalKVCacheTensor(tensor, page_size_bytes)],
        group_data_refs=[[CanonicalKVCacheRef(0, page_size_bytes)]],
    )


def _bare_handler() -> experimental_mapped.MappedOffloadingHandler:
    handler = object.__new__(experimental_mapped.MappedOffloadingHandler)
    handler._closed = False
    handler._store_state = experimental_mapped._AsyncState()
    handler._load_state = experimental_mapped._AsyncState()
    handler._states = (handler._store_state, handler._load_state)
    handler._store_descriptor_pool = []
    handler._load_id_pool = []
    handler._mapped_handles = []
    handler._gather_op = object()
    handler._unregister_op = lambda handle: True
    handler.cpu_tensors = [torch.empty((2, 32), dtype=torch.int8)]
    handler.npu_tensors = [torch.empty((2, 32), dtype=torch.int8)]
    handler.kv_cache_groups_data_refs = [[CanonicalKVCacheRef(0, 32)]]
    handler._load_jobs = 0
    handler._load_pages = 0
    handler._load_bytes = 0
    handler._store_jobs = 0
    handler._store_pages = 0
    handler._store_bytes = 0
    return handler


class _FakeEvent:
    def __init__(self, *, record_failures: int = 0, synchronize_failures: int = 0):
        self.record_failures = record_failures
        self.synchronize_failures = synchronize_failures
        self.record_calls = 0
        self.query_calls = 0
        self.synchronize_calls = 0

    def record(self, stream):
        self.record_calls += 1
        if self.record_failures:
            self.record_failures -= 1
            raise RuntimeError("event record failed")

    def query(self):
        self.query_calls += 1
        return True

    def synchronize(self):
        self.synchronize_calls += 1
        if self.synchronize_failures:
            self.synchronize_failures -= 1
            raise RuntimeError("event synchronize failed")

    def elapsed_time(self, other):
        return 0.0


class _FakeStream:
    def __init__(self, *, synchronize_failures: int = 0):
        self.synchronize_failures = synchronize_failures
        self.synchronize_calls = 0

    def synchronize(self):
        self.synchronize_calls += 1
        if self.synchronize_failures:
            self.synchronize_failures -= 1
            raise RuntimeError("stream synchronize failed")


def _fake_transfer(
    *,
    job_id: int = 1,
    stream: _FakeStream | None = None,
    end_event: _FakeEvent | None = None,
) -> experimental_mapped._Transfer:
    return experimental_mapped._Transfer(
        job_id=job_id,
        stream=stream or _FakeStream(),
        start_event=_FakeEvent(),
        end_event=end_event or _FakeEvent(),
        num_bytes=32,
        batch_src=None,
        batch_dst=None,
        batch_sizes=None,
        mapped_id_buffers=None,
    )


def test_mapped_worker_rejects_unsupported_configuration_before_allocating():
    caches = _canonical_caches()

    with pytest.raises(ValueError, match="block_size_factor == 1"):
        experimental_mapped.MappedOffloadingHandler(caches, 2, 1)
    with pytest.raises(ValueError, match="greater than zero"):
        experimental_mapped.MappedOffloadingHandler(caches, 1, 0)


def test_mapped_worker_fails_when_runtime_lifecycle_ops_are_missing(monkeypatch):
    monkeypatch.setattr(
        experimental_mapped,
        "_load_mapped_gather_ops",
        lambda: (None, None, None),
    )

    with pytest.raises(RuntimeError, match="lifecycle ops are unavailable"):
        experimental_mapped.MappedOffloadingHandler(_canonical_caches(), 1, 1)


def test_partial_registration_failure_rolls_back_existing_leases(monkeypatch):
    tensors = [
        CanonicalKVCacheTensor(torch.empty((4, 32), dtype=torch.int8), 32),
        CanonicalKVCacheTensor(torch.empty((4, 32), dtype=torch.int8), 32),
    ]
    caches = CanonicalKVCaches(
        tensors=tensors,
        group_data_refs=[
            [CanonicalKVCacheRef(0, 32)],
            [CanonicalKVCacheRef(1, 32)],
        ],
    )
    register_calls = 0
    unregistered = []

    def register(tensor):
        nonlocal register_calls
        register_calls += 1
        if register_calls == 2:
            raise RuntimeError("second registration failed")
        return 101

    def unregister(handle):
        unregistered.append(handle)
        return True

    monkeypatch.setattr(experimental_mapped, "is_pin_memory_available", lambda: False)
    monkeypatch.setattr(
        experimental_mapped,
        "_load_mapped_gather_ops",
        lambda: (object(), register, unregister),
    )

    with pytest.raises(RuntimeError, match="second registration failed"):
        experimental_mapped.MappedOffloadingHandler(caches, 1, 2)

    assert unregistered == [101]


def test_unregister_failure_preserves_lease_for_retry():
    handler = _bare_handler()
    handler._mapped_handles = [17]
    attempts = 0

    def unregister(handle):
        nonlocal attempts
        attempts += 1
        return attempts > 1

    handler._unregister_op = unregister

    with pytest.raises(RuntimeError, match="failed to unregister 1"):
        handler._close_mapped_pool()
    assert handler._mapped_handles == [17]
    assert handler._unregister_op is unregister

    handler._close_mapped_pool()
    assert attempts == 2
    assert handler._mapped_handles == []
    assert handler._unregister_op is None


def test_mapped_layout_rejects_unaligned_pages():
    page_size_bytes = 31
    npu_tensors = [torch.empty((4, page_size_bytes), dtype=torch.int8)]
    cpu_tensors = [torch.empty((4, page_size_bytes), dtype=torch.int8)]

    with pytest.raises(ValueError, match="32-byte aligned"):
        experimental_mapped._validate_mapped_gather_layout(
            npu_tensors,
            cpu_tensors,
            [[CanonicalKVCacheRef(0, page_size_bytes)]],
        )


def test_mapped_spec_creates_one_worker_for_both_directions(monkeypatch):
    calls = []
    sentinel = object()

    def create_handler(kv_caches, block_size_factor, num_cpu_blocks):
        calls.append((kv_caches, block_size_factor, num_cpu_blocks))
        return sentinel

    monkeypatch.setattr(
        experimental_mapped,
        "MappedOffloadingHandler",
        create_handler,
    )
    spec = object.__new__(experimental_mapped.MappedOffloadingSpec)
    spec.block_size_factor = 1
    spec.num_blocks = 17
    caches = _canonical_caches()

    worker = spec.create_worker(caches)

    assert worker is sentinel
    assert spec.create_worker(caches) is worker
    assert calls == [(caches, 1, 17)]


def test_mapped_handler_dispatches_both_routes():
    handler = object.__new__(experimental_mapped.MappedOffloadingHandler)
    calls = []
    handler._submit_store = lambda job_id, src, dst: calls.append(("store", job_id, src, dst)) or True
    handler._submit_load = lambda job_id, src, dst: calls.append(("load", job_id, src, dst)) or True
    cpu_spec = CPULoadStoreSpec([1])
    gpu_spec = GPULoadStoreSpec([2], group_sizes=[1], block_indices=[0])

    assert handler.transfer_async(10, (gpu_spec, cpu_spec))
    assert handler.transfer_async(11, (cpu_spec, gpu_spec))
    assert [call[0] for call in calls] == ["store", "load"]


def test_load_submission_failure_keeps_inflight_resources_reachable(monkeypatch):
    handler = _bare_handler()
    state = handler._load_state
    stream = _FakeStream()
    start_event = _FakeEvent()
    end_event = _FakeEvent()
    buffers = experimental_mapped._MappedIDBuffers(
        src_host=torch.empty(1, dtype=torch.int32),
        dst_host=torch.empty(1, dtype=torch.int32),
        src_device=torch.empty(1, dtype=torch.int32),
        dst_device=torch.empty(1, dtype=torch.int32),
        capacity=1,
    )
    handler._make_transfer_plan = lambda *args: (
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        [1],
        1,
        32,
    )
    handler._validate_mapped_ids = lambda *args: None
    handler._acquire_mapped_ids = lambda count: buffers
    handler._acquire_async_resources = lambda state, wait_for_current_stream: (
        stream,
        start_event,
        end_event,
    )

    def fail_after_possible_submission(*args):
        raise RuntimeError("gather submission failed")

    handler._run_mapped_gather = fail_after_possible_submission
    monkeypatch.setattr(torch.npu, "stream", lambda stream: nullcontext())

    cpu_spec = CPULoadStoreSpec([0])
    gpu_spec = GPULoadStoreSpec([1], group_sizes=[1], block_indices=[0])
    with pytest.raises(RuntimeError, match="gather submission failed"):
        handler._submit_load(23, cpu_spec, gpu_spec)

    assert len(state.transfers) == 1
    assert state.transfers[0].success is False
    assert state.transfers[0].mapped_id_buffers is buffers
    assert state.transfer_events == {23: end_event}
    assert end_event.record_calls == 1


def test_failed_event_record_and_stream_sync_poison_until_shutdown_retry():
    handler = _bare_handler()
    unregistered = []
    handler._mapped_handles = [31]
    handler._unregister_op = lambda handle: unregistered.append(handle) or True
    state = handler._load_state
    stream = _FakeStream(synchronize_failures=1)
    end_event = _FakeEvent(record_failures=1)
    transfer = _fake_transfer(stream=stream, end_event=end_event)

    handler._preserve_failed_submission(
        state,
        transfer,
        work_may_be_inflight=True,
        completion_recorded=False,
    )

    assert state.poisoned
    assert list(state.transfers) == [transfer]
    assert transfer.needs_stream_sync
    assert handler.get_finished() == []
    assert end_event.query_calls == 0
    with pytest.raises(RuntimeError, match="handler shutdown"):
        handler._acquire_async_resources(state, wait_for_current_stream=False)

    handler.shutdown()
    handler.shutdown()
    assert stream.synchronize_calls == 2
    assert unregistered == [31]
    assert handler._closed


def test_failed_event_record_recycles_only_after_stream_synchronizes():
    handler = _bare_handler()
    state = handler._load_state
    stream = _FakeStream()
    end_event = _FakeEvent(record_failures=1)
    buffers = experimental_mapped._MappedIDBuffers(
        src_host=torch.empty(1, dtype=torch.int32),
        dst_host=torch.empty(1, dtype=torch.int32),
        src_device=torch.empty(1, dtype=torch.int32),
        dst_device=torch.empty(1, dtype=torch.int32),
        capacity=1,
    )
    transfer = _fake_transfer(stream=stream, end_event=end_event)
    transfer.mapped_id_buffers = buffers

    handler._preserve_failed_submission(
        state,
        transfer,
        work_may_be_inflight=True,
        completion_recorded=False,
    )

    assert not state.poisoned
    assert not state.transfers
    assert state.stream_pool == [stream]
    assert buffers in handler._load_id_pool
    assert stream.synchronize_calls == 1


def test_empty_load_completion_failure_still_synchronizes_recorded_start_event(monkeypatch):
    handler = _bare_handler()
    state = handler._load_state
    stream = _FakeStream()
    start_event = _FakeEvent()
    end_event = _FakeEvent(record_failures=2)
    handler._make_transfer_plan = lambda *args: (
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        [0],
        0,
        0,
    )
    handler._acquire_async_resources = lambda state, wait_for_current_stream: (
        stream,
        start_event,
        end_event,
    )
    monkeypatch.setattr(torch.npu, "stream", lambda stream: nullcontext())

    cpu_spec = CPULoadStoreSpec([])
    gpu_spec = GPULoadStoreSpec([], group_sizes=[0], block_indices=[0])
    with pytest.raises(RuntimeError, match="event record failed"):
        handler._submit_load(24, cpu_spec, gpu_spec)

    assert start_event.record_calls == 1
    assert end_event.record_calls == 2
    assert stream.synchronize_calls == 1
    assert not state.transfers
    assert state.stream_pool == [stream]


def test_shutdown_retries_failed_completion_event_synchronization():
    handler = _bare_handler()
    handler._mapped_handles = [41]
    unregistered = []
    handler._unregister_op = lambda handle: unregistered.append(handle) or True
    end_event = _FakeEvent(synchronize_failures=1)
    transfer = _fake_transfer(end_event=end_event)
    handler._load_state.transfers.append(transfer)
    handler._load_state.transfer_events[transfer.job_id] = end_event

    with pytest.raises(RuntimeError, match="event synchronize failed"):
        handler.shutdown()
    assert list(handler._load_state.transfers) == [transfer]
    assert handler._mapped_handles == [41]

    handler.shutdown()
    handler.shutdown()
    assert end_event.synchronize_calls == 2
    assert unregistered == [41]
    assert handler._closed


def test_store_descriptor_plan_covers_each_group_and_tensor(monkeypatch):
    monkeypatch.setattr(experimental_mapped, "is_pin_memory_available", lambda: False)
    handler = object.__new__(experimental_mapped.MappedOffloadingHandler)
    handler._store_descriptor_pool = []
    handler.npu_tensors = [
        torch.empty((4, 32), dtype=torch.int8),
        torch.empty((4, 64), dtype=torch.int8),
    ]
    handler.cpu_tensors = [
        torch.empty((4, 32), dtype=torch.int8),
        torch.empty((4, 64), dtype=torch.int8),
    ]
    handler.kv_cache_groups_data_refs = [
        [CanonicalKVCacheRef(0, 32)],
        [CanonicalKVCacheRef(1, 64)],
    ]

    _, _, _, copy_src, copy_dst, copy_sizes = handler._prepare_store_descriptors(
        np.array([1, 3], dtype=np.int64),
        np.array([2, 0], dtype=np.int64),
        [1, 1],
        2,
    )

    assert copy_src.tolist() == [
        handler.npu_tensors[0].data_ptr() + 32,
        handler.npu_tensors[1].data_ptr() + 3 * 64,
    ]
    assert copy_dst.tolist() == [
        handler.cpu_tensors[0].data_ptr() + 2 * 32,
        handler.cpu_tensors[1].data_ptr(),
    ]
    assert copy_sizes.tolist() == [32, 64]


def test_native_npu_spec_remains_the_default():
    assert CPUOffloadingSpec is NPUOffloadingSpec
    assert NPUOffloadingSpec.create_worker is not experimental_mapped.MappedOffloadingSpec.create_worker


def test_legacy_num_blocks_translation_preserves_explicit_bytes():
    extra_config = {"num_cpu_blocks": 10, "cpu_bytes_to_use": 1234}
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config=extra_config,
        ),
    )

    _set_cpu_bytes_from_legacy_num_blocks(vllm_config, SimpleNamespace())

    assert extra_config["cpu_bytes_to_use"] == 1234
