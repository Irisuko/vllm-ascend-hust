# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import weakref

import torch
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401

STRESS_CYCLES = 32


def test_worker_local_host_pool_registration_lifecycle():
    """Exercise independent leases and owned-versus-borrowed teardown."""
    torch.npu.set_device(0)
    ops = torch.ops._C_ascend
    pool = torch.empty((16, 4096), dtype=torch.int8, pin_memory=True)

    stats_before = dict(ops.get_kv_cache_block_gather_host_mapping_stats())
    first = ops.register_kv_cache_block_gather_host_pool(pool)
    second = ops.register_kv_cache_block_gather_host_pool(pool)

    assert first > 0
    assert second > 0
    assert first != second
    first_info = dict(ops.inspect_kv_cache_block_gather_host_pool(first))
    assert first_info["known"] == 1
    assert first_info["active"] == 1
    assert first_info["requested_bytes"] == pool.numel() * pool.element_size()
    assert first_info["device_ptr"] != 0
    assert first_info["owner_pid"] == stats_before["registry_pid"]
    assert first_info["device_id"] == 0
    assert first_info["owned"] + first_info["already_mapped"] == 1
    assert ops.is_kv_cache_block_gather_host_mapping_cached(pool[1:])

    # Releasing one lease must not invalidate another owner of the same pool.
    assert ops.unregister_kv_cache_block_gather_host_pool(first)
    assert not ops.unregister_kv_cache_block_gather_host_pool(first)
    assert ops.is_kv_cache_block_gather_host_mapping_cached(pool)
    assert ops.inspect_kv_cache_block_gather_host_pool(second)["active"] == 1

    assert ops.unregister_kv_cache_block_gather_host_pool(second)
    assert not ops.is_kv_cache_block_gather_host_mapping_cached(pool)
    assert ops.inspect_kv_cache_block_gather_host_pool(first)["known"] == 0
    stats_after = dict(ops.get_kv_cache_block_gather_host_mapping_stats())
    assert stats_after["mapping_count"] == stats_before["mapping_count"]
    assert stats_after["explicit_handle_count"] == stats_before["explicit_handle_count"]
    assert stats_after["explicit_handle_storage_count"] == stats_before["explicit_handle_storage_count"]


def test_worker_local_host_pool_registration_is_bounded_under_repeated_cycles():
    torch.npu.set_device(0)
    ops = torch.ops._C_ascend
    pool = torch.empty((16, 4096), dtype=torch.int8, pin_memory=True)
    stats_before = dict(ops.get_kv_cache_block_gather_host_mapping_stats())

    for _ in range(STRESS_CYCLES):
        handle = ops.register_kv_cache_block_gather_host_pool(pool)
        assert ops.unregister_kv_cache_block_gather_host_pool(handle)

    stats_after = dict(ops.get_kv_cache_block_gather_host_mapping_stats())
    assert stats_after["mapping_count"] == stats_before["mapping_count"]
    assert stats_after["mapped_bytes_current"] == stats_before["mapped_bytes_current"]
    assert stats_after["explicit_handle_count"] == stats_before["explicit_handle_count"]
    assert stats_after["explicit_handle_storage_count"] == stats_before["explicit_handle_storage_count"]
    assert stats_after["explicit_register_call_count"] == stats_before["explicit_register_call_count"] + STRESS_CYCLES
    assert (
        stats_after["explicit_unregister_call_count"] == stats_before["explicit_unregister_call_count"] + STRESS_CYCLES
    )


def test_explicit_lease_retains_tensor_until_unregister():
    torch.npu.set_device(0)
    ops = torch.ops._C_ascend
    pool = torch.empty((16, 4096), dtype=torch.int8, pin_memory=True)
    pool_ref = weakref.ref(pool)

    handle = ops.register_kv_cache_block_gather_host_pool(pool)
    del pool
    gc.collect()
    assert pool_ref() is not None
    assert ops.inspect_kv_cache_block_gather_host_pool(handle)["active"] == 1

    assert ops.unregister_kv_cache_block_gather_host_pool(handle)
    gc.collect()
    assert pool_ref() is None
