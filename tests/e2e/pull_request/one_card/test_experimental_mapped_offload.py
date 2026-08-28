# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch
import torch_npu  # noqa: F401
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

from vllm_ascend.kv_offload.experimental_mapped import MappedOffloadingHandler


def test_mapped_handler_store_then_fragmented_restore():
    torch.npu.set_device(0)
    page_bytes = 4096
    device_pages = torch.zeros((8, page_bytes), dtype=torch.int8, device="npu:0")
    caches = CanonicalKVCaches(
        tensors=[CanonicalKVCacheTensor(device_pages, page_bytes)],
        group_data_refs=[[CanonicalKVCacheRef(0, page_bytes)]],
    )
    handler = MappedOffloadingHandler(caches, block_size_factor=1, num_cpu_blocks=8)
    try:
        pattern = torch.arange(device_pages.numel(), dtype=torch.int64).view_as(device_pages)
        device_pages.copy_(((pattern * 17 + 3) % 251 - 125).to(torch.int8).npu())
        original = device_pages.cpu()

        gpu_ids = np.asarray([0, 3, 6], dtype=np.int64)
        cpu_ids = np.asarray([5, 1, 7], dtype=np.int64)
        gpu_spec = GPULoadStoreSpec(gpu_ids, group_sizes=[3], block_indices=[0])
        cpu_spec = CPULoadStoreSpec(cpu_ids)
        assert handler.transfer_async(1, (gpu_spec, cpu_spec))
        handler.wait({1})

        device_pages.zero_()
        restore_gpu_ids = np.asarray([7, 2, 4], dtype=np.int64)
        restore_spec = GPULoadStoreSpec(
            restore_gpu_ids,
            group_sizes=[3],
            block_indices=[0],
        )
        assert handler.transfer_async(2, (cpu_spec, restore_spec))
        handler.wait({2})

        actual = device_pages.cpu()
        for source_gpu_id, restored_gpu_id in zip(gpu_ids, restore_gpu_ids):
            assert torch.equal(actual[restored_gpu_id], original[source_gpu_id])
    finally:
        handler.shutdown()
