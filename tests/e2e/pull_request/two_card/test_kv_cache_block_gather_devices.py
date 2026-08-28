# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401

from vllm_ascend.custom_op_package import activate_kv_cache_block_gather_runtime


@pytest.mark.parametrize("mismatched_ids", ["src", "dst"])
def test_gather_rejects_block_ids_from_another_npu(mismatched_ids: str):
    if torch.npu.device_count() < 2:
        pytest.skip("requires two NPUs")

    torch.npu.set_device(0)
    activate_kv_cache_block_gather_runtime(torch)
    ops = torch.ops._C_ascend

    src = torch.zeros((2, 128), dtype=torch.int8, pin_memory=True)
    out = torch.zeros((2, 128), dtype=torch.int8, device="npu:0")
    src_device = "npu:1" if mismatched_ids == "src" else "npu:0"
    dst_device = "npu:1" if mismatched_ids == "dst" else "npu:0"
    src_ids = torch.tensor([0], dtype=torch.int32, device=src_device)
    dst_ids = torch.tensor([1], dtype=torch.int32, device=dst_device)

    torch.npu.set_device(0)
    handle = ops.register_kv_cache_block_gather_host_pool(src)
    try:
        with pytest.raises(RuntimeError, match="must be on the same NPU device"):
            ops.kv_cache_block_gather(src_ids, src, dst_ids, out)
    finally:
        torch.npu.set_device(0)
        assert ops.unregister_kv_cache_block_gather_host_pool(handle)
