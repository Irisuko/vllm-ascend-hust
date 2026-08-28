# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401

from vllm_ascend.custom_op_package import activate_kv_cache_block_gather_runtime


@pytest.mark.parametrize(
    "dtype",
    [torch.int8, torch.float16, torch.bfloat16, torch.float32],
)
def test_fragmented_mapped_host_gather_is_byte_exact(dtype: torch.dtype):
    torch.npu.set_device(0)
    activate_kv_cache_block_gather_runtime(torch)
    ops = torch.ops._C_ascend

    # The int8 kernel requires each page to be 32-byte aligned. The same shape
    # gives all dtype specializations a realistic, non-trivial payload.
    src = torch.arange(8 * 128, dtype=torch.float32).reshape(8, 128).to(dtype)
    out = torch.zeros((8, 128), dtype=dtype, device="npu:0")
    src_ids = torch.tensor([6, 1, 7, 2], dtype=torch.int32, device="npu:0")
    dst_ids = torch.tensor([0, 5, 2, 7], dtype=torch.int32, device="npu:0")

    handle = ops.register_kv_cache_block_gather_host_pool(src)
    try:
        ops.kv_cache_block_gather(src_ids, src, dst_ids, out)
        torch.npu.synchronize()
        expected = torch.zeros_like(out.cpu())
        expected[dst_ids.cpu().long()] = src[src_ids.cpu().long()]
        assert torch.equal(out.cpu(), expected)
    finally:
        assert ops.unregister_kv_cache_block_gather_host_pool(handle)


def test_gather_rejects_an_unregistered_host_tensor():
    torch.npu.set_device(0)
    activate_kv_cache_block_gather_runtime(torch)
    src = torch.zeros((2, 128), dtype=torch.int8)
    out = torch.zeros((2, 128), dtype=torch.int8, device="npu:0")
    ids = torch.tensor([0], dtype=torch.int32, device="npu:0")

    with pytest.raises(RuntimeError, match="registration"):
        torch.ops._C_ascend.kv_cache_block_gather(ids, src, ids, out)
