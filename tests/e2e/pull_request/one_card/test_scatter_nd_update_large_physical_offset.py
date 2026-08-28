# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise scatter element addresses on both sides of INT32_MAX.

The production block stride makes this a roughly 4.01 GiB BF16 allocation. The
case initializes and reads only a few rows; it is a correctness gate rather
than a memory-bandwidth benchmark.
"""

import torch
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401

_INT32_MAX = (1 << 31) - 1
_PRODUCTION_BLOCK_STRIDE = 1_616_512
_WIDTH = 512
_LAST_INT32_BLOCK = _INT32_MAX // _PRODUCTION_BLOCK_STRIDE
_FIRST_INT64_OFFSET_BLOCK = _LAST_INT32_BLOCK + 1
_SENTINEL = -7.0


def _boundary_cache() -> tuple[torch.Tensor, torch.Tensor]:
    backing_elements = _FIRST_INT64_OFFSET_BLOCK * _PRODUCTION_BLOCK_STRIDE + _WIDTH
    backing = torch.empty(
        (backing_elements,),
        dtype=torch.bfloat16,
        device="npu:0",
    )
    cache = backing.as_strided(
        size=(_FIRST_INT64_OFFSET_BLOCK + 1, 1, 1, _WIDTH),
        stride=(_PRODUCTION_BLOCK_STRIDE, _WIDTH, _WIDTH, 1),
    )
    return backing, cache


def _assert_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0, rtol=0)


def test_scatter_crosses_int32_physical_offset_under_replay() -> None:
    torch.npu.set_device(0)
    assert _LAST_INT32_BLOCK * _PRODUCTION_BLOCK_STRIDE <= _INT32_MAX
    assert _FIRST_INT64_OFFSET_BLOCK * _PRODUCTION_BLOCK_STRIDE > _INT32_MAX

    backing, cache = _boundary_cache()
    guarded_blocks = (0, 1, _LAST_INT32_BLOCK, _FIRST_INT64_OFFSET_BLOCK)
    for block in guarded_blocks:
        cache[block, 0].fill_(_SENTINEL)

    indices = torch.tensor(
        [[_LAST_INT32_BLOCK, 0], [0, 0]],
        dtype=torch.int32,
        device="npu:0",
    )
    updates = (
        torch.arange(2 * _WIDTH, dtype=torch.float32, device="npu:0")
        .to(torch.bfloat16)
        .reshape(2, 1, _WIDTH)
    )
    below_boundary_expected = updates[0].clone()

    # Seed the last address representable by int32 before graph capture.  The
    # cache's full physical range already selects the 64-bit-offset kernel;
    # replay then moves the same int32 index tensor across the boundary.
    torch.ops._C_ascend.npu_scatter_nd_update_v2(cache, indices, updates)
    torch.npu.synchronize()
    _assert_exact(cache[_LAST_INT32_BLOCK, 0], updates[0])
    _assert_exact(cache[0, 0], updates[1])
    _assert_exact(cache[_FIRST_INT64_OFFSET_BLOCK, 0], torch.full_like(updates[0], _SENTINEL))

    graph = torch.npu.NPUGraph()
    pool = torch.npu.graph_pool_handle()
    stream = torch.npu.Stream()
    with torch.npu.graph(graph, pool=pool, stream=stream):
        torch.ops._C_ascend.npu_scatter_nd_update_v2(cache, indices, updates)
    torch.npu.synchronize()

    # Reuse the captured int32 index tensors after moving the first row to the
    # first physical address that requires 64-bit multiplication.
    indices.copy_(
        torch.tensor(
            [[_FIRST_INT64_OFFSET_BLOCK, 0], [1, 0]],
            dtype=torch.int32,
            device="npu:0",
        )
    )
    updates[0].fill_(23)
    updates[1].fill_(29)
    cache[_FIRST_INT64_OFFSET_BLOCK, 0].fill_(_SENTINEL)
    cache[1, 0].fill_(_SENTINEL)
    graph.replay()
    torch.npu.synchronize()

    _assert_exact(cache[_FIRST_INT64_OFFSET_BLOCK, 0], updates[0])
    _assert_exact(cache[1, 0], updates[1])
    _assert_exact(cache[_LAST_INT32_BLOCK, 0], below_boundary_expected)

    # The large-index kernel allocates its index tile according to indexDim and
    # index dtype. Exercise the widest supported index rank for both key 40
    # (int32 indices) and key 30 (int64 indices) against the same backing store.
    rank_eight_cache = backing.as_strided(
        size=(_FIRST_INT64_OFFSET_BLOCK + 1, 1, 1, 1, 1, 1, 1, 1),
        stride=(_PRODUCTION_BLOCK_STRIDE, 1, 1, 1, 1, 1, 1, 1),
    )
    for index_dtype, values in (
        (torch.int32, (31.0, 37.0)),
        (torch.int64, (41.0, 43.0)),
    ):
        high_rank_indices = torch.zeros((2, 8), dtype=index_dtype, device="npu:0")
        high_rank_indices[0, 0] = _FIRST_INT64_OFFSET_BLOCK
        high_rank_updates = torch.tensor(values, dtype=torch.bfloat16, device="npu:0")
        rank_eight_cache[_FIRST_INT64_OFFSET_BLOCK].fill_(_SENTINEL)
        rank_eight_cache[0].fill_(_SENTINEL)

        torch.ops._C_ascend.npu_scatter_nd_update_v2(
            rank_eight_cache,
            high_rank_indices,
            high_rank_updates,
        )
        torch.npu.synchronize()
        _assert_exact(
            rank_eight_cache[_FIRST_INT64_OFFSET_BLOCK].reshape(()),
            high_rank_updates[0],
        )
        _assert_exact(rank_eight_cache[0].reshape(()), high_rank_updates[1])

    assert backing.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
