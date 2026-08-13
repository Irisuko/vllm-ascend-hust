# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch_npu
from torch.distributed.distributed_c10d import _get_default_group

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


def _get_hcomm_name(rank: int) -> str:
    default_group = _get_default_group()
    backend = default_group._get_backend(torch.device("npu"))
    # TORCH_DISTRIBUTED_DEBUG may wrap the device backend in a
    # _ProcessGroupWrapper.  The communicator belongs to the wrapped HCCL
    # process group rather than to the debug wrapper.
    while hasattr(backend, "wrapped_pg"):
        backend = backend.wrapped_pg
    return backend.get_hccl_comm_name(rank)


def _make_expert_idx(
    tokens: int,
    top_k: int,
    global_experts: int,
    source_rank: int,
    generation: int,
) -> torch.Tensor:
    flat_idx = torch.arange(tokens * top_k, dtype=torch.int32)
    if generation == 0:
        expert_idx = (flat_idx + source_rank * top_k) % global_experts
    elif generation == 1:
        expert_idx = (flat_idx * 3 + source_rank * 5) % (global_experts - 3)
    else:
        # Leave most experts empty to exercise changing and skewed count rows.
        expert_idx = (flat_idx * 5 + source_rank * 3) % 7
    return expert_idx.reshape(tokens, top_k)


def _run_rank(rank: int, world_size: int, port: int) -> None:
    torch_npu.npu.set_device(rank)
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
    )

    try:
        local_experts = 8
        tokens = 64
        top_k = 4
        hidden_size = 256
        ffn_size = 256
        gate_up_size = 2 * ffn_size

        torch_npu.npu.config.allow_internal_format = True

        x = torch.zeros((tokens, hidden_size), dtype=torch.bfloat16)
        x[:, 0] = 1

        # Every expert receives the same one-hot gate/up input. GMM2 then
        # encodes the global expert id in output column zero, which makes
        # cross-rank dispatch and combine errors directly observable.
        weight1 = torch.zeros((local_experts, hidden_size, gate_up_size), dtype=torch.bfloat16)
        weight1[:, 0, 0] = 1
        weight1[:, 0, ffn_size] = 1
        weight2 = torch.zeros((local_experts, ffn_size, hidden_size), dtype=torch.bfloat16)
        for local_expert in range(local_experts):
            global_expert = rank * local_experts + local_expert
            weight2[local_expert, 0, 0] = global_expert + 1

        global_experts = world_size * local_experts
        probs_cpu = torch.arange(1, top_k + 1, dtype=torch.float32).repeat(tokens, 1)
        probs_cpu /= probs_cpu.sum(dim=-1, keepdim=True)
        silu_one = F.silu(torch.tensor(1, dtype=torch.bfloat16))

        x = x.npu()
        probs = probs_cpu.npu()
        weight1_nz = [torch_npu.npu_format_cast(weight1.npu(), 29)]
        weight2_nz = [torch_npu.npu_format_cast(weight2.npu(), 29)]
        scale1 = [torch.empty(0, dtype=torch.int64, device="npu")]
        scale2 = [torch.empty(0, dtype=torch.int64, device="npu")]
        empty_bias = [torch.empty(0, dtype=torch.float32, device="npu")]

        out = torch.empty_like(x)
        expert_token_nums = torch.zeros(local_experts, dtype=torch.int32).npu()
        x_before = x.clone()
        for generation in range(3):
            expert_idx_cpu = _make_expert_idx(tokens, top_k, global_experts, rank, generation)
            expected = torch.zeros((tokens, hidden_size), dtype=torch.bfloat16)
            expected[:, 0] = ((expert_idx_cpu + 1) * probs_cpu).sum(dim=-1).to(torch.bfloat16) * silu_one

            all_routes = torch.cat(
                [
                    _make_expert_idx(tokens, top_k, global_experts, source_rank, generation).reshape(-1)
                    for source_rank in range(world_size)
                ]
            )
            global_counts = torch.bincount(all_routes.to(torch.int64), minlength=global_experts).to(torch.int32)
            expected_tokens_per_local_expert = global_counts[rank * local_experts : (rank + 1) * local_experts]

            expert_idx = expert_idx_cpu.npu()
            expert_idx_before = expert_idx.clone()
            out.fill_(torch.nan)
            expert_token_nums.fill_(-1)
            torch.ops._C_ascend.dispatch_ffn_combine(
                x=x,
                weight1=weight1_nz,
                weight2=weight2_nz,
                expert_idx=expert_idx,
                scale1=scale1,
                scale2=scale2,
                bias1=empty_bias,
                bias2=empty_bias,
                probs=probs,
                group=_get_hcomm_name(rank),
                max_output_size=512,
                out=out,
                expert_token_nums=expert_token_nums,
            )
            torch_npu.npu.synchronize()

            torch.testing.assert_close(out.cpu(), expected, rtol=0.02, atol=0.02)
            torch.testing.assert_close(expert_token_nums.cpu(), expected_tokens_per_local_expert)
            torch.testing.assert_close(expert_idx, expert_idx_before, rtol=0, atol=0)
            torch.testing.assert_close(x, x_before, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@torch.inference_mode()
def test_dispatch_ffn_combine_bf16_two_ranks():
    world_size = 2
    port = 29501 + random.randint(0, 10000)
    mp.spawn(_run_rank, args=(world_size, port), nprocs=world_size, join=True)
