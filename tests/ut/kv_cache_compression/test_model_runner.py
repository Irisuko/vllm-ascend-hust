# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

LAYER_NAMES = [f"model.layers.{index}.self_attn.attn" for index in range(32)]
QWEN_LAYER_NAMES = [f"model.layers.{index}.self_attn.attn" for index in range(48)]


class FakeProvider:
    def __init__(self, *, compress: bool = True) -> None:
        self.compress = compress
        self.cleaned: list[str] = []
        self.commits = []
        self.build_kwargs = None
        self.finished = None

    def cleanup_request(self, request_id: str) -> None:
        self.cleaned.append(request_id)

    def mark_committed(self, request_id, block_ids) -> None:
        self.commits.append((request_id, block_ids))

    def build_attention_batch_view(self, **kwargs):
        self.build_kwargs = kwargs
        return SimpleNamespace(requests=(SimpleNamespace(compress=self.compress),))

    def finish_model_forward(self, view, **kwargs):
        self.finished = (view, kwargs)
        return ["plan"]

    @staticmethod
    def _layer_indices(layer_names):
        return {name: int(name.split(".layers.", 1)[1].split(".", 1)[0]) for name in layer_names}


class FakeBlockTable:
    def __init__(self) -> None:
        self.rows = []

    def add_row(self, block_ids, row_index) -> None:
        self.rows.append((block_ids, row_index))


def _runner(
    provider: FakeProvider,
    *,
    layer_names=LAYER_NAMES,
    num_hidden_layers: int = 32,
    num_attention_heads: int = 32,
) -> NPUModelRunner:
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.kv_cache_compression_provider = provider
    runner._kv_cache_compression_step_view = None
    runner._kv_cache_compression_plans = None
    runner._kv_cache_compression_destination_block_ids = None
    runner.requests = {
        "request": SimpleNamespace(block_ids=([1, 2, 3],)),
    }
    runner.input_batch = SimpleNamespace(
        req_ids=["request"],
        req_id_to_index={"request": 0},
        block_table=FakeBlockTable(),
        num_computed_tokens_cpu=np.array([0], dtype=np.int32),
        num_prompt_tokens=np.array([20], dtype=np.int32),
    )
    runner.optimistic_seq_lens_cpu = torch.tensor([20], dtype=torch.int32)
    group = SimpleNamespace(
        layer_names=layer_names,
        kv_cache_spec=SimpleNamespace(block_size=128),
    )
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[group])
    runner.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            hidden_size=num_attention_heads * 128,
        )
    )
    runner.vllm_config = SimpleNamespace(kv_cache_compression_config=SimpleNamespace(schema_version=1))
    return runner


def test_disabled_runner_does_not_build_provider_view() -> None:
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.kv_cache_compression_provider = None

    assert (
        runner._build_kv_cache_compression_view(
            num_reqs=1,
            num_scheduled_tokens_np=np.array([1], dtype=np.int32),
        )
        is None
    )


def test_commit_ack_replaces_request_and_persistent_block_table() -> None:
    provider = FakeProvider()
    runner = _runner(provider)
    scheduler_output = SimpleNamespace(kv_cache_compression_block_table_updates={"request": ([1, 2],)})

    runner._apply_kv_cache_compression_block_table_updates(scheduler_output)

    assert runner.requests["request"].block_ids == ([1, 2],)
    assert runner.input_batch.block_table.rows == [(([1, 2],), 0)]
    assert provider.commits == [("request", ((1, 2),))]


def test_multiple_commit_acks_keep_request_provider_and_view_in_sync() -> None:
    provider = FakeProvider()
    runner = _runner(provider)
    runner.requests["other"] = SimpleNamespace(block_ids=([4, 5, 6],))
    runner.input_batch.req_ids = ["request", "other"]
    runner.input_batch.req_id_to_index["other"] = 1
    runner.input_batch.num_computed_tokens_cpu = np.array([20, 20], dtype=np.int32)
    runner.input_batch.num_prompt_tokens = np.array([20, 20], dtype=np.int32)
    runner.optimistic_seq_lens_cpu = torch.tensor([21, 21], dtype=torch.int32)
    scheduler_output = SimpleNamespace(
        kv_cache_compression_block_table_updates={
            "request": ([1, 2],),
            "other": ([4, 5],),
        }
    )

    runner._apply_kv_cache_compression_block_table_updates(scheduler_output)
    runner._build_kv_cache_compression_view(
        num_reqs=2,
        num_scheduled_tokens_np=np.array([1, 1], dtype=np.int32),
    )

    assert runner.requests["request"].block_ids == ([1, 2],)
    assert runner.requests["other"].block_ids == ([4, 5],)
    assert provider.commits == [
        ("request", ((1, 2),)),
        ("other", ((4, 5),)),
    ]
    assert provider.build_kwargs["block_ids"] == (
        ((1, 2),),
        ((4, 5),),
    )


def test_commit_ack_without_active_provider_is_rejected() -> None:
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.kv_cache_compression_provider = None
    scheduler_output = SimpleNamespace(kv_cache_compression_block_table_updates={"request": ([1],)})

    with pytest.raises(RuntimeError, match="without an active provider"):
        runner._apply_kv_cache_compression_block_table_updates(scheduler_output)


def test_unknown_commit_ack_is_rejected_without_provider_mutation() -> None:
    provider = FakeProvider()
    runner = _runner(provider)
    scheduler_output = SimpleNamespace(kv_cache_compression_block_table_updates={"unknown": ([1],)})

    with pytest.raises(RuntimeError, match="unknown request"):
        runner._apply_kv_cache_compression_block_table_updates(scheduler_output)

    assert provider.commits == []
    assert runner.input_batch.block_table.rows == []


def test_finished_preempted_and_resumed_states_are_cleaned() -> None:
    provider = FakeProvider()
    runner = _runner(provider)
    scheduler_output = SimpleNamespace(
        finished_req_ids={"finished"},
        preempted_req_ids={"preempted"},
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids={"resumed"}),
    )

    runner._cleanup_kv_cache_compression_states(scheduler_output)

    assert set(provider.cleaned) == {"finished", "preempted", "resumed"}


def test_runner_builds_view_from_semantic_state_and_physical_blocks() -> None:
    provider = FakeProvider()
    runner = _runner(provider)

    view = runner._build_kv_cache_compression_view(
        num_reqs=1,
        num_scheduled_tokens_np=np.array([20], dtype=np.int32),
    )

    assert view is runner._kv_cache_compression_step_view
    assert provider.build_kwargs == {
        "request_ids": ("request",),
        "query_lengths": (20,),
        "semantic_num_tokens": (20,),
        "num_computed_tokens": (0,),
        "num_prompt_tokens": (20,),
        "block_ids": (((1, 2, 3),),),
        "layer_names": tuple(LAYER_NAMES),
        "block_size": 128,
        "destination_block_ids": None,
    }


def test_runner_forwards_private_destination_table_to_provider() -> None:
    provider = FakeProvider()
    runner = _runner(provider)
    destinations = {"request": ((7, 8),)}
    runner._kv_cache_compression_destination_block_ids = destinations

    runner._build_kv_cache_compression_view(
        num_reqs=1,
        num_scheduled_tokens_np=np.array([20], dtype=np.int32),
    )

    assert provider.build_kwargs["destination_block_ids"] is destinations


def test_below_threshold_batch_keeps_attention_view_none() -> None:
    provider = FakeProvider(compress=False)
    runner = _runner(provider)

    view = runner._build_kv_cache_compression_view(
        num_reqs=1,
        num_scheduled_tokens_np=np.array([20], dtype=np.int32),
    )

    assert view is None
    assert runner._kv_cache_compression_step_view is None


def test_full_decode_keeps_uncompressed_view_without_step_state() -> None:
    provider = FakeProvider(compress=False)
    runner = _runner(provider)

    view = runner._build_kv_cache_compression_view(
        num_reqs=1,
        num_scheduled_tokens_np=np.array([1], dtype=np.int32),
        include_uncompressed=True,
    )

    assert view is not None
    assert view.requests[0].compress is False
    assert runner._kv_cache_compression_step_view is None


def test_successful_forward_finishes_plans_and_clears_step_view() -> None:
    provider = FakeProvider()
    runner = _runner(provider)
    view = SimpleNamespace(requests=())
    runner._kv_cache_compression_step_view = view
    runner._kv_cache_compression_destination_block_ids = {"request": ((7, 8),)}

    runner._finish_kv_cache_compression_forward()

    assert provider.finished == (
        view,
        {"layer_names": tuple(LAYER_NAMES), "schema_version": 1},
    )
    assert runner._kv_cache_compression_plans == ["plan"]
    assert runner._kv_cache_compression_step_view is None
    assert runner._kv_cache_compression_destination_block_ids is None


def test_runner_takes_an_independent_plan_snapshot_for_async_output() -> None:
    runner = _runner(FakeProvider())
    mutable_plans = ["plan"]
    runner._kv_cache_compression_plans = mutable_plans

    snapshot = runner._take_kv_cache_compression_plans()
    mutable_plans.append("later-plan")

    assert snapshot == ["plan"]
    assert snapshot is not mutable_plans
    assert runner._kv_cache_compression_plans is None


@pytest.mark.parametrize(
    ("layer_names", "num_hidden_layers", "num_attention_heads"),
    [
        (LAYER_NAMES, 32, 32),
        (QWEN_LAYER_NAMES, 48, 40),
        (list(reversed(QWEN_LAYER_NAMES)), 48, 40),
    ],
    ids=["llama-3-8b", "qwen2.5-14b", "qwen2.5-14b-unordered"],
)
def test_full_decode_buffers_keep_addresses_across_updates(
    layer_names,
    num_hidden_layers: int,
    num_attention_heads: int,
) -> None:
    provider = FakeProvider()
    runner = _runner(
        provider,
        layer_names=layer_names,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
    )
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY)
    runner.max_num_reqs = 4
    runner.pin_memory = False
    runner.device = torch.device("cpu")
    runner.activate_kv_cache_compression_provider(provider)
    slots = runner._kv_cache_compression_full_slots
    lengths = runner._kv_cache_compression_full_lengths
    assert slots is not None
    assert lengths is not None
    assert tuple(slots.shape) == (num_hidden_layers, 4)
    assert tuple(lengths.shape) == (num_hidden_layers, 4)
    slot_ptr = slots.untyped_storage().data_ptr()
    length_ptr = lengths.untyped_storage().data_ptr()

    class FakeFullView:
        def __init__(self, value: int) -> None:
            self.value = value

        def fill_full_decode_metadata(
            self,
            *,
            layer_names,
            slot_staging,
            length_staging,
            num_reqs_padded,
        ) -> None:
            slot_staging.fill_(-1)
            length_staging.fill_(1)
            slot_staging[:, :2].fill_(self.value)
            length_staging[:, :2].fill_(self.value + 1)

    runner._prepare_full_kv_cache_compression_metadata(view=FakeFullView(7), num_reqs=2, num_reqs_padded=4)
    assert torch.equal(slots[0], torch.tensor([7, 7, -1, -1]))
    assert torch.equal(lengths[0], torch.tensor([8, 8, 1, 1]))

    runner._prepare_full_kv_cache_compression_metadata(view=FakeFullView(11), num_reqs=2, num_reqs_padded=4)
    assert slots.untyped_storage().data_ptr() == slot_ptr
    assert lengths.untyped_storage().data_ptr() == length_ptr
    assert torch.equal(slots[0], torch.tensor([11, 11, -1, -1]))
    assert torch.equal(lengths[0], torch.tensor([12, 12, 1, 1]))


@pytest.mark.parametrize(
    "layer_names",
    [LAYER_NAMES, QWEN_LAYER_NAMES],
    ids=["llama-3-8b", "qwen2.5-14b"],
)
def test_full_graph_completion_marks_layers_only_in_post_forward_finish(
    layer_names,
) -> None:
    provider = FakeProvider()
    runner = _runner(
        provider,
        layer_names=layer_names,
        num_hidden_layers=len(layer_names),
        num_attention_heads=40 if len(layer_names) == 48 else 32,
    )
    view = SimpleNamespace(
        requests=(SimpleNamespace(is_prefill=False),),
        completed_decode_layers=set(),
    )
    runner._kv_cache_compression_step_view = view

    assert view.completed_decode_layers == set()
    runner._finish_kv_cache_compression_forward(full_graph_decode=True)

    assert view.completed_decode_layers == set(layer_names)
    assert provider.finished == (
        view,
        {"layer_names": tuple(layer_names), "schema_version": 1},
    )


def test_full_graph_completion_rejects_prefill_without_finishing() -> None:
    provider = FakeProvider()
    runner = _runner(provider)
    view = SimpleNamespace(
        requests=(SimpleNamespace(is_prefill=True),),
        completed_decode_layers=set(),
    )
    runner._kv_cache_compression_step_view = view

    with pytest.raises(RuntimeError, match="cannot contain prefill"):
        runner._finish_kv_cache_compression_forward(full_graph_decode=True)

    assert provider.finished is None
    assert view.completed_decode_layers == set()


def test_one_token_final_prefill_disables_full_graph_but_decode_allows_it() -> None:
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.kv_cache_compression_provider = object()
    runner.input_batch = SimpleNamespace(
        num_computed_tokens_cpu=np.array([511], dtype=np.int32),
        num_prompt_tokens=np.array([512], dtype=np.int32),
        lora_id_to_lora_request={},
    )
    runner.speculative_config = None
    runner.uniform_decode_query_len = 1
    runner.model_config = SimpleNamespace(is_encoder_decoder=False)
    runner.parallel_config = SimpleNamespace(
        data_parallel_size=1,
        tensor_parallel_size=1,
    )
    runner.vllm_config = SimpleNamespace(
        additional_config={"enable_flashcomm1": False, "refresh": True},
        parallel_config=runner.parallel_config,
        observability_config=SimpleNamespace(cudagraph_metrics=False),
    )
    runner._pad_for_sequence_parallelism = lambda value: value

    class FakeDispatcher:
        def __init__(self) -> None:
            self.calls = []

        def dispatch(self, **kwargs):
            self.calls.append(kwargs)
            invalid = kwargs.get("invalid_modes") or set()
            mode = CUDAGraphMode.PIECEWISE if CUDAGraphMode.FULL in invalid else CUDAGraphMode.FULL
            return mode, BatchDescriptor(kwargs["num_tokens"])

    runner.cudagraph_dispatcher = FakeDispatcher()
    common = {
        "num_tokens": 1,
        "num_reqs": 1,
        "num_scheduled_tokens_np": np.array([1], dtype=np.int32),
        "max_num_scheduled_tokens": 1,
        "use_cascade_attn": False,
        "force_uniform_decode": True,
    }

    prefill_mode, *_ = runner._determine_batch_execution_and_padding(**common)
    assert prefill_mode == CUDAGraphMode.PIECEWISE
    assert runner.cudagraph_dispatcher.calls[-1]["invalid_modes"] == {CUDAGraphMode.FULL}

    runner.input_batch.num_computed_tokens_cpu[0] = 512
    decode_mode, *_ = runner._determine_batch_execution_and_padding(**common)
    assert decode_mode == CUDAGraphMode.FULL
    assert runner.cudagraph_dispatcher.calls[-1]["invalid_modes"] is None
