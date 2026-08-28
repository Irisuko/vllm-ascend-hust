#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.

import pickle
from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_ascend.core.layered_prefill_scheduler import (
    LayeredPrefillScheduler,
    _ActiveChunk,
)
from vllm_ascend.layered_prefill import (
    LAYERED_PREFILL_MODEL_ADAPTERS,
    LayeredPrefillMetadata,
    LayeredPrefillRequestData,
    get_layer_stage_range,
    get_moe_layer_cursors,
)
from vllm_ascend.platform import NPUPlatform


@pytest.mark.parametrize(
    ("num_layers", "num_stages"),
    [(32, 4), (32, 3), (7, 7)],
)
def test_layer_stage_ranges_cover_model_once(
    num_layers: int,
    num_stages: int,
) -> None:
    ranges = [get_layer_stage_range(num_layers, num_stages, stage) for stage in range(num_stages)]

    assert ranges[0][0] == 0
    assert ranges[-1][1] == num_layers
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
    sizes = [stop - start for start, stop in ranges]
    assert max(sizes) - min(sizes) <= 1


def test_layer_stage_range_rejects_more_stages_than_layers() -> None:
    with pytest.raises(ValueError, match="stages but only"):
        get_layer_stage_range(2, 3, 0)


def test_moe_layer_cursors_cover_dense_prefix_and_multiple_experts() -> None:
    names = (
        "model.layers.2.mlp.experts",
        "model.layers.4.mlp.experts",
        "model.layers.4.shared_expert.experts",
    )

    assert get_moe_layer_cursors(names, 0, 6) == (0, 0, 0, 1, 1, 3, 3)


@pytest.mark.parametrize(
    ("names", "match"),
    [
        (("model.experts",), "cannot map"),
        (
            (
                "model.layers.3.mlp.experts",
                "model.layers.2.mlp.experts",
            ),
            "execution order",
        ),
        (("model.layers.8.mlp.experts",), "outside decoder"),
    ],
)
def test_moe_layer_cursors_reject_invalid_registry(
    names: tuple[str, ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        get_moe_layer_cursors(names, 0, 8)


def test_layered_prefill_registers_standard_moe_architectures() -> None:
    expected_moe_architectures = {
        "Qwen3MoeForCausalLM",
        "GptOssForCausalLM",
        "MixtralForCausalLM",
        "Glm4MoeForCausalLM",
        "Ernie4_5_MoeForCausalLM",
        "DeepseekForCausalLM",
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
    }

    assert expected_moe_architectures <= LAYERED_PREFILL_MODEL_ADAPTERS.keys()
    assert all(LAYERED_PREFILL_MODEL_ADAPTERS[architecture].is_moe for architecture in expected_moe_architectures)
    assert LAYERED_PREFILL_MODEL_ADAPTERS["GptOssForCausalLM"].layer_call_order == "hidden_first"


def test_runner_restores_precomputed_moe_cursor(monkeypatch) -> None:
    import vllm_ascend.worker.layered_prefill_model_runner as runner_module

    registry = [
        "model.layers.2.mlp.experts",
        "model.layers.4.mlp.experts",
    ]
    context = SimpleNamespace(
        all_moe_layers=registry,
        layer_idx=0,
        is_first_layer=True,
        moe_layer_index=0,
    )
    model = SimpleNamespace(start_layer=0, end_layer=6)
    runner = object.__new__(runner_module.LayeredPrefillNPUModelRunner)
    runner._moe_layer_registry = registry
    runner._moe_layer_names = tuple(registry)
    runner._moe_layer_cursors = get_moe_layer_cursors(registry, 0, 6)
    monkeypatch.setattr(
        runner_module,
        "get_forward_context",
        lambda: context,
    )

    runner._align_moe_cursor(model, 4)

    assert context.layer_idx == 4
    assert not context.is_first_layer
    assert context.moe_layer_index == 1


def test_layered_prefill_metadata_marks_only_last_stage_final() -> None:
    request = LayeredPrefillRequestData("request", 0, 16)

    assert not LayeredPrefillMetadata(0, 2, (request,)).is_final_stage
    assert LayeredPrefillMetadata(1, 2, (request,)).is_final_stage


def test_scheduler_output_transports_opt_in_metadata() -> None:
    request = LayeredPrefillRequestData("request", 0, 16)
    output = SchedulerOutput.make_empty()
    output.layered_prefill = LayeredPrefillMetadata(  # type: ignore[attr-defined]
        0,
        2,
        (request,),
    )

    restored = pickle.loads(pickle.dumps(output))
    assert restored.layered_prefill.requests == (request,)  # type: ignore[attr-defined]


def test_logical_tokens_advance_only_on_final_stage(monkeypatch) -> None:
    request = SimpleNamespace(
        request_id="request",
        num_computed_tokens=0,
        num_prompt_tokens=16,
    )
    scheduler = object.__new__(LayeredPrefillScheduler)
    scheduler._num_stages = 2
    scheduler._stage = 0
    scheduler._active_chunks = {}
    scheduler._active_token_lists = {}
    scheduler.requests = {request.request_id: request}
    scheduler.prev_step_scheduled_req_ids = set()

    def advance_tokens(_scheduler, output) -> None:
        for req_id, num_tokens in output.num_scheduled_tokens.items():
            scheduler.requests[req_id].num_computed_tokens += num_tokens

    monkeypatch.setattr(Scheduler, "_update_after_schedule", advance_tokens)
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {request.request_id: 16}
    output.total_num_scheduled_tokens = 16

    scheduler._update_after_schedule(output)
    assert request.num_computed_tokens == 0
    assert request.request_id in scheduler._active_chunks
    assert output.total_num_scheduled_tokens == 16

    scheduler._stage = 1
    scheduler._update_after_schedule(output)
    assert request.num_computed_tokens == 16


def test_later_stage_caps_and_restores_the_original_chunk() -> None:
    class Request:
        request_id = "request"

        def __init__(self) -> None:
            self._all_token_ids = list(range(32))

        @property
        def num_tokens(self) -> int:
            return len(self._all_token_ids)

    request = Request()
    original_token_ids = request._all_token_ids
    scheduler = object.__new__(LayeredPrefillScheduler)
    scheduler._active_chunks = {request.request_id: _ActiveChunk(start_token=8, num_tokens=4)}
    scheduler._active_token_lists = {}
    scheduler.requests = {request.request_id: request}

    scheduler._cap_active_token_lists([request])
    assert request._all_token_ids == list(range(12))

    scheduler._restore_active_token_lists()
    assert request._all_token_ids is original_token_ids


@pytest.mark.parametrize(
    "architecture",
    [
        "Qwen3ForCausalLM",
        "MixtralForCausalLM",
        "DeepseekV3ForCausalLM",
    ],
)
def test_platform_mounts_isolated_scheduler_and_eager_runner(
    monkeypatch,
    architecture: str,
) -> None:
    # Keep the test independent from earlier tests that exercise the
    # process-global sequence-parallel feature cache.
    monkeypatch.setattr("vllm_ascend.utils._ENABLE_SP", False)
    model_config = SimpleNamespace(
        architecture=architecture,
        model_impl="auto",
        get_total_num_hidden_layers=lambda: 32,
        is_multimodal_model=False,
        enable_return_routed_experts=False,
        disable_cascade_attn=False,
        enforce_eager=False,
    )
    scheduler_config = SimpleNamespace(
        scheduler_cls=None,
        async_scheduling=True,
        policy="fcfs",
        runner_type="generate",
    )
    parallel_config = SimpleNamespace(
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        data_parallel_size=1,
        use_ubatching=False,
        use_sequence_parallel_moe=False,
        worker_cls="auto",
    )
    compilation_config = SimpleNamespace(
        pass_config=SimpleNamespace(enable_sp=False),
        mode=None,
        cudagraph_mode=None,
        splitting_ops=None,
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
        compilation_config=compilation_config,
        speculative_config=None,
        lora_config=None,
        kv_transfer_config=None,
        ec_transfer_config=None,
        use_v2_model_runner=False,
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=False),
    )
    ascend_config = SimpleNamespace(
        layered_prefill_num_stages=4,
        scheduler_config=SimpleNamespace(
            enable_balance_scheduling=False,
            recompute_scheduler_enable=False,
            short_request_first_config=SimpleNamespace(enabled=False),
            profiling_chunk_config=SimpleNamespace(enabled=False),
            batch_job_sched_config=SimpleNamespace(enabled=False),
            dyntra_lb_config=SimpleNamespace(enabled=False),
        ),
        xlite_graph_config=SimpleNamespace(enabled=False),
        eplb_config=SimpleNamespace(dynamic_eplb=False),
        multistream_overlap_shared_expert=False,
        ascend_compilation_config=SimpleNamespace(
            enable_npugraph_ex=True,
            enable_static_kernel=True,
        ),
    )
    monkeypatch.setattr("vllm_ascend.platform.is_310p", lambda: False)

    NPUPlatform._configure_layered_prefill(vllm_config, ascend_config)
    # EngineCore's multiprocessing handshake validates the same config again.
    NPUPlatform._configure_layered_prefill(vllm_config, ascend_config)

    assert scheduler_config.async_scheduling is False
    assert scheduler_config.scheduler_cls.endswith("LayeredPrefillScheduler")
    assert model_config.enforce_eager
    assert model_config.disable_cascade_attn
    assert not ascend_config.ascend_compilation_config.enable_npugraph_ex
    assert not ascend_config.ascend_compilation_config.enable_static_kernel


def test_platform_rejects_unsupported_layered_prefill_model(
    monkeypatch,
) -> None:
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            architecture="LlamaForCausalLM",
        ),
        scheduler_config=SimpleNamespace(),
        parallel_config=SimpleNamespace(),
        compilation_config=SimpleNamespace(),
    )
    ascend_config = SimpleNamespace(layered_prefill_num_stages=2)
    monkeypatch.setattr("vllm_ascend.platform.is_310p", lambda: False)

    with pytest.raises(ValueError, match="currently supports"):
        NPUPlatform._configure_layered_prefill(vllm_config, ascend_config)
