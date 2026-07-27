# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import math
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from vllm.config import KVCacheCompressionConfig

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackendImpl,
    AscendAttentionState,
    AscendMetadata,
)
from vllm_ascend.kv_cache_compression.pyramidkv import (
    PyramidKVAscendConfig,
    PyramidKVAscendProvider,
    PyramidKVAttentionBatchView,
    PyramidKVAttentionRequest,
    PyramidKVCapabilityContext,
    PyramidKVSelection,
    select_pyramid_kv,
)

LAYER_NAMES = tuple(f"model.layers.{index}.self_attn.attn" for index in range(32))


def _config(**updates) -> PyramidKVAscendConfig:
    values = {
        "max_capacity_prompt": 12,
        "window_size": 4,
        "kernel_size": 1,
        "pooling": "maxpool",
        "beta": 2,
        "kv_cache_granularity": "kv_head",
        "gqa_score_aggregation": "mean",
        "merge": None,
    }
    values.update(updates)
    return PyramidKVAscendConfig.from_dict(values)


def _context(**updates) -> PyramidKVCapabilityContext:
    values = {
        "platform": "npu",
        "device_name": "Ascend910B2",
        "cann_version": "8.5.1",
        "use_v2_model_runner": False,
        "enforce_eager": True,
        "cudagraph_mode": "NONE",
        "pa_shape_list": (),
        "backend": "AscendAttentionBackend",
        "model_architecture": "LlamaForCausalLM",
        "dtype": "torch.bfloat16",
        "quantization": None,
        "num_attention_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "num_hidden_layers": 32,
        "cache_layout": "standard_bf16_paged",
        "block_size": 128,
        "hash_block_size": 128,
        "max_model_len": 8192,
        "num_kv_cache_groups": 1,
        "full_attention_only": True,
        "prefix_caching": False,
        "chunked_prefill": False,
        "sliding_window": False,
        "speculative_decoding": False,
        "kv_transfer": False,
        "kv_offload": False,
        "cache_dtype": "auto",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "async_scheduling": False,
        "balance_scheduling": False,
        "dbo_enabled": False,
        "knorm_enabled": False,
        "missing_ops": (),
    }
    values.update(updates)
    return PyramidKVCapabilityContext(**values)


def _core_config() -> KVCacheCompressionConfig:
    return KVCacheCompressionConfig(
        provider="pyramidkv_ascend",
        provider_config={
            "max_capacity_prompt": 12,
            "window_size": 4,
            "kernel_size": 1,
            "pooling": "maxpool",
            "beta": 2,
            "kv_cache_granularity": "kv_head",
            "gqa_score_aggregation": "mean",
            "merge": None,
        },
    )


@pytest.mark.parametrize(
    ("updates", "error_match"),
    [
        ({"max_capacity_prompt": 4}, "greater than window_size"),
        ({"window_size": True}, "positive integer"),
        ({"kernel_size": 2}, "must be odd"),
        ({"pooling": "avgpool"}, "maxpool"),
        ({"beta": 0}, "positive integer"),
        ({"kv_cache_granularity": "query_head"}, "kv_head"),
        ({"gqa_score_aggregation": "max"}, "mean"),
        ({"merge": "pivot"}, "must be null"),
        ({"extra": 1}, "unknown PyramidKV"),
    ],
)
def test_config_rejects_unsupported_values(updates, error_match) -> None:
    with pytest.raises((TypeError, ValueError), match=error_match):
        _config(**updates)


def test_layer_capacity_schedule_and_thresholds() -> None:
    config = _config()

    assert config.retained_tokens(11, 0, 2) == 11
    assert config.retained_tokens(12, 0, 2) == 12
    assert config.retained_tokens(15, 1, 2) == 12
    assert config.retained_tokens(20, 0, 2) == 16
    assert config.retained_tokens(20, 1, 2) == 8


def _independent_selection(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    retained_tokens: int,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    groups = query.shape[1] // key.shape[1]
    repeated_key = key.repeat_interleave(groups, dim=1)
    scores = torch.matmul(query[:, :, -window_size:, :], repeated_key.transpose(2, 3)) / math.sqrt(query.shape[-1])
    scores[..., -window_size:] += torch.triu(
        torch.full(
            (window_size, window_size),
            torch.finfo(scores.dtype).min,
            dtype=scores.dtype,
        ),
        diagonal=1,
    )
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    history = probabilities[..., :-window_size].sum(dim=-2)
    history = history.reshape(query.shape[0], key.shape[1], groups, history.shape[-1]).mean(dim=2)
    pooled = F.max_pool1d(history, 1, stride=1)
    selected = pooled.topk(retained_tokens - window_size, dim=-1).indices
    gather_index = selected.unsqueeze(-1).expand(-1, -1, -1, key.shape[-1])
    compact_key = torch.cat(
        [
            key[:, :, :-window_size, :].gather(2, gather_index),
            key[:, :, -window_size:, :],
        ],
        dim=2,
    )
    compact_value = torch.cat(
        [
            value[:, :, :-window_size, :].gather(2, gather_index),
            value[:, :, -window_size:, :],
        ],
        dim=2,
    )
    return compact_key, compact_value, selected


def test_gqa_mean_topk_and_gather_match_cpu_oracle() -> None:
    generator = torch.Generator().manual_seed(123)
    query = torch.randn(1, 4, 20, 8, generator=generator)
    key = torch.randn(1, 2, 20, 8, generator=generator)
    value = torch.randn(1, 2, 20, 8, generator=generator)
    config = _config()

    result = select_pyramid_kv(query, key, value, config, layer_index=1, num_hidden_layers=2)
    expected_key, expected_value, expected_indices = _independent_selection(
        query, key, value, retained_tokens=8, window_size=4
    )

    assert result.compressed
    assert result.retained_tokens == 8
    assert torch.equal(result.selected_past_indices, expected_indices)
    torch.testing.assert_close(result.key, expected_key)
    torch.testing.assert_close(result.value, expected_value)
    torch.testing.assert_close(result.key[:, :, -4:, :], key[:, :, -4:, :])
    torch.testing.assert_close(result.value[:, :, -4:, :], value[:, :, -4:, :])


def test_below_threshold_returns_original_tensors_without_selection() -> None:
    query = torch.randn(1, 4, 11, 8)
    key = torch.randn(1, 2, 11, 8)
    value = torch.randn(1, 2, 11, 8)

    result = select_pyramid_kv(query, key, value, _config(), layer_index=0, num_hidden_layers=2)

    assert result.key is key
    assert result.value is value
    assert result.selected_past_indices is None
    assert not result.compressed


def _run_chunked_prefill(
    chunk_lengths: tuple[int, ...],
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, PyramidKVSelection],
    list,
    PyramidKVAscendProvider,
]:
    prompt_tokens = sum(chunk_lengths)
    layer_names = tuple(f"model.layers.{index}.self_attn.attn" for index in range(2))
    provider = PyramidKVAscendProvider(
        _config(
            max_capacity_prompt=128,
            window_size=8,
            kernel_size=7,
            beta=20,
        )
    )
    first_end = chunk_lengths[0]
    first_blocks = tuple(range((first_end + 127) // 128))
    provider.begin_request("request", prompt_tokens, (first_blocks,))
    generator = torch.Generator().manual_seed(20260725)
    layer_inputs = {
        layer_name: (
            torch.randn(prompt_tokens, 4, 8, generator=generator, dtype=torch.bfloat16),
            torch.randn(prompt_tokens, 2, 8, generator=generator, dtype=torch.bfloat16),
            torch.randn(prompt_tokens, 2, 8, generator=generator, dtype=torch.bfloat16),
        )
        for layer_name in layer_names
    }
    expected = {
        layer_name: select_pyramid_kv(
            query.permute(1, 0, 2).unsqueeze(0),
            key.permute(1, 0, 2).unsqueeze(0),
            value.permute(1, 0, 2).unsqueeze(0),
            provider.config,
            layer_index=layer_index,
            num_hidden_layers=len(layer_names),
        )
        for layer_index, (layer_name, (query, key, value)) in enumerate(layer_inputs.items())
    }
    num_blocks = (prompt_tokens + 127) // 128
    layer_caches = {
        layer_name: (
            torch.zeros(num_blocks, 128, 2, 8, dtype=torch.bfloat16),
            torch.zeros(num_blocks, 128, 2, 8, dtype=torch.bfloat16),
        )
        for layer_name in layer_names
    }

    def write_cache(layer, write_key, write_value, kv_cache, slots):
        kv_cache[0].view(-1, 2, 8)[slots.long()] = write_key
        kv_cache[1].view(-1, 2, 8)[slots.long()] = write_value

    backend = SimpleNamespace(do_kv_cache_update=write_cache)
    computed = 0
    selected_by_layer: dict[str, torch.Tensor] = {}
    final_plans = None
    for chunk_index, chunk_length in enumerate(chunk_lengths):
        end = computed + chunk_length
        block_ids = tuple(range((end + 127) // 128))
        request = PyramidKVAttentionRequest(
            request_id="request",
            query_start=0,
            query_end=chunk_length,
            semantic_num_tokens=end,
            num_computed_tokens=computed,
            num_prompt_tokens=prompt_tokens,
            block_ids=block_ids,
            is_prefill=True,
        )
        view = PyramidKVAttentionBatchView(
            provider=provider,
            requests=(request,),
            layer_indices={layer_name: index for index, layer_name in enumerate(layer_names)},
            num_hidden_layers=len(layer_names),
        )
        if len(chunk_lengths) == 1:
            attention_state = AscendAttentionState.PrefillNoCache
        elif chunk_length == 1:
            attention_state = AscendAttentionState.DecodeOnly
        else:
            attention_state = AscendAttentionState.ChunkedPrefill
        metadata = AscendMetadata(
            attn_state=attention_state,
            num_actual_tokens=chunk_length,
            slot_mapping=torch.arange(computed, end, dtype=torch.int32),
            seq_lens=torch.tensor([end], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([end], dtype=torch.int32),
            seq_lens_list=[end],
        )
        for layer_name in layer_names:
            query, key, value = layer_inputs[layer_name]
            view.before_cache_write(
                layer=SimpleNamespace(layer_name=layer_name),
                backend=backend,
                query=query[computed:end],
                key=key[computed:end],
                value=value[computed:end],
                kv_cache=layer_caches[layer_name],
                attn_metadata=metadata,
            )
        if end < prompt_tokens:
            assert provider.finish_model_forward(view, layer_names=layer_names, schema_version=1) is None
            state = provider.get_request_state("request")
            assert state.prefill_num_computed_tokens == end
            assert state.expected_block_ids == (block_ids,)
            assert not state.layers
            assert not state.plan_emitted
        else:
            selected_by_layer = {
                deferred.layer.layer_name: deferred.selected_past_indices for deferred in view.deferred_prefills
            }
            final_plans = provider.finish_model_forward(view, layer_names=layer_names, schema_version=1)
        computed = end

    assert final_plans is not None and len(final_plans) == 1
    for layer_name, selection in expected.items():
        key_cache, value_cache = layer_caches[layer_name]
        actual_key = key_cache.view(-1, 2, 8)[: selection.retained_tokens]
        actual_value = value_cache.view(-1, 2, 8)[: selection.retained_tokens]
        torch.testing.assert_close(
            actual_key,
            selection.key.squeeze(0).permute(1, 0, 2),
            rtol=1e-2,
            atol=1e-2,
        )
        torch.testing.assert_close(
            actual_value,
            selection.value.squeeze(0).permute(1, 0, 2),
            rtol=1e-2,
            atol=1e-2,
        )
    return selected_by_layer, expected, final_plans, provider


@pytest.mark.parametrize(
    "chunk_lengths",
    [
        (256, 256, 256),
        (512, 255, 1),
        (128, 320, 200, 120),
        (768,),
    ],
    ids=["equal", "one-token-final", "dynamic-budget", "single-prefill"],
)
def test_chunked_prefill_matches_single_prefill_selection_and_compact_kv(
    chunk_lengths: tuple[int, ...],
) -> None:
    selected, expected, plans, provider = _run_chunked_prefill(chunk_lengths)

    if len(chunk_lengths) > 1:
        assert set(selected) == set(expected)
        for layer_name, indices in selected.items():
            assert torch.equal(
                indices,
                expected[layer_name].selected_past_indices,
            )
    plan = plans[0]
    assert plan.semantic_num_tokens == 768
    assert plan.physical_num_tokens < plan.semantic_num_tokens
    state = provider.get_request_state("request")
    assert state.plan_emitted
    assert state.prefill_query_tail is None


@pytest.mark.parametrize(
    "num_computed_tokens",
    [0, 128],
    ids=["cold", "prefix-hit"],
)
def test_prefix_cached_prefill_materializes_private_destination_without_source_mutation(
    num_computed_tokens: int,
) -> None:
    prompt_tokens = 256
    query_length = prompt_tokens - num_computed_tokens
    layer_names = (
        "model.layers.0.self_attn.attn",
        "model.layers.1.self_attn.attn",
    )
    provider = PyramidKVAscendProvider(
        _config(
            max_capacity_prompt=128,
            window_size=8,
            kernel_size=7,
            beta=20,
        )
    )
    provider.prefix_caching = True
    provider.begin_request(
        "request",
        prompt_tokens,
        ((0, 1),),
        prefill_num_computed_tokens=num_computed_tokens,
    )
    generator = torch.Generator().manual_seed(20260726)
    layer_inputs = {
        layer_name: (
            torch.randn(
                prompt_tokens,
                4,
                8,
                generator=generator,
                dtype=torch.bfloat16,
            ),
            torch.randn(
                prompt_tokens,
                2,
                8,
                generator=generator,
                dtype=torch.bfloat16,
            ),
            torch.randn(
                prompt_tokens,
                2,
                8,
                generator=generator,
                dtype=torch.bfloat16,
            ),
        )
        for layer_name in layer_names
    }
    expected = {
        layer_name: select_pyramid_kv(
            query.permute(1, 0, 2).unsqueeze(0),
            key.permute(1, 0, 2).unsqueeze(0),
            value.permute(1, 0, 2).unsqueeze(0),
            provider.config,
            layer_index=layer_index,
            num_hidden_layers=len(layer_names),
        )
        for layer_index, (layer_name, (query, key, value)) in enumerate(layer_inputs.items())
    }
    layer_caches = {
        layer_name: (
            torch.zeros(4, 128, 2, 8, dtype=torch.bfloat16),
            torch.zeros(4, 128, 2, 8, dtype=torch.bfloat16),
        )
        for layer_name in layer_names
    }
    for layer_name, (_, key, value) in layer_inputs.items():
        if num_computed_tokens:
            layer_caches[layer_name][0].view(-1, 2, 8)[:num_computed_tokens] = key[:num_computed_tokens]
            layer_caches[layer_name][1].view(-1, 2, 8)[:num_computed_tokens] = value[:num_computed_tokens]

    def write_cache(layer, write_key, write_value, kv_cache, slots):
        kv_cache[0].view(-1, 2, 8)[slots.long()] = write_key
        kv_cache[1].view(-1, 2, 8)[slots.long()] = write_value

    backend = SimpleNamespace(do_kv_cache_update=write_cache)
    request = PyramidKVAttentionRequest(
        request_id="request",
        query_start=0,
        query_end=query_length,
        semantic_num_tokens=prompt_tokens,
        num_computed_tokens=num_computed_tokens,
        num_prompt_tokens=prompt_tokens,
        block_ids=(0, 1),
        is_prefill=True,
        destination_block_ids=(2, 3),
    )
    view = PyramidKVAttentionBatchView(
        provider=provider,
        requests=(request,),
        layer_indices={layer_name: index for index, layer_name in enumerate(layer_names)},
        num_hidden_layers=len(layer_names),
    )
    metadata = AscendMetadata(
        attn_state=(
            AscendAttentionState.PrefillNoCache if num_computed_tokens == 0 else AscendAttentionState.PrefillCacheHit
        ),
        num_actual_tokens=query_length,
        slot_mapping=torch.arange(
            num_computed_tokens,
            prompt_tokens,
            dtype=torch.int32,
        ),
        seq_lens=torch.tensor([prompt_tokens], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([prompt_tokens], dtype=torch.int32),
        seq_lens_list=[prompt_tokens],
    )

    source_snapshots = {}
    for layer_name in layer_names:
        query, key, value = layer_inputs[layer_name]
        cache = layer_caches[layer_name]
        suppress_default_write = view.before_cache_write(
            layer=SimpleNamespace(layer_name=layer_name),
            backend=backend,
            query=query[num_computed_tokens:],
            key=key[num_computed_tokens:],
            value=value[num_computed_tokens:],
            kv_cache=cache,
            attn_metadata=metadata,
        )
        assert not suppress_default_write
        write_cache(
            SimpleNamespace(layer_name=layer_name),
            key[num_computed_tokens:],
            value[num_computed_tokens:],
            cache,
            metadata.slot_mapping,
        )
        source_snapshots[layer_name] = (
            cache[0][:2].clone(),
            cache[1][:2].clone(),
        )

    assert provider.get_request_state("request").layers == {}
    plans = provider.finish_model_forward(
        view,
        layer_names=layer_names,
        schema_version=1,
    )

    assert plans is not None and len(plans) == 1
    assert plans[0].expected_block_ids == ((0, 1),)
    assert plans[0].physical_num_tokens == max(selection.retained_tokens for selection in expected.values())
    for layer_name, selection in expected.items():
        key_cache, value_cache = layer_caches[layer_name]
        source_key, source_value = source_snapshots[layer_name]
        torch.testing.assert_close(key_cache[:2], source_key)
        torch.testing.assert_close(value_cache[:2], source_value)
        retained_tokens = selection.retained_tokens
        torch.testing.assert_close(
            key_cache[2:].reshape(-1, 2, 8)[:retained_tokens],
            selection.key.squeeze(0).permute(1, 0, 2),
            rtol=1e-2,
            atol=1e-2,
        )
        torch.testing.assert_close(
            value_cache[2:].reshape(-1, 2, 8)[:retained_tokens],
            selection.value.squeeze(0).permute(1, 0, 2),
            rtol=1e-2,
            atol=1e-2,
        )


@pytest.mark.skipif(
    os.getenv("RUN_PYRAMIDKV_NPU_TEST") != "1",
    reason="set RUN_PYRAMIDKV_NPU_TEST=1 on an Ascend worker",
)
def test_npu_bf16_private_materialization_preserves_source_and_decodes() -> None:
    device = torch.device("npu")
    layer_name = "model.layers.0.self_attn.attn"
    provider = PyramidKVAscendProvider(
        _config(
            max_capacity_prompt=128,
            window_size=8,
            kernel_size=7,
            beta=20,
        )
    )
    provider.prefix_caching = True
    provider.begin_request(
        "request",
        256,
        ((0, 1),),
        prefill_num_computed_tokens=128,
    )
    torch.manual_seed(20260726)
    query = torch.randn(256, 4, 8, dtype=torch.bfloat16, device=device)
    key = torch.randn(256, 2, 8, dtype=torch.bfloat16, device=device)
    value = torch.randn(256, 2, 8, dtype=torch.bfloat16, device=device)
    expected = select_pyramid_kv(
        query.permute(1, 0, 2).unsqueeze(0),
        key.permute(1, 0, 2).unsqueeze(0),
        value.permute(1, 0, 2).unsqueeze(0),
        provider.config,
        layer_index=0,
        num_hidden_layers=2,
    )
    cache = (
        torch.zeros(4, 128, 2, 8, dtype=torch.bfloat16, device=device),
        torch.zeros(4, 128, 2, 8, dtype=torch.bfloat16, device=device),
    )
    cache[0].view(-1, 2, 8)[:128].copy_(key[:128])
    cache[1].view(-1, 2, 8)[:128].copy_(value[:128])

    def write_cache(layer, write_key, write_value, kv_cache, slots):
        indices = slots.to(dtype=torch.int64)
        kv_cache[0].view(-1, 2, 8).index_copy_(0, indices, write_key)
        kv_cache[1].view(-1, 2, 8).index_copy_(0, indices, write_value)

    backend = SimpleNamespace(do_kv_cache_update=write_cache)
    request = PyramidKVAttentionRequest(
        request_id="request",
        query_start=0,
        query_end=128,
        semantic_num_tokens=256,
        num_computed_tokens=128,
        num_prompt_tokens=256,
        block_ids=(0, 1),
        is_prefill=True,
        destination_block_ids=(2, 3),
    )
    view = PyramidKVAttentionBatchView(
        provider=provider,
        requests=(request,),
        layer_indices={layer_name: 0},
        num_hidden_layers=2,
    )
    prefill_metadata = AscendMetadata(
        attn_state=AscendAttentionState.PrefillCacheHit,
        num_actual_tokens=128,
        slot_mapping=torch.arange(128, 256, dtype=torch.int32, device=device),
        seq_lens=torch.tensor([256], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([256], dtype=torch.int32),
        seq_lens_list=[256],
    )

    assert not view.before_cache_write(
        layer=SimpleNamespace(layer_name=layer_name),
        backend=backend,
        query=query[128:],
        key=key[128:],
        value=value[128:],
        kv_cache=cache,
        attn_metadata=prefill_metadata,
    )
    write_cache(
        SimpleNamespace(layer_name=layer_name),
        key[128:],
        value[128:],
        cache,
        prefill_metadata.slot_mapping,
    )
    source_key = cache[0][:2].clone()
    source_value = cache[1][:2].clone()

    plans = provider.finish_model_forward(
        view,
        layer_names=(layer_name,),
        schema_version=1,
    )
    torch.npu.synchronize()

    assert plans is not None and len(plans) == 1
    torch.testing.assert_close(cache[0][:2], source_key)
    torch.testing.assert_close(cache[1][:2], source_value)
    retained_tokens = expected.retained_tokens
    torch.testing.assert_close(
        cache[0][2:].reshape(-1, 2, 8)[:retained_tokens],
        expected.key.squeeze(0).permute(1, 0, 2),
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        cache[1][2:].reshape(-1, 2, 8)[:retained_tokens],
        expected.value.squeeze(0).permute(1, 0, 2),
        rtol=1e-2,
        atol=1e-2,
    )

    provider.mark_committed("request", ((2, 3),))
    decode_view = PyramidKVAttentionBatchView(
        provider=provider,
        requests=(
            PyramidKVAttentionRequest(
                request_id="request",
                query_start=0,
                query_end=1,
                semantic_num_tokens=257,
                num_computed_tokens=256,
                num_prompt_tokens=256,
                block_ids=(2, 3),
                is_prefill=False,
            ),
        ),
        layer_indices={layer_name: 0},
        num_hidden_layers=2,
    )
    decode_metadata = AscendMetadata(
        attn_state=AscendAttentionState.DecodeOnly,
        num_actual_tokens=1,
        slot_mapping=torch.tensor([-1], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([257], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([257], dtype=torch.int32),
        seq_lens_list=[257],
    )
    assert decode_view.before_cache_write(
        layer=SimpleNamespace(layer_name=layer_name),
        backend=backend,
        query=query[:1],
        key=key[:1],
        value=value[:1],
        kv_cache=cache,
        attn_metadata=decode_metadata,
    )
    assert (
        provider.finish_model_forward(
            decode_view,
            layer_names=(layer_name,),
            schema_version=1,
        )
        is None
    )
    torch.npu.synchronize()


def test_prefix_hit_view_initializes_at_aligned_offset_and_requires_destination() -> None:
    provider = PyramidKVAscendProvider(PyramidKVAscendConfig.from_dict({}))
    provider.prefix_caching = True
    common = {
        "request_ids": ("request",),
        "query_lengths": (129,),
        "semantic_num_tokens": (769,),
        "num_computed_tokens": (640,),
        "num_prompt_tokens": (769,),
        "block_ids": ((tuple(range(7)),),),
        "layer_names": LAYER_NAMES,
        "block_size": 128,
    }

    with pytest.raises(RuntimeError, match="missing its private destination"):
        provider.build_attention_batch_view(**common)

    view = provider.build_attention_batch_view(
        **common,
        destination_block_ids={"request": (tuple(range(7, 15)),)},
    )

    state = provider.get_request_state("request")
    assert view.requests[0].num_computed_tokens == 640
    assert view.requests[0].destination_block_ids == tuple(range(7, 15))
    assert state.initial_prefill_num_computed_tokens == 640
    assert state.prefill_num_computed_tokens == 640


@pytest.mark.parametrize(
    ("computed", "query_length", "destination", "error_match"),
    [
        (641, 128, tuple(range(7, 15)), "unaligned cached offset"),
        (768, 1, tuple(range(7, 15)), "fewer than query window"),
        (640, 129, (0, 7, 8, 9), "source and destination.*overlap"),
        (640, 129, (7,), "capacity 128 is smaller than required 512"),
    ],
)
def test_prefix_hit_view_rejects_invalid_cached_or_destination_state(
    computed: int,
    query_length: int,
    destination: tuple[int, ...],
    error_match: str,
) -> None:
    provider = PyramidKVAscendProvider(PyramidKVAscendConfig.from_dict({}))
    provider.prefix_caching = True

    with pytest.raises(RuntimeError, match=error_match):
        provider.build_attention_batch_view(
            request_ids=("request",),
            query_lengths=(query_length,),
            semantic_num_tokens=(769,),
            num_computed_tokens=(computed,),
            num_prompt_tokens=(769,),
            block_ids=((tuple(range(7)),),),
            layer_names=LAYER_NAMES,
            block_size=128,
            destination_block_ids={"request": (destination,)},
        )


@pytest.mark.parametrize("prompt_tokens", [256, 4096, 7168])
def test_planned_prompt_lengths_produce_valid_compact_shapes(
    prompt_tokens: int,
) -> None:
    generator = torch.Generator().manual_seed(prompt_tokens)
    query = torch.randn(1, 4, prompt_tokens, 8, generator=generator)
    key = torch.randn(1, 2, prompt_tokens, 8, generator=generator)
    value = torch.randn(1, 2, prompt_tokens, 8, generator=generator)
    config = _config(max_capacity_prompt=128)

    selection = select_pyramid_kv(
        query,
        key,
        value,
        config,
        layer_index=31,
        num_hidden_layers=32,
    )

    assert selection.compressed
    assert selection.retained_tokens < prompt_tokens
    assert selection.key.shape == (1, 2, selection.retained_tokens, 8)
    assert selection.value.shape == selection.key.shape
    assert selection.selected_past_indices is not None


def test_capability_report_aggregates_all_reasons() -> None:
    provider = PyramidKVAscendProvider(_config())
    invalid = replace(
        _context(),
        device_name="Ascend910A",
        backend="OtherBackend",
        quantization="w8a8",
        num_kv_heads=4,
        prefix_caching=True,
        tensor_parallel_size=2,
        dbo_enabled=True,
        knorm_enabled=True,
        missing_ops=("reshape_and_cache", "fused_infer_attention"),
    )

    report = provider.compatibility_report(_core_config(), invalid, "registry:get")

    assert not report.supported
    message = "\n".join(report.reasons)
    assert "Ascend910A" in message
    assert "OtherBackend" in message
    assert "model quantization" in message
    assert "KV heads must be 8, got 4" in message
    assert "tensor parallel size" in message
    assert "dual-batch overlap" in message
    assert "VLLM_KNORM_ENABLED" in message
    assert "reshape_and_cache" in message
    assert "fused_infer_attention" in message


def test_supported_capability_has_no_reasons() -> None:
    provider = PyramidKVAscendProvider(_config())

    report = provider.compatibility_report(_core_config(), _context(), "registry:get")

    assert report.supported
    assert report.reasons == ()
    assert report.runtime_spec is not None
    assert report.runtime_spec.compression_threshold_tokens == 12
    assert report.runtime_spec.required_recompute_tokens == 4
    assert report.runtime_spec.max_physical_num_tokens == 16


def test_async_scheduling_is_supported_without_relaxing_other_guards() -> None:
    provider = PyramidKVAscendProvider(_config())

    supported = provider.compatibility_report(
        _core_config(),
        _context(async_scheduling=True),
        "registry:get",
    )
    rejected = provider.compatibility_report(
        _core_config(),
        _context(async_scheduling=True, balance_scheduling=True),
        "registry:get",
    )

    assert supported.supported
    assert supported.reasons == ()
    assert not rejected.supported
    assert rejected.reasons == ("balance scheduling is unsupported",)


def test_prefix_caching_requires_matching_128_token_hash_blocks() -> None:
    provider = PyramidKVAscendProvider(_config())

    supported = provider.compatibility_report(
        _core_config(),
        _context(prefix_caching=True),
        "registry:get",
    )
    rejected = provider.compatibility_report(
        _core_config(),
        _context(prefix_caching=True, hash_block_size=64),
        "registry:get",
    )

    assert supported.supported
    assert supported.runtime_spec is not None
    assert supported.runtime_spec.requires_private_destination
    assert not rejected.supported
    assert rejected.runtime_spec is None
    assert "hash_block_size == block_size == 128" in rejected.reasons[0]


def test_balance_scheduling_is_rejected() -> None:
    report = PyramidKVAscendProvider(_config()).compatibility_report(
        _core_config(),
        _context(balance_scheduling=True),
        "registry:get",
    )

    assert not report.supported
    assert report.runtime_spec is None
    assert "balance scheduling is unsupported" in report.reasons


def test_default_runtime_spec_reserves_eight_private_blocks() -> None:
    provider = PyramidKVAscendProvider(PyramidKVAscendConfig.from_dict({}))
    core_config = KVCacheCompressionConfig(
        provider="pyramidkv_ascend",
        provider_config={},
    )

    report = provider.compatibility_report(
        core_config,
        _context(prefix_caching=True, max_model_len=8192),
        "registry:get",
    )

    assert report.supported
    assert report.runtime_spec is not None
    assert report.runtime_spec.compression_threshold_tokens == 512
    assert report.runtime_spec.required_recompute_tokens == 8
    assert report.runtime_spec.max_physical_num_tokens == 991
    assert math.ceil(report.runtime_spec.max_physical_num_tokens / 128) == 8


def test_cann_9_capability_is_supported() -> None:
    provider = PyramidKVAscendProvider(_config())

    report = provider.compatibility_report(_core_config(), _context(cann_version="9.0.0"), "registry:get")

    assert report.supported
    assert report.reasons == ()


@pytest.mark.parametrize(
    ("cann_version", "supported"),
    [("8.5.1", False), ("9.0.0", True)],
)
def test_chunked_prefill_requires_cann_9(cann_version: str, supported: bool) -> None:
    provider = PyramidKVAscendProvider(_config())

    report = provider.compatibility_report(
        _core_config(),
        _context(
            cann_version=cann_version,
            chunked_prefill=True,
        ),
        "registry:get",
    )

    assert report.supported is supported
    if not supported:
        assert report.reasons == ("chunked prefill requires CANN 9.0, got '8.5.1'",)


def test_chunked_prefill_rejects_page_attention_shapes() -> None:
    provider = PyramidKVAscendProvider(_config())

    report = provider.compatibility_report(
        _core_config(),
        _context(
            cann_version="9.0.0",
            chunked_prefill=True,
            pa_shape_list=(16,),
        ),
        "registry:get",
    )

    assert not report.supported
    assert report.reasons == ("chunked prefill currently requires default FIA with an empty pa_shape_list, got (16,)",)


@pytest.mark.parametrize("cudagraph_mode", ["PIECEWISE", "FULL_DECODE_ONLY"])
def test_cann_9_graph_capability_is_supported(cudagraph_mode: str) -> None:
    provider = PyramidKVAscendProvider(_config())

    report = provider.compatibility_report(
        _core_config(),
        _context(
            cann_version="9.0.0",
            enforce_eager=False,
            cudagraph_mode=cudagraph_mode,
        ),
        "registry:get",
    )

    assert report.supported
    assert report.reasons == ()


def test_cann_8_graph_capability_is_rejected() -> None:
    provider = PyramidKVAscendProvider(_config())

    report = provider.compatibility_report(
        _core_config(),
        _context(enforce_eager=False, cudagraph_mode="PIECEWISE"),
        "registry:get",
    )

    assert not report.supported
    assert report.reasons == ("graph execution requires CANN 9.0, got '8.5.1'",)


@pytest.mark.parametrize(
    ("enforce_eager", "cudagraph_mode"),
    [(False, "NONE"), (True, "PIECEWISE"), (False, "FULL_AND_PIECEWISE")],
)
def test_conflicting_or_unsupported_graph_mode_is_rejected(enforce_eager: bool, cudagraph_mode: str) -> None:
    provider = PyramidKVAscendProvider(_config())

    report = provider.compatibility_report(
        _core_config(),
        _context(
            cann_version="9.0.0",
            enforce_eager=enforce_eager,
            cudagraph_mode=cudagraph_mode,
        ),
        "registry:get",
    )

    assert not report.supported
    assert "execution must be eager/NONE" in report.reasons[0]


def test_graph_page_attention_is_rejected() -> None:
    provider = PyramidKVAscendProvider(_config())

    report = provider.compatibility_report(
        _core_config(),
        _context(
            cann_version="9.0.0",
            enforce_eager=False,
            cudagraph_mode="FULL_DECODE_ONLY",
            pa_shape_list=(16,),
        ),
        "registry:get",
    )

    assert not report.supported
    assert report.reasons == ("graph execution currently requires default FIA with an empty pa_shape_list, got (16,)",)


def test_unsupported_cann_version_is_rejected() -> None:
    provider = PyramidKVAscendProvider(_config())

    report = provider.compatibility_report(_core_config(), _context(cann_version="9.1.0"), "registry:get")

    assert not report.supported
    assert report.reasons == ("CANN version must start with one of ('8.5.1', '9.0'), got '9.1.0'",)


def _selection(retained_tokens: int, index: int) -> PyramidKVSelection:
    tensor = torch.tensor([[[index]]])
    return PyramidKVSelection(
        key=torch.empty(0),
        value=torch.empty(0),
        selected_past_indices=tensor,
        retained_tokens=retained_tokens,
        compressed=True,
    )


def test_request_states_are_isolated_finalize_commit_decode_and_cleanup() -> None:
    provider = PyramidKVAscendProvider(_config())
    provider.begin_request("a", 20, ((1, 2),))
    provider.begin_request("b", 20, ((3, 4),))
    provider.record_prefill_layer("a", "layer0", _selection(16, 1))
    provider.record_prefill_layer("a", "layer1", _selection(8, 2))
    provider.record_prefill_layer("b", "layer0", _selection(12, 3))
    provider.record_prefill_layer("b", "layer1", _selection(10, 4))

    plan_a = provider.finalize_plan("a", ("layer0", "layer1"), 1)
    plan_b = provider.finalize_plan("b", ("layer0", "layer1"), 1)

    assert plan_a.expected_block_ids == ((1, 2),)
    assert plan_a.physical_num_tokens == 16
    assert plan_b.expected_block_ids == ((3, 4),)
    assert plan_b.physical_num_tokens == 12

    provider.mark_committed("a", ((1,),))
    provider.advance_decode("a", {"layer0", "layer1"})
    state_a = provider.get_request_state("a")
    state_b = provider.get_request_state("b")
    assert state_a.layers["layer0"].physical_num_tokens == 17
    assert state_a.layers["layer1"].physical_num_tokens == 9
    assert state_a.semantic_num_tokens == 21
    assert state_b.layers["layer0"].physical_num_tokens == 12
    assert not state_b.committed

    provider.cleanup_request("a")
    with pytest.raises(KeyError):
        provider.get_request_state("a")
    assert provider.get_request_state("b") is state_b


def test_partial_layer_or_partial_decode_cannot_advance_state() -> None:
    provider = PyramidKVAscendProvider(_config())
    provider.begin_request("request", 20, ((1, 2),))
    provider.record_prefill_layer("request", "layer0", _selection(16, 1))

    with pytest.raises(RuntimeError, match="incomplete"):
        provider.finalize_plan("request", ("layer0", "layer1"), 1)

    provider.record_prefill_layer("request", "layer1", _selection(8, 2))
    provider.finalize_plan("request", ("layer0", "layer1"), 1)
    provider.mark_committed("request", ((1,),))
    before = provider.get_request_state("request").layers["layer0"].physical_num_tokens

    with pytest.raises(RuntimeError, match="every layer"):
        provider.advance_decode("request", {"layer0"})

    assert provider.get_request_state("request").layers["layer0"].physical_num_tokens == before


def _fake_attention_backend():
    calls = []
    backend = SimpleNamespace(
        enable_hamming_sparse=False,
        _use_layer_aware_fia_graph_replay=False,
        key_cache=None,
        value_cache=None,
    )

    def reshape_and_cache(query, key, value, kv_cache, metadata, output):
        calls.append("default_cache_write")
        return query, key, value, output

    def forward_impl(query, key, value, kv_cache, metadata, output):
        calls.append("attention")
        return output.fill_(1)

    backend.reshape_and_cache = reshape_and_cache
    backend.forward_impl = forward_impl
    return backend, calls


def test_dense_attention_default_path_keeps_original_cache_write() -> None:
    backend, calls = _fake_attention_backend()
    layer = SimpleNamespace(
        layer_name="model.layers.0.self_attn.attn",
        _k_scale_float=1.0,
        _v_scale_float=1.0,
    )
    query = torch.zeros(2, 4, 8)
    key = torch.zeros(2, 2, 8)
    value = torch.zeros_like(key)
    output = torch.zeros_like(query)
    metadata = AscendMetadata(
        num_actual_tokens=2,
        kv_cache_compression_view=None,
    )

    result = AscendAttentionBackendImpl.forward(
        backend,
        layer,
        query,
        key,
        value,
        (torch.empty(1), torch.empty(1)),
        metadata,
        output,
    )

    assert result is output
    assert calls == ["default_cache_write", "attention"]


def test_dense_attention_enabled_view_replaces_only_default_cache_write() -> None:
    backend, calls = _fake_attention_backend()
    layer = SimpleNamespace(
        layer_name="model.layers.0.self_attn.attn",
        _k_scale_float=1.0,
        _v_scale_float=1.0,
    )

    class FakeView:
        def before_cache_write(self, **kwargs):
            calls.append(("provider_cache_write", kwargs["layer"].layer_name))
            return True

    metadata = AscendMetadata(
        num_actual_tokens=2,
        kv_cache_compression_view=FakeView(),
    )
    query = torch.zeros(2, 4, 8)
    key = torch.zeros(2, 2, 8)
    output = torch.zeros_like(query)

    AscendAttentionBackendImpl.forward(
        backend,
        layer,
        query,
        key,
        torch.zeros_like(key),
        (torch.empty(1), torch.empty(1)),
        metadata,
        output,
    )

    assert calls == [
        ("provider_cache_write", layer.layer_name),
        "attention",
    ]


def test_attention_view_writes_compact_prefill_and_physical_decode_slot() -> None:
    provider = PyramidKVAscendProvider(_config())
    provider.begin_request("request", 20, ((1, 2),))
    layer_name = "model.layers.1.self_attn.attn"
    prefill_view = PyramidKVAttentionBatchView(
        provider=provider,
        requests=(
            PyramidKVAttentionRequest(
                request_id="request",
                query_start=0,
                query_end=20,
                semantic_num_tokens=20,
                num_computed_tokens=0,
                num_prompt_tokens=20,
                block_ids=(1, 2),
                is_prefill=True,
            ),
        ),
        layer_indices={layer_name: 1},
        num_hidden_layers=2,
    )
    generator = torch.Generator().manual_seed(987)
    query = torch.randn(20, 4, 8, generator=generator)
    key = torch.randn(20, 2, 8, generator=generator)
    value = torch.randn(20, 2, 8, generator=generator)
    writes = []
    backend = SimpleNamespace(
        do_kv_cache_update=lambda layer, write_key, write_value, cache, slots: writes.append(
            (write_key, write_value, slots)
        )
    )
    layer = SimpleNamespace(layer_name=layer_name)
    prefill_metadata = AscendMetadata(
        attn_state=AscendAttentionState.PrefillNoCache,
        num_actual_tokens=20,
        slot_mapping=torch.arange(20, dtype=torch.int32),
        seq_lens=torch.tensor([20], dtype=torch.int32),
        seq_lens_list=[20],
    )

    assert prefill_view.before_cache_write(
        layer=layer,
        backend=backend,
        query=query,
        key=key,
        value=value,
        kv_cache=(torch.empty(1), torch.empty(1)),
        attn_metadata=prefill_metadata,
    )
    assert writes[0][0].shape == (8, 2, 8)
    assert writes[0][1].shape == (8, 2, 8)
    assert torch.equal(writes[0][2], torch.arange(128, 136, dtype=torch.int32))

    plans = provider.finish_model_forward(prefill_view, layer_names=(layer_name,), schema_version=1)
    assert plans is not None
    plan = plans[0]
    assert plan.physical_num_tokens == 8
    provider.mark_committed("request", ((1,),))
    decode_view = PyramidKVAttentionBatchView(
        provider=provider,
        requests=(
            PyramidKVAttentionRequest(
                request_id="request",
                query_start=0,
                query_end=1,
                semantic_num_tokens=21,
                num_computed_tokens=20,
                num_prompt_tokens=20,
                block_ids=(1,),
                is_prefill=False,
            ),
        ),
        layer_indices={layer_name: 1},
        num_hidden_layers=2,
    )
    decode_metadata = AscendMetadata(
        attn_state=AscendAttentionState.DecodeOnly,
        num_actual_tokens=1,
        slot_mapping=torch.tensor([999], dtype=torch.int32),
        seq_lens=torch.tensor([21], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([21], dtype=torch.int32),
        seq_lens_list=[21],
    )

    decode_view.before_cache_write(
        layer=layer,
        backend=backend,
        query=query[:1],
        key=key[:1],
        value=value[:1],
        kv_cache=(torch.empty(1), torch.empty(1)),
        attn_metadata=decode_metadata,
    )

    assert torch.equal(writes[1][2], torch.tensor([136], dtype=torch.int32))
    assert decode_metadata.seq_lens_list == [9]
    assert decode_metadata.seq_lens.tolist() == [9]


def test_decode_slot_matrix_batches_requests_and_materializes_once() -> None:
    provider = PyramidKVAscendProvider(_config())
    layer_names = (
        "model.layers.0.self_attn.attn",
        "model.layers.1.self_attn.attn",
    )
    for request_id, block_id, lengths in (
        ("a", 1, (10, 8)),
        ("b", 2, (12, 9)),
    ):
        provider.begin_request(request_id, 20, ((block_id,),))
        for layer_name, retained_tokens in zip(layer_names, lengths):
            provider.record_prefill_layer(
                request_id,
                layer_name,
                _selection(retained_tokens, 1),
            )
        provider.finalize_plan(request_id, layer_names, 1)
        provider.mark_committed(request_id, ((block_id,),))

    view = PyramidKVAttentionBatchView(
        provider=provider,
        requests=(
            PyramidKVAttentionRequest(
                request_id="a",
                query_start=0,
                query_end=1,
                semantic_num_tokens=21,
                num_computed_tokens=20,
                num_prompt_tokens=20,
                block_ids=(1,),
                is_prefill=False,
            ),
            PyramidKVAttentionRequest(
                request_id="b",
                query_start=1,
                query_end=2,
                semantic_num_tokens=21,
                num_computed_tokens=20,
                num_prompt_tokens=20,
                block_ids=(2,),
                is_prefill=False,
            ),
        ),
        layer_indices={name: index for index, name in enumerate(layer_names)},
        num_hidden_layers=2,
    )
    query = torch.randn(2, 4, 8)
    key = torch.randn(2, 2, 8)
    value = torch.randn(2, 2, 8)
    writes = []
    backend = SimpleNamespace(
        do_kv_cache_update=lambda layer, write_key, write_value, cache, slots: writes.append(
            (write_key, write_value, slots)
        )
    )
    metadata = AscendMetadata(
        attn_state=AscendAttentionState.DecodeOnly,
        num_actual_tokens=2,
        slot_mapping=torch.tensor([999, 999], dtype=torch.int32),
        seq_lens=torch.tensor([21, 21], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([21, 21], dtype=torch.int32),
        seq_lens_list=[21, 21],
    )
    decode_lengths = []

    for layer_name in layer_names:
        view.before_cache_write(
            layer=SimpleNamespace(layer_name=layer_name),
            backend=backend,
            query=query,
            key=key,
            value=value,
            kv_cache=(torch.empty(1), torch.empty(1)),
            attn_metadata=metadata,
        )
        decode_lengths.append((tuple(metadata.seq_lens_list), metadata.seq_lens))

    assert writes[0][0] is key
    assert writes[0][1] is value
    assert torch.equal(writes[0][2], torch.tensor([138, 268], dtype=torch.int32))
    assert torch.equal(writes[1][2], torch.tensor([136, 265], dtype=torch.int32))
    assert writes[0][2].untyped_storage().data_ptr() == writes[1][2].untyped_storage().data_ptr()
    assert decode_lengths[0][0] == (11, 13)
    assert decode_lengths[1][0] == (9, 10)
    assert torch.equal(decode_lengths[0][1], torch.tensor([11, 13], dtype=torch.int32))
    assert torch.equal(decode_lengths[1][1], torch.tensor([9, 10], dtype=torch.int32))
    assert decode_lengths[0][1].untyped_storage().data_ptr() == decode_lengths[1][1].untyped_storage().data_ptr()
    assert writes[0][2].untyped_storage().data_ptr() != decode_lengths[0][1].untyped_storage().data_ptr()
    assert decode_lengths[0][1].device.type == "cpu"
    assert metadata.seq_lens_cpu is metadata.seq_lens
    assert len(view.decode_slot_tensors_by_layer) == 2
    assert len(view.decode_length_tensors_by_layer) == 2
    assert view.completed_decode_layers == set(layer_names)

    assert provider.finish_model_forward(view, layer_names=layer_names, schema_version=1) is None
    assert provider.get_request_state("a").semantic_num_tokens == 21
    assert provider.get_request_state("b").semantic_num_tokens == 21


def test_full_decode_staging_is_layer_specific_and_padding_is_safe() -> None:
    provider = PyramidKVAscendProvider(_config())
    layer_names = (
        "model.layers.0.self_attn.attn",
        "model.layers.1.self_attn.attn",
    )
    provider.begin_request("compressed", 20, ((1,),))
    provider.record_prefill_layer("compressed", layer_names[0], _selection(10, 1))
    provider.record_prefill_layer("compressed", layer_names[1], _selection(8, 2))
    provider.finalize_plan("compressed", layer_names, 1)
    provider.mark_committed("compressed", ((1,),))
    view = PyramidKVAttentionBatchView(
        provider=provider,
        requests=(
            PyramidKVAttentionRequest(
                request_id="compressed",
                query_start=0,
                query_end=1,
                semantic_num_tokens=21,
                num_computed_tokens=20,
                num_prompt_tokens=20,
                block_ids=(1,),
                is_prefill=False,
            ),
            PyramidKVAttentionRequest(
                request_id="ordinary",
                query_start=1,
                query_end=2,
                semantic_num_tokens=21,
                num_computed_tokens=20,
                num_prompt_tokens=20,
                block_ids=(3,),
                is_prefill=False,
                compress=False,
            ),
        ),
        layer_indices={name: index for index, name in enumerate(layer_names)},
        num_hidden_layers=2,
    )
    slots = torch.empty((2, 4), dtype=torch.int32)
    lengths = torch.empty((2, 4), dtype=torch.int32)

    view.fill_full_decode_metadata(
        layer_names=layer_names,
        slot_staging=slots,
        length_staging=lengths,
        num_reqs_padded=4,
    )

    assert torch.equal(
        slots,
        torch.tensor(
            [[138, 404, -1, -1], [136, 404, -1, -1]],
            dtype=torch.int32,
        ),
    )
    assert torch.equal(
        lengths,
        torch.tensor([[11, 21, 1, 1], [9, 21, 1, 1]], dtype=torch.int32),
    )
    assert view.completed_decode_layers == set()
    with pytest.raises(RuntimeError, match="expected all 32 layers"):
        provider.finish_model_forward(view, layer_names=layer_names, schema_version=1)
    assert provider.get_request_state("compressed").semantic_num_tokens == 20


def test_mixed_attention_defers_prefill_compact_until_model_finishes() -> None:
    provider = PyramidKVAscendProvider(_config())
    layer_name = "model.layers.1.self_attn.attn"
    provider.begin_request("decode", 20, ((1, 2),))
    provider.record_prefill_layer("decode", layer_name, _selection(8, 1))
    provider.finalize_plan("decode", (layer_name,), 1)
    provider.mark_committed("decode", ((1,),))
    provider.begin_request("prefill", 20, ((2, 3),))

    view = PyramidKVAttentionBatchView(
        provider=provider,
        requests=(
            PyramidKVAttentionRequest(
                request_id="decode",
                query_start=0,
                query_end=1,
                semantic_num_tokens=21,
                num_computed_tokens=20,
                num_prompt_tokens=20,
                block_ids=(1,),
                is_prefill=False,
            ),
            PyramidKVAttentionRequest(
                request_id="prefill",
                query_start=1,
                query_end=21,
                semantic_num_tokens=20,
                num_computed_tokens=0,
                num_prompt_tokens=20,
                block_ids=(2, 3),
                is_prefill=True,
            ),
        ),
        layer_indices={layer_name: 1},
        num_hidden_layers=2,
    )
    generator = torch.Generator().manual_seed(2468)
    query = torch.randn(21, 4, 8, generator=generator)
    key = torch.randn(21, 2, 8, generator=generator)
    value = torch.randn(21, 2, 8, generator=generator)
    expected = select_pyramid_kv(
        query[1:].permute(1, 0, 2).unsqueeze(0),
        key[1:].permute(1, 0, 2).unsqueeze(0),
        value[1:].permute(1, 0, 2).unsqueeze(0),
        provider.config,
        layer_index=1,
        num_hidden_layers=2,
    )
    cache = (
        torch.zeros(4, 128, 2, 8),
        torch.zeros(4, 128, 2, 8),
    )

    def write_cache(layer, write_key, write_value, kv_cache, slots):
        kv_cache[0].view(-1, 2, 8)[slots.long()] = write_key
        kv_cache[1].view(-1, 2, 8)[slots.long()] = write_value

    backend = SimpleNamespace(do_kv_cache_update=write_cache)
    metadata = AscendMetadata(
        attn_state=AscendAttentionState.PrefillCacheHit,
        num_actual_tokens=21,
        slot_mapping=torch.cat(
            (
                torch.tensor([999], dtype=torch.int32),
                torch.arange(256, 276, dtype=torch.int32),
            )
        ),
        seq_lens=torch.tensor([21, 20], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([21, 20], dtype=torch.int32),
        seq_lens_list=[21, 20],
    )

    assert view.before_cache_write(
        layer=SimpleNamespace(layer_name=layer_name),
        backend=backend,
        query=query,
        key=key,
        value=value,
        kv_cache=cache,
        attn_metadata=metadata,
    )
    assert metadata.seq_lens_list == [9, 20]
    assert len(view.deferred_prefills) == 1
    assert provider.get_request_state("prefill").layers == {}
    assert torch.equal(cache[0][2, :20], key[1:])
    assert torch.equal(cache[1][2, :20], value[1:])

    plans = provider.finish_model_forward(view, layer_names=(layer_name,), schema_version=1)

    assert plans is not None and plans[0].request_id == "prefill"
    assert plans[0].physical_num_tokens == 8
    assert not view.deferred_prefills
    assert torch.equal(cache[0][2, :8], expected.key.squeeze(0).permute(1, 0, 2))
    assert torch.equal(cache[1][2, :8], expected.value.squeeze(0).permute(1, 0, 2))
    assert torch.equal(cache[0][1, 8], key[0])
    assert provider.get_request_state("decode").semantic_num_tokens == 21


def test_runner_batch_view_prefill_plan_commit_and_decode_lifecycle() -> None:
    provider = PyramidKVAscendProvider(_config())
    view = provider.build_attention_batch_view(
        request_ids=("request",),
        query_lengths=(20,),
        semantic_num_tokens=(20,),
        num_computed_tokens=(0,),
        num_prompt_tokens=(20,),
        block_ids=(((1, 2),),),
        layer_names=LAYER_NAMES,
        block_size=128,
    )
    assert view.requests[0].compress
    assert view.requests[0].query_start == 0
    assert view.requests[0].query_end == 20

    view.completed_prefill_lengths["request"] = {
        layer_name: 8 if index else 12 for index, layer_name in enumerate(LAYER_NAMES)
    }
    plans = provider.finish_model_forward(view, layer_names=LAYER_NAMES, schema_version=1)
    assert plans is not None
    assert len(plans) == 1
    assert plans[0].physical_num_tokens == 12

    provider.mark_committed("request", ((1,),))
    decode_view = provider.build_attention_batch_view(
        request_ids=("request",),
        query_lengths=(1,),
        semantic_num_tokens=(21,),
        num_computed_tokens=(20,),
        num_prompt_tokens=(20,),
        block_ids=(((1,),),),
        layer_names=LAYER_NAMES,
        block_size=128,
    )
    decode_view.completed_decode_layers.update(LAYER_NAMES)
    assert provider.finish_model_forward(decode_view, layer_names=LAYER_NAMES, schema_version=1) is None
    state = provider.get_request_state("request")
    assert state.semantic_num_tokens == 21
    assert state.layers[LAYER_NAMES[0]].physical_num_tokens == 13
    assert state.layers[LAYER_NAMES[1]].physical_num_tokens == 9


def test_chunked_prefill_rejects_duplicate_gap_and_nonmonotonic_blocks() -> None:
    provider = PyramidKVAscendProvider(_config())
    first = provider.build_attention_batch_view(
        request_ids=("request",),
        query_lengths=(256,),
        semantic_num_tokens=(256,),
        num_computed_tokens=(0,),
        num_prompt_tokens=(768,),
        block_ids=(((0, 1),),),
        layer_names=LAYER_NAMES,
        block_size=128,
    )
    assert first.requests[0].is_prefill
    state = provider.get_request_state("request")
    state.prefill_num_computed_tokens = 256
    state.prefill_query_tail = torch.zeros(32, 4, 4, 8)
    state.prefill_query_tail_length = 4

    common = {
        "request_ids": ("request",),
        "num_prompt_tokens": (768,),
        "layer_names": LAYER_NAMES,
        "block_size": 128,
    }
    with pytest.raises(RuntimeError, match="chunk starts at 0, expected 256"):
        provider.build_attention_batch_view(
            query_lengths=(256,),
            semantic_num_tokens=(256,),
            num_computed_tokens=(0,),
            block_ids=(((0, 1),),),
            **common,
        )
    with pytest.raises(RuntimeError, match="chunk starts at 384, expected 256"):
        provider.build_attention_batch_view(
            query_lengths=(128,),
            semantic_num_tokens=(512,),
            num_computed_tokens=(384,),
            block_ids=(((0, 1, 2, 3),),),
            **common,
        )
    with pytest.raises(RuntimeError, match="not a monotonic extension"):
        provider.build_attention_batch_view(
            query_lengths=(256,),
            semantic_num_tokens=(512,),
            num_computed_tokens=(256,),
            block_ids=(((9, 1, 2, 3),),),
            **common,
        )

    second = provider.build_attention_batch_view(
        query_lengths=(256,),
        semantic_num_tokens=(512,),
        num_computed_tokens=(256,),
        block_ids=(((0, 1, 2, 3),),),
        **common,
    )
    assert second.requests[0].num_computed_tokens == 256
    assert state.prefill_num_computed_tokens == 256


def test_missing_layer_or_failed_forward_does_not_commit_chunk_state() -> None:
    provider = PyramidKVAscendProvider(_config())
    layer_names = (
        "model.layers.0.self_attn.attn",
        "model.layers.1.self_attn.attn",
    )
    provider.begin_request("request", 20, ((0,),))
    request = PyramidKVAttentionRequest(
        request_id="request",
        query_start=0,
        query_end=10,
        semantic_num_tokens=10,
        num_computed_tokens=0,
        num_prompt_tokens=20,
        block_ids=(0,),
        is_prefill=True,
    )
    query = torch.randn(10, 4, 8)
    key = torch.randn(10, 2, 8)
    value = torch.randn(10, 2, 8)
    cache = (torch.zeros(1, 128, 2, 8), torch.zeros(1, 128, 2, 8))

    def write_cache(layer, write_key, write_value, kv_cache, slots):
        kv_cache[0].view(-1, 2, 8)[slots.long()] = write_key
        kv_cache[1].view(-1, 2, 8)[slots.long()] = write_value

    backend = SimpleNamespace(do_kv_cache_update=write_cache)

    def new_view() -> PyramidKVAttentionBatchView:
        return PyramidKVAttentionBatchView(
            provider=provider,
            requests=(request,),
            layer_indices={name: index for index, name in enumerate(layer_names)},
            num_hidden_layers=2,
        )

    metadata = AscendMetadata(
        attn_state=AscendAttentionState.ChunkedPrefill,
        num_actual_tokens=10,
        slot_mapping=torch.arange(10, dtype=torch.int32),
        seq_lens=torch.tensor([10], dtype=torch.int32),
        seq_lens_list=[10],
    )
    incomplete = new_view()
    incomplete.before_cache_write(
        layer=SimpleNamespace(layer_name=layer_names[0]),
        backend=backend,
        query=query,
        key=key,
        value=value,
        kv_cache=cache,
        attn_metadata=metadata,
    )
    with pytest.raises(RuntimeError, match="expected all 32 layers"):
        provider.finish_model_forward(incomplete, layer_names=layer_names, schema_version=1)
    state = provider.get_request_state("request")
    assert state.prefill_num_computed_tokens == 0
    assert state.prefill_query_tail is None

    # A forward exception means finish_model_forward is not invoked at all.
    # The same scheduler range can therefore be retried from committed state.
    retry = new_view()
    for layer_name in layer_names:
        retry.before_cache_write(
            layer=SimpleNamespace(layer_name=layer_name),
            backend=backend,
            query=query,
            key=key,
            value=value,
            kv_cache=cache,
            attn_metadata=metadata,
        )
    assert provider.finish_model_forward(retry, layer_names=layer_names, schema_version=1) is None
    assert state.prefill_num_computed_tokens == 10
    assert state.prefill_query_tail is not None


def test_decode_view_accepts_only_required_monotonic_block_extension() -> None:
    provider = PyramidKVAscendProvider(_config())
    provider.begin_request("request", 128, ((1, 2),))
    for index, layer_name in enumerate(LAYER_NAMES):
        provider.record_prefill_layer("request", layer_name, _selection(128, index + 1))
    provider.finalize_plan("request", LAYER_NAMES, 1)
    provider.mark_committed("request", ((1,),))

    view = provider.build_attention_batch_view(
        request_ids=("request",),
        query_lengths=(1,),
        semantic_num_tokens=(129,),
        num_computed_tokens=(128,),
        num_prompt_tokens=(128,),
        block_ids=(((1, 7),),),
        layer_names=LAYER_NAMES,
        block_size=128,
    )

    assert view.requests[0].block_ids == (1, 7)
    assert provider.get_request_state("request").expected_block_ids == ((1, 7),)

    provider.get_request_state("request").expected_block_ids = ((1,),)
    with pytest.raises(RuntimeError, match="required monotonic extension"):
        provider.build_attention_batch_view(
            request_ids=("request",),
            query_lengths=(1,),
            semantic_num_tokens=(129,),
            num_computed_tokens=(128,),
            num_prompt_tokens=(128,),
            block_ids=(((9, 7),),),
            layer_names=LAYER_NAMES,
            block_size=128,
        )


def test_runner_batch_view_leaves_below_threshold_request_unadapted() -> None:
    provider = PyramidKVAscendProvider(_config())
    view = provider.build_attention_batch_view(
        request_ids=("short",),
        query_lengths=(8,),
        semantic_num_tokens=(8,),
        num_computed_tokens=(0,),
        num_prompt_tokens=(8,),
        block_ids=(((7,),),),
        layer_names=LAYER_NAMES,
        block_size=128,
    )
    assert not view.requests[0].compress
    with pytest.raises(KeyError):
        provider.get_request_state("short")


def test_runner_batch_view_supports_committed_decode_and_new_prefill() -> None:
    provider = PyramidKVAscendProvider(_config())
    provider.begin_request("decode", 8, ((3,),))
    for index, layer_name in enumerate(LAYER_NAMES):
        provider.record_prefill_layer("decode", layer_name, _selection(8, index + 1))
    provider.finalize_plan("decode", LAYER_NAMES, 1)
    provider.mark_committed("decode", ((3,),))

    view = provider.build_attention_batch_view(
        request_ids=("decode", "prefill"),
        query_lengths=(1, 20),
        semantic_num_tokens=(9, 20),
        num_computed_tokens=(8, 0),
        num_prompt_tokens=(8, 20),
        block_ids=(((3,),), ((1, 2),)),
        layer_names=LAYER_NAMES,
        block_size=128,
    )

    assert [request.is_prefill for request in view.requests] == [False, True]
    assert all(request.compress for request in view.requests)
    assert provider.get_request_state("prefill").semantic_num_tokens == 20
