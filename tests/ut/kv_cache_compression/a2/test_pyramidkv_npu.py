# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from vllm.config import CUDAGraphMode, KVCacheCompressionConfig
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.v1.attention.backend import AttentionType

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackendImpl,
    AscendAttentionState,
    AscendMetadata,
)
from vllm_ascend.kv_cache_compression.pyramidkv import (
    PyramidKVAscendProvider,
    PyramidKVAttentionBatchView,
    PyramidKVAttentionRequest,
    select_pyramid_kv,
)

DEVICE = torch.device("npu:0")
DTYPE = torch.bfloat16
BLOCK_SIZE = 128
NUM_BLOCKS = 8
NUM_HEADS = 32
QWEN_NUM_HEADS = 40
NUM_KV_HEADS = 8
HEAD_DIM = 128
PROMPT_TOKENS = 384
LAYER_NAMES = (
    "model.layers.0.self_attn.attn",
    "model.layers.1.self_attn.attn",
)
QWEN_LAYER_NAMES = tuple(f"model.layers.{index}.self_attn.attn" for index in range(48))


def _core_config() -> KVCacheCompressionConfig:
    return KVCacheCompressionConfig(
        provider="pyramidkv_ascend",
        provider_config={
            "max_capacity_prompt": 128,
            "window_size": 64,
            "kernel_size": 7,
            "pooling": "maxpool",
            "beta": 20,
            "kv_cache_granularity": "kv_head",
            "gqa_score_aggregation": "mean",
            "merge": None,
        },
    )


def _backend(num_heads: int = NUM_HEADS) -> AscendAttentionBackendImpl:
    backend = object.__new__(AscendAttentionBackendImpl)
    backend.vllm_config = SimpleNamespace(
        speculative_config=None,
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
    )
    backend.num_heads = num_heads
    backend.num_kv_heads = NUM_KV_HEADS
    backend.head_size = HEAD_DIM
    backend.hidden_size = num_heads * HEAD_DIM
    backend.scale = HEAD_DIM**-0.5
    backend.sliding_window = None
    backend.sinks = None
    backend.attn_type = AttentionType.DECODER
    backend.key_cache = None
    backend.value_cache = None
    backend.is_kv_producer = False
    backend.enable_hamming_sparse = False
    backend._use_layer_aware_fia_graph_replay = False
    return backend


def _layer(layer_name: str):
    return SimpleNamespace(
        layer_name=layer_name,
        _k_scale_float=1.0,
        _v_scale_float=1.0,
    )


def _cache() -> tuple[torch.Tensor, torch.Tensor]:
    shape = (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    return (
        torch.zeros(shape, dtype=DTYPE, device=DEVICE),
        torch.zeros(shape, dtype=DTYPE, device=DEVICE),
    )


def _slots(block_ids: tuple[int, ...], num_tokens: int) -> torch.Tensor:
    positions = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
    blocks = torch.tensor(block_ids, dtype=torch.int32, device=DEVICE)
    return blocks[positions // BLOCK_SIZE] * BLOCK_SIZE + positions % BLOCK_SIZE


def _read_cache(
    cache: torch.Tensor,
    block_ids: tuple[int, ...],
    num_tokens: int,
) -> torch.Tensor:
    return torch.cat([cache[block_id] for block_id in block_ids], dim=0)[:num_tokens]


def _metadata(
    state: AscendAttentionState,
    block_ids: tuple[int, ...],
    seq_len: int,
    num_tokens: int,
    view=None,
) -> AscendMetadata:
    return AscendMetadata(
        attn_mask=torch.triu(
            torch.ones(2048, 2048, dtype=torch.int8, device=DEVICE),
            diagonal=1,
        ),
        attn_state=state,
        num_actual_tokens=num_tokens,
        num_decode_tokens=num_tokens if state == AscendAttentionState.DecodeOnly else 0,
        num_decodes=1 if state == AscendAttentionState.DecodeOnly else 0,
        num_prefills=1 if state == AscendAttentionState.PrefillNoCache else 0,
        seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=DEVICE),
        seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int32),
        seq_lens_list=[seq_len],
        actual_seq_lengths_q=[num_tokens],
        block_tables=torch.tensor([block_ids], dtype=torch.int32, device=DEVICE),
        slot_mapping=_slots(block_ids, num_tokens),
        causal=True,
        kv_cache_compression_view=view,
    )


def _forward(
    backend: AscendAttentionBackendImpl,
    layer_name: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cache: tuple[torch.Tensor, torch.Tensor],
    metadata: AscendMetadata,
) -> torch.Tensor:
    output = torch.empty_like(query)
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
    )
    context.capturing = False
    with override_forward_context(context):
        backend.forward(
            _layer(layer_name),
            query,
            key,
            value,
            cache,
            metadata,
            output,
        )
    torch.npu.synchronize()
    return output


def _inputs(seed: int, num_heads: int = NUM_HEADS):
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(
        PROMPT_TOKENS,
        num_heads,
        HEAD_DIM,
        generator=generator,
        dtype=torch.float32,
    ).to(device=DEVICE, dtype=DTYPE)
    key = torch.randn(
        PROMPT_TOKENS,
        NUM_KV_HEADS,
        HEAD_DIM,
        generator=generator,
        dtype=torch.float32,
    ).to(device=DEVICE, dtype=DTYPE)
    value = torch.randn(
        PROMPT_TOKENS,
        NUM_KV_HEADS,
        HEAD_DIM,
        generator=generator,
        dtype=torch.float32,
    ).to(device=DEVICE, dtype=DTYPE)
    return query, key, value


def _legacy_repeat_selection(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    retained_tokens: int,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Frozen pre-optimization GQA path used only as an NPU oracle."""
    groups = query.shape[1] // key.shape[1]
    repeated_key = key.repeat_interleave(groups, dim=1)
    scores = torch.matmul(query[:, :, -window_size:, :], repeated_key.transpose(2, 3)) / math.sqrt(query.shape[-1])
    scores[..., -window_size:] += torch.triu(
        torch.full(
            (window_size, window_size),
            torch.finfo(scores.dtype).min,
            dtype=scores.dtype,
            device=scores.device,
        ),
        diagonal=1,
    )
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    history = probabilities[..., :-window_size].sum(dim=-2)
    history = history.reshape(query.shape[0], key.shape[1], groups, history.shape[-1]).mean(dim=2)
    pooled = F.max_pool1d(
        history,
        kernel_size=7,
        stride=1,
        padding=3,
    )
    selected = pooled.topk(retained_tokens - window_size, dim=-1).indices
    gather_index = selected.unsqueeze(-1).expand(-1, -1, -1, key.shape[-1])
    compact_key = torch.cat(
        (
            key[:, :, :-window_size, :].gather(2, gather_index),
            key[:, :, -window_size:, :],
        ),
        dim=2,
    )
    compact_value = torch.cat(
        (
            value[:, :, :-window_size, :].gather(2, gather_index),
            value[:, :, -window_size:, :],
        ),
        dim=2,
    )
    return compact_key, compact_value, selected


def test_grouped_gqa_selection_matches_legacy_repeat_bf16() -> None:
    query, key, value = _inputs(777)
    query = query.permute(1, 0, 2).unsqueeze(0)
    key = key.permute(1, 0, 2).unsqueeze(0)
    value = value.permute(1, 0, 2).unsqueeze(0)
    provider = PyramidKVAscendProvider.from_core_config(_core_config())
    retained_tokens = provider.config.retained_tokens(PROMPT_TOKENS, 1, len(LAYER_NAMES))

    actual = select_pyramid_kv(
        query,
        key,
        value,
        provider.config,
        layer_index=1,
        num_hidden_layers=len(LAYER_NAMES),
    )
    expected_key, expected_value, expected_indices = _legacy_repeat_selection(
        query,
        key,
        value,
        retained_tokens,
        provider.config.window_size,
    )
    torch.npu.synchronize()

    assert torch.equal(actual.selected_past_indices, expected_indices)
    torch.testing.assert_close(actual.key, expected_key, rtol=0, atol=0)
    torch.testing.assert_close(actual.value, expected_value, rtol=0, atol=0)


def test_qwen_grouped_gqa_selection_matches_legacy_repeat_bf16() -> None:
    query, key, value = _inputs(778, QWEN_NUM_HEADS)
    query = query.permute(1, 0, 2).unsqueeze(0)
    key = key.permute(1, 0, 2).unsqueeze(0)
    value = value.permute(1, 0, 2).unsqueeze(0)
    provider = PyramidKVAscendProvider.from_core_config(_core_config())
    retained_tokens = provider.config.retained_tokens(
        PROMPT_TOKENS,
        47,
        len(QWEN_LAYER_NAMES),
    )

    actual = select_pyramid_kv(
        query,
        key,
        value,
        provider.config,
        layer_index=47,
        num_hidden_layers=len(QWEN_LAYER_NAMES),
    )
    expected_key, expected_value, expected_indices = _legacy_repeat_selection(
        query,
        key,
        value,
        retained_tokens,
        provider.config.window_size,
    )
    torch.npu.synchronize()

    assert torch.equal(actual.selected_past_indices, expected_indices)
    torch.testing.assert_close(actual.key, expected_key, rtol=0, atol=0)
    torch.testing.assert_close(actual.value, expected_value, rtol=0, atol=0)


def test_qwen_prefill_cache_write_uses_48_layer_40_to_8_geometry() -> None:
    provider = PyramidKVAscendProvider.from_core_config(_core_config())
    block_ids = (1, 2, 3)
    provider.begin_request("qwen", PROMPT_TOKENS, (block_ids,))
    layer_name = QWEN_LAYER_NAMES[-1]
    layer_indices = {name: index for index, name in enumerate(QWEN_LAYER_NAMES)}
    view = PyramidKVAttentionBatchView(
        provider=provider,
        requests=(
            PyramidKVAttentionRequest(
                request_id="qwen",
                query_start=0,
                query_end=PROMPT_TOKENS,
                semantic_num_tokens=PROMPT_TOKENS,
                num_computed_tokens=0,
                num_prompt_tokens=PROMPT_TOKENS,
                block_ids=block_ids,
                is_prefill=True,
            ),
        ),
        layer_indices=layer_indices,
        num_hidden_layers=len(QWEN_LAYER_NAMES),
    )
    query, key, value = _inputs(779, QWEN_NUM_HEADS)
    expected = select_pyramid_kv(
        query.permute(1, 0, 2).unsqueeze(0),
        key.permute(1, 0, 2).unsqueeze(0),
        value.permute(1, 0, 2).unsqueeze(0),
        provider.config,
        layer_index=47,
        num_hidden_layers=len(QWEN_LAYER_NAMES),
    )
    cache = _cache()

    _forward(
        _backend(QWEN_NUM_HEADS),
        layer_name,
        query,
        key,
        value,
        cache,
        _metadata(
            AscendAttentionState.PrefillNoCache,
            block_ids,
            PROMPT_TOKENS,
            PROMPT_TOKENS,
            view,
        ),
    )

    torch.testing.assert_close(
        _read_cache(cache[0], block_ids, expected.retained_tokens),
        expected.key.squeeze(0).permute(1, 0, 2),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        _read_cache(cache[1], block_ids, expected.retained_tokens),
        expected.value.squeeze(0).permute(1, 0, 2),
        rtol=0,
        atol=0,
    )
    assert view.completed_prefill_lengths == {"qwen": {layer_name: expected.retained_tokens}}
    provider.cleanup_request("qwen")


def _attention_oracle(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    repeated_key = key.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)
    repeated_value = value.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)
    scores = torch.einsum("hd,shd->hs", query.float().cpu(), repeated_key.float().cpu()) * (HEAD_DIM**-0.5)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("hs,shd->hd", probabilities, repeated_value.float().cpu())


def test_full_prefill_compact_write_two_layers_and_decode_fia() -> None:
    provider = PyramidKVAscendProvider.from_core_config(_core_config())
    original_block_ids = (1, 2, 3)
    provider.begin_request(
        "request",
        PROMPT_TOKENS,
        (original_block_ids,),
    )
    layer_indices = {name: index for index, name in enumerate(LAYER_NAMES)}
    prefill_view = PyramidKVAttentionBatchView(
        provider=provider,
        requests=(
            PyramidKVAttentionRequest(
                request_id="request",
                query_start=0,
                query_end=PROMPT_TOKENS,
                semantic_num_tokens=PROMPT_TOKENS,
                num_computed_tokens=0,
                num_prompt_tokens=PROMPT_TOKENS,
                block_ids=original_block_ids,
                is_prefill=True,
            ),
        ),
        layer_indices=layer_indices,
        num_hidden_layers=len(LAYER_NAMES),
    )

    layer_runtime = {}
    for layer_index, layer_name in enumerate(LAYER_NAMES):
        query, key, value = _inputs(1000 + layer_index)
        expected = select_pyramid_kv(
            query.permute(1, 0, 2).unsqueeze(0),
            key.permute(1, 0, 2).unsqueeze(0),
            value.permute(1, 0, 2).unsqueeze(0),
            provider.config,
            layer_index=layer_index,
            num_hidden_layers=len(LAYER_NAMES),
        )
        cache = _cache()
        backend = _backend()
        enabled_metadata = _metadata(
            AscendAttentionState.PrefillNoCache,
            original_block_ids,
            PROMPT_TOKENS,
            PROMPT_TOKENS,
            prefill_view,
        )
        enabled_output = _forward(
            backend,
            layer_name,
            query,
            key,
            value,
            cache,
            enabled_metadata,
        )

        disabled_cache = _cache()
        disabled_output = _forward(
            _backend(),
            layer_name,
            query,
            key,
            value,
            disabled_cache,
            _metadata(
                AscendAttentionState.PrefillNoCache,
                original_block_ids,
                PROMPT_TOKENS,
                PROMPT_TOKENS,
            ),
        )
        torch.testing.assert_close(enabled_output, disabled_output, rtol=0, atol=0)

        actual_key = _read_cache(cache[0], original_block_ids, expected.retained_tokens)
        actual_value = _read_cache(cache[1], original_block_ids, expected.retained_tokens)
        torch.testing.assert_close(
            actual_key,
            expected.key.squeeze(0).permute(1, 0, 2),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            actual_value,
            expected.value.squeeze(0).permute(1, 0, 2),
            rtol=0,
            atol=0,
        )
        layer_runtime[layer_name] = (backend, cache, expected, query, key, value)

    plans = provider.finish_model_forward(prefill_view, layer_names=LAYER_NAMES, schema_version=1)
    assert plans is not None
    plan = plans[0]
    retained_lengths = dict(plan.per_layer_physical_num_tokens)
    assert retained_lengths[LAYER_NAMES[0]] == 189
    assert retained_lengths[LAYER_NAMES[1]] == 67
    assert plan.physical_num_tokens == 189
    committed_block_ids = original_block_ids[:2]
    provider.mark_committed("request", (committed_block_ids,))

    completed_layers = set()
    for layer_index, layer_name in enumerate(LAYER_NAMES):
        backend, cache, expected, _, _, _ = layer_runtime[layer_name]
        generator = torch.Generator().manual_seed(2000 + layer_index)
        decode_query = torch.randn(1, NUM_HEADS, HEAD_DIM, generator=generator).to(device=DEVICE, dtype=DTYPE)
        decode_key = torch.randn(1, NUM_KV_HEADS, HEAD_DIM, generator=generator).to(device=DEVICE, dtype=DTYPE)
        decode_value = torch.randn(1, NUM_KV_HEADS, HEAD_DIM, generator=generator).to(device=DEVICE, dtype=DTYPE)
        decode_view = PyramidKVAttentionBatchView(
            provider=provider,
            requests=(
                PyramidKVAttentionRequest(
                    request_id="request",
                    query_start=0,
                    query_end=1,
                    semantic_num_tokens=PROMPT_TOKENS + 1,
                    num_computed_tokens=PROMPT_TOKENS,
                    num_prompt_tokens=PROMPT_TOKENS,
                    block_ids=committed_block_ids,
                    is_prefill=False,
                ),
            ),
            layer_indices=layer_indices,
            num_hidden_layers=len(LAYER_NAMES),
        )
        metadata = _metadata(
            AscendAttentionState.DecodeOnly,
            committed_block_ids,
            PROMPT_TOKENS + 1,
            1,
            decode_view,
        )
        decode_output = _forward(
            backend,
            layer_name,
            decode_query,
            decode_key,
            decode_value,
            cache,
            metadata,
        )
        physical_length = retained_lengths[layer_name]
        assert metadata.seq_lens_list == [physical_length + 1]
        actual_key = _read_cache(cache[0], committed_block_ids, physical_length + 1)
        actual_value = _read_cache(cache[1], committed_block_ids, physical_length + 1)
        torch.testing.assert_close(actual_key[-1], decode_key[0], rtol=0, atol=0)
        torch.testing.assert_close(actual_value[-1], decode_value[0], rtol=0, atol=0)
        oracle = _attention_oracle(decode_query[0], actual_key, actual_value)
        torch.testing.assert_close(
            decode_output[0].float().cpu(),
            oracle,
            rtol=1e-2,
            atol=1e-2,
        )
        completed_layers.add(layer_name)

    provider.advance_decode("request", completed_layers)
    state = provider.get_request_state("request")
    assert state.semantic_num_tokens == PROMPT_TOKENS + 1
    assert state.layers[LAYER_NAMES[0]].physical_num_tokens == 190
    assert state.layers[LAYER_NAMES[1]].physical_num_tokens == 68
    provider.cleanup_request("request")


def test_chunked_prefill_cross_chunk_gather_tail_compact_and_decode() -> None:
    provider = PyramidKVAscendProvider.from_core_config(_core_config())
    layer_indices = {name: index for index, name in enumerate(LAYER_NAMES)}
    chunk_lengths = (256, 127, 1)
    all_block_ids = (1, 2, 3)
    provider.begin_request("chunked", PROMPT_TOKENS, ((1, 2),))
    layer_inputs = {layer_name: _inputs(3000 + layer_index) for layer_index, layer_name in enumerate(LAYER_NAMES)}
    expected = {
        layer_name: select_pyramid_kv(
            query.permute(1, 0, 2).unsqueeze(0),
            key.permute(1, 0, 2).unsqueeze(0),
            value.permute(1, 0, 2).unsqueeze(0),
            provider.config,
            layer_index=layer_index,
            num_hidden_layers=len(LAYER_NAMES),
        )
        for layer_index, (layer_name, (query, key, value)) in enumerate(layer_inputs.items())
    }
    layer_runtime = {layer_name: (_backend(), _cache()) for layer_name in LAYER_NAMES}

    computed = 0
    plans = None
    for chunk_length in chunk_lengths:
        end = computed + chunk_length
        block_ids = all_block_ids[: (end + BLOCK_SIZE - 1) // BLOCK_SIZE]
        view = PyramidKVAttentionBatchView(
            provider=provider,
            requests=(
                PyramidKVAttentionRequest(
                    request_id="chunked",
                    query_start=0,
                    query_end=chunk_length,
                    semantic_num_tokens=end,
                    num_computed_tokens=computed,
                    num_prompt_tokens=PROMPT_TOKENS,
                    block_ids=block_ids,
                    is_prefill=True,
                ),
            ),
            layer_indices=layer_indices,
            num_hidden_layers=len(LAYER_NAMES),
        )
        state = AscendAttentionState.DecodeOnly if chunk_length == 1 else AscendAttentionState.ChunkedPrefill
        metadata = _metadata(
            state,
            block_ids,
            end,
            chunk_length,
            view,
        )
        positions = torch.arange(computed, end, dtype=torch.int32, device=DEVICE)
        blocks = torch.tensor(block_ids, dtype=torch.int32, device=DEVICE)
        metadata.slot_mapping = blocks[positions // BLOCK_SIZE] * BLOCK_SIZE + positions % BLOCK_SIZE
        for layer_name in LAYER_NAMES:
            backend, cache = layer_runtime[layer_name]
            query, key, value = layer_inputs[layer_name]
            _forward(
                backend,
                layer_name,
                query[computed:end],
                key[computed:end],
                value[computed:end],
                cache,
                metadata,
            )
        if end < PROMPT_TOKENS:
            assert provider.finish_model_forward(view, layer_names=LAYER_NAMES, schema_version=1) is None
            request_state = provider.get_request_state("chunked")
            assert request_state.prefill_num_computed_tokens == end
            assert not request_state.layers
        else:
            selected = {
                deferred.layer.layer_name: deferred.selected_past_indices for deferred in view.deferred_prefills
            }
            for layer_name in LAYER_NAMES:
                assert torch.equal(
                    selected[layer_name],
                    expected[layer_name].selected_past_indices,
                )
            plans = provider.finish_model_forward(view, layer_names=LAYER_NAMES, schema_version=1)
        computed = end

    assert plans is not None and len(plans) == 1
    assert plans[0].semantic_num_tokens == PROMPT_TOKENS
    assert plans[0].physical_num_tokens == 189
    for layer_name in LAYER_NAMES:
        _, cache = layer_runtime[layer_name]
        selection = expected[layer_name]
        torch.testing.assert_close(
            _read_cache(cache[0], all_block_ids, selection.retained_tokens),
            selection.key.squeeze(0).permute(1, 0, 2),
            rtol=1e-2,
            atol=1e-2,
        )
        torch.testing.assert_close(
            _read_cache(cache[1], all_block_ids, selection.retained_tokens),
            selection.value.squeeze(0).permute(1, 0, 2),
            rtol=1e-2,
            atol=1e-2,
        )

    committed_block_ids = (1, 2)
    provider.mark_committed("chunked", (committed_block_ids,))
    decode_view = PyramidKVAttentionBatchView(
        provider=provider,
        requests=(
            PyramidKVAttentionRequest(
                request_id="chunked",
                query_start=0,
                query_end=1,
                semantic_num_tokens=PROMPT_TOKENS + 1,
                num_computed_tokens=PROMPT_TOKENS,
                num_prompt_tokens=PROMPT_TOKENS,
                block_ids=committed_block_ids,
                is_prefill=False,
            ),
        ),
        layer_indices=layer_indices,
        num_hidden_layers=len(LAYER_NAMES),
    )
    for layer_index, layer_name in enumerate(LAYER_NAMES):
        backend, cache = layer_runtime[layer_name]
        query, key, value = _inputs(4000 + layer_index)
        _forward(
            backend,
            layer_name,
            query[:1],
            key[:1],
            value[:1],
            cache,
            _metadata(
                AscendAttentionState.DecodeOnly,
                committed_block_ids,
                PROMPT_TOKENS + 1,
                1,
                decode_view,
            ),
        )
    assert provider.finish_model_forward(decode_view, layer_names=LAYER_NAMES, schema_version=1) is None
    assert provider.get_request_state("chunked").semantic_num_tokens == 385
    provider.cleanup_request("chunked")
