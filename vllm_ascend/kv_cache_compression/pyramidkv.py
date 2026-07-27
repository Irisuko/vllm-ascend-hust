# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Ascend PyramidKV algorithm and request state.

This is a clean-room implementation of the observable PyramidKV selection
semantics described by KVCache-Factory at commit
fc6f8f4c3d8ca7a1849a2ef67ff5fca8d285a6f0 (MIT). It does not import or copy
that repository's Hugging Face patches, CUDA, Triton, or custom kernels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import regex as re
import torch
import torch.nn.functional as F
from vllm.logger import logger
from vllm.v1.kv_cache_compression import (
    KVCacheCompressionCompatibility,
    KVCacheCompressionPlan,
    KVCacheCompressionRuntimeSpec,
)

if TYPE_CHECKING:
    from vllm.config import KVCacheCompressionConfig

PYRAMIDKV_ASCEND_PROVIDER = "pyramidkv_ascend"
SUPPORTED_DEVICE_NAME = "Ascend910B2"
SUPPORTED_CANN_VERSION_PREFIXES = ("8.5.1", "9.0")
SUPPORTED_BACKEND = "AscendAttentionBackend"
SUPPORTED_MODEL_ARCHITECTURE = "LlamaForCausalLM"
SUPPORTED_CACHE_LAYOUT = "standard_bf16_paged"
SUPPORTED_BLOCK_SIZE = 128


@dataclass(frozen=True)
class PyramidKVAscendConfig:
    """Validated provider-owned algorithm configuration."""

    max_capacity_prompt: int = 512
    window_size: int = 8
    kernel_size: int = 7
    pooling: str = "maxpool"
    beta: int = 20
    kv_cache_granularity: str = "kv_head"
    gqa_score_aggregation: str = "mean"
    merge: None = None

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> PyramidKVAscendConfig:
        expected = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - expected)
        if unknown:
            raise ValueError("unknown PyramidKV provider_config fields: " + ", ".join(unknown))
        config = cls(**values)
        config._validate()
        return config

    def _validate(self) -> None:
        integer_fields = {
            "max_capacity_prompt": self.max_capacity_prompt,
            "window_size": self.window_size,
            "kernel_size": self.kernel_size,
            "beta": self.beta,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.max_capacity_prompt <= self.window_size:
            raise ValueError(
                "max_capacity_prompt must be greater than window_size, got "
                f"{self.max_capacity_prompt} and {self.window_size}"
            )
        if self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd to preserve score length, got {self.kernel_size}")
        if self.pooling != "maxpool":
            raise ValueError(f"pooling must be 'maxpool' in the first release, got {self.pooling!r}")
        if self.kv_cache_granularity != "kv_head":
            raise ValueError(
                f"kv_cache_granularity must be 'kv_head' in the first release, got {self.kv_cache_granularity!r}"
            )
        if self.gqa_score_aggregation != "mean":
            raise ValueError(
                f"gqa_score_aggregation must be 'mean' in the first release, got {self.gqa_score_aggregation!r}"
            )
        if self.merge is not None:
            raise ValueError(f"merge must be null in the first release, got {self.merge!r}")

    def retained_tokens(
        self,
        prompt_tokens: int,
        layer_index: int,
        num_hidden_layers: int,
    ) -> int:
        """Return the layer's final retained length, including recent window."""
        if prompt_tokens <= 0:
            raise ValueError(f"prompt_tokens must be positive, got {prompt_tokens}")
        if num_hidden_layers <= 1:
            raise ValueError(f"num_hidden_layers must be greater than one, got {num_hidden_layers}")
        if not 0 <= layer_index < num_hidden_layers:
            raise ValueError(f"layer_index {layer_index} is outside [0, {num_hidden_layers})")
        if prompt_tokens < self.max_capacity_prompt:
            return prompt_tokens

        past_budget = self.max_capacity_prompt - self.window_size
        if prompt_tokens < 2 * past_budget:
            return self.max_capacity_prompt

        minimum_past = past_budget // self.beta
        maximum_past = 2 * past_budget - minimum_past
        available_past = prompt_tokens - self.window_size
        if maximum_past >= available_past:
            maximum_past = available_past
            minimum_past = 2 * past_budget - maximum_past
        step = (maximum_past - minimum_past) // (num_hidden_layers - 1)
        layer_past = maximum_past - layer_index * step
        return layer_past + self.window_size


@dataclass(frozen=True)
class PyramidKVCapabilityContext:
    """Actual worker/backend properties checked before KV allocation."""

    platform: str
    device_name: str
    cann_version: str
    use_v2_model_runner: bool
    enforce_eager: bool
    cudagraph_mode: str
    pa_shape_list: tuple[int, ...]
    backend: str
    model_architecture: str
    dtype: str
    quantization: str | None
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    num_hidden_layers: int
    cache_layout: str
    block_size: int
    hash_block_size: int
    max_model_len: int
    num_kv_cache_groups: int
    full_attention_only: bool
    prefix_caching: bool
    chunked_prefill: bool
    sliding_window: bool
    speculative_decoding: bool
    kv_transfer: bool
    kv_offload: bool
    cache_dtype: str
    tensor_parallel_size: int
    pipeline_parallel_size: int
    prefill_context_parallel_size: int
    decode_context_parallel_size: int
    async_scheduling: bool
    balance_scheduling: bool
    dbo_enabled: bool
    knorm_enabled: bool
    missing_ops: tuple[str, ...] = ()


@dataclass(frozen=True)
class PyramidKVSelection:
    """One layer's selected compact K/V representation."""

    key: torch.Tensor
    value: torch.Tensor
    selected_past_indices: torch.Tensor | None
    retained_tokens: int
    compressed: bool


def select_pyramid_indices(
    query_window: torch.Tensor,
    key: torch.Tensor,
    config: PyramidKVAscendConfig,
    *,
    layer_index: int,
    num_hidden_layers: int,
) -> tuple[torch.Tensor | None, int]:
    """Select retained history positions from the final query window."""
    if query_window.ndim != 4 or key.ndim != 4:
        raise ValueError("query_window and key must have [batch, heads, tokens, dim]")
    if query_window.shape[0] != key.shape[0] or query_window.shape[-1] != key.shape[-1]:
        raise ValueError(f"query/key batch and head dimensions must match, got {query_window.shape} and {key.shape}")
    num_query_heads = query_window.shape[1]
    num_kv_heads = key.shape[1]
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(f"query heads {num_query_heads} must be divisible by KV heads {num_kv_heads}")

    prompt_tokens = key.shape[2]
    retained_tokens = config.retained_tokens(prompt_tokens, layer_index, num_hidden_layers)
    if retained_tokens >= prompt_tokens:
        return None, prompt_tokens
    if query_window.shape[2] != config.window_size:
        raise ValueError(
            "compressed PyramidKV selection requires exactly window_size query "
            f"tokens, got {query_window.shape[2]} and {config.window_size}"
        )

    groups = num_query_heads // num_kv_heads
    grouped_query = query_window.reshape(
        query_window.shape[0],
        num_kv_heads,
        groups,
        config.window_size,
        query_window.shape[-1],
    )
    scores = torch.matmul(
        grouped_query,
        key.unsqueeze(2).transpose(-2, -1),
    ) / math.sqrt(query_window.shape[-1])
    causal_mask = torch.triu(
        torch.full(
            (config.window_size, config.window_size),
            torch.finfo(scores.dtype).min,
            dtype=scores.dtype,
            device=scores.device,
        ),
        diagonal=1,
    )
    scores[..., -config.window_size :] += causal_mask
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query_window.dtype)
    history_scores = probabilities[..., : -config.window_size].sum(dim=-2)
    history_scores = history_scores.mean(dim=2)
    pooled_scores = F.max_pool1d(
        history_scores,
        kernel_size=config.kernel_size,
        stride=1,
        padding=config.kernel_size // 2,
    )

    retained_past = retained_tokens - config.window_size
    selected = pooled_scores.topk(retained_past, dim=-1).indices
    return selected, retained_tokens


def materialize_pyramid_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    selected_past_indices: torch.Tensor,
    retained_tokens: int,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize compact K/V from provider-selected history positions."""
    if key.shape != value.shape or key.ndim != 4:
        raise ValueError("key and value must be matching [batch, heads, tokens, dim] tensors")
    retained_past = retained_tokens - window_size
    expected_shape = (key.shape[0], key.shape[1], retained_past)
    if tuple(selected_past_indices.shape) != expected_shape:
        raise ValueError(f"selected indices must have shape {expected_shape}, got {tuple(selected_past_indices.shape)}")
    gather_index = selected_past_indices.unsqueeze(-1).expand(-1, -1, -1, key.shape[-1])
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
    return compact_key, compact_value


def select_pyramid_kv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    config: PyramidKVAscendConfig,
    *,
    layer_index: int,
    num_hidden_layers: int,
) -> PyramidKVSelection:
    """Select history independently per KV head using GQA mean scores."""
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have [batch, heads, tokens, dim]")
    if key.shape != value.shape:
        raise ValueError(f"key and value shapes must match, got {key.shape} and {value.shape}")
    if query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
        raise ValueError(f"query/key batch, token, and head dimensions must match, got {query.shape} and {key.shape}")
    prompt_tokens = query.shape[2]
    selected, retained_tokens = select_pyramid_indices(
        query[:, :, -config.window_size :, :],
        key,
        config,
        layer_index=layer_index,
        num_hidden_layers=num_hidden_layers,
    )
    if selected is None:
        return PyramidKVSelection(
            key=key,
            value=value,
            selected_past_indices=None,
            retained_tokens=prompt_tokens,
            compressed=False,
        )
    if prompt_tokens <= config.window_size:
        raise ValueError(f"prompt length {prompt_tokens} must exceed window_size {config.window_size}")
    compact_key, compact_value = materialize_pyramid_kv(
        key,
        value,
        selected,
        retained_tokens,
        config.window_size,
    )
    return PyramidKVSelection(
        key=compact_key,
        value=compact_value,
        selected_past_indices=selected,
        retained_tokens=retained_tokens,
        compressed=True,
    )


def _slots_for_positions(
    block_ids: tuple[int, ...],
    start_position: int,
    num_positions: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    if start_position < 0 or num_positions <= 0:
        raise ValueError(
            f"start_position must be non-negative and num_positions positive, got {start_position} and {num_positions}"
        )
    end_position = start_position + num_positions
    required_blocks = (end_position + block_size - 1) // block_size
    if required_blocks > len(block_ids):
        raise RuntimeError(
            f"PyramidKV needs {required_blocks} blocks through physical position "
            f"{end_position}, "
            f"but the request block table has {len(block_ids)}"
        )
    positions = torch.arange(start_position, end_position, dtype=torch.int32, device=device)
    block_table = torch.tensor(block_ids, dtype=torch.int32, device=device)
    return block_table[torch.div(positions, block_size, rounding_mode="floor")] * block_size + positions.remainder(
        block_size
    )


def _slot_for_position(
    block_ids: tuple[int, ...],
    position: int,
    block_size: int,
) -> int:
    """Resolve one physical slot without launching per-request device ops."""
    if position < 0:
        raise ValueError(f"position must be non-negative, got {position}")
    block_index, block_offset = divmod(position, block_size)
    if block_index >= len(block_ids):
        raise RuntimeError(
            f"PyramidKV needs block {block_index} for physical position "
            f"{position}, but the request block table has {len(block_ids)} blocks"
        )
    return block_ids[block_index] * block_size + block_offset


def _gather_paged_prefix(
    cache: torch.Tensor,
    block_ids: tuple[int, ...],
    num_tokens: int,
    block_size: int,
) -> torch.Tensor:
    """Gather one request prefix from standard paged KV storage."""
    if cache.ndim != 4 or cache.shape[1] != block_size:
        raise RuntimeError(
            "PyramidKV chunked prefill requires [blocks, block, heads, dim] "
            f"cache with block_size {block_size}, got {tuple(cache.shape)}"
        )
    if num_tokens <= 0:
        return cache.new_empty((0, cache.shape[2], cache.shape[3]))
    required_blocks = (num_tokens + block_size - 1) // block_size
    if required_blocks > len(block_ids):
        raise RuntimeError(
            f"PyramidKV needs {required_blocks} blocks for {num_tokens} cached tokens, got {len(block_ids)}"
        )
    block_index = torch.tensor(block_ids[:required_blocks], dtype=torch.long, device=cache.device)
    return cache.index_select(0, block_index).reshape(-1, cache.shape[2], cache.shape[3])[:num_tokens]


@dataclass
class PyramidKVLayerState:
    physical_num_tokens: int


@dataclass
class PyramidKVRequestState:
    request_id: str
    semantic_num_tokens: int
    expected_block_ids: tuple[tuple[int, ...], ...]
    prefill_num_computed_tokens: int = 0
    initial_prefill_num_computed_tokens: int = 0
    prefill_query_tail: torch.Tensor | None = None
    prefill_query_tail_length: int = 0
    layers: dict[str, PyramidKVLayerState] = field(default_factory=dict)
    plan_emitted: bool = False
    committed: bool = False


@dataclass(frozen=True)
class PyramidKVAttentionRequest:
    """One request slice in a provider-owned attention batch view."""

    request_id: str
    query_start: int
    query_end: int
    semantic_num_tokens: int
    num_computed_tokens: int
    num_prompt_tokens: int
    block_ids: tuple[int, ...]
    is_prefill: bool
    destination_block_ids: tuple[int, ...] | None = None
    compress: bool = True

    @property
    def query_length(self) -> int:
        return self.query_end - self.query_start

    @property
    def is_complete_prefill(self) -> bool:
        return self.is_prefill and self.num_computed_tokens == 0 and self.query_length == self.num_prompt_tokens

    @property
    def is_final_prefill_chunk(self) -> bool:
        return self.is_prefill and self.semantic_num_tokens == self.num_prompt_tokens


@dataclass(frozen=True)
class PyramidKVDeferredPrefill:
    """One mixed-batch prefill compacted only after attention succeeds."""

    request_id: str
    layer: Any
    backend: Any
    kv_cache: tuple[torch.Tensor, ...]
    block_ids: tuple[int, ...]
    semantic_num_tokens: int
    retained_tokens: int
    selected_past_indices: torch.Tensor
    destination_block_ids: tuple[int, ...] | None = None


@dataclass(frozen=True)
class PyramidKVAttentionBatchView:
    """Layer-aware cache-write data attached to Ascend attention metadata."""

    provider: PyramidKVAscendProvider
    requests: tuple[PyramidKVAttentionRequest, ...]
    layer_indices: dict[str, int]
    num_hidden_layers: int
    block_size: int = SUPPORTED_BLOCK_SIZE
    completed_decode_layers: set[str] = field(default_factory=set)
    deferred_prefills: list[PyramidKVDeferredPrefill] = field(default_factory=list)
    completed_prefill_lengths: dict[str, dict[str, int]] = field(default_factory=dict)
    pending_query_tails: dict[str, torch.Tensor] = field(default_factory=dict)
    pending_query_tail_lengths: dict[str, int] = field(default_factory=dict)
    completed_prefill_query_layers: dict[str, set[str]] = field(default_factory=dict)
    decode_slot_values_by_layer: dict[str, tuple[int, ...]] = field(default_factory=dict)
    decode_lengths_by_layer: dict[str, list[int]] = field(default_factory=dict)
    decode_slot_tensors_by_layer: dict[str, torch.Tensor] = field(default_factory=dict)
    decode_length_tensors_by_layer: dict[str, torch.Tensor] = field(default_factory=dict)

    def _prepare_decode_values(self) -> tuple[str, ...]:
        """Validate decode state and compute per-layer CPU slot/length values."""
        ordered_layers = tuple(name for name, _ in sorted(self.layer_indices.items(), key=lambda item: item[1]))
        if self.decode_slot_values_by_layer:
            return ordered_layers

        for current_layer in ordered_layers:
            slots: list[int] = []
            lengths: list[int] = []
            for request in self.requests:
                if request.is_prefill:
                    raise RuntimeError("decode slot matrix cannot contain a prefill request")
                if request.query_end - request.query_start != 1:
                    raise RuntimeError("PyramidKV supports exactly one decode token per request")
                if request.compress:
                    request_state = self.provider.get_request_state(request.request_id)
                    if not request_state.committed:
                        raise RuntimeError(f"request {request.request_id!r} has no core commit ack")
                    if request.block_ids != request_state.expected_block_ids[0]:
                        raise RuntimeError(
                            f"request {request.request_id!r} decode block "
                            "table does not match the core commit ack: "
                            f"runner={request.block_ids}, provider="
                            f"{request_state.expected_block_ids[0]}"
                        )
                    if request.semantic_num_tokens != request_state.semantic_num_tokens + 1:
                        raise RuntimeError(
                            f"request {request.request_id!r} decode semantic "
                            "length must be "
                            f"{request_state.semantic_num_tokens + 1}, got "
                            f"{request.semantic_num_tokens}"
                        )
                    try:
                        position = request_state.layers[current_layer].physical_num_tokens
                    except KeyError as error:
                        raise RuntimeError(
                            f"request {request.request_id!r} has no state for layer {current_layer!r}"
                        ) from error
                    lengths.append(position + 1)
                else:
                    position = request.semantic_num_tokens - 1
                    lengths.append(request.semantic_num_tokens)
                slots.append(
                    _slot_for_position(
                        request.block_ids,
                        position,
                        self.block_size,
                    )
                )
            self.decode_slot_values_by_layer[current_layer] = tuple(slots)
            self.decode_lengths_by_layer[current_layer] = lengths
        return ordered_layers

    def fill_full_decode_metadata(
        self,
        *,
        layer_names: tuple[str, ...],
        slot_staging: torch.Tensor,
        length_staging: torch.Tensor,
        num_reqs_padded: int,
    ) -> None:
        """Fill address-stable FULL graph staging matrices without advancing state."""
        ordered_layers = self._prepare_decode_values()
        if ordered_layers != layer_names:
            raise RuntimeError(
                "PyramidKV FULL decode layer order does not match the KV cache "
                f"group: view={ordered_layers}, group={layer_names}"
            )
        expected_shape = (len(layer_names), slot_staging.shape[1])
        if (
            slot_staging.device.type != "cpu"
            or length_staging.device.type != "cpu"
            or slot_staging.dtype != torch.int32
            or length_staging.dtype != torch.int32
            or slot_staging.shape != length_staging.shape
            or tuple(slot_staging.shape) != expected_shape
        ):
            raise RuntimeError(
                f"PyramidKV FULL decode staging buffers must be matching CPU int32 matrices shaped {expected_shape}"
            )
        num_reqs = len(self.requests)
        if not num_reqs <= num_reqs_padded <= slot_staging.shape[1]:
            raise RuntimeError(
                "PyramidKV FULL decode padded request count is invalid: "
                f"actual={num_reqs}, padded={num_reqs_padded}, "
                f"capacity={slot_staging.shape[1]}"
            )

        # Padding must never write a real KV slot. FIA nevertheless requires a
        # positive length for every padded request row.
        slot_staging.fill_(-1)
        length_staging.fill_(1)
        for layer_index, layer_name in enumerate(layer_names):
            slot_staging[layer_index, :num_reqs] = torch.tensor(
                self.decode_slot_values_by_layer[layer_name],
                dtype=torch.int32,
            )
            length_staging[layer_index, :num_reqs] = torch.tensor(
                self.decode_lengths_by_layer[layer_name],
                dtype=torch.int32,
            )

    def _decode_tensors(self, layer_name: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize all decode slots in one device transfer; keep lengths CPU."""
        if not self.decode_slot_tensors_by_layer:
            ordered_layers = self._prepare_decode_values()
            slot_matrix = torch.tensor(
                [self.decode_slot_values_by_layer[name] for name in ordered_layers],
                dtype=torch.int32,
                device=device,
            )
            length_matrix = torch.tensor(
                [self.decode_lengths_by_layer[name] for name in ordered_layers],
                dtype=torch.int32,
            )
            self.decode_slot_tensors_by_layer.update(
                {name: slot_matrix[index] for index, name in enumerate(ordered_layers)}
            )
            self.decode_length_tensors_by_layer.update(
                {name: length_matrix[index] for index, name in enumerate(ordered_layers)}
            )
        slots = self.decode_slot_tensors_by_layer[layer_name]
        lengths = self.decode_length_tensors_by_layer[layer_name]
        if slots.device != device or lengths.device.type != "cpu":
            raise RuntimeError(
                "PyramidKV decode metadata device changed within one model step: "
                f"slots={slots.device}, lengths={lengths.device}, "
                f"expected slots={device} and CPU lengths"
            )
        return slots, lengths

    def _record_prefill_length(self, request_id: str, layer_name: str, retained_tokens: int) -> None:
        layer_lengths = self.completed_prefill_lengths.setdefault(request_id, {})
        if layer_name in layer_lengths:
            raise RuntimeError(f"request {request_id!r} layer {layer_name!r} completed twice")
        layer_lengths[layer_name] = retained_tokens

    def _stage_query_tail(
        self,
        request: PyramidKVAttentionRequest,
        layer_name: str,
        request_query: torch.Tensor,
    ) -> torch.Tensor:
        """Stage one layer's next query tail without advancing provider state."""
        layer_index = self.layer_indices[layer_name]
        completed_layers = self.completed_prefill_query_layers.setdefault(request.request_id, set())
        if layer_name in completed_layers:
            raise RuntimeError(f"request {request.request_id!r} layer {layer_name!r} staged its query tail twice")
        state = self.provider.get_request_state(request.request_id)
        previous_tail = state.prefill_query_tail
        previous_length = state.prefill_query_tail_length
        if request.num_computed_tokens != state.prefill_num_computed_tokens:
            raise RuntimeError(
                f"request {request.request_id!r} chunk starts at "
                f"{request.num_computed_tokens}, expected "
                f"{state.prefill_num_computed_tokens}"
            )
        if previous_tail is None:
            if previous_length != 0:
                raise RuntimeError(f"request {request.request_id!r} has stale prefill query tail length")
            if request.num_computed_tokens != state.initial_prefill_num_computed_tokens:
                raise RuntimeError(f"request {request.request_id!r} is missing its committed query tail")
            combined = request_query
        else:
            if not 0 < previous_length <= self.provider.config.window_size:
                raise RuntimeError(f"request {request.request_id!r} has an invalid committed query tail")
            if (
                previous_tail.device != request_query.device
                or previous_tail.dtype != request_query.dtype
                or previous_tail.shape[0] != self.num_hidden_layers
                or previous_tail.shape[1] != self.provider.config.window_size
                or tuple(previous_tail.shape[2:]) != tuple(request_query.shape[1:])
            ):
                raise RuntimeError(
                    f"request {request.request_id!r} query tail metadata does not match the current chunk"
                )
            combined = torch.cat(
                (previous_tail[layer_index, :previous_length], request_query),
                dim=0,
            )

        tail = combined[-self.provider.config.window_size :]
        tail_length = tail.shape[0]
        staged = self.pending_query_tails.get(request.request_id)
        if staged is None:
            staged = torch.empty(
                (
                    self.num_hidden_layers,
                    self.provider.config.window_size,
                    request_query.shape[1],
                    request_query.shape[2],
                ),
                dtype=request_query.dtype,
                device=request_query.device,
            )
            self.pending_query_tails[request.request_id] = staged
            self.pending_query_tail_lengths[request.request_id] = tail_length
        elif self.pending_query_tail_lengths[request.request_id] != tail_length:
            raise RuntimeError(f"request {request.request_id!r} produced inconsistent query tail lengths across layers")
        staged[layer_index, :tail_length].copy_(tail)
        completed_layers.add(layer_name)
        return tail

    def _select_final_chunk(
        self,
        request: PyramidKVAttentionRequest,
        layer_name: str,
        request_query: torch.Tensor,
        request_key: torch.Tensor,
        key_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        query_tail = self._stage_query_tail(request, layer_name, request_query)
        if query_tail.shape[0] != self.provider.config.window_size:
            raise RuntimeError(
                f"request {request.request_id!r} final query tail has "
                f"{query_tail.shape[0]} tokens, expected "
                f"{self.provider.config.window_size}"
            )
        prefix_key = _gather_paged_prefix(
            key_cache,
            request.block_ids,
            request.num_computed_tokens,
            self.block_size,
        )
        full_key = torch.cat((prefix_key, request_key), dim=0)
        if full_key.shape[0] != request.num_prompt_tokens:
            raise RuntimeError(
                f"request {request.request_id!r} reconstructed "
                f"{full_key.shape[0]} key tokens, expected "
                f"{request.num_prompt_tokens}"
            )
        selected, retained_tokens = select_pyramid_indices(
            query_tail.permute(1, 0, 2).unsqueeze(0),
            full_key.permute(1, 0, 2).unsqueeze(0),
            self.provider.config,
            layer_index=self.layer_indices[layer_name],
            num_hidden_layers=self.num_hidden_layers,
        )
        if selected is None:
            raise RuntimeError(f"request {request.request_id!r} did not cross the PyramidKV compression threshold")
        return selected, retained_tokens

    def before_cache_write(
        self,
        *,
        layer: Any,
        backend: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: Any,
    ) -> bool:
        """Write compact prefill/decode K/V and suppress the default writer."""
        layer_name = layer.layer_name
        if layer_name not in self.layer_indices:
            raise RuntimeError(f"PyramidKV batch view has no layer index for {layer_name!r}")
        if self.block_size != SUPPORTED_BLOCK_SIZE:
            raise RuntimeError(f"PyramidKV attention view requires block_size 128, got {self.block_size}")
        if not self.requests:
            raise RuntimeError("PyramidKV attention view has no requests")
        if key is None or value is None:
            raise RuntimeError("PyramidKV cache write requires key and value tensors")
        if len(self.requests) != len(attn_metadata.seq_lens_list):
            raise RuntimeError(
                "PyramidKV attention view request count does not match attention "
                f"metadata: {len(self.requests)} vs "
                f"{len(attn_metadata.seq_lens_list)}"
            )
        expected_query_start = 0
        for request in self.requests:
            if request.query_start != expected_query_start:
                raise RuntimeError(
                    "PyramidKV attention request ranges must be contiguous and "
                    f"ordered; expected start {expected_query_start}, got "
                    f"{request.query_start}"
                )
            expected_query_start = request.query_end
        if expected_query_start != attn_metadata.num_actual_tokens:
            raise RuntimeError(
                "PyramidKV attention request ranges do not cover all actual "
                f"tokens: covered {expected_query_start}, expected "
                f"{attn_metadata.num_actual_tokens}"
            )

        phases = {request.is_prefill for request in self.requests}
        is_decode_only = phases == {False}
        state_name = getattr(attn_metadata.attn_state, "name", "")
        if is_decode_only:
            if state_name != "DecodeOnly":
                raise RuntimeError(
                    f"PyramidKV decode requires DecodeOnly attention state, got {attn_metadata.attn_state}"
                )
        elif state_name == "PrefillNoCache":
            if False in phases or any(request.num_computed_tokens != 0 for request in self.requests):
                raise RuntimeError("PyramidKV PrefillNoCache metadata cannot contain cached prefill or decode requests")
        elif state_name == "DecodeOnly":
            if any(request.query_length != 1 for request in self.requests):
                raise RuntimeError("PyramidKV DecodeOnly mixed metadata requires one token for every request")
        elif state_name not in {"ChunkedPrefill", "PrefillCacheHit"}:
            raise RuntimeError(
                "PyramidKV prefill requires PrefillNoCache, ChunkedPrefill, "
                "PrefillCacheHit, or one-token DecodeOnly metadata; got "
                f"{attn_metadata.attn_state}"
            )

        if is_decode_only:
            actual_tokens = attn_metadata.num_actual_tokens
            write_key = key if actual_tokens == key.shape[0] else key[:actual_tokens]
            write_value = value if actual_tokens == value.shape[0] else value[:actual_tokens]
            write_slots, decode_length_tensor = self._decode_tensors(layer_name, key.device)
            backend.do_kv_cache_update(
                layer,
                write_key,
                write_value,
                kv_cache,
                write_slots,
            )
            decode_lengths = self.decode_lengths_by_layer[layer_name]
            if any(request.compress for request in self.requests):
                self.completed_decode_layers.add(layer_name)
            attn_metadata.seq_lens_list = decode_lengths
            attn_metadata.seq_lens = decode_length_tensor
            attn_metadata.seq_lens_cpu = decode_length_tensor
            return True

        compact_keys: list[torch.Tensor] = []
        compact_values: list[torch.Tensor] = []
        slot_tensors: list[torch.Tensor] = []
        decode_lengths = list(attn_metadata.seq_lens_list)
        compact_before_attention = (
            not self.provider.prefix_caching
            and state_name == "PrefillNoCache"
            and all(request.is_complete_prefill for request in self.requests)
        )

        for request_index, request in enumerate(self.requests):
            if not 0 <= request.query_start < request.query_end <= key.shape[0]:
                raise RuntimeError(
                    f"request {request.request_id!r} has invalid query range "
                    f"[{request.query_start}, {request.query_end})"
                )
            request_key = key[request.query_start : request.query_end]
            request_value = value[request.query_start : request.query_end]
            if request.is_prefill and request.compress:
                request_state = self.provider.get_request_state(request.request_id)
                if request.num_prompt_tokens != request_state.semantic_num_tokens:
                    raise RuntimeError(
                        f"request {request.request_id!r} prompt length changed "
                        f"from {request_state.semantic_num_tokens} to "
                        f"{request.num_prompt_tokens}"
                    )
                request_query = query[request.query_start : request.query_end]
                if request.is_complete_prefill:
                    selection = select_pyramid_kv(
                        request_query.permute(1, 0, 2).unsqueeze(0),
                        request_key.permute(1, 0, 2).unsqueeze(0),
                        request_value.permute(1, 0, 2).unsqueeze(0),
                        self.provider.config,
                        layer_index=self.layer_indices[layer_name],
                        num_hidden_layers=self.num_hidden_layers,
                    )
                    if not selection.compressed:
                        raise RuntimeError(
                            f"request {request.request_id!r} did not cross the PyramidKV compression threshold"
                        )
                    selected = selection.selected_past_indices
                    retained_tokens = selection.retained_tokens
                    if selected is None:
                        raise RuntimeError(f"request {request.request_id!r} has no selected history indices")
                elif request.is_final_prefill_chunk:
                    if len(kv_cache) < 2:
                        raise RuntimeError("PyramidKV final chunk requires K/V cache tensors")
                    selected, retained_tokens = self._select_final_chunk(
                        request,
                        layer_name,
                        request_query,
                        request_key,
                        kv_cache[0],
                    )
                else:
                    self._stage_query_tail(request, layer_name, request_query)
                    selected = None
                    retained_tokens = 0

                if compact_before_attention:
                    if not request.is_complete_prefill or selected is None:
                        raise RuntimeError("PyramidKV can only compact complete uncached prefills before attention")
                    compact_key, compact_value = materialize_pyramid_kv(
                        request_key.permute(1, 0, 2).unsqueeze(0),
                        request_value.permute(1, 0, 2).unsqueeze(0),
                        selected,
                        retained_tokens,
                        self.provider.config.window_size,
                    )
                    compact_keys.append(compact_key.squeeze(0).permute(1, 0, 2))
                    compact_values.append(compact_value.squeeze(0).permute(1, 0, 2))
                    slot_tensors.append(
                        _slots_for_positions(
                            request.block_ids,
                            0,
                            retained_tokens,
                            self.block_size,
                            key.device,
                        )
                    )
                    self._record_prefill_length(request.request_id, layer_name, retained_tokens)
                else:
                    compact_keys.append(request_key)
                    compact_values.append(request_value)
                    slot_tensors.append(
                        attn_metadata.slot_mapping[request.query_start : request.query_end].to(dtype=torch.int32)
                    )
                    if request.is_final_prefill_chunk:
                        if selected is None:
                            raise RuntimeError(
                                f"request {request.request_id!r} final prefill has no selected history indices"
                            )
                        self.deferred_prefills.append(
                            PyramidKVDeferredPrefill(
                                request_id=request.request_id,
                                layer=layer,
                                backend=backend,
                                kv_cache=kv_cache,
                                block_ids=request.block_ids,
                                semantic_num_tokens=request.num_prompt_tokens,
                                retained_tokens=retained_tokens,
                                selected_past_indices=selected,
                                destination_block_ids=(request.destination_block_ids),
                            )
                        )
            elif not request.is_prefill and request.compress:
                if request.query_end - request.query_start != 1:
                    raise RuntimeError("PyramidKV supports exactly one decode token per request")
                request_state = self.provider.get_request_state(request.request_id)
                if not request_state.committed:
                    raise RuntimeError(f"request {request.request_id!r} has no core commit ack")
                if request.block_ids != request_state.expected_block_ids[0]:
                    raise RuntimeError(
                        f"request {request.request_id!r} decode block table does "
                        "not match the core commit ack: runner="
                        f"{request.block_ids}, provider="
                        f"{request_state.expected_block_ids[0]}"
                    )
                if request.semantic_num_tokens != request_state.semantic_num_tokens + 1:
                    raise RuntimeError(
                        f"request {request.request_id!r} decode semantic length "
                        f"must be {request_state.semantic_num_tokens + 1}, got "
                        f"{request.semantic_num_tokens}"
                    )
                try:
                    physical_tokens = request_state.layers[layer_name].physical_num_tokens
                except KeyError as error:
                    raise RuntimeError(
                        f"request {request.request_id!r} has no state for layer {layer_name!r}"
                    ) from error
                if not is_decode_only:
                    compact_keys.append(request_key)
                    compact_values.append(request_value)
                    slot_tensors.append(
                        _slots_for_positions(
                            request.block_ids,
                            physical_tokens,
                            1,
                            self.block_size,
                            key.device,
                        )
                    )
                decode_lengths[request_index] = physical_tokens + 1
            else:
                if not is_decode_only:
                    compact_keys.append(request_key)
                    compact_values.append(request_value)
                    slot_tensors.append(
                        attn_metadata.slot_mapping[request.query_start : request.query_end].to(dtype=torch.int32)
                    )

        if self.provider.prefix_caching and phases == {True}:
            # Preserve the ordinary full-KV writer for source blocks. Selection
            # and materialization remain deferred until every model layer has
            # completed successfully.
            return False

        write_key = torch.cat(compact_keys, dim=0)
        write_value = torch.cat(compact_values, dim=0)
        write_slots = torch.cat(slot_tensors, dim=0)
        backend.do_kv_cache_update(layer, write_key, write_value, kv_cache, write_slots)
        if False in phases:
            if any(request.compress and not request.is_prefill for request in self.requests):
                self.completed_decode_layers.add(layer_name)
            attn_metadata.seq_lens_list = decode_lengths
            attn_metadata.seq_lens = attn_metadata.seq_lens.new_tensor(decode_lengths)
            attn_metadata.seq_lens_cpu = attn_metadata.seq_lens
        return True


class PyramidKVAscendProvider:
    """Provider-owned compatibility, algorithm, and per-request state."""

    def __init__(self, config: PyramidKVAscendConfig) -> None:
        self.config = config
        self._requests: dict[str, PyramidKVRequestState] = {}
        self.prefix_caching = False

    @staticmethod
    def _layer_indices(layer_names: tuple[str, ...]) -> dict[str, int]:
        indices: dict[str, int] = {}
        for layer_name in layer_names:
            match = re.fullmatch(r"model\.layers\.(\d+)\.self_attn\.attn", layer_name)
            if match is None:
                raise RuntimeError(f"PyramidKV requires canonical Llama attention layer names, got {layer_name!r}")
            index = int(match.group(1))
            if index in indices.values():
                raise RuntimeError(f"duplicate PyramidKV layer index {index} in {layer_names}")
            indices[layer_name] = index
        expected = set(range(len(layer_names)))
        if set(indices.values()) != expected:
            raise RuntimeError(
                f"PyramidKV layer indices must be contiguous from zero, got {tuple(sorted(indices.values()))}"
            )
        return indices

    def build_attention_batch_view(
        self,
        *,
        request_ids: tuple[str, ...],
        query_lengths: tuple[int, ...],
        semantic_num_tokens: tuple[int, ...],
        num_computed_tokens: tuple[int, ...],
        num_prompt_tokens: tuple[int, ...],
        block_ids: tuple[tuple[tuple[int, ...], ...], ...],
        layer_names: tuple[str, ...],
        block_size: int,
        destination_block_ids: (dict[str, tuple[tuple[int, ...], ...]] | None) = None,
    ) -> PyramidKVAttentionBatchView:
        """Build a step-local view after runner state and commit-ack updates."""
        counts = {
            len(request_ids),
            len(query_lengths),
            len(semantic_num_tokens),
            len(num_computed_tokens),
            len(num_prompt_tokens),
            len(block_ids),
        }
        if counts != {len(request_ids)} or not request_ids:
            raise RuntimeError("PyramidKV runner batch fields must have the same non-zero request count")
        if block_size != SUPPORTED_BLOCK_SIZE:
            raise RuntimeError(f"PyramidKV requires block_size 128, got {block_size}")
        if len(layer_names) != 32:
            raise RuntimeError(f"PyramidKV requires 32 attention layers, got {len(layer_names)}")
        layer_indices = self._layer_indices(layer_names)
        destinations = destination_block_ids or {}
        unknown_destinations = set(destinations).difference(request_ids)
        if unknown_destinations:
            raise RuntimeError(
                f"PyramidKV destination tables contain unscheduled requests: {tuple(sorted(unknown_destinations))}"
            )

        phases = tuple(computed < prompt for computed, prompt in zip(num_computed_tokens, num_prompt_tokens))
        requests: list[PyramidKVAttentionRequest] = []
        query_start = 0
        for index, request_id in enumerate(request_ids):
            query_length = query_lengths[index]
            semantic_length = semantic_num_tokens[index]
            computed = num_computed_tokens[index]
            prompt = num_prompt_tokens[index]
            request_blocks = block_ids[index]
            request_destination = destinations.get(request_id)
            if query_length <= 0:
                raise RuntimeError(f"request {request_id!r} has no scheduled tokens")
            if len(request_blocks) != 1 or not request_blocks[0]:
                raise RuntimeError(f"request {request_id!r} requires one non-empty block table")

            is_prefill = phases[index]
            compress = False
            if is_prefill:
                if semantic_length != computed + query_length or semantic_length > prompt:
                    raise RuntimeError(
                        f"request {request_id!r} prefill must cover the exact "
                        f"range [{computed}, {computed + query_length}) within "
                        f"prompt length {prompt}, got semantic length "
                        f"{semantic_length}"
                    )
                compress = any(
                    self.config.retained_tokens(prompt, layer_index, len(layer_names)) < prompt
                    for layer_index in range(len(layer_names))
                )
            destination = None
            if request_destination is not None:
                if len(request_destination) != 1 or not request_destination[0]:
                    raise RuntimeError(f"request {request_id!r} requires one non-empty PyramidKV destination table")
                destination = request_destination[0]
                if set(destination).intersection(request_blocks[0]):
                    raise RuntimeError(f"request {request_id!r} source and destination block tables overlap")
                if not is_prefill or not compress:
                    raise RuntimeError(
                        f"request {request_id!r} received a destination table outside a compressing prefill"
                    )
            if compress:
                if self.prefix_caching and semantic_length == prompt:
                    if destination is None:
                        raise RuntimeError(
                            f"request {request_id!r} final prefill is missing its private destination table"
                        )
                    max_retained = max(
                        self.config.retained_tokens(prompt, layer_index, len(layer_names))
                        for layer_index in range(len(layer_names))
                    )
                    if len(destination) * block_size < max_retained:
                        raise RuntimeError(
                            f"request {request_id!r} private destination "
                            f"capacity {len(destination) * block_size} is "
                            f"smaller than required {max_retained}"
                        )
                elif destination is not None:
                    raise RuntimeError(
                        f"request {request_id!r} received a destination table outside a prefix-cached final prefill"
                    )
                request_state = self._requests.get(request_id)
                if request_state is None:
                    if computed % block_size != 0:
                        raise RuntimeError(
                            f"request {request_id!r} starts PyramidKV prefill at unaligned cached offset {computed}"
                        )
                    if prompt - computed < self.config.window_size:
                        raise RuntimeError(
                            f"request {request_id!r} cached suffix has "
                            f"{prompt - computed} tokens, fewer than query "
                            f"window {self.config.window_size}"
                        )
                    request_state = self.begin_request(
                        request_id,
                        prompt,
                        request_blocks,
                        prefill_num_computed_tokens=computed,
                    )
                if request_state.plan_emitted or request_state.committed:
                    raise RuntimeError(f"request {request_id!r} prefill state is sealed")
                if request_state.semantic_num_tokens != prompt:
                    raise RuntimeError(
                        f"request {request_id!r} prompt length changed "
                        f"from {request_state.semantic_num_tokens} to "
                        f"{prompt}"
                    )
                if request_state.prefill_num_computed_tokens != computed:
                    raise RuntimeError(
                        f"request {request_id!r} chunk starts at "
                        f"{computed}, expected "
                        f"{request_state.prefill_num_computed_tokens}"
                    )
                expected_blocks = request_state.expected_block_ids[0]
                actual_blocks = request_blocks[0]
                if (
                    len(actual_blocks) < len(expected_blocks)
                    or actual_blocks[: len(expected_blocks)] != expected_blocks
                ):
                    raise RuntimeError(
                        f"request {request_id!r} prefill block table is not "
                        "a monotonic extension: expected prefix "
                        f"{expected_blocks}, got {actual_blocks}"
                    )
                required_blocks = (semantic_length + block_size - 1) // block_size
                if len(actual_blocks) < required_blocks:
                    raise RuntimeError(
                        f"request {request_id!r} needs {required_blocks} "
                        f"blocks through token {semantic_length}, got "
                        f"{len(actual_blocks)}"
                    )
            elif not is_prefill:
                if query_length != 1 or semantic_length != computed + 1:
                    raise RuntimeError(f"request {request_id!r} must execute ordinary one-token decode")
                request_state = self._requests.get(request_id)
                if request_state is not None:
                    if not request_state.committed:
                        raise RuntimeError(f"request {request_id!r} is waiting for the core compression commit ack")
                    self.accept_decode_block_table(request_id, request_blocks)
                    compress = True

            query_end = query_start + query_length
            requests.append(
                PyramidKVAttentionRequest(
                    request_id=request_id,
                    query_start=query_start,
                    query_end=query_end,
                    semantic_num_tokens=semantic_length,
                    num_computed_tokens=computed,
                    num_prompt_tokens=prompt,
                    block_ids=request_blocks[0],
                    is_prefill=is_prefill,
                    destination_block_ids=destination,
                    compress=compress,
                )
            )
            query_start = query_end

        compressing_prefills = [request for request in requests if request.is_prefill and request.compress]
        if compressing_prefills:
            logger.info(
                "Prepared PyramidKV prefill step: chunks=%s prefix_hit_tokens=%s mixed_decode=%s",
                tuple(
                    (
                        request.request_id,
                        request.num_computed_tokens,
                        request.semantic_num_tokens,
                        request.num_prompt_tokens,
                        "final" if request.is_final_prefill_chunk else "intermediate",
                    )
                    for request in compressing_prefills
                ),
                tuple(
                    (
                        request.request_id,
                        self._requests[request.request_id].initial_prefill_num_computed_tokens,
                    )
                    for request in compressing_prefills
                ),
                any(not request.is_prefill for request in requests),
            )

        return PyramidKVAttentionBatchView(
            provider=self,
            requests=tuple(requests),
            layer_indices=layer_indices,
            num_hidden_layers=len(layer_names),
            block_size=block_size,
        )

    def finish_model_forward(
        self,
        view: PyramidKVAttentionBatchView,
        *,
        layer_names: tuple[str, ...],
        schema_version: int,
    ) -> list[KVCacheCompressionPlan] | None:
        """Commit successful prefill chunks or atomically advance decodes."""
        expected_layers = set(layer_names)
        deferred_layers: dict[str, set[str]] = {}
        for deferred_prefill in view.deferred_prefills:
            layers = deferred_layers.setdefault(deferred_prefill.request_id, set())
            layer_name = deferred_prefill.layer.layer_name
            if layer_name in layers:
                raise RuntimeError(
                    f"request {deferred_prefill.request_id!r} deferred layer {layer_name!r} was recorded twice"
                )
            layers.add(layer_name)

        # Validate the whole step before mutating persistent provider state or
        # compacting paged cache. The runner calls this only after model forward
        # succeeds, so a failed/partial forward leaves the prior chunk committed.
        for request in view.requests:
            if not request.compress:
                continue
            if request.is_prefill:
                query_layers = view.completed_prefill_query_layers.get(request.request_id, set())
                if request.is_complete_prefill:
                    completed = set(view.completed_prefill_lengths.get(request.request_id, {}))
                    recorded_deferred_layers = deferred_layers.get(request.request_id, set())
                    if completed != expected_layers and recorded_deferred_layers != expected_layers:
                        raise RuntimeError(
                            f"request {request.request_id!r} full prefill "
                            "completed/deferred layers "
                            f"{tuple(sorted(completed))}/"
                            f"{tuple(sorted(recorded_deferred_layers))}, "
                            "expected all 32 layers"
                        )
                else:
                    if query_layers != expected_layers:
                        raise RuntimeError(
                            f"request {request.request_id!r} chunk staged query "
                            f"layers {tuple(sorted(query_layers))}, expected all "
                            "32 layers"
                        )
                    if request.is_final_prefill_chunk:
                        completed = deferred_layers.get(request.request_id, set())
                        if completed != expected_layers:
                            raise RuntimeError(
                                f"request {request.request_id!r} final chunk "
                                f"deferred layers {tuple(sorted(completed))}, "
                                "expected all 32 layers"
                            )
            else:
                completed = view.completed_decode_layers
                if completed != expected_layers:
                    raise RuntimeError(
                        f"request {request.request_id!r} decode completed layers "
                        f"{tuple(sorted(completed))}, expected all 32 layers"
                    )
        self._compact_deferred_prefills(view)

        plans: list[KVCacheCompressionPlan] = []
        for request in view.requests:
            if not request.compress:
                continue
            if request.is_prefill:
                state = self.get_request_state(request.request_id)
                state.prefill_num_computed_tokens = request.semantic_num_tokens
                state.expected_block_ids = (request.block_ids,)
                if request.is_final_prefill_chunk:
                    completed_lengths = view.completed_prefill_lengths[request.request_id]
                    state.layers = {
                        layer_name: PyramidKVLayerState(physical_num_tokens=completed_lengths[layer_name])
                        for layer_name in layer_names
                    }
                    state.prefill_query_tail = None
                    state.prefill_query_tail_length = 0
                    plans.append(self.finalize_plan(request.request_id, layer_names, schema_version))
                    logger.info(
                        "Finalized PyramidKV prefill compression: request_id=%s "
                        "semantic_tokens=%d physical_tokens=%d "
                        "source_blocks=%d destination_blocks=%d",
                        request.request_id,
                        request.semantic_num_tokens,
                        plans[-1].physical_num_tokens,
                        len(request.block_ids),
                        len(request.destination_block_ids or request.block_ids),
                    )
                else:
                    state.prefill_query_tail = view.pending_query_tails[request.request_id]
                    state.prefill_query_tail_length = view.pending_query_tail_lengths[request.request_id]
                    logger.info(
                        "Committed PyramidKV intermediate prefill chunk: "
                        "request_id=%s chunk_end=%d prompt_tokens=%d "
                        "plan_emitted=false",
                        request.request_id,
                        request.semantic_num_tokens,
                        request.num_prompt_tokens,
                    )
            else:
                self.advance_decode(request.request_id, view.completed_decode_layers)
        return plans or None

    def _compact_deferred_prefills(self, view: PyramidKVAttentionBatchView) -> None:
        """Compact mixed-batch prefills after every attention layer succeeds."""
        for deferred in view.deferred_prefills:
            if len(deferred.kv_cache) < 2:
                raise RuntimeError("PyramidKV deferred compact requires K/V cache")
            key_cache, value_cache = deferred.kv_cache[:2]
            if key_cache.shape != value_cache.shape or key_cache.ndim != 4:
                raise RuntimeError(
                    "PyramidKV deferred compact requires matching [blocks, block, heads, dim] K/V caches"
                )
            full_key = _gather_paged_prefix(
                key_cache,
                deferred.block_ids,
                deferred.semantic_num_tokens,
                view.block_size,
            )
            full_value = _gather_paged_prefix(
                value_cache,
                deferred.block_ids,
                deferred.semantic_num_tokens,
                view.block_size,
            )
            selected = deferred.selected_past_indices.to(device=key_cache.device, dtype=torch.long)
            compact_key, compact_value = materialize_pyramid_kv(
                full_key.permute(1, 0, 2).unsqueeze(0),
                full_value.permute(1, 0, 2).unsqueeze(0),
                selected,
                deferred.retained_tokens,
                self.config.window_size,
            )
            compact_key = compact_key.squeeze(0).permute(1, 0, 2)
            compact_value = compact_value.squeeze(0).permute(1, 0, 2)
            deferred.backend.do_kv_cache_update(
                deferred.layer,
                compact_key,
                compact_value,
                deferred.kv_cache,
                _slots_for_positions(
                    (
                        deferred.destination_block_ids
                        if deferred.destination_block_ids is not None
                        else deferred.block_ids
                    ),
                    0,
                    deferred.retained_tokens,
                    view.block_size,
                    key_cache.device,
                ),
            )
            view._record_prefill_length(
                deferred.request_id,
                deferred.layer.layer_name,
                deferred.retained_tokens,
            )
            if deferred.destination_block_ids is not None:
                logger.debug(
                    "Materialized PyramidKV request %s layer %s from %d "
                    "source blocks into %d private destination blocks",
                    deferred.request_id,
                    deferred.layer.layer_name,
                    len(deferred.block_ids),
                    len(deferred.destination_block_ids),
                )
        view.deferred_prefills.clear()

    @classmethod
    def from_core_config(cls, config: KVCacheCompressionConfig) -> PyramidKVAscendProvider:
        return cls(PyramidKVAscendConfig.from_dict(config.provider_config))

    def compatibility_report(
        self,
        core_config: KVCacheCompressionConfig,
        context: PyramidKVCapabilityContext,
        provider_factory: str,
    ) -> KVCacheCompressionCompatibility:
        reasons = self._compatibility_reasons(context)
        runtime_spec = None
        if not reasons:
            max_physical_num_tokens = max(
                self.config.retained_tokens(
                    context.max_model_len,
                    layer_index,
                    context.num_hidden_layers,
                )
                for layer_index in range(context.num_hidden_layers)
            )
            runtime_spec = KVCacheCompressionRuntimeSpec(
                schema_version=core_config.schema_version,
                provider=core_config.provider,
                requires_private_destination=True,
                compression_threshold_tokens=(self.config.max_capacity_prompt),
                required_recompute_tokens=self.config.window_size,
                max_physical_num_tokens=max_physical_num_tokens,
            )
        return KVCacheCompressionCompatibility(
            schema_version=core_config.schema_version,
            provider=core_config.provider,
            supported=not reasons,
            reasons=reasons,
            platform=context.platform,
            provider_factory=provider_factory,
            backend=context.backend,
            model_architecture=context.model_architecture,
            dtype=context.dtype,
            cache_layout=context.cache_layout,
            block_size=context.block_size,
            runtime_spec=runtime_spec,
        )

    def validate_worker(
        self,
        core_config: KVCacheCompressionConfig,
        worker: Any,
        provider_factory: str,
    ) -> KVCacheCompressionCompatibility:
        """Inspect the initialized worker without allocating a KV tensor."""
        context = _capability_context_from_worker(worker)
        report = self.compatibility_report(core_config, context, provider_factory)
        if report.supported:
            self.prefix_caching = context.prefix_caching
        return report

    @staticmethod
    def _compatibility_reasons(
        context: PyramidKVCapabilityContext,
    ) -> tuple[str, ...]:
        reasons: list[str] = []

        def require(condition: bool, message: str) -> None:
            if not condition:
                reasons.append(message)

        require(context.platform == "npu", f"platform must be 'npu', got {context.platform!r}")
        require(
            context.device_name == SUPPORTED_DEVICE_NAME,
            f"device must be {SUPPORTED_DEVICE_NAME}, got {context.device_name!r}",
        )
        eager_mode = context.enforce_eager and context.cudagraph_mode == "NONE"
        graph_mode = not context.enforce_eager and context.cudagraph_mode in {"PIECEWISE", "FULL_DECODE_ONLY"}
        require(
            eager_mode or graph_mode,
            "execution must be eager/NONE, PIECEWISE, or FULL_DECODE_ONLY "
            "without an eager/graph configuration conflict; got "
            f"enforce_eager={context.enforce_eager}, "
            f"cudagraph_mode={context.cudagraph_mode!r}",
        )
        if graph_mode:
            require(
                context.cann_version.startswith("9.0"),
                f"graph execution requires CANN 9.0, got {context.cann_version!r}",
            )
            require(
                not context.pa_shape_list,
                "graph execution currently requires default FIA with an empty "
                f"pa_shape_list, got {context.pa_shape_list}",
            )
        else:
            require(
                context.cann_version.startswith(SUPPORTED_CANN_VERSION_PREFIXES),
                f"CANN version must start with one of {SUPPORTED_CANN_VERSION_PREFIXES}, got {context.cann_version!r}",
            )
        require(not context.use_v2_model_runner, "V2 model runner is unsupported")
        require(
            context.backend == SUPPORTED_BACKEND,
            f"attention backend must be {SUPPORTED_BACKEND}, got {context.backend!r}",
        )
        require(
            context.model_architecture == SUPPORTED_MODEL_ARCHITECTURE,
            f"model architecture must be {SUPPORTED_MODEL_ARCHITECTURE}, got {context.model_architecture!r}",
        )
        require(
            context.dtype in {"bfloat16", "torch.bfloat16"},
            f"model dtype must be bfloat16, got {context.dtype!r}",
        )
        require(
            context.quantization is None,
            f"model quantization is unsupported, got {context.quantization!r}",
        )
        require(
            context.num_attention_heads == 32,
            f"query heads must be 32, got {context.num_attention_heads}",
        )
        require(
            context.num_kv_heads == 8,
            f"KV heads must be 8, got {context.num_kv_heads}",
        )
        require(
            context.head_dim == 128,
            f"head_dim must be 128, got {context.head_dim}",
        )
        require(
            context.num_hidden_layers == 32,
            f"hidden layers must be 32, got {context.num_hidden_layers}",
        )
        require(
            context.cache_layout == SUPPORTED_CACHE_LAYOUT,
            f"cache layout must be {SUPPORTED_CACHE_LAYOUT}, got {context.cache_layout!r}",
        )
        require(
            context.block_size == SUPPORTED_BLOCK_SIZE,
            f"block_size must be {SUPPORTED_BLOCK_SIZE}, got {context.block_size}",
        )
        require(
            context.num_kv_cache_groups == 1,
            f"exactly one KV cache group is required, got {context.num_kv_cache_groups}",
        )
        require(
            context.full_attention_only,
            "all KV cache layers must use full attention",
        )
        if context.prefix_caching:
            require(
                context.hash_block_size == context.block_size == SUPPORTED_BLOCK_SIZE,
                "prefix caching requires hash_block_size == block_size == "
                f"{SUPPORTED_BLOCK_SIZE}, got hash={context.hash_block_size} "
                f"and block={context.block_size}",
            )
        if context.chunked_prefill:
            require(
                context.cann_version.startswith("9.0"),
                f"chunked prefill requires CANN 9.0, got {context.cann_version!r}",
            )
            require(
                not context.pa_shape_list,
                "chunked prefill currently requires default FIA with an "
                f"empty pa_shape_list, got {context.pa_shape_list}",
            )
        require(
            not context.sliding_window,
            "sliding window/chunked attention is unsupported",
        )
        require(not context.speculative_decoding, "speculative decoding is unsupported")
        require(not context.kv_transfer, "KV transfer is unsupported")
        require(not context.kv_offload, "KV offload is unsupported")
        require(
            context.cache_dtype in {"auto", "bfloat16", "torch.bfloat16"},
            f"KV cache dtype must resolve to bfloat16, got {context.cache_dtype!r}",
        )
        require(
            context.tensor_parallel_size == 1,
            f"tensor parallel size must be 1, got {context.tensor_parallel_size}",
        )
        require(
            context.pipeline_parallel_size == 1,
            f"pipeline parallel size must be 1, got {context.pipeline_parallel_size}",
        )
        require(
            context.prefill_context_parallel_size == 1,
            f"prefill context parallel size must be 1, got {context.prefill_context_parallel_size}",
        )
        require(
            context.decode_context_parallel_size == 1,
            f"decode context parallel size must be 1, got {context.decode_context_parallel_size}",
        )
        require(not context.async_scheduling, "async scheduling is unsupported")
        require(not context.balance_scheduling, "balance scheduling is unsupported")
        require(not context.dbo_enabled, "dual-batch overlap is unsupported")
        require(not context.knorm_enabled, "VLLM_KNORM_ENABLED must be 0")
        reasons.extend(f"required NPU op is unavailable: {op}" for op in context.missing_ops)
        return tuple(reasons)

    def begin_request(
        self,
        request_id: str,
        semantic_num_tokens: int,
        expected_block_ids: tuple[tuple[int, ...], ...],
        prefill_num_computed_tokens: int = 0,
    ) -> PyramidKVRequestState:
        if request_id in self._requests:
            raise RuntimeError(f"request {request_id!r} already has PyramidKV state")
        if semantic_num_tokens <= 0:
            raise ValueError("semantic_num_tokens must be positive")
        if len(expected_block_ids) != 1 or not expected_block_ids[0]:
            raise ValueError("PyramidKV requires one non-empty block table")
        if prefill_num_computed_tokens < 0:
            raise ValueError("prefill_num_computed_tokens must be non-negative")
        state = PyramidKVRequestState(
            request_id=request_id,
            semantic_num_tokens=semantic_num_tokens,
            expected_block_ids=expected_block_ids,
            prefill_num_computed_tokens=prefill_num_computed_tokens,
            initial_prefill_num_computed_tokens=(prefill_num_computed_tokens),
        )
        self._requests[request_id] = state
        return state

    def record_prefill_layer(
        self,
        request_id: str,
        layer_name: str,
        selection: PyramidKVSelection,
    ) -> None:
        state = self._requests[request_id]
        if state.plan_emitted or state.committed:
            raise RuntimeError(f"request {request_id!r} prefill state is sealed")
        if layer_name in state.layers:
            raise RuntimeError(f"request {request_id!r} layer {layer_name!r} was recorded twice")
        if not selection.compressed or selection.selected_past_indices is None:
            raise ValueError("only compressed layer selections can produce a plan")
        state.layers[layer_name] = PyramidKVLayerState(
            physical_num_tokens=selection.retained_tokens,
        )

    def finalize_plan(
        self,
        request_id: str,
        expected_layer_names: tuple[str, ...],
        schema_version: int,
    ) -> KVCacheCompressionPlan:
        state = self._requests[request_id]
        if state.plan_emitted:
            raise RuntimeError(f"request {request_id!r} plan was already emitted")
        if set(state.layers) != set(expected_layer_names):
            raise RuntimeError(
                f"request {request_id!r} layers are incomplete: expected "
                f"{expected_layer_names}, got {tuple(state.layers)}"
            )
        per_layer = tuple((name, state.layers[name].physical_num_tokens) for name in expected_layer_names)
        plan = KVCacheCompressionPlan(
            schema_version=schema_version,
            provider=PYRAMIDKV_ASCEND_PROVIDER,
            request_id=request_id,
            semantic_num_tokens=state.semantic_num_tokens,
            physical_num_tokens=max(length for _, length in per_layer),
            per_layer_physical_num_tokens=per_layer,
            expected_block_ids=state.expected_block_ids,
        )
        state.plan_emitted = True
        return plan

    def mark_committed(
        self,
        request_id: str,
        block_ids: tuple[tuple[int, ...], ...],
    ) -> None:
        state = self._requests[request_id]
        if not state.plan_emitted or state.committed:
            raise RuntimeError(f"request {request_id!r} has no pending PyramidKV commit")
        if len(block_ids) != 1 or not block_ids[0]:
            raise ValueError("PyramidKV commit requires one non-empty block table")
        state.expected_block_ids = block_ids
        state.committed = True

    def accept_decode_block_table(
        self,
        request_id: str,
        block_ids: tuple[tuple[int, ...], ...],
    ) -> None:
        """Accept only the core allocation needed by the next decode token."""
        state = self._requests[request_id]
        if not state.committed:
            raise RuntimeError(f"request {request_id!r} is not committed")
        if len(block_ids) != 1 or not block_ids[0]:
            raise ValueError("PyramidKV decode requires one non-empty block table")
        expected = state.expected_block_ids[0]
        actual = block_ids[0]
        next_physical_tokens = max(layer.physical_num_tokens for layer in state.layers.values()) + 1
        required_blocks = (next_physical_tokens + SUPPORTED_BLOCK_SIZE - 1) // SUPPORTED_BLOCK_SIZE
        if len(actual) != required_blocks or len(actual) < len(expected) or actual[: len(expected)] != expected:
            raise RuntimeError(
                f"request {request_id!r} decode block table is not the "
                "required monotonic extension: expected prefix "
                f"{expected}, required blocks {required_blocks}, got {actual}"
            )
        state.expected_block_ids = block_ids

    def advance_decode(self, request_id: str, completed_layers: set[str]) -> None:
        state = self._requests[request_id]
        if not state.committed:
            raise RuntimeError(f"request {request_id!r} is not committed")
        if completed_layers != set(state.layers):
            raise RuntimeError(f"request {request_id!r} decode did not complete every layer")
        for layer in state.layers.values():
            layer.physical_num_tokens += 1
        state.semantic_num_tokens += 1

    def get_request_state(self, request_id: str) -> PyramidKVRequestState:
        return self._requests[request_id]

    def cleanup_request(self, request_id: str) -> None:
        self._requests.pop(request_id, None)


def _capability_context_from_worker(worker: Any) -> PyramidKVCapabilityContext:
    """Collect first-release capability facts from an initialized NPU worker."""
    import torch_npu
    import vllm.envs as envs_vllm
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    model_config = worker.vllm_config.model_config
    hf_config = model_config.hf_text_config
    cache_config = worker.vllm_config.cache_config
    scheduler_config = worker.vllm_config.scheduler_config
    parallel_config = worker.vllm_config.parallel_config
    specs = worker.get_kv_cache_spec()
    spec_values = list(specs.values())
    full_attention_only = bool(spec_values) and all(isinstance(spec, FullAttentionSpec) for spec in spec_values)

    signatures = {
        (
            type(spec).__name__,
            getattr(spec, "block_size", None),
            getattr(spec, "num_kv_heads", None),
            getattr(spec, "head_size", None),
            str(getattr(spec, "dtype", None)),
            getattr(spec, "sliding_window", None),
            getattr(spec, "attention_chunk_size", None),
        )
        for spec in spec_values
    }
    block_sizes = {getattr(spec, "block_size", -1) for spec in spec_values}
    block_size = next(iter(block_sizes)) if len(block_sizes) == 1 else -1
    standard_layout = (
        full_attention_only
        and len(signatures) == 1
        and all(str(spec.dtype) == "torch.bfloat16" for spec in spec_values)
    )
    sliding_window = any(
        getattr(spec, "sliding_window", None) is not None or getattr(spec, "attention_chunk_size", None) is not None
        for spec in spec_values
    )

    architectures = getattr(model_config, "architectures", ()) or ()
    architecture = architectures[0] if architectures else ""
    num_attention_heads = int(getattr(hf_config, "num_attention_heads", 0))
    head_dim = int(
        getattr(hf_config, "head_dim", 0)
        or (getattr(hf_config, "hidden_size", 0) // num_attention_heads if num_attention_heads else 0)
    )
    backend_type = type(worker.model_runner.attn_backend)
    backend = (
        worker.model_runner.attn_backend.__name__
        if isinstance(worker.model_runner.attn_backend, type)
        else backend_type.__name__
    )
    missing_ops = tuple(
        name
        for name in (
            "_npu_reshape_and_cache",
            "npu_fused_infer_attention_score",
        )
        if not hasattr(torch_npu, name)
    )
    compilation_config = worker.vllm_config.compilation_config
    configured_cudagraph_mode = getattr(compilation_config, "cudagraph_mode", "")
    cudagraph_mode = str(getattr(configured_cudagraph_mode, "name", configured_cudagraph_mode))

    return PyramidKVCapabilityContext(
        platform=worker.current_platform.device_type,
        device_name=torch.npu.get_device_name(worker.local_rank),
        cann_version=str(torch.version.cann or ""),
        use_v2_model_runner=bool(worker.use_v2_model_runner),
        enforce_eager=bool(model_config.enforce_eager),
        cudagraph_mode=cudagraph_mode,
        pa_shape_list=tuple(worker.model_runner.ascend_config.pa_shape_list),
        backend=backend,
        model_architecture=architecture,
        dtype=str(model_config.dtype),
        quantization=getattr(model_config, "quantization", None),
        num_attention_heads=num_attention_heads,
        num_kv_heads=int(getattr(hf_config, "num_key_value_heads", 0)),
        head_dim=head_dim,
        num_hidden_layers=int(getattr(hf_config, "num_hidden_layers", 0)),
        cache_layout=(SUPPORTED_CACHE_LAYOUT if standard_layout else "unsupported"),
        block_size=block_size,
        hash_block_size=int(cache_config.hash_block_size or cache_config.block_size),
        max_model_len=int(model_config.max_model_len),
        num_kv_cache_groups=len(signatures),
        full_attention_only=full_attention_only,
        prefix_caching=bool(cache_config.enable_prefix_caching),
        chunked_prefill=bool(scheduler_config.enable_chunked_prefill),
        sliding_window=sliding_window,
        speculative_decoding=worker.vllm_config.speculative_config is not None,
        kv_transfer=worker.vllm_config.kv_transfer_config is not None,
        kv_offload=getattr(cache_config, "kv_offloading_size", None) is not None,
        cache_dtype=str(worker.cache_dtype),
        tensor_parallel_size=parallel_config.tensor_parallel_size,
        pipeline_parallel_size=parallel_config.pipeline_parallel_size,
        prefill_context_parallel_size=(parallel_config.prefill_context_parallel_size),
        decode_context_parallel_size=parallel_config.decode_context_parallel_size,
        async_scheduling=bool(scheduler_config.async_scheduling),
        balance_scheduling=bool(worker.model_runner.ascend_config.enable_balance_scheduling),
        dbo_enabled=bool(parallel_config.enable_dbo),
        knorm_enabled=bool(envs_vllm.VLLM_KNORM_ENABLED),
        missing_ops=missing_ops,
    )
