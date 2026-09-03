"""Compatibility wrapper for vLLM's KV-cache binding transition.

Current vLLM provides ``AttentionLayerBase.bind_kv_cache`` and its core
``bind_kv_cache`` helper handles standardized strided cache views. Older
vLLM-Ascend releases replaced that helper because the layer method used to be
abstract. Keeping the replacement on current core drops cache-group metadata
and bypasses the layer-specific binding contract, so this module intentionally
does not monkey-patch the upstream helper anymore. The exported wrapper remains
for the v2 adaptor, which imports it as an explicit dependency.
"""

from collections.abc import Sequence

import torch
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.v1.kv_cache_interface import KVCacheGroupSpec
from vllm.v1.worker.utils import bind_kv_cache as _core_bind_kv_cache

_core_bind_mamba_cache = MambaBase.bind_kv_cache


def bind_mamba_cache(self, kv_cache: torch.Tensor | tuple[torch.Tensor, ...]) -> None:
    """Accept native dense state planes, retaining core binding for raw pages.

    Ascend's layer-compact hybrid allocator has already unpacked each state
    into a contiguous plane. Sending those views through the core's raw-page
    unpacker would reinterpret the bytes a second time.
    """
    if not isinstance(kv_cache, tuple):
        return _core_bind_mamba_cache(self, kv_cache)
    shapes = tuple(self.get_state_shape())
    dtypes = tuple(self.get_state_dtype())
    if len(kv_cache) != len(shapes) or len(shapes) != len(dtypes):
        raise ValueError("Native Mamba cache state count does not match the layer")
    block_counts = set()
    for state, shape, dtype in zip(kv_cache, shapes, dtypes):
        if state.shape[1:] != shape or state.dtype != dtype or not state.is_contiguous():
            raise ValueError("Native Mamba cache state shape, dtype or contiguity is invalid")
        block_counts.add(state.shape[0])
    if len(block_counts) != 1:
        raise ValueError("Native Mamba cache states disagree on block count")
    self.kv_cache = kv_cache


MambaBase.bind_kv_cache = bind_mamba_cache


def bind_kv_cache(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, Attention],
    runner_kv_caches: list[torch.Tensor],
    num_attn_module: int = 1,
    kv_cache_groups: Sequence[KVCacheGroupSpec] | None = None,
) -> None:
    """Delegate to the current core binding contract without replacing it."""
    _core_bind_kv_cache(
        kv_caches,
        forward_context,
        runner_kv_caches,
        num_attn_module,
        kv_cache_groups,
    )
