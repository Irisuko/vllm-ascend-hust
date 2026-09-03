# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, MambaSpec
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_ascend.worker.hybrid_cache import allocate_native_hybrid_cache


def fixture_config():
    # Two aliased groups, three block IDs, two layer pools. Mixed dtypes make
    # byte arithmetic failures visible instead of masking them with FP16 only.
    attn = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=4, dtype=torch.float16, page_size_padded=72)
    mamba = MambaSpec(block_size=4, shapes=((4,), (8,)), dtypes=(torch.float16, torch.float32), page_size_padded=72)
    groups = [KVCacheGroupSpec(["a0", "a1"], attn), KVCacheGroupSpec(["m0", "m1"], mamba)]
    tensors = [KVCacheTensor(size=432, layers=g.layer_names, layer_stride=216, block_stride=72) for g in groups]
    return KVCacheConfig(num_blocks=3, kv_cache_tensors=tensors, kv_cache_groups=groups)


def test_native_hybrid_planes_preserve_aliases_and_block_isolation():
    config = fixture_config()
    caches = allocate_native_hybrid_cache(config, torch.device("cpu"), KVCacheLayout.LBHNC, [2, 4])
    key, value = caches["a0"]
    conv, ssm = caches["m0"]
    assert key.shape == (6, 2, 1, 4)
    assert conv.shape == (3, 4) and ssm.shape == (3, 8)
    assert all(t.is_contiguous() for views in caches.values() for t in views)
    assert len({t.untyped_storage().data_ptr() for views in caches.values() for t in views}) == 1
    # Mamba block 1 is live while full attention owns blocks 0 and 2.
    conv[1].fill_(3)
    ssm[1].fill_(4)
    key[:2].fill_(5)
    value[:2].fill_(6)
    key[4:].fill_(7)
    value[4:].fill_(8)
    assert torch.all(conv[1] == 3) and torch.all(ssm[1] == 4)
    # Same block intentionally aliases K/SSM, but not the conv or V planes.
    key[2:4].zero_()
    assert torch.all(ssm[1] == 0) and torch.all(conv[1] == 3)
    assert all(torch.count_nonzero(t) == 0 for t in caches["a1"] + caches["m1"])


@pytest.mark.parametrize("problem", ["layout", "stride", "bounds", "planes"])
def test_invalid_pool_contract_fails_before_allocation(problem, monkeypatch):
    config = fixture_config()
    layout = KVCacheLayout.LBHNC
    if problem == "layout":
        layout = KVCacheLayout.BLHNC
    elif problem == "stride":
        config.kv_cache_tensors[0].block_stride += 1
    elif problem == "bounds":
        config.kv_cache_tensors[0].offset = 432
    else:
        group = config.kv_cache_groups[1]
        group.kv_cache_spec = replace(group.kv_cache_spec, shapes=((8,), (8,)))
    monkeypatch.setattr(torch, "zeros", lambda *a, **k: pytest.fail("invalid descriptor allocated memory"))
    with pytest.raises(ValueError):
        allocate_native_hybrid_cache(config, torch.device("cpu"), layout, [2, 4])
