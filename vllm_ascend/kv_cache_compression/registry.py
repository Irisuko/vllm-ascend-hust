# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Lazy registry for Ascend KV cache compression providers."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from vllm.v1.kv_cache_compression import KV_CACHE_COMPRESSION_SCHEMA_VERSION

if TYPE_CHECKING:
    from vllm.config import KVCacheCompressionConfig

_PYRAMIDKV_PROVIDER = "pyramidkv_ascend"


def get_kv_cache_compression_provider(
    config: "KVCacheCompressionConfig",
) -> Any:
    """Resolve and construct a provider only for a valid enabled config."""
    if config.schema_version != KV_CACHE_COMPRESSION_SCHEMA_VERSION:
        raise ValueError(
            "unsupported KV cache compression schema_version "
            f"{config.schema_version}; expected "
            f"{KV_CACHE_COMPRESSION_SCHEMA_VERSION}"
        )
    if config.provider != _PYRAMIDKV_PROVIDER:
        raise ValueError(
            f"unknown Ascend KV cache compression provider {config.provider!r}; expected {_PYRAMIDKV_PROVIDER!r}"
        )

    module = import_module("vllm_ascend.kv_cache_compression.pyramidkv")
    return module.PyramidKVAscendProvider.from_core_config(config)
