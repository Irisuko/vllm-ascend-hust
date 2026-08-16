# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Fail-closed LMCache CacheBlend compatibility detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm.logger import logger

LMCache_LOCAL_BLEND = "lmcache_local_blend"
KV_TRANSFER_NONE = "none"
KV_TRANSFER_UNSUPPORTED = "unsupported"

_CONNECTOR = "LMCacheAscendConnectorV1Dynamic"
_CONNECTOR_MODULE = "lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"


@dataclass(frozen=True)
class LMCacheCompatibility:
    mode: str
    connector: str | None
    role: str | None
    use_layerwise: bool
    enable_blending: bool
    reason: str | None = None


def _load_lmcache_config() -> Any:
    from lmcache.integration.vllm.utils import lmcache_get_or_create_config

    return lmcache_get_or_create_config()


def classify_lmcache_compatibility(vllm_config: Any) -> LMCacheCompatibility:
    """Classify KV transfer without initializing the connector."""
    transfer = vllm_config.kv_transfer_config
    if transfer is None:
        return LMCacheCompatibility(
            mode=KV_TRANSFER_NONE,
            connector=None,
            role=None,
            use_layerwise=False,
            enable_blending=False,
        )

    connector = transfer.kv_connector
    role = transfer.kv_role
    module = transfer.kv_connector_module_path
    if connector != _CONNECTOR or module != _CONNECTOR_MODULE:
        return LMCacheCompatibility(
            mode=KV_TRANSFER_UNSUPPORTED,
            connector=connector,
            role=role,
            use_layerwise=False,
            enable_blending=False,
            reason=(f"only LMCacheAscendConnectorV1Dynamic from {_CONNECTOR_MODULE!r} is supported"),
        )
    if role != "kv_both":
        return LMCacheCompatibility(
            mode=KV_TRANSFER_UNSUPPORTED,
            connector=connector,
            role=role,
            use_layerwise=False,
            enable_blending=False,
            reason=f"LMCache CacheBlend requires kv_role='kv_both', got {role!r}",
        )

    try:
        lmcache_config = _load_lmcache_config()
    except Exception as error:
        return LMCacheCompatibility(
            mode=KV_TRANSFER_UNSUPPORTED,
            connector=connector,
            role=role,
            use_layerwise=False,
            enable_blending=False,
            reason=(f"LMCache configuration could not be loaded: {type(error).__name__}: {error}"),
        )

    use_layerwise = bool(getattr(lmcache_config, "use_layerwise", False))
    enable_blending = bool(getattr(lmcache_config, "enable_blending", False))
    if not (use_layerwise and enable_blending):
        return LMCacheCompatibility(
            mode=KV_TRANSFER_UNSUPPORTED,
            connector=connector,
            role=role,
            use_layerwise=use_layerwise,
            enable_blending=enable_blending,
            reason=("LMCache CacheBlend requires use_layerwise=true and enable_blending=true"),
        )
    return LMCacheCompatibility(
        mode=LMCache_LOCAL_BLEND,
        connector=connector,
        role=role,
        use_layerwise=True,
        enable_blending=True,
    )


def _register_model(model: Any) -> None:
    from lmcache.integration.vllm.utils import ENGINE_NAME
    from lmcache.v1.compute.models.utils import VLLMModelTracker

    VLLMModelTracker.register_model(ENGINE_NAME, model)


def register_lmcache_blend_model(vllm_config: Any, model: Any) -> bool:
    """Register a loaded model before the LMCache connector is initialized."""
    compatibility = classify_lmcache_compatibility(vllm_config)
    if compatibility.mode != LMCache_LOCAL_BLEND:
        return False

    _register_model(model)
    logger.info(
        "Registered LMCache CacheBlend model before connector initialization: "
        "connector=%s role=%s use_layerwise=%s enable_blending=%s",
        compatibility.connector,
        compatibility.role,
        compatibility.use_layerwise,
        compatibility.enable_blending,
    )
    return True
