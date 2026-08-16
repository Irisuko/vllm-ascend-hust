# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import pytest

from vllm_ascend.kv_cache_compression import lmcache_compat


def _vllm_config(
    *,
    connector="LMCacheAscendConnectorV1Dynamic",
    role="kv_both",
    module=("lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"),
):
    transfer = SimpleNamespace(
        kv_connector=connector,
        kv_role=role,
        kv_connector_module_path=module,
    )
    return SimpleNamespace(kv_transfer_config=transfer)


@pytest.mark.parametrize(
    ("connector", "role", "module", "reason"),
    [
        ("OtherConnector", "kv_both", "other.module", "only LMCache"),
        (
            "LMCacheAscendConnectorV1Dynamic",
            "kv_consumer",
            "lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1",
            "kv_role='kv_both'",
        ),
    ],
)
def test_non_local_blend_transfer_modes_fail_closed(
    connector: str,
    role: str,
    module: str,
    reason: str,
) -> None:
    result = lmcache_compat.classify_lmcache_compatibility(_vllm_config(connector=connector, role=role, module=module))

    assert result.mode == "unsupported"
    assert result.reason is not None and reason in result.reason


@pytest.mark.parametrize(
    ("use_layerwise", "enable_blending", "expected"),
    [
        (True, True, "lmcache_local_blend"),
        (False, True, "unsupported"),
        (True, False, "unsupported"),
    ],
)
def test_lmcache_config_controls_local_blend_mode(
    monkeypatch,
    use_layerwise: bool,
    enable_blending: bool,
    expected: str,
) -> None:
    monkeypatch.setattr(
        lmcache_compat,
        "_load_lmcache_config",
        lambda: SimpleNamespace(
            use_layerwise=use_layerwise,
            enable_blending=enable_blending,
        ),
    )

    result = lmcache_compat.classify_lmcache_compatibility(_vllm_config())

    assert result.mode == expected


def test_model_is_registered_only_for_local_blend(monkeypatch) -> None:
    model = object()
    registered = []
    compatibility = lmcache_compat.LMCacheCompatibility(
        mode="lmcache_local_blend",
        connector="LMCacheAscendConnectorV1Dynamic",
        role="kv_both",
        use_layerwise=True,
        enable_blending=True,
    )
    monkeypatch.setattr(
        lmcache_compat,
        "classify_lmcache_compatibility",
        lambda config: compatibility,
    )
    monkeypatch.setattr(
        lmcache_compat,
        "_register_model",
        registered.append,
    )

    assert lmcache_compat.register_lmcache_blend_model(SimpleNamespace(), model)
    assert registered == [model]

    monkeypatch.setattr(
        lmcache_compat,
        "classify_lmcache_compatibility",
        lambda config: SimpleNamespace(mode="none"),
    )

    assert not lmcache_compat.register_lmcache_blend_model(SimpleNamespace(), model)
    assert registered == [model]
