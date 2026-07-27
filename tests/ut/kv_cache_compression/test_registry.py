# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib
import sys
from types import SimpleNamespace

import pytest
from vllm.config import KVCacheCompressionConfig

from vllm_ascend.kv_cache_compression import registry
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.worker import worker as worker_module
from vllm_ascend.worker.worker import NPUWorker

PROVIDER_MODULE = "vllm_ascend.kv_cache_compression.pyramidkv"


def test_package_and_platform_factory_are_lazy() -> None:
    sys.modules.pop(PROVIDER_MODULE, None)
    importlib.reload(registry)

    assert NPUPlatform.get_kv_cache_compression_provider_factory() == (
        "vllm_ascend.kv_cache_compression.registry:get_kv_cache_compression_provider"
    )
    assert PROVIDER_MODULE not in sys.modules


def test_unknown_provider_does_not_import_implementation() -> None:
    sys.modules.pop(PROVIDER_MODULE, None)
    config = KVCacheCompressionConfig(provider="unknown", provider_config={})

    with pytest.raises(ValueError, match="unknown Ascend"):
        registry.get_kv_cache_compression_provider(config)

    assert PROVIDER_MODULE not in sys.modules


def test_wrong_schema_does_not_import_implementation() -> None:
    sys.modules.pop(PROVIDER_MODULE, None)
    config = SimpleNamespace(
        schema_version=2,
        provider="pyramidkv_ascend",
        provider_config={},
    )

    with pytest.raises(ValueError, match="schema_version 2"):
        registry.get_kv_cache_compression_provider(config)

    assert PROVIDER_MODULE not in sys.modules


def test_valid_provider_is_imported_only_when_resolved() -> None:
    sys.modules.pop(PROVIDER_MODULE, None)
    config = KVCacheCompressionConfig(
        provider="pyramidkv_ascend",
        provider_config={"max_capacity_prompt": 512},
    )

    provider = registry.get_kv_cache_compression_provider(config)

    assert provider.config.max_capacity_prompt == 512
    assert PROVIDER_MODULE in sys.modules


def test_provider_config_errors_are_explicit() -> None:
    config = KVCacheCompressionConfig(
        provider="pyramidkv_ascend",
        provider_config={"unknown": 1},
    )

    with pytest.raises(ValueError, match="unknown PyramidKV.*unknown"):
        registry.get_kv_cache_compression_provider(config)


def test_worker_reports_missing_provider_module_before_cache_allocation(
    monkeypatch,
) -> None:
    config = KVCacheCompressionConfig(provider="pyramidkv_ascend", provider_config={})
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(kv_cache_compression_config=config),
        current_platform=SimpleNamespace(
            device_type="npu",
            get_kv_cache_compression_provider_factory=lambda: ("missing.provider:get_provider"),
        ),
    )

    def missing_module(_name):
        raise ModuleNotFoundError("No module named 'missing.provider'")

    monkeypatch.setattr(worker_module, "import_module", missing_module)
    report = NPUWorker.validate_kv_cache_compression(worker)

    assert not report.supported
    assert "provider initialization failed" in report.reasons[0]
    assert "missing.provider" in report.reasons[0]
